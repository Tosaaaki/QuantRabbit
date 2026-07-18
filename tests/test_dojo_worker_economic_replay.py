from __future__ import annotations

import gzip
import hashlib
import io
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant_rabbit.dojo_worker_economic_replay import (
    ACTION_POSITION,
    DojoWorkerEconomicReplayError,
    canonical_sha256,
    replay_worker_economics,
    seal_action_intents,
)


UTC = timezone.utc
CANDIDATE = "worker_candidate"
OWNER = "worker:worker_candidate"


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def _ohlc(o: str, h: str, low: str, c: str) -> dict[str, str]:
    return {"o": o, "h": h, "l": low, "c": c}


def _row(
    stamp: str,
    *,
    bid: dict[str, str],
    ask: dict[str, str],
) -> dict[str, object]:
    return {
        "ask": ask,
        "bid": bid,
        "complete": True,
        "granularity": "M1",
        "pair": "USD_JPY",
        "price": "BA",
        "time": stamp,
        "volume": 10,
    }


def _write_corpus(
    tmp_path: Path, rows: list[dict[str, object]]
) -> tuple[Path, dict[str, object], Path]:
    root = tmp_path / "corpus"
    shard = root / "day-001" / "USD_JPY" / "quotes.jsonl.gz"
    shard.parent.mkdir(parents=True)
    plain = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
        for row in rows
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(plain)
    shard.write_bytes(buffer.getvalue())
    relative = shard.relative_to(root).as_posix()
    shards = [
        {
            "path": relative,
            "size_bytes": shard.stat().st_size,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        }
    ]
    body = {"root": str(root.resolve()), "shards": shards}
    return root, {**body, "corpus_sha256": canonical_sha256(body)}, shard


def _cursor(stamp: str, phase: str) -> dict[str, object]:
    return {
        "epoch": _epoch(stamp),
        "phase": phase,
        "position": ACTION_POSITION,
    }


def _market(
    cursor: dict[str, object],
    *,
    side: str = "LONG",
    units: float = 100.0,
    tp_pips: float | None = None,
    sl_pips: float | None = None,
) -> dict[str, object]:
    return {
        "cursor": cursor,
        "action": "MARKET",
        "parameters": {
            "pair": "USD_JPY",
            "side": side,
            "units": units,
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
        },
    }


def _replay(
    root: Path,
    manifest: dict[str, object],
    actions: list[dict[str, object]],
    *,
    start: str,
    end: str,
    intrabar: str = "OHLC",
    balance: float = 10_000.0,
    slippage: float = 0.0,
    financing: float = 0.0,
) -> dict[str, object]:
    return replay_worker_economics(
        corpus_root=root,
        expected_corpus=manifest,
        window_start_utc=start,
        window_end_utc=end,
        intrabar=intrabar,
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        actions=actions,
        initial_balance_jpy=balance,
        slippage_pips_per_fill=slippage,
        financing_pips_per_day=financing,
    )


def test_market_and_manual_close_are_recomputed_from_cursor_quotes(
    tmp_path: Path,
) -> None:
    first = "2026-07-20T00:00:00Z"
    second = "2026-07-20T00:01:00Z"
    root, manifest, _ = _write_corpus(
        tmp_path,
        [
            _row(
                first,
                bid=_ohlc("100.000", "100.500", "99.500", "100.000"),
                ask=_ohlc("100.020", "100.520", "99.520", "100.020"),
            ),
            _row(
                second,
                bid=_ohlc("100.000", "101.500", "99.500", "101.000"),
                ask=_ohlc("100.020", "101.520", "99.520", "101.020"),
            ),
        ],
    )
    actions = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        intents=[
            _market(_cursor(first, "O")),
            {
                "cursor": _cursor(second, "C"),
                "action": "CLOSE",
                "parameters": {"trade_id": "T000001", "units": None},
            },
        ],
    )

    result = _replay(
        root,
        manifest,
        actions,
        start=first,
        end="2026-07-20T00:02:00Z",
    )

    assert result["promotion_eligible"] is False
    assert result["score"] == {
        "entries": 1,
        "resolved_exits": 1,
        "wins": 1,
        "win_rate_resolved": 1.0,
        "margin_closeouts": 0,
        "gross_pl_jpy": 98.0,
        "financing_jpy": 0.0,
        "realized_net_jpy": 98.0,
        "terminal_net_jpy": 98.0,
        "final_balance_jpy": 10098.0,
        "final_equity_jpy": 10098.0,
        "terminal_resolved": True,
    }
    assert result["account"]["open_positions"] == 0
    assert [event["event"] for event in result["events"]] == [
        "FILL_MARKET",
        "CLOSE",
        "PERIOD_END_SETTLEMENT",
    ]
    assert result["events"][1]["payload"]["price"] == 101.0
    assert "pl_jpy" not in actions[1]["parameters"]


def test_limit_fill_and_tp_follow_intrabar_phases_not_worker_claims(
    tmp_path: Path,
) -> None:
    stamp = "2026-07-20T00:00:00Z"
    root, manifest, _ = _write_corpus(
        tmp_path,
        [
            _row(
                stamp,
                bid=_ohlc("100.000", "101.000", "99.000", "100.000"),
                ask=_ohlc("100.020", "101.020", "99.020", "100.020"),
            )
        ],
    )
    actions = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        intents=[
            {
                "cursor": _cursor(stamp, "O"),
                "action": "LIMIT",
                "parameters": {
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "units": 100.0,
                    "price": 99.5,
                    "tp_pips": 50.0,
                    "sl_pips": None,
                },
            }
        ],
    )

    result = _replay(
        root,
        manifest,
        actions,
        start=stamp,
        end="2026-07-20T00:01:00Z",
    )

    events = {event["event"]: event for event in result["events"]}
    assert events["FILL_LIMIT"]["cursor"]["phase"] == "L"
    assert events["FILL_LIMIT"]["payload"]["price"] == 99.02
    assert events["EXIT_TP"]["cursor"]["phase"] == "C"
    assert events["EXIT_TP"]["payload"]["price"] == 99.52
    assert events["EXIT_TP"]["payload"]["gross_pl_jpy"] == 50.0


def test_financing_and_period_end_settlement_are_derived(
    tmp_path: Path,
) -> None:
    first = "2026-07-20T00:00:00Z"
    second = "2026-07-21T00:00:00Z"
    flat_bid = _ohlc("100.000", "100.000", "100.000", "100.000")
    flat_ask = _ohlc("100.020", "100.020", "100.020", "100.020")
    root, manifest, _ = _write_corpus(
        tmp_path,
        [
            _row(first, bid=flat_bid, ask=flat_ask),
            _row(second, bid=flat_bid, ask=flat_ask),
        ],
    )
    actions = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        intents=[
            _market(_cursor(first, "O")),
            {
                "cursor": _cursor(first, "O"),
                "action": "LIMIT",
                "parameters": {
                    "pair": "USD_JPY",
                    "side": "LONG",
                    "units": 1.0,
                    "price": 90.0,
                    "tp_pips": None,
                    "sl_pips": None,
                },
            },
        ],
    )

    result = _replay(
        root,
        manifest,
        actions,
        start=first,
        end="2026-07-21T00:01:00Z",
        financing=1.0,
    )

    assert result["period_end"] == {
        "owner_id": OWNER,
        "requested_order_ids": ["O000002"],
        "requested_trade_ids": ["T000001"],
        "remaining_order_ids": [],
        "remaining_trade_ids": [],
        "errors": [],
        "complete": True,
    }
    close = next(
        event for event in result["events"] if event["event"] == "PERIOD_END_CLOSE"
    )
    assert close["payload"]["gross_pl_jpy"] == -2.0
    assert close["payload"]["financing_jpy"] == 1.0
    assert close["payload"]["pl_jpy"] == -3.0
    assert result["score"]["financing_jpy"] == 1.0
    assert result["account"]["resting_orders"] == 0


def test_adverse_quote_triggers_independently_recomputed_margin_closeout(
    tmp_path: Path,
) -> None:
    first = "2026-07-20T00:00:00Z"
    second = "2026-07-20T00:01:00Z"
    root, manifest, _ = _write_corpus(
        tmp_path,
        [
            _row(
                first,
                bid=_ohlc("100.000", "100.000", "100.000", "100.000"),
                ask=_ohlc("100.000", "100.000", "100.000", "100.000"),
            ),
            _row(
                second,
                bid=_ohlc("95.000", "95.000", "95.000", "95.000"),
                ask=_ohlc("95.000", "95.000", "95.000", "95.000"),
            ),
        ],
    )
    actions = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        intents=[_market(_cursor(first, "O"), units=200.0)],
    )

    result = _replay(
        root,
        manifest,
        actions,
        start=first,
        end="2026-07-20T00:02:00Z",
        balance=1_000.0,
    )

    closeout = next(
        event for event in result["events"] if event["event"] == "MARGIN_CLOSEOUT"
    )
    assert closeout["cursor"]["phase"] == "O"
    assert closeout["payload"]["pl_jpy"] == -1000.0
    assert result["score"]["margin_closeouts"] == 1
    assert result["score"]["final_balance_jpy"] == 0.0


def test_action_hash_cursor_owner_and_unknown_trade_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    stamp = "2026-07-20T00:00:00Z"
    root, manifest, _ = _write_corpus(
        tmp_path,
        [
            _row(
                stamp,
                bid=_ohlc("100.000", "100.000", "100.000", "100.000"),
                ask=_ohlc("100.020", "100.020", "100.020", "100.020"),
            )
        ],
    )
    valid = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        intents=[_market(_cursor(stamp, "O"))],
    )

    changed_pl_basis = deepcopy(valid)
    changed_pl_basis[0]["parameters"]["units"] = 1_000_000.0
    with pytest.raises(DojoWorkerEconomicReplayError, match="chain verification"):
        _replay(
            root,
            manifest,
            changed_pl_basis,
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )

    missing_cursor = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        intents=[_market(_cursor("2026-07-20T00:01:00Z", "O"))],
    )
    with pytest.raises(DojoWorkerEconomicReplayError, match="not in sealed replay"):
        _replay(
            root,
            manifest,
            missing_cursor,
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )

    wrong_owner = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id="worker:attacker",
        intents=[_market(_cursor(stamp, "O"))],
    )
    with pytest.raises(DojoWorkerEconomicReplayError, match="chain verification"):
        _replay(
            root,
            manifest,
            wrong_owner,
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )

    boolean_schema_version = deepcopy(valid)
    boolean_schema_version[0]["schema_version"] = True
    boolean_body = {
        key: value
        for key, value in boolean_schema_version[0].items()
        if key != "action_sha256"
    }
    boolean_schema_version[0]["action_sha256"] = canonical_sha256(boolean_body)
    with pytest.raises(DojoWorkerEconomicReplayError, match="chain verification"):
        _replay(
            root,
            manifest,
            boolean_schema_version,
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )

    unknown_trade = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        intents=[
            {
                "cursor": _cursor(stamp, "O"),
                "action": "CLOSE",
                "parameters": {"trade_id": "T999999", "units": None},
            }
        ],
    )
    with pytest.raises(DojoWorkerEconomicReplayError, match="unknown owned trade"):
        _replay(
            root,
            manifest,
            unknown_trade,
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )


def test_sealed_corpus_with_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    stamp = "2026-07-20T00:00:00Z"
    row = _row(
        stamp,
        bid=_ohlc("100.000", "100.000", "100.000", "100.000"),
        ask=_ohlc("100.020", "100.020", "100.020", "100.020"),
    )
    root, manifest, shard = _write_corpus(tmp_path, [row])
    encoded = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    duplicate = (encoded[:-1] + ',"volume":11}\n').encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(duplicate)
    shard.write_bytes(buffer.getvalue())
    seal = manifest["shards"][0]
    seal["size_bytes"] = shard.stat().st_size
    seal["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    body = {"root": manifest["root"], "shards": manifest["shards"]}
    manifest["corpus_sha256"] = canonical_sha256(body)

    with pytest.raises(DojoWorkerEconomicReplayError, match="invalid corpus JSON"):
        _replay(
            root,
            manifest,
            [],
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )


def test_corpus_byte_tamper_extra_file_and_row_schema_fail_closed(
    tmp_path: Path,
) -> None:
    stamp = "2026-07-20T00:00:00Z"
    valid_row = _row(
        stamp,
        bid=_ohlc("100.000", "100.000", "100.000", "100.000"),
        ask=_ohlc("100.020", "100.020", "100.020", "100.020"),
    )
    root, manifest, shard = _write_corpus(tmp_path / "tamper", [valid_row])
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(DojoWorkerEconomicReplayError, match="bytes differ"):
        _replay(
            root,
            manifest,
            [],
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )

    root, manifest, _ = _write_corpus(tmp_path / "extra", [valid_row])
    (root / "unsealed.txt").write_text("not sealed")
    with pytest.raises(DojoWorkerEconomicReplayError, match="file set differs"):
        _replay(
            root,
            manifest,
            [],
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )

    malformed = deepcopy(valid_row)
    malformed["worker_pl_jpy"] = 9_999_999
    root, manifest, _ = _write_corpus(tmp_path / "schema", [malformed])
    with pytest.raises(DojoWorkerEconomicReplayError, match="schema is not exact"):
        _replay(
            root,
            manifest,
            [],
            start=stamp,
            end="2026-07-20T00:01:00Z",
        )


def test_action_order_must_be_monotonic_in_selected_intrabar_path(
    tmp_path: Path,
) -> None:
    stamp = "2026-07-20T00:00:00Z"
    root, manifest, _ = _write_corpus(
        tmp_path,
        [
            _row(
                stamp,
                bid=_ohlc("100.000", "101.000", "99.000", "100.000"),
                ask=_ohlc("100.020", "101.020", "99.020", "100.020"),
            )
        ],
    )
    actions = seal_action_intents(
        candidate_id=CANDIDATE,
        owner_id=OWNER,
        intents=[
            _market(_cursor(stamp, "H"), units=1.0),
            _market(_cursor(stamp, "L"), units=1.0),
        ],
    )
    with pytest.raises(DojoWorkerEconomicReplayError, match="not monotonic"):
        _replay(
            root,
            manifest,
            actions,
            start=stamp,
            end="2026-07-20T00:01:00Z",
            intrabar="OLHC",
        )


def test_replay_api_has_no_worker_reported_pl_or_account_input(
    tmp_path: Path,
) -> None:
    stamp = "2026-07-20T00:00:00Z"
    root, manifest, _ = _write_corpus(
        tmp_path,
        [
            _row(
                stamp,
                bid=_ohlc("100.000", "100.000", "100.000", "100.000"),
                ask=_ohlc("100.020", "100.020", "100.020", "100.020"),
            )
        ],
    )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        replay_worker_economics(
            corpus_root=root,
            expected_corpus=manifest,
            window_start_utc=stamp,
            window_end_utc="2026-07-20T00:01:00Z",
            intrabar="OHLC",
            candidate_id=CANDIDATE,
            owner_id=OWNER,
            actions=[],
            initial_balance_jpy=10_000.0,
            slippage_pips_per_fill=0.0,
            financing_pips_per_day=0.0,
            worker_reported_pl_jpy=1_000_000.0,  # type: ignore[call-arg]
        )
