"""Completed-M1 decision feed for isolated future DOJO paper rooms.

The existing fixed-control runner intentionally remains byte-for-byte frozen.
This module supplies an opt-in live loop for future diagnostic rooms:

* executable quotes still drive the virtual broker every five seconds;
* only broker-authored, completed M1 bid/ask candles drive bot decisions;
* a cutoff record is appended before each decision, so restart recovery is
  fail-closed and cannot duplicate an order after an ambiguous crash;
* stale missed bars warm state only and never create hindsight orders.

No execution client is imported.  The caller supplies the existing
``OandaReadOnlyClient`` through the frozen runner's environment.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time as time_mod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


UTC = timezone.utc
COMPLETED_M1_SOURCE = "OANDA_COMPLETED_M1_BID_ASK_V1"
COMPLETED_M1_CUTOFF_CONTRACT = "QR_DOJO_COMPLETED_M1_CUTOFF_V1"
COMPLETED_M1_MISSED_CONTRACT = "QR_DOJO_COMPLETED_M1_MISSED_V1"
COMPLETED_M1_SEED_CONTRACT = "QR_DOJO_COMPLETED_M1_SEED_V1"
COMPLETED_M1_SOURCE_ERROR_CONTRACT = "QR_DOJO_COMPLETED_M1_SOURCE_ERROR_V1"
MAX_DECISION_BAR_AGE_SECONDS = 90.0
MIN_OBSERVED_LAB_SEED_BARS = 1441


class CompletedM1EvidenceError(ValueError):
    """Completed-candle evidence is absent, malformed, or non-causal."""


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: Any, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CompletedM1EvidenceError(
                f"{label} must be an ISO timestamp"
            ) from exc
    else:
        raise CompletedM1EvidenceError(f"{label} must be a timestamp")
    if parsed.tzinfo is None:
        raise CompletedM1EvidenceError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _price_block(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise CompletedM1EvidenceError(f"{label} must be an object")
    result: dict[str, float] = {}
    for key in ("o", "h", "l", "c"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            raise CompletedM1EvidenceError(f"{label}.{key} is invalid")
        try:
            number = float(raw)
        except ValueError as exc:
            raise CompletedM1EvidenceError(
                f"{label}.{key} is invalid"
            ) from exc
        if number <= 0:
            raise CompletedM1EvidenceError(
                f"{label}.{key} must be positive"
            )
        result[key] = number
    if (
        result["h"] < max(result["o"], result["l"], result["c"])
        or result["l"] > min(result["o"], result["h"], result["c"])
    ):
        raise CompletedM1EvidenceError(f"{label} OHLC geometry is invalid")
    return result


def completed_m1_bars(
    payload: Mapping[str, Any],
    *,
    pair: str,
    cutoff_utc: datetime | str,
    after_epoch: int,
) -> list[dict[str, Any]]:
    """Seal unseen completed M1 bid/ask bars available at ``cutoff_utc``.

    Incomplete rows and rows whose one-minute end is after the cutoff are
    discarded.  Conflicting duplicate timestamps fail closed.
    """

    if not isinstance(payload, Mapping):
        raise CompletedM1EvidenceError("candle payload must be an object")
    cutoff = _utc(cutoff_utc, "cutoff_utc")
    raw_rows = payload.get("candles")
    if not isinstance(raw_rows, list):
        raise CompletedM1EvidenceError("candle payload has no rows")
    pair_key = str(pair or "").upper()
    if not pair_key:
        raise CompletedM1EvidenceError("pair is required")

    by_epoch: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise CompletedM1EvidenceError(
                f"candle row {index} must be an object"
            )
        if raw.get("complete") is not True:
            continue
        started = _utc(raw.get("time"), f"candle row {index} time")
        if started.second != 0 or started.microsecond != 0:
            raise CompletedM1EvidenceError(
                f"candle row {index} is not M1-aligned"
            )
        ended = started + timedelta(minutes=1)
        epoch = int(started.timestamp())
        if epoch <= int(after_epoch) or ended > cutoff:
            continue
        bid = _price_block(raw.get("bid"), f"candle row {index} bid")
        ask = _price_block(raw.get("ask"), f"candle row {index} ask")
        for key in ("o", "h", "l", "c"):
            if ask[key] < bid[key]:
                raise CompletedM1EvidenceError(
                    f"candle row {index} ask is below bid"
                )
        source = {
            "pair": pair_key,
            "granularity": "M1",
            "start_utc": started.isoformat(),
            "end_utc": ended.isoformat(),
            "complete": True,
            "volume": int(raw.get("volume") or 0),
            "bid": bid,
            "ask": ask,
        }
        bar = {
            "epoch": epoch,
            "bid_o": bid["o"],
            "bid_h": bid["h"],
            "bid_l": bid["l"],
            "bid_c": bid["c"],
            "ask_o": ask["o"],
            "ask_h": ask["h"],
            "ask_l": ask["l"],
            "ask_c": ask["c"],
            "source_sha256": _canonical_sha256(source),
            "source": source,
        }
        previous = by_epoch.get(epoch)
        if previous is not None and previous != bar:
            raise CompletedM1EvidenceError(
                f"conflicting completed M1 duplicate at {started.isoformat()}"
            )
        by_epoch[epoch] = bar

    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _bar_for_bot(bar: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: bar[key]
        for key in (
            "epoch",
            "bid_o",
            "bid_h",
            "bid_l",
            "bid_c",
            "ask_o",
            "ask_h",
            "ask_l",
            "ask_c",
        )
    }


def seed_completed_m1_history(
    *,
    seed_root: Path,
    bot: Any,
    pairs: list[str],
    window_start_utc: datetime | str,
    seed_hours: float,
) -> dict[str, Any]:
    """Warm indicators from immutable pre-window completed M1 BA shards."""

    root = seed_root.resolve()
    if not root.is_dir():
        raise CompletedM1EvidenceError(
            f"completed-M1 seed root is absent: {root}"
        )
    window_start = _utc(window_start_utc, "window_start_utc")
    hours = float(seed_hours)
    if hours <= 0:
        raise CompletedM1EvidenceError("seed_hours must be positive")
    seed_start = window_start - timedelta(hours=hours)
    pair_counts: dict[str, int] = {}
    source_records: list[dict[str, Any]] = []

    for pair in pairs:
        files = sorted(root.glob(f"*/{pair}/{pair}_M1_BA_*.jsonl.gz"))
        if not files:
            raise CompletedM1EvidenceError(
                f"{pair} has no completed-M1 seed shard"
            )
        raw_rows: list[dict[str, Any]] = []
        for path in files:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise CompletedM1EvidenceError(
                            f"invalid seed JSON: {path}:{line_number}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise CompletedM1EvidenceError(
                            f"seed row is not an object: {path}:{line_number}"
                        )
                    if row.get("pair") not in {None, pair}:
                        raise CompletedM1EvidenceError(
                            f"seed pair mismatch: {path}:{line_number}"
                        )
                    if row.get("granularity") not in {None, "M1"}:
                        raise CompletedM1EvidenceError(
                            f"seed granularity mismatch: {path}:{line_number}"
                        )
                    if row.get("price") not in {None, "BA"}:
                        raise CompletedM1EvidenceError(
                            f"seed price component mismatch: {path}:{line_number}"
                        )
                    if row.get("complete") is not True:
                        raise CompletedM1EvidenceError(
                            f"seed contains an incomplete bar: {path}:{line_number}"
                        )
                    started = _utc(
                        row.get("time"),
                        f"seed time {path}:{line_number}",
                    )
                    if started + timedelta(minutes=1) > window_start:
                        raise CompletedM1EvidenceError(
                            "seed source contains post-cutoff evidence: "
                            f"{path}:{line_number}"
                        )
                    if started >= seed_start:
                        raw_rows.append(row)
            source_records.append(
                {
                    "pair": pair,
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        bars = completed_m1_bars(
            {"candles": raw_rows},
            pair=pair,
            cutoff_utc=window_start,
            after_epoch=int(seed_start.timestamp()) - 60,
        )
        if len(bars) < MIN_OBSERVED_LAB_SEED_BARS:
            raise CompletedM1EvidenceError(
                f"{pair} seed has only {len(bars)} completed M1 bars; "
                f"{MIN_OBSERVED_LAB_SEED_BARS} required"
            )
        for bar in bars:
            bot.seed_bar(pair, _bar_for_bot(bar))
        pair_counts[pair] = len(bars)

    body = {
        "contract": COMPLETED_M1_SEED_CONTRACT,
        "source": COMPLETED_M1_SOURCE,
        "seed_root": str(root),
        "seed_start_utc": seed_start.isoformat(),
        "window_start_utc": window_start.isoformat(),
        "seed_hours": hours,
        "pair_counts": pair_counts,
        "files": sorted(
            source_records,
            key=lambda row: (row["pair"], row["path"]),
        ),
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "seed_manifest_sha256": _canonical_sha256(body)}


def cutoff_payload(
    *,
    pair: str,
    bar: Mapping[str, Any],
    cutoff_utc: datetime | str,
    quote_timestamp_utc: datetime | str,
    decision_mode: str,
) -> dict[str, Any]:
    if decision_mode not in {"ACTION", "SEED_ONLY_MISSED"}:
        raise CompletedM1EvidenceError("decision_mode is invalid")
    cutoff = _utc(cutoff_utc, "cutoff_utc")
    quote_timestamp = _utc(quote_timestamp_utc, "quote_timestamp_utc")
    if quote_timestamp > cutoff:
        raise CompletedM1EvidenceError("quote timestamp is after cutoff")
    source = bar.get("source")
    source_sha = str(bar.get("source_sha256") or "")
    if not isinstance(source, Mapping) or _canonical_sha256(source) != source_sha:
        raise CompletedM1EvidenceError("bar source binding is invalid")
    ended = _utc(source.get("end_utc"), "bar end_utc")
    if ended > cutoff:
        raise CompletedM1EvidenceError("decision bar ends after cutoff")
    body = {
        "contract": COMPLETED_M1_CUTOFF_CONTRACT,
        "pair": str(pair).upper(),
        "bar_start_utc": source["start_utc"],
        "bar_end_utc": source["end_utc"],
        "bar_epoch": int(bar["epoch"]),
        "bar_source_sha256": source_sha,
        "bar": _bar_for_bot(bar),
        "cutoff_utc": cutoff.isoformat(),
        "quote_timestamp_utc": quote_timestamp.isoformat(),
        "decision_mode": decision_mode,
        "paper_only": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "cutoff_sha256": _canonical_sha256(body)}


def restore_consumed_bars(
    ledger_path: Path,
    *,
    bot: Any,
    pairs: list[str],
    initial_epoch: int,
) -> dict[str, int]:
    """Warm post-seed bot state from pre-decision cutoff records only."""

    cursors = {pair: int(initial_epoch) for pair in pairs}
    if not ledger_path.is_file():
        return cursors
    seen: dict[tuple[str, int], str] = {}
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompletedM1EvidenceError(
                f"invalid ledger JSON at line {line_number}"
            ) from exc
        if row.get("event") != "BOT_M1_CUTOFF":
            continue
        payload = (row.get("payload") or {}).get("payload")
        if not isinstance(payload, Mapping):
            raise CompletedM1EvidenceError(
                f"cutoff payload is absent at line {line_number}"
            )
        body = {
            key: value
            for key, value in payload.items()
            if key != "cutoff_sha256"
        }
        if (
            payload.get("contract") != COMPLETED_M1_CUTOFF_CONTRACT
            or payload.get("cutoff_sha256") != _canonical_sha256(body)
            or payload.get("order_authority") != "NONE"
            or payload.get("paper_only") is not True
            or payload.get("live_permission") is not False
        ):
            raise CompletedM1EvidenceError(
                f"cutoff contract is invalid at line {line_number}"
            )
        pair = str(payload.get("pair") or "")
        if pair not in cursors:
            raise CompletedM1EvidenceError(
                f"cutoff pair is outside bot scope at line {line_number}"
            )
        epoch = int(payload.get("bar_epoch"))
        key = (pair, epoch)
        digest = str(payload.get("bar_source_sha256") or "")
        previous = seen.get(key)
        if previous is not None and previous != digest:
            raise CompletedM1EvidenceError(
                f"conflicting cutoff at line {line_number}"
            )
        if previous is not None:
            continue
        if epoch <= cursors[pair]:
            raise CompletedM1EvidenceError(
                f"cutoff order regresses at line {line_number}"
            )
        if epoch != cursors[pair] + 60:
            raise CompletedM1EvidenceError(
                f"cutoff sequence has a gap at line {line_number}"
            )
        if not hasattr(bot, "seed_bar"):
            raise CompletedM1EvidenceError(
                "bot cannot restore completed-M1 state"
            )
        bot.seed_bar(pair, dict(payload["bar"]))
        cursors[pair] = epoch
        seen[key] = digest
    return cursors


def _format_oanda_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _fetch_completed_m1(
    client: Any,
    *,
    pair: str,
    after_epoch: int,
    cutoff: datetime,
) -> list[dict[str, Any]]:
    start = datetime.fromtimestamp(after_epoch + 60, tz=UTC)
    payload = client.get_json(
        f"/v3/instruments/{pair}/candles",
        {
            "granularity": "M1",
            "from": _format_oanda_time(start),
            "to": _format_oanda_time(cutoff),
            "price": "BA",
            "includeFirst": "true",
        },
    )
    return completed_m1_bars(
        payload,
        pair=pair,
        cutoff_utc=cutoff,
        after_epoch=after_epoch,
    )


def fetch_completed_m1_fail_closed(
    client: Any,
    *,
    broker: Any,
    bot: Any,
    pair: str,
    after_epoch: int,
    cutoff: datetime,
    error_fingerprints: dict[str, str],
) -> list[dict[str, Any]] | None:
    """Fetch one causal M1 suffix without letting source failure stale orders.

    A source/transport/shape exception is evidence unavailability, not a
    reason to terminate the paper owner and strand its resting orders.  The
    cursor remains unchanged, every strategy-owned pending order for the pair
    is cancelled, and the same suffix is retried on the next poll.  Repeated
    identical errors at the same cursor are ledger-idempotent.
    """

    try:
        bars = _fetch_completed_m1(
            client,
            pair=pair,
            after_epoch=after_epoch,
            cutoff=cutoff,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)[:200]
        fingerprint_body = {
            "pair": pair,
            "after_epoch": int(after_epoch),
            "error_type": error_type,
            "error_message": error_message,
        }
        fingerprint = _canonical_sha256(fingerprint_body)
        if error_fingerprints.get(pair) != fingerprint:
            body = {
                "contract": COMPLETED_M1_SOURCE_ERROR_CONTRACT,
                **fingerprint_body,
                "as_of_utc": cutoff.astimezone(UTC).isoformat(),
                "source": COMPLETED_M1_SOURCE,
                "retry_same_cursor": True,
                "new_entries_allowed": False,
                "paper_only": True,
                "order_authority": "NONE",
                "live_permission": False,
                "error_fingerprint_sha256": fingerprint,
            }
            broker._log(
                "BOT_M1_SOURCE_ERROR",
                {
                    "payload": {
                        **body,
                        "source_error_sha256": _canonical_sha256(body),
                    }
                },
            )
            error_fingerprints[pair] = fingerprint
        strategy_tag = str(getattr(bot, "strategy_tag", "") or "")
        for order_id, order in list(broker.orders.items()):
            if (
                order.pair == pair
                and strategy_tag
                and order.strategy_tag == strategy_tag
            ):
                broker._log(
                    "BOT_M1_SOURCE_ORDER_CANCEL",
                    {
                        "order_id": order_id,
                        "pair": pair,
                        "strategy_tag": strategy_tag,
                        "error_fingerprint_sha256": fingerprint,
                        "new_entries_allowed": False,
                        "order_authority": "NONE",
                    },
                )
                broker.cancel_order(order_id)
        return None

    previous = error_fingerprints.pop(pair, None)
    if previous is not None:
        body = {
            "contract": COMPLETED_M1_SOURCE_ERROR_CONTRACT,
            "pair": pair,
            "after_epoch": int(after_epoch),
            "recovered_at_utc": cutoff.astimezone(UTC).isoformat(),
            "previous_error_fingerprint_sha256": previous,
            "source": COMPLETED_M1_SOURCE,
            "paper_only": True,
            "order_authority": "NONE",
            "live_permission": False,
        }
        broker._log(
            "BOT_M1_SOURCE_RECOVERED",
            {
                "payload": {
                    **body,
                    "source_recovery_sha256": _canonical_sha256(body),
                }
            },
        )
    return bars


def make_completed_m1_run_live(runtime: Any) -> Callable[..., None]:
    """Bind the frozen paper runtime helpers to the completed-M1 live loop."""

    def run_live(args: Any, broker: Any, session_dir: Path, bot: Any = None) -> None:
        from quant_rabbit.broker.oanda import OandaReadOnlyClient

        client = OandaReadOnlyClient()
        feed_pairs = [pair for pair in args.pairs.split(",") if pair]
        bot_pairs = list(getattr(bot, "pairs", [])) if bot is not None else []
        if bot is not None and (
            not bot_pairs or not set(bot_pairs).issubset(feed_pairs)
        ):
            raise CompletedM1EvidenceError(
                "bot pairs must be a non-empty subset of feed pairs"
            )

        window_start = (
            datetime.fromisoformat(args.window_start_utc).astimezone(UTC)
            if args.window_start_utc
            else datetime.now(UTC)
        )
        deadline = (
            datetime.fromisoformat(args.window_end_utc).timestamp()
            if args.window_end_utc
            else time_mod.time() + args.minutes * 60.0
        )
        if bot is not None:
            if not args.seed_m1_root or args.seed_hours <= 0:
                raise CompletedM1EvidenceError(
                    "completed-M1 diagnostic requires fixed seed history"
                )
            seed_manifest = seed_completed_m1_history(
                seed_root=Path(args.seed_m1_root),
                bot=bot,
                pairs=bot_pairs,
                window_start_utc=window_start,
                seed_hours=args.seed_hours,
            )
            broker._log("BOT_SEEDED", seed_manifest)
        initial_epoch = int(window_start.timestamp() // 60) * 60 - 60
        cursors = restore_consumed_bars(
            session_dir / "ledger.jsonl",
            bot=bot,
            pairs=bot_pairs,
            initial_epoch=initial_epoch,
        )
        source_error_fingerprints: dict[str, str] = {}

        while time_mod.time() < deadline:
            now = datetime.now(UTC)
            if now < window_start:
                runtime._write_state(
                    session_dir,
                    broker,
                    now.isoformat(),
                    "live",
                    "WAITING_FOR_PRECOMMITTED_WINDOW: no quote consumed",
                )
                time_mod.sleep(
                    min(
                        runtime.POLL_SECONDS,
                        max((window_start - now).total_seconds(), 0.1),
                    )
                )
                continue
            if not runtime.compute_market_status(now).is_fx_open:
                runtime._write_state(
                    session_dir,
                    broker,
                    now.isoformat(),
                    "live",
                    "MARKET_CLOSED: no fills, orders not processed",
                )
                time_mod.sleep(30.0)
                continue
            try:
                quotes = client.quotes(feed_pairs)
            except Exception as exc:
                broker._log("QUOTE_ERROR", {"error": str(exc)[:200]})
                time_mod.sleep(runtime.POLL_SECONDS)
                continue
            if set(quotes) != set(feed_pairs):
                raise CompletedM1EvidenceError(
                    "read-only quote response does not cover feed pairs"
                )
            stale = any(
                (now - quote.timestamp_utc).total_seconds()
                > runtime.STALE_QUOTE_MAX_S
                for quote in quotes.values()
            )
            if stale:
                runtime._write_state(
                    session_dir,
                    broker,
                    now.isoformat(),
                    "live",
                    "STALE_QUOTES: refusing fills and order processing",
                )
                time_mod.sleep(runtime.POLL_SECONDS)
                continue

            for pair, quote in quotes.items():
                broker.on_quote(
                    pair,
                    quote.bid,
                    quote.ask,
                    quote.timestamp_utc.isoformat(),
                )

            source_blocked_pairs: list[str] = []
            if bot is not None:
                for pair in bot_pairs:
                    quote = quotes[pair]
                    cutoff = min(now, quote.timestamp_utc)
                    latest_complete_epoch = (
                        int(cutoff.timestamp() // 60) * 60 - 60
                    )
                    if latest_complete_epoch <= cursors[pair]:
                        continue
                    bars = fetch_completed_m1_fail_closed(
                        client,
                        broker=broker,
                        bot=bot,
                        pair=pair,
                        after_epoch=cursors[pair],
                        cutoff=cutoff,
                        error_fingerprints=source_error_fingerprints,
                    )
                    if bars is None:
                        source_blocked_pairs.append(pair)
                        continue
                    if not bars:
                        continue
                    for bar in bars:
                        if bar["epoch"] != cursors[pair] + 60:
                            raise CompletedM1EvidenceError(
                                f"{pair} completed-M1 source has a gap"
                            )
                        ended = _utc(
                            bar["source"]["end_utc"],
                            "completed M1 end",
                        )
                        age_seconds = (cutoff - ended).total_seconds()
                        decision_mode = (
                            "ACTION"
                            if age_seconds <= MAX_DECISION_BAR_AGE_SECONDS
                            else "SEED_ONLY_MISSED"
                        )
                        sealed = cutoff_payload(
                            pair=pair,
                            bar=bar,
                            cutoff_utc=cutoff,
                            quote_timestamp_utc=quote.timestamp_utc,
                            decision_mode=decision_mode,
                        )
                        # Persist the cutoff before any bot side effect.  A
                        # crash after this line skips the decision on restart,
                        # avoiding duplicate virtual orders.
                        broker._log("BOT_M1_CUTOFF", {"payload": sealed})
                        if decision_mode == "ACTION":
                            bot.on_bar_closed(
                                pair,
                                _bar_for_bot(bar),
                                int(bar["epoch"]),
                            )
                        else:
                            if not hasattr(bot, "seed_bar"):
                                raise CompletedM1EvidenceError(
                                    "bot cannot seed a missed completed M1"
                                )
                            bot.seed_bar(pair, _bar_for_bot(bar))
                            broker._log(
                                "BOT_M1_UNSUPERVISED",
                                {
                                    "contract": COMPLETED_M1_MISSED_CONTRACT,
                                    "pair": pair,
                                    "bar_epoch": int(bar["epoch"]),
                                    "bar_source_sha256": bar[
                                        "source_sha256"
                                    ],
                                    "age_seconds": age_seconds,
                                    "new_entries_allowed": False,
                                    "paper_only": True,
                                    "order_authority": "NONE",
                                },
                            )
                        cursors[pair] = int(bar["epoch"])

            if not args.drain_only and not source_blocked_pairs:
                runtime._process_inbox(session_dir, broker)
            runtime._write_state(
                session_dir,
                broker,
                now.isoformat(),
                "live",
                (
                    "BOT_M1_SOURCE_UNAVAILABLE: no new entries; retrying "
                    + ",".join(sorted(source_blocked_pairs))
                    if source_blocked_pairs
                    else ""
                ),
            )
            if args.drain_only and not broker.positions and not broker.orders:
                return
            time_mod.sleep(runtime.POLL_SECONDS)

    return run_live


__all__ = [
    "COMPLETED_M1_CUTOFF_CONTRACT",
    "COMPLETED_M1_SOURCE",
    "COMPLETED_M1_SOURCE_ERROR_CONTRACT",
    "CompletedM1EvidenceError",
    "completed_m1_bars",
    "cutoff_payload",
    "fetch_completed_m1_fail_closed",
    "make_completed_m1_run_live",
    "restore_consumed_bars",
    "seed_completed_m1_history",
]
