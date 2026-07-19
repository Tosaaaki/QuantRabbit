#!/usr/bin/env python3
"""Virtual market session: real-mechanics paper trading for the duty agent.

Binds the VirtualBroker (OANDA mechanics, fills only at real quotes,
hash-chained ledger) to one of two honest feeds:

  --feed live    the account's own live pricing, polled read-only every
                 5s.  Wall-clock time.  Market closed / stale quotes =>
                 no fills, no order processing, refusal logged.
  --feed replay  the sealed M1 corpus between --from/--to, delivered
                 strictly in order as 4 quotes per bar (O,H,L,C bid/ask).
                 The sim clock is the bar's historical timestamp; no
                 lookahead is possible because the state file never
                 contains anything past the cursor.

Agent protocol (file-based, works from any agent harness):
  session-dir/
    state.json        account, positions, resting orders, latest quotes,
                      sim_time — rewritten atomically every tick
    inbox/            the agent drops one JSON per action:
                      {"action":"MARKET","pair":"USD_JPY","side":"LONG",
                       "units":10000,"tp_pips":5,"sl_pips":null}
                      {"action":"LIMIT","pair":...,"price":...}
                      {"action":"CLOSE","trade_id":"T000001","units":null}
                      {"action":"CANCEL","order_id":"O000001"}
                      {"action":"SET_EXIT","trade_id":...,"tp_price":...,
                       "sl_price":...}
    inbox/processed/  handled action files are renamed here (never deleted)
    ledger.jsonl      the broker's hash-chained ledger (every quote-caused
                      fill records the exact quote)

In replay mode --step makes the session turn-based: it advances one bar
whenever the agent writes inbox/STEP (renamed after consumption), so a
discretionary agent can think per bar.  ORDER AUTHORITY: NONE (paper
account only; this process cannot reach the real broker's order API).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.analysis.market_status import compute_market_status  # noqa: E402
from quant_rabbit.dojo_lab_provenance import (  # noqa: E402
    strategy_ownership_registry,
)
from quant_rabbit.virtual_broker import (  # noqa: E402
    VirtualBroker,
    VirtualBrokerError,
)


PHASE_ORDERS = {
    "OHLC": {"O": 0, "H": 1, "L": 2, "C": 3},
    "OLHC": {"O": 0, "L": 1, "H": 2, "C": 3},
}


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(value: str):
    def reject_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=reject_duplicates,
    )


def _finite_number(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _normalized_pairs(raw: str) -> list[str]:
    pairs = [pair.strip() for pair in raw.split(",") if pair.strip()]
    if not pairs:
        raise ValueError("at least one pair is required")
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate pairs are forbidden")
    for pair in pairs:
        parts = pair.split("_")
        if len(parts) != 2 or any(
            len(part) != 3 or not part.isalpha() or not part.isupper() for part in parts
        ):
            raise ValueError(f"invalid pair: {pair}")
    return sorted(pairs)


def _parse_utc_bound(value: str) -> datetime:
    text = str(value).strip()
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})?",
        text,
    )
    if match:
        head, fraction, zone = match.groups()
        text = head
        if fraction:
            text += "." + fraction[:6]
        text += "+00:00" if zone in {None, "Z"} else zone
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def _validate_time_window(time_from: str, time_to: str) -> tuple[datetime, datetime]:
    start = _parse_utc_bound(time_from)
    end = _parse_utc_bound(time_to)
    if start >= end:
        raise ValueError("--from must be earlier than --to")
    return start, end


def _parse_corpus_time(value: object) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("corpus row time is missing")
    return _parse_utc_bound(text)


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_dependency_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    resolved = (
        (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("bot dependencies must stay inside the repository") from exc
    if not resolved.is_file():
        raise ValueError(f"bot dependency is not a file: {resolved}")
    return resolved


def _verify_custom_bot_seal(bot_manifest: dict) -> bytes:
    """Read the exact custom-module bytes and recheck its declared closure."""

    module_path = Path(bot_manifest["module_path"])
    module_bytes = module_path.read_bytes()
    if hashlib.sha256(module_bytes).hexdigest() != bot_manifest["module_sha256"]:
        raise ValueError("custom bot module changed after SESSION_START seal")
    for relative, expected_sha in bot_manifest["dependency_sha256"].items():
        dependency = _repo_dependency_path(relative)
        if _file_sha256(dependency) != expected_sha:
            raise ValueError(f"custom bot dependency changed after seal: {relative}")
    return module_bytes


def _settle_custom_bot_at_end(broker: VirtualBroker, owner_id: str) -> None:
    """Resolve only the declared custom worker's orders and positions."""

    ownership = strategy_ownership_registry(broker)
    order_ids = ownership.active_order_ids(owner_id)
    trade_ids = ownership.active_trade_ids(owner_id)
    errors: list[str] = []
    for order_id in order_ids:
        try:
            broker.cancel_order(order_id)
        except VirtualBrokerError as exc:
            errors.append(f"cancel:{order_id}:{str(exc)[:120]}")
    for trade_id in trade_ids:
        try:
            broker.close_trade(trade_id)
        except VirtualBrokerError as exc:
            errors.append(f"close:{trade_id}:{str(exc)[:120]}")
    broker._log(
        "PERIOD_END_SETTLEMENT",
        {
            "strategy_owner_id": owner_id,
            "requested_order_ids": list(order_ids),
            "requested_trade_ids": list(trade_ids),
            "errors": errors,
            "complete": not errors
            and not ownership.active_order_ids(owner_id)
            and not ownership.active_trade_ids(owner_id),
        },
    )


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
        raise RuntimeError("unable to bind a valid git HEAD")
    return head


def _selected_corpus_shards(
    root: Path,
    pairs: list[str],
    time_from: str,
    time_to: str,
    granularity: str,
) -> list[Path]:
    start, end = _validate_time_window(time_from, time_to)
    shards: set[Path] = set()
    for year in range(start.year, end.year + 1):
        for pair in pairs:
            shards.update(
                path.resolve()
                for path in root.glob(
                    f"*/{pair}/{pair}_{granularity}_BA_{year}*.jsonl.gz"
                )
            )
    return sorted(shards)


def _validate_replay_row(row: object, pair: str, *, expected_year: int) -> datetime:
    if not isinstance(row, dict):
        raise ValueError(f"corpus row for {pair} must be an object")
    stamp = _parse_corpus_time(row.get("time"))
    if stamp.year != expected_year:
        raise ValueError(f"corpus row year/file mismatch for {pair}")
    for side in ("bid", "ask"):
        values = row.get(side)
        if not isinstance(values, dict) or set(values) < {"o", "h", "l", "c"}:
            raise ValueError(f"corpus {side} OHLC is incomplete for {pair}")
        parsed = {
            key: _finite_number(f"{pair} {side}.{key}", values[key], positive=True)
            for key in ("o", "h", "l", "c")
        }
        if parsed["h"] < max(parsed["o"], parsed["l"], parsed["c"]):
            raise ValueError(f"corpus {side} high geometry is invalid for {pair}")
        if parsed["l"] > min(parsed["o"], parsed["h"], parsed["c"]):
            raise ValueError(f"corpus {side} low geometry is invalid for {pair}")
    for key in ("o", "h", "l", "c"):
        if float(row["ask"][key]) < float(row["bid"][key]):
            raise ValueError(f"corpus ask is below bid for {pair} at {key}")
    return stamp


def _validate_corpus_pair_coverage(
    root: Path,
    pairs: list[str],
    time_from: str,
    time_to: str,
    granularity: str,
    shards: list[Path],
) -> None:
    start, end = _validate_time_window(time_from, time_to)
    by_pair: dict[str, list[Path]] = {pair: [] for pair in pairs}
    for path in shards:
        for pair in pairs:
            if path.parent.name == pair:
                by_pair[pair].append(path)
                break
    missing_shards = [pair for pair, paths in by_pair.items() if not paths]
    if missing_shards:
        raise ValueError(
            "replay corpus missing requested pair shards: " + ",".join(missing_shards)
        )
    covered: set[str] = set()
    for pair, paths in by_pair.items():
        for shard in paths:
            try:
                expected_year = int(shard.name.split(f"_{granularity}_BA_", 1)[1][:4])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid corpus shard name: {shard.name}") from exc
            with gzip.open(shard, "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        row = _strict_json_loads(line)
                        stamp = _validate_replay_row(
                            row, pair, expected_year=expected_year
                        )
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(
                            f"invalid corpus row {shard.name}:{line_number}: {exc}"
                        ) from exc
                    if start <= stamp < end:
                        covered.add(pair)
                        break
            if pair in covered:
                break
    missing_period = [pair for pair in pairs if pair not in covered]
    if missing_period:
        raise ValueError(
            "replay corpus has no rows in requested period for: "
            + ",".join(missing_period)
        )


def _replay_coordinate(
    *, epoch: int, phase: str, granularity: str, intrabar: str
) -> dict:
    return {
        "mode": "replay",
        "epoch": epoch,
        "phase": phase,
        "granularity": granularity,
        "intrabar": intrabar,
    }


def _expected_quote_batch_receipt(
    quotes: list[tuple[str, float, float, str]],
    *,
    coordinate: dict,
    feed_pairs: list[str],
    batch_index: int,
    previous_batch_sha256: str,
) -> dict:
    """Mirror the broker's public quote-batch receipt for pre-registration.

    The manifest must commit the terminal batch-chain digest before the bot is
    loaded.  Keeping this pure predictor next to manifest construction makes
    any drift from ``VirtualBroker.record_quote_batch_begin`` fail the actual
    replay's terminal comparison rather than silently changing the evidence.
    """

    canonical_quotes = [
        {"pair": pair, "bid": bid, "ask": ask, "ts": ts}
        for pair, bid, ask, ts in sorted(quotes, key=lambda row: row[0])
    ]
    canonical_feed_pairs = sorted(feed_pairs)
    batch_pairs = [row["pair"] for row in canonical_quotes]
    payload = {
        "contract": "QR_VIRTUAL_QUOTE_BATCH_V1",
        "batch_index": batch_index,
        "coordinate": coordinate,
        "feed_pairs": canonical_feed_pairs,
        "batch_pairs": batch_pairs,
        "coverage_complete": batch_pairs == canonical_feed_pairs,
        "quotes": canonical_quotes,
        "quotes_sha256": _canonical_sha256(canonical_quotes),
        "previous_batch_sha256": previous_batch_sha256,
    }
    return {**payload, "batch_sha256": _canonical_sha256(payload)}


def _build_replay_mtm_commitment(
    *,
    root: Path,
    pairs: list[str],
    time_from: str,
    time_to: str,
    intrabar: str,
    granularity: str,
    expected_shards: list[dict],
) -> dict:
    """Pre-register every causal MTM coordinate and quote-batch chain tip."""

    phase_order = [
        phase
        for phase, _ in sorted(PHASE_ORDERS[intrabar].items(), key=lambda item: item[1])
    ]
    previous_batch_sha256 = "0" * 64
    phase_mark_count = 0
    partial_coordinates: list[dict] = []
    first_coordinate = None
    last_coordinate = None
    for epoch, phase, quote_batch in _iter_replay_quote_batches(
        root,
        pairs,
        time_from,
        time_to,
        intrabar,
        granularity,
        expected_shards=expected_shards,
    ):
        coordinate = _replay_coordinate(
            epoch=epoch,
            phase=phase,
            granularity=granularity,
            intrabar=intrabar,
        )
        receipt = _expected_quote_batch_receipt(
            quote_batch,
            coordinate=coordinate,
            feed_pairs=pairs,
            batch_index=phase_mark_count,
            previous_batch_sha256=previous_batch_sha256,
        )
        if not receipt["coverage_complete"]:
            partial_coordinates.append(coordinate)
        if first_coordinate is None:
            first_coordinate = coordinate
        last_coordinate = coordinate
        previous_batch_sha256 = receipt["batch_sha256"]
        phase_mark_count += 1

    if phase_mark_count == 0 or first_coordinate is None or last_coordinate is None:
        raise ValueError("replay MTM coordinate set is empty")
    if partial_coordinates:
        sample = ",".join(
            f"{row['epoch']}#{row['phase']}" for row in partial_coordinates[:3]
        )
        raise ValueError(
            "replay MTM requires full feed-pair coverage at every epoch/phase; "
            f"partial_phase_count={len(partial_coordinates)} sample={sample}"
        )
    return {
        "mtm_coordinate_contract": "QR_REPLAY_MTM_COORDINATES_V1",
        "phase_order": phase_order,
        "feed_pairs": sorted(pairs),
        "expected_phase_mark_count": phase_mark_count,
        "expected_batch_chain_terminal_sha256": previous_batch_sha256,
        "full_pair_phase_coverage": True,
        "partial_phase_count": 0,
        "first_coordinate": first_coordinate,
        "last_coordinate": last_coordinate,
    }


def _build_reproducibility_manifest(args) -> dict:
    """Bind every material replay, cost, source, and worker input.

    The corpus and code digests make a SESSION_START self-contained proof of
    which bytes were eligible to influence the run.  Known bot-configuration
    environment values are digest-bound without leaking their contents.
    """

    pairs = _normalized_pairs(args.pairs)
    _validate_time_window(args.time_from, args.time_to)
    _finite_number("balance", args.balance, positive=True)
    _finite_number("slippage_pips", args.slippage_pips)
    _finite_number("financing_pips_day", args.financing_pips_day)
    if args.slippage_pips < 0 or args.financing_pips_day < 0:
        raise ValueError("cost parameters must be non-negative")
    settle_at_end = bool(getattr(args, "settle_at_end", False))
    if settle_at_end and (args.feed != "replay" or not args.bot_module):
        raise ValueError("--settle-at-end requires replay with a custom bot")
    root = Path(args.corpus_root).expanduser().resolve()
    shards = (
        _selected_corpus_shards(
            root, pairs, args.time_from, args.time_to, args.granularity
        )
        if args.feed == "replay"
        else []
    )
    if args.feed == "replay" and not shards:
        raise ValueError("replay corpus contains no selected shards")
    if args.feed == "replay":
        _validate_corpus_pair_coverage(
            root,
            pairs,
            args.time_from,
            args.time_to,
            args.granularity,
            shards,
        )
    shard_rows = [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in shards
    ]

    continuous_mtm = bool(getattr(args, "continuous_mtm", False))
    if continuous_mtm and args.feed != "replay":
        raise ValueError("--continuous-mtm requires --feed replay")

    if args.bot and args.bot_module:
        raise ValueError("--bot and --bot-module are mutually exclusive")
    if args.bot_module:
        module_path_raw, _, class_name = args.bot_module.partition(":")
        module_path = Path(module_path_raw).expanduser().resolve()
        if not module_path.is_file():
            raise ValueError(f"bot module is not a file: {module_path}")
        strategy_owner_id = getattr(args, "strategy_owner_id", None)
        if (
            not isinstance(strategy_owner_id, str)
            or not strategy_owner_id
            or len(strategy_owner_id) > 128
            or any(ord(char) < 33 or ord(char) > 126 for char in strategy_owner_id)
        ):
            raise ValueError("custom bot requires a visible --strategy-owner-id")
        raw_dependencies = list(getattr(args, "bot_dependency", None) or [])
        if not raw_dependencies:
            raise ValueError("custom bot requires a non-empty dependency closure")
        dependency_paths = [_repo_dependency_path(item) for item in raw_dependencies]
        relative_dependencies = [
            str(path.relative_to(REPO_ROOT)) for path in dependency_paths
        ]
        if len(relative_dependencies) != len(set(relative_dependencies)):
            raise ValueError("custom bot dependency paths must be unique")
        bot_manifest = {
            "kind": "custom_module",
            "name": None,
            "module_path": str(module_path),
            "module_sha256": _file_sha256(module_path),
            "class": class_name or "Bot",
            "strategy_owner_id": strategy_owner_id,
            "dependency_sha256": {
                relative: _file_sha256(path)
                for relative, path in sorted(
                    zip(relative_dependencies, dependency_paths, strict=True)
                )
            },
        }
    elif args.bot:
        vendor = REPO_ROOT / "vendored" / "golden_20251209"
        dependency_paths = [
            vendor / "analysis_ma_projection.py",
            vendor / "indicators_calc_core.py",
            vendor / "strategies_micro_momentum_burst.py",
        ]
        bot_manifest = {
            "kind": "built_in",
            "name": args.bot,
            "module_path": str(Path(__file__).resolve()),
            "module_sha256": _file_sha256(Path(__file__).resolve()),
            "class": "GoldenBurstBot",
            "dependency_sha256": {
                str(path.relative_to(REPO_ROOT)): _file_sha256(path)
                for path in dependency_paths
            },
        }
    else:
        bot_manifest = {
            "kind": "none",
            "name": None,
            "module_path": None,
            "module_sha256": None,
            "class": None,
            "dependency_sha256": {},
        }
    if "dependency_sha256" not in bot_manifest:
        bot_manifest["dependency_sha256"] = {}
    bot_manifest["configuration_bindings"] = {
        key: {
            "sha256": hashlib.sha256(os.environ[key].encode("utf-8")).hexdigest(),
            "length": len(os.environ[key]),
        }
        for key in ("DOJO_BOT_CONFIG", "DOJO_BOT_COMBO")
        if key in os.environ
    }
    session_dir = getattr(args, "session_dir", None)
    snapshot_path = Path(session_dir) / "broker_snapshot.json" if session_dir else None
    resume_snapshot = (
        {
            "path": str(snapshot_path.resolve()),
            "size_bytes": snapshot_path.stat().st_size,
            "sha256": _file_sha256(snapshot_path),
        }
        if snapshot_path is not None and snapshot_path.is_file()
        else None
    )
    if continuous_mtm and resume_snapshot is not None:
        raise ValueError(
            "resumed replay cannot attest QR_REPLAY_MTM_COORDINATES_V1; "
            "start a fresh session for trainer evidence"
        )

    corpus_manifest = {
        "root": str(root) if args.feed == "replay" else None,
        "shards": shard_rows,
    }
    corpus_manifest["corpus_sha256"] = _canonical_sha256(corpus_manifest)
    replay_mtm_commitment = (
        _build_replay_mtm_commitment(
            root=root,
            pairs=pairs,
            time_from=args.time_from,
            time_to=args.time_to,
            intrabar=args.intrabar,
            granularity=args.granularity,
            expected_shards=shard_rows,
        )
        if continuous_mtm
        else {}
    )
    manifest = {
        "schema": "QR_VIRTUAL_SESSION_REPRODUCIBILITY_V1",
        "source": {
            "git_head": _git_head(),
            "session_script_sha256": _file_sha256(Path(__file__).resolve()),
            "virtual_broker_sha256": _file_sha256(
                REPO_ROOT / "src" / "quant_rabbit" / "virtual_broker.py"
            ),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
        },
        "replay": {
            "feed": args.feed,
            "pairs": pairs,
            "time_from": args.time_from,
            "time_to": args.time_to,
            "granularity": args.granularity,
            "intrabar": args.intrabar,
            "bot_bar": args.bot_bar,
            "period_end_settlement": settle_at_end,
            "continuous_mtm": continuous_mtm,
            **replay_mtm_commitment,
        },
        "corpus": corpus_manifest,
        "costs": {
            "slippage_pips_per_fill": args.slippage_pips,
            "financing_pips_per_day": args.financing_pips_day,
            "leverage": 25.0,
        },
        "initial_balance_jpy": args.balance,
        "resume_snapshot": resume_snapshot,
        "bot": bot_manifest,
        "pacing": {
            "bars_per_second": getattr(args, "bars_per_second", None),
            "step": getattr(args, "step", False),
            "state_every": getattr(args, "state_every", 1),
        },
        "order_authority": "NONE",
    }
    return {**manifest, "manifest_sha256": _canonical_sha256(manifest)}


def _replay_identity_sha256(manifest: dict) -> str:
    """Material identity that a replay cursor is allowed to resume."""

    material = {
        "source": manifest["source"],
        "replay": manifest["replay"],
        "corpus": manifest["corpus"],
        "costs": manifest["costs"],
        "initial_balance_jpy": manifest["initial_balance_jpy"],
        "bot": manifest["bot"],
        "order_authority": manifest["order_authority"],
    }
    return _canonical_sha256(material)


class GoldenBurstBot:
    """Worker bot living INSIDE the virtual session: same broker, same
    fill engine, same ledger as the duty agent.  Runs the vendored
    golden-day MomentumBurst with arsenal protections (max 3 concurrent,
    4h hard ceiling).  Sizing: NAV-proportional 4.3x per position."""

    WARMUP = 40
    MAX_CONCURRENT = 3
    CEILING_S = 4 * 3600
    PAIR = "USD_JPY"

    def __init__(self, broker: VirtualBroker, blind_spread: bool = False):
        import importlib.util
        from types import ModuleType

        vendor = REPO_ROOT / "vendored" / "golden_20251209"

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        pkg = ModuleType("analysis")
        pkg.__path__ = []
        sys.modules["analysis"] = pkg
        pkg.ma_projection = load(
            "analysis.ma_projection", vendor / "analysis_ma_projection.py"
        )
        self.calc_core = load("golden_calc_core", vendor / "indicators_calc_core.py")
        self.strategy = load(
            "golden_momentum_burst", vendor / "strategies_micro_momentum_burst.py"
        ).MomentumBurstMicro
        self.broker = broker
        self.blind_spread = blind_spread
        self.bars: list[dict] = []
        self.my_trades: dict[str, float] = {}  # trade_id -> entry_epoch

    def on_bar_closed(self, pair: str, bar: dict, bar_epoch: int) -> None:
        if pair != self.PAIR:
            return
        self.bars.append(bar)
        if len(self.bars) > 2000:
            self.bars.pop(0)
        # ceiling exits for my open trades
        for trade_id in list(self.my_trades):
            if trade_id not in self.broker.positions:
                del self.my_trades[trade_id]
                continue
            if bar_epoch - self.my_trades[trade_id] >= self.CEILING_S:
                try:
                    self.broker.close_trade(trade_id)
                except VirtualBrokerError:
                    pass
                del self.my_trades[trade_id]
        if len(self.bars) < self.WARMUP:
            return
        live_mine = [t for t in self.my_trades if t in self.broker.positions]
        if len(live_mine) >= self.MAX_CONCURRENT:
            return
        import pandas as pd

        mid_c = pd.Series([(b["bid_c"] + b["ask_c"]) / 2 for b in self.bars])
        mid_h = pd.Series([(b["bid_h"] + b["ask_h"]) / 2 for b in self.bars])
        mid_l = pd.Series([(b["bid_l"] + b["ask_l"]) / 2 for b in self.bars])
        ma10 = mid_c.rolling(10, min_periods=10).mean().iloc[-1]
        ma20 = mid_c.rolling(20, min_periods=20).mean().iloc[-1]
        ema20 = mid_c.ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
        vol_5m = (mid_c.diff().abs().rolling(5, min_periods=5).mean() / 0.01).iloc[-1]
        rsi = self.calc_core._rsi(mid_c, 14).iloc[-1]
        atr = self.calc_core._atr(mid_h, mid_l, mid_c, 14).iloc[-1]
        adx = self.calc_core._adx(mid_h, mid_l, mid_c, 14).iloc[-1]
        if any(pd.isna(v) for v in (ma10, ma20, ema20, vol_5m, rsi, atr, adx)):
            return
        last = self.bars[-1]
        fac = {
            "close": float(mid_c.iloc[-1]),
            "ma10": float(ma10),
            "ma20": float(ma20),
            "ema20": float(ema20),
            "rsi": float(rsi),
            "atr": float(atr),
            "adx": float(adx),
            "vol_5m": float(vol_5m),
            "spread_pips": 0.0
            if self.blind_spread
            else (last["ask_c"] - last["bid_c"]) / 0.01,
            "candles": [
                {
                    "high": (b["bid_h"] + b["ask_h"]) / 2,
                    "low": (b["bid_l"] + b["ask_l"]) / 2,
                    "open": (b["bid_o"] + b["ask_o"]) / 2,
                    "close": (b["bid_c"] + b["ask_c"]) / 2,
                }
                for b in self.bars[-4:]
            ],
        }
        signal = self.strategy.check(fac)
        if not signal or signal.get("action") not in {"OPEN_LONG", "OPEN_SHORT"}:
            return
        side = "LONG" if signal["action"] == "OPEN_LONG" else "SHORT"
        try:
            acct = self.broker.account()
            units = max(acct["equity_jpy"], 0.0) * 4.3 / fac["close"]
            if units <= 0:
                return
            trade_id = self.broker.market_order(
                self.PAIR,
                side,
                units,
                tp_pips=float(signal["tp_pips"]),
                sl_pips=float(signal["sl_pips"]),
            )
            self.my_trades[trade_id] = bar_epoch
        except VirtualBrokerError:
            return


UTC = timezone.utc
POLL_SECONDS = 5.0
STALE_QUOTE_MAX_S = 90.0


def _write_state(
    session_dir: Path, broker: VirtualBroker, sim_time: str, mode: str, note: str = ""
) -> None:
    state = {
        "mode": mode,
        "sim_time_utc": sim_time,
        "wall_time_utc": datetime.now(UTC).isoformat(),
        "account": broker.account() if broker.last_quotes else None,
        "positions": [vars(p) for p in broker.positions.values()],
        "resting_orders": [vars(o) for o in broker.orders.values()],
        "quotes": {
            pair: {"bid": q[0], "ask": q[1], "ts": q[2]}
            for pair, q in broker.last_quotes.items()
        },
        "note": note,
    }
    tmp = session_dir / ".state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(tmp, session_dir / "state.json")


def _process_inbox(
    session_dir: Path,
    broker: VirtualBroker,
    *,
    allowed_pairs: set[str] | None = None,
) -> int:
    inbox = session_dir / "inbox"
    done = inbox / "processed"
    done.mkdir(parents=True, exist_ok=True)
    handled = 0
    now = time_mod.time()
    for path in sorted(inbox.glob("*.json")):
        try:
            if now - path.stat().st_mtime < 0.5:
                continue  # writer may still be mid-write; pick up next tick
        except OSError:
            continue
        try:
            action = _strict_json_loads(path.read_text())
            if not isinstance(action, dict):
                raise ValueError("action must be a JSON object")
            kind = action.get("action")
            action_pair = action.get("pair")
            if (
                action_pair is not None
                and allowed_pairs is not None
                and action_pair not in allowed_pairs
            ):
                raise ValueError(f"pair is not active in this session: {action_pair}")
            if kind == "MARKET":
                broker.market_order(
                    action["pair"],
                    action["side"],
                    float(action["units"]),
                    tp_pips=action.get("tp_pips"),
                    sl_pips=action.get("sl_pips"),
                )
            elif kind == "LIMIT":
                broker.limit_order(
                    action["pair"],
                    action["side"],
                    float(action["units"]),
                    price=float(action["price"]),
                    tp_pips=action.get("tp_pips"),
                    sl_pips=action.get("sl_pips"),
                )
            elif kind == "CLOSE":
                units = action.get("units")
                broker.close_trade(
                    action["trade_id"],
                    units=float(units) if units is not None else None,
                )
            elif kind == "CANCEL":
                broker.cancel_order(action["order_id"])
            elif kind == "SET_EXIT":
                broker.set_exit(
                    action["trade_id"],
                    tp_price=action.get("tp_price"),
                    sl_price=action.get("sl_price"),
                )
            else:
                broker._log(
                    "AGENT_ACTION_REJECTED",
                    {"file": path.name, "error": f"unknown action {kind}"},
                )
        except (VirtualBrokerError, KeyError, ValueError, json.JSONDecodeError) as exc:
            broker._log(
                "AGENT_ACTION_REJECTED", {"file": path.name, "error": str(exc)[:200]}
            )
        os.replace(path, done / f"{int(time_mod.time()*1000)}_{path.name}")
        handled += 1
    return handled


def run_live(args, broker: VirtualBroker, session_dir: Path, bot=None) -> None:
    from quant_rabbit.broker.oanda import OandaReadOnlyClient

    client = OandaReadOnlyClient()
    pairs = _normalized_pairs(args.pairs)
    exposure_pairs = {position.pair for position in broker.positions.values()} | {
        order.pair for order in broker.orders.values()
    }
    if not exposure_pairs.issubset(set(pairs)):
        raise VirtualBrokerError(
            "live session pairs do not cover restored exposure: "
            + ",".join(sorted(exposure_pairs - set(pairs)))
        )
    deadline = time_mod.time() + args.minutes * 60.0
    live_bars: dict[str, dict] = {}
    while time_mod.time() < deadline:
        now = datetime.now(UTC)
        if not compute_market_status(now).is_fx_open:
            _write_state(
                session_dir,
                broker,
                now.isoformat(),
                "live",
                "MARKET_CLOSED: no fills, orders not processed",
            )
            time_mod.sleep(30.0)
            continue
        try:
            quotes = client.quotes(pairs)
        except Exception as exc:
            broker._log("QUOTE_ERROR", {"error": str(exc)[:200]})
            time_mod.sleep(POLL_SECONDS)
            continue
        missing = sorted(set(pairs) - set(quotes))
        unexpected = sorted(set(quotes) - set(pairs))
        invalid_freshness: list[str] = []
        for pair, q in quotes.items():
            age_s = (now - q.timestamp_utc).total_seconds()
            prior = broker.last_quotes.get(pair)
            prior_epoch = broker._ts_epoch(prior[2]) if prior is not None else None
            quote_epoch = q.timestamp_utc.timestamp()
            if age_s < -1.0 or age_s > STALE_QUOTE_MAX_S:
                invalid_freshness.append(f"{pair}:age={age_s:.3f}")
            elif prior_epoch is not None and quote_epoch < prior_epoch:
                invalid_freshness.append(f"{pair}:timestamp_rewind")
        if missing or unexpected or invalid_freshness:
            detail = {
                "missing_pairs": missing,
                "unexpected_pairs": unexpected,
                "invalid_freshness": invalid_freshness,
            }
            broker._log("QUOTE_BATCH_REJECTED", detail)
            _write_state(
                session_dir,
                broker,
                now.isoformat(),
                "live",
                "INCOMPLETE_OR_STALE_QUOTES: refusing actions",
            )
            time_mod.sleep(POLL_SECONDS)
            continue
        broker.on_quote_batch(
            [
                (
                    pair,
                    quotes[pair].bid,
                    quotes[pair].ask,
                    quotes[pair].timestamp_utc.isoformat(),
                )
                for pair in pairs
            ]
        )
        for pair in pairs:
            q = quotes[pair]
            minute = int(q.timestamp_utc.timestamp() // 60) * 60
            bar = live_bars.get(pair)
            if bar is not None and bar["epoch"] != minute:
                if bot is not None:
                    bot.on_bar_closed(pair, bar, bar["epoch"])
                bar = None
            if bar is None:
                live_bars[pair] = {
                    "epoch": minute,
                    "bid_o": q.bid,
                    "bid_h": q.bid,
                    "bid_l": q.bid,
                    "bid_c": q.bid,
                    "ask_o": q.ask,
                    "ask_h": q.ask,
                    "ask_l": q.ask,
                    "ask_c": q.ask,
                }
            else:
                bar["bid_h"] = max(bar["bid_h"], q.bid)
                bar["bid_l"] = min(bar["bid_l"], q.bid)
                bar["ask_h"] = max(bar["ask_h"], q.ask)
                bar["ask_l"] = min(bar["ask_l"], q.ask)
                bar["bid_c"] = q.bid
                bar["ask_c"] = q.ask
        _process_inbox(session_dir, broker, allowed_pairs=set(pairs))
        _write_state(session_dir, broker, now.isoformat(), "live")
        time_mod.sleep(POLL_SECONDS)


def _iter_replay_quotes(
    root: Path,
    pairs: list[str],
    time_from: str,
    time_to: str,
    intrabar: str = "OHLC",
    granularity: str = "M1",
    expected_shards=None,
):
    """Merge pairs' M1 bars in time order; yield (epoch, pair, bid, ask, phase).

    Streams year by year so full-history sessions stay in bounded memory.
    ``intrabar`` declares the synthetic intrabar path (the true tick path
    is unknown): OHLC favors longs' TP on both-touch bars, OLHC favors
    shorts'/SL — run both to bracket ambiguous outcomes.
    """

    root = root.expanduser().resolve()
    pairs = sorted(pairs)
    if len(pairs) != len(set(pairs)) or not pairs:
        raise ValueError("replay pairs must be unique and non-empty")
    start, end = _validate_time_window(time_from, time_to)
    phase_keys = {
        "OHLC": (("O", "o"), ("H", "h"), ("L", "l"), ("C", "c")),
        "OLHC": (("O", "o"), ("L", "l"), ("H", "h"), ("C", "c")),
    }[intrabar]
    discovered = _selected_corpus_shards(root, pairs, time_from, time_to, granularity)
    expected_by_path: dict[Path, dict] = {}
    if expected_shards is not None:
        for row in expected_shards:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("invalid expected corpus shard seal")
            path = (root / row["path"]).resolve()
            if root not in path.parents:
                raise ValueError("corpus shard escapes corpus root")
            expected_by_path[path] = row
        if set(discovered) != set(expected_by_path):
            raise ValueError("corpus shard set changed after manifest sealing")
        for path, row in expected_by_path.items():
            if path.stat().st_size != int(row["size_bytes"]):
                raise ValueError(f"corpus shard size changed: {path.name}")
    eligible_pairs: set[str] = set()
    for year in range(start.year, end.year + 1):
        rows = []
        for pair in pairs:
            pair_shards = [
                shard
                for shard in discovered
                if shard.parent.name == pair
                and f"_{granularity}_BA_{year}" in shard.name
            ]
            for shard in pair_shards:
                with gzip.open(shard, "rt", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        try:
                            row = _strict_json_loads(line)
                            stamp = _validate_replay_row(row, pair, expected_year=year)
                        except (json.JSONDecodeError, ValueError) as exc:
                            raise ValueError(
                                f"invalid corpus row {shard.name}:{line_number}: {exc}"
                            ) from exc
                        if not (start <= stamp < end):
                            continue
                        eligible_pairs.add(pair)
                        epoch = int(stamp.timestamp())
                        rows.append((epoch, pair, row))
                expected = expected_by_path.get(shard)
                if expected is not None and _file_sha256(shard) != expected["sha256"]:
                    raise ValueError(
                        f"corpus shard changed during replay: {shard.name}"
                    )
        rows.sort(key=lambda r: (r[0], r[1]))
        cursor = 0
        while cursor < len(rows):
            epoch = rows[cursor][0]
            epoch_rows: dict[str, dict] = {}
            while cursor < len(rows) and rows[cursor][0] == epoch:
                _, pair, row = rows[cursor]
                if pair in epoch_rows:
                    raise ValueError(
                        f"duplicate corpus row for {pair} at epoch {epoch}"
                    )
                epoch_rows[pair] = row
                cursor += 1
            # Synchronize every pair at O before any H/L/C, then repeat for
            # each phase.  This prevents one pair's close from leaking into a
            # different pair's open at the same epoch.
            for phase, key in phase_keys:
                for pair in sorted(epoch_rows):
                    row = epoch_rows[pair]
                    bid_values, ask_values = row["bid"], row["ask"]
                    yield (
                        epoch,
                        pair,
                        float(bid_values[key]),
                        float(ask_values[key]),
                        phase,
                    )
    missing = [pair for pair in pairs if pair not in eligible_pairs]
    if missing:
        raise ValueError("replay corpus has no eligible rows for: " + ",".join(missing))


def _iter_replay_quote_batches(*args, **kwargs):
    batch_key = None
    batch: list[tuple[str, float, float, str]] = []
    for epoch, pair, bid, ask, phase in _iter_replay_quotes(*args, **kwargs):
        key = (epoch, phase)
        if batch_key is not None and key != batch_key:
            yield batch_key[0], batch_key[1], batch
            batch = []
        batch_key = key
        ts = datetime.fromtimestamp(epoch, tz=UTC).isoformat() + f"#{phase}"
        batch.append((pair, bid, ask, ts))
    if batch_key is not None:
        yield batch_key[0], batch_key[1], batch


def _cursor_key(epoch: int, phase: str, intrabar: str) -> tuple[int, int]:
    try:
        return epoch, PHASE_ORDERS[intrabar][phase]
    except KeyError as exc:
        raise ValueError("invalid replay cursor phase") from exc


def run_replay(
    args,
    broker: VirtualBroker,
    session_dir: Path,
    bot=None,
    *,
    replay_identity_sha256: str | None = None,
    expected_shards=None,
    mtm_contract: dict | None = None,
) -> dict:
    pairs = _normalized_pairs(args.pairs)
    root = Path(args.corpus_root).expanduser().resolve()
    bars_per_second = _finite_number(
        "bars_per_second", args.bars_per_second, positive=True
    )
    state_every_value = _finite_number("state_every", args.state_every, positive=True)
    if not state_every_value.is_integer():
        raise ValueError("state_every must be an integer")
    state_every = int(state_every_value)
    bar_sleep = 1.0 / bars_per_second
    no_sleep = bars_per_second >= 10000
    step_file = session_dir / "inbox" / "STEP"
    current_epoch = None
    pending_bars: dict[str, dict] = {}
    bot_minute: dict[str, dict] = {}
    aggregate_bot_bars = (
        bot is not None and args.bot_bar == "M1" and args.granularity != "M1"
    )
    cursor = broker.feed_cursor
    resume_key: tuple[int, int] | None = None
    bar_count = 0
    if cursor is not None:
        if mtm_contract is not None:
            raise VirtualBrokerError(
                "resumed replay cannot attest the continuous MTM contract"
            )
        if cursor.get("mode") != "replay":
            raise VirtualBrokerError("snapshot feed mode is not replay")
        if (
            replay_identity_sha256 is None
            or cursor.get("replay_identity_sha256") != replay_identity_sha256
        ):
            raise VirtualBrokerError("replay resume identity mismatch")
        if bot is not None:
            raise VirtualBrokerError("bot replay resume requires persisted bot state")
        epoch_value = _finite_number("resume cursor epoch", cursor.get("epoch"))
        if not epoch_value.is_integer():
            raise VirtualBrokerError("resume cursor epoch must be an integer")
        resume_key = _cursor_key(
            int(epoch_value), str(cursor.get("phase")), args.intrabar
        )
        current_epoch = resume_key[0]
        bar_count_value = _finite_number(
            "resume cursor bar_count", cursor.get("bar_count", 0)
        )
        if not bar_count_value.is_integer() or bar_count_value < 0:
            raise VirtualBrokerError(
                "resume cursor bar_count must be non-negative integer"
            )
        bar_count = int(bar_count_value)
    elif broker.positions or broker.orders:
        raise VirtualBrokerError(
            "replay snapshot with exposure has no causal feed cursor"
        )

    bar_seconds = 5 if args.granularity == "S5" else 60
    processed_any = False
    phase_mark_count = 0
    first_coordinate = None
    last_coordinate = None
    last_batch_receipt = None
    last_phase_mark = None
    expected_previous_batch_sha256 = "0" * 64
    for epoch, phase, quote_batch in _iter_replay_quote_batches(
        root,
        pairs,
        args.time_from,
        args.time_to,
        args.intrabar,
        args.granularity,
        expected_shards=expected_shards,
    ):
        key = _cursor_key(epoch, phase, args.intrabar)
        if resume_key is not None and key <= resume_key:
            continue
        coordinate = _replay_coordinate(
            epoch=epoch,
            phase=phase,
            granularity=args.granularity,
            intrabar=args.intrabar,
        )
        boundary = phase == "O" and epoch != current_epoch
        publish_state = False
        closed_bars: list[tuple[str, dict]] = []
        closed_minutes: list[tuple[str, dict]] = []
        if boundary and current_epoch is not None:
            bar_count += 1
            publish_state = bar_count % state_every == 0
            if publish_state or args.step:
                _write_state(
                    session_dir,
                    broker,
                    datetime.fromtimestamp(
                        current_epoch + bar_seconds, tz=UTC
                    ).isoformat(),
                    "replay",
                )
            if args.step:
                while not step_file.exists():
                    time_mod.sleep(0.2)
                os.replace(
                    step_file,
                    session_dir
                    / "inbox"
                    / "processed"
                    / f"{int(time_mod.time()*1000)}_STEP",
                )
            elif not no_sleep:
                time_mod.sleep(bar_sleep)
            if bot is not None and not aggregate_bot_bars:
                closed_bars = [
                    (bpair, bbar)
                    for bpair, bbar in pending_bars.items()
                    if bbar["epoch"] == current_epoch
                ]
            if aggregate_bot_bars:
                minute = epoch // 60 * 60
                closed_minutes = [
                    (bpair, bbar)
                    for bpair, bbar in bot_minute.items()
                    if bbar["epoch"] != minute
                ]

        # Existing orders/exits see the next executable phase first.  Only
        # after the new O is staged may a closed-bar decision submit MARKET.
        batch_receipt = None
        if mtm_contract is not None:
            expected_receipt = _expected_quote_batch_receipt(
                quote_batch,
                coordinate=coordinate,
                feed_pairs=pairs,
                batch_index=phase_mark_count,
                previous_batch_sha256=expected_previous_batch_sha256,
            )
            batch_receipt = broker.record_quote_batch_begin(
                quote_batch,
                coordinate=coordinate,
                feed_pairs=pairs,
            )
            if batch_receipt != expected_receipt:
                raise VirtualBrokerError(
                    "runtime quote-batch receipt differs from sealed commitment"
                )
            if not batch_receipt["coverage_complete"]:
                raise VirtualBrokerError(
                    "runtime quote batch lacks complete feed-pair coverage"
                )
            expected_previous_batch_sha256 = batch_receipt["batch_sha256"]
        broker.on_quote_batch(quote_batch)
        processed_any = True
        broker.feed_cursor = {
            "mode": "replay",
            "epoch": epoch,
            "phase": phase,
            "bar_count": bar_count,
            "completed": False,
            "replay_identity_sha256": replay_identity_sha256,
        }
        if boundary:
            for bpair, bbar in closed_bars:
                bot.on_bar_closed(bpair, bbar, bbar["epoch"])
            for bpair, bbar in closed_minutes:
                bot.on_bar_closed(bpair, bbar, bbar["epoch"])
                del bot_minute[bpair]
            if publish_state or args.step:
                _process_inbox(session_dir, broker, allowed_pairs=set(pairs))
            current_epoch = epoch

        for pair, bid, ask, _ in quote_batch:
            if phase == "O":
                pending_bars[pair] = {
                    "bid_o": bid,
                    "ask_o": ask,
                    "bid_h": bid,
                    "bid_l": bid,
                    "ask_h": ask,
                    "ask_l": ask,
                    "bid_c": bid,
                    "ask_c": ask,
                    "epoch": epoch,
                }
            else:
                pb = pending_bars.get(pair)
                if pb is not None:
                    pb["bid_h"] = max(pb["bid_h"], bid)
                    pb["bid_l"] = min(pb["bid_l"], bid)
                    pb["ask_h"] = max(pb["ask_h"], ask)
                    pb["ask_l"] = min(pb["ask_l"], ask)
                    pb["bid_c"] = bid
                    pb["ask_c"] = ask
            if aggregate_bot_bars:
                minute = epoch // 60 * 60
                mb = bot_minute.get(pair)
                if mb is None:
                    bot_minute[pair] = {
                        "epoch": minute,
                        "bid_o": bid,
                        "bid_h": bid,
                        "bid_l": bid,
                        "bid_c": bid,
                        "ask_o": ask,
                        "ask_h": ask,
                        "ask_l": ask,
                        "ask_c": ask,
                    }
                else:
                    mb["bid_h"] = max(mb["bid_h"], bid)
                    mb["bid_l"] = min(mb["bid_l"], bid)
                    mb["ask_h"] = max(mb["ask_h"], ask)
                    mb["ask_l"] = min(mb["ask_l"], ask)
                    mb["bid_c"] = bid
                    mb["ask_c"] = ask

        if mtm_contract is not None:
            last_phase_mark = broker.account_mark(
                "PHASE",
                coordinate=coordinate,
                batch_receipt=batch_receipt,
                feed_cursor=dict(broker.feed_cursor),
            )
            if first_coordinate is None:
                first_coordinate = coordinate
            last_coordinate = coordinate
            last_batch_receipt = batch_receipt
            phase_mark_count += 1

    if processed_any and broker.feed_cursor is not None:
        broker.feed_cursor["completed"] = True
    elif cursor is not None:
        broker.feed_cursor = dict(cursor)
        broker.feed_cursor["completed"] = True
    _write_state(session_dir, broker, "REPLAY_END", "replay", "replay finished")
    runtime = {
        "phase_mark_count": phase_mark_count,
        "first_coordinate": first_coordinate,
        "last_coordinate": last_coordinate,
        "last_batch_receipt": last_batch_receipt,
        "last_phase_mark": last_phase_mark,
    }
    if mtm_contract is not None:
        actual = {
            "expected_phase_mark_count": phase_mark_count,
            "expected_batch_chain_terminal_sha256": (
                last_batch_receipt["batch_sha256"]
                if last_batch_receipt is not None
                else None
            ),
            "first_coordinate": first_coordinate,
            "last_coordinate": last_coordinate,
        }
        for field, actual_value in actual.items():
            if mtm_contract.get(field) != actual_value:
                raise VirtualBrokerError(
                    f"runtime MTM coordinate commitment mismatch: {field}"
                )
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feed", choices=["live", "replay"], required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--pairs", default="USD_JPY,EUR_USD")
    parser.add_argument("--balance", type=float, default=200_000.0)
    parser.add_argument(
        "--minutes", type=float, default=480.0, help="live mode duration"
    )
    parser.add_argument(
        "--corpus-root",
        default=(
            "/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026"
        ),
    )
    parser.add_argument("--from", dest="time_from", default="2026-01-05T00:00:00")
    parser.add_argument("--to", dest="time_to", default="2026-01-10T00:00:00")
    parser.add_argument("--bars-per-second", type=float, default=20.0)
    parser.add_argument(
        "--step",
        action="store_true",
        help="replay: advance one bar per inbox/STEP file",
    )
    bot_group = parser.add_mutually_exclusive_group()
    bot_group.add_argument(
        "--bot",
        choices=["golden_burst", "golden_burst_blindspread"],
        default=None,
        help="built-in worker bot (same broker/ledger)",
    )
    bot_group.add_argument(
        "--bot-module",
        default=None,
        help="path to ANY bot file: <file.py>[:ClassName]; the class "
        "takes (broker) and implements on_bar_closed(pair, bar, epoch)",
    )
    parser.add_argument(
        "--granularity",
        choices=["M1", "S5"],
        default="M1",
        help="replay feed granularity (S5 = 12x finer fill realism)",
    )
    parser.add_argument(
        "--bot-bar",
        choices=["feed", "M1"],
        default="feed",
        help="bot decision cadence: per feed bar, or aggregated M1",
    )
    parser.add_argument(
        "--state-every",
        type=int,
        default=1,
        help="replay: write state/process inbox every N bars (lab speed)",
    )
    parser.add_argument(
        "--fast-ledger",
        action="store_true",
        help="ledger flush without fsync (lab runs)",
    )
    parser.add_argument(
        "--continuous-mtm",
        action="store_true",
        help="replay: seal coordinate-complete account marks; fresh sessions only",
    )
    parser.add_argument(
        "--slippage-pips",
        type=float,
        default=0.0,
        help="stress: extra pips against the trader on every fill",
    )
    parser.add_argument(
        "--financing-pips-day",
        type=float,
        default=0.0,
        help="holding cost in pips per 24h held (pro-rata)",
    )
    parser.add_argument(
        "--intrabar",
        choices=["OHLC", "OLHC"],
        default="OHLC",
        help="declared synthetic intrabar path; run both to bracket "
        "both-touch ambiguity",
    )
    parser.add_argument(
        "--strategy-owner-id",
        default=None,
        help="required custom-bot owner identity, ledger-bound and scorer-checked",
    )
    parser.add_argument(
        "--bot-dependency",
        action="append",
        default=[],
        help="repo-local custom-bot dependency path; repeat for the complete closure",
    )
    parser.add_argument(
        "--settle-at-end",
        action="store_true",
        help="replay custom bot: cancel its owned orders and close its owned trades",
    )
    args = parser.parse_args()

    session_dir = args.session_dir
    (session_dir / "inbox" / "processed").mkdir(parents=True, exist_ok=True)
    broker = VirtualBroker(
        ledger_path=session_dir / "ledger.jsonl",
        balance_jpy=args.balance,
        fast_ledger=args.fast_ledger,
        slippage_pips=args.slippage_pips,
        financing_pips_per_day=args.financing_pips_day,
    )
    snap_path = session_dir / "broker_snapshot.json"
    if snap_path.exists():
        broker.restore(_strict_json_loads(snap_path.read_text()))
    reproducibility_manifest = _build_reproducibility_manifest(args)
    replay_identity = (
        _replay_identity_sha256(reproducibility_manifest)
        if args.feed == "replay"
        else None
    )
    broker._log(
        "SESSION_START",
        {
            "contract": "QR_VIRTUAL_MARKET_SESSION_V1",
            "feed": args.feed,
            "pairs": args.pairs,
            "balance": broker.balance_jpy,
            "order_authority": "NONE",
            "reproducibility_manifest": reproducibility_manifest,
            "reproducibility_manifest_sha256": reproducibility_manifest[
                "manifest_sha256"
            ],
        },
    )
    bot = None
    if args.bot == "golden_burst":
        bot = GoldenBurstBot(broker)
    elif args.bot == "golden_burst_blindspread":
        # live-faithful configuration: the 2025-12 live worker's spread
        # monitor supplied nothing, so its gate never saw the spread
        bot = GoldenBurstBot(broker, blind_spread=True)
    elif args.bot_module:
        import types as _types

        spec_str = args.bot_module
        module_path, _, class_name = spec_str.partition(":")
        module_path = str(Path(module_path).expanduser().resolve())
        module_bytes = _verify_custom_bot_seal(reproducibility_manifest["bot"])
        _mod = _types.ModuleType("dojo_custom_bot")
        _mod.__file__ = module_path
        sys.modules["dojo_custom_bot"] = _mod
        exec(compile(module_bytes, module_path, "exec"), _mod.__dict__)
        bot_cls = getattr(_mod, class_name or "Bot")
        bot = bot_cls(broker)
        _verify_custom_bot_seal(reproducibility_manifest["bot"])
        broker._log(
            "BOT_LOADED",
            {
                "module": module_path,
                "class": class_name or "Bot",
                "strategy_owner_id": args.strategy_owner_id,
            },
        )
    mtm_contract = (
        reproducibility_manifest["replay"]
        if reproducibility_manifest["replay"].get("mtm_coordinate_contract")
        else None
    )
    mtm_start_mark = None
    replay_runtime = None
    replay_completed = False
    terminal_mark = None
    try:
        if mtm_contract is not None:
            # START deliberately follows custom BOT_LOADED (or completion of
            # built-in/none construction), so no constructor side effect can
            # be hidden before the continuous account curve begins.
            mtm_start_mark = broker.account_mark("START")
        if args.feed == "live":
            run_live(args, broker, session_dir, bot=bot)
        else:
            replay_runtime = run_replay(
                args,
                broker,
                session_dir,
                bot=bot,
                replay_identity_sha256=replay_identity,
                expected_shards=reproducibility_manifest["corpus"]["shards"],
                mtm_contract=mtm_contract,
            )
            if args.settle_at_end:
                _settle_custom_bot_at_end(broker, args.strategy_owner_id)
            replay_completed = True
    finally:
        terminal_failure = None
        mtm_complete = False
        if mtm_contract is not None and replay_completed:
            try:
                if replay_runtime is None:
                    raise VirtualBrokerError("continuous MTM replay result is missing")
                terminal_mark = broker.account_mark(
                    "TERMINAL",
                    batch_receipt=replay_runtime["last_batch_receipt"],
                    feed_cursor=(
                        dict(broker.feed_cursor)
                        if broker.feed_cursor is not None
                        else None
                    ),
                )
                expected_total_marks = mtm_contract["expected_phase_mark_count"] + 2
                if mtm_start_mark is None or mtm_start_mark.get("kind") != "START":
                    raise VirtualBrokerError("continuous MTM START mark is missing")
                if (
                    replay_runtime["phase_mark_count"]
                    != mtm_contract["expected_phase_mark_count"]
                ):
                    raise VirtualBrokerError("continuous MTM phase count mismatch")
                if (
                    replay_runtime["last_batch_receipt"]["batch_sha256"]
                    != mtm_contract["expected_batch_chain_terminal_sha256"]
                ):
                    raise VirtualBrokerError("continuous MTM batch chain mismatch")
                if terminal_mark["mark_index"] + 1 != expected_total_marks:
                    raise VirtualBrokerError("continuous MTM mark count mismatch")
                mtm_complete = True
            except Exception as exc:  # fail closed, but still seal SESSION_STOP
                terminal_failure = exc

        try:
            stop_account = (
                terminal_mark["account"]
                if mtm_complete and terminal_mark is not None
                else (broker.account() if broker.last_quotes else None)
            )
            stop_error = (
                str(terminal_failure)[:200] if terminal_failure is not None else None
            )
        except VirtualBrokerError as exc:
            stop_account = None
            stop_error = str(exc)[:200]
        stop_payload = {"account": stop_account, "account_error": stop_error}
        if mtm_contract is not None:
            stop_payload.update(
                {
                    "mtm_complete": mtm_complete,
                    "mtm_mark_count": (
                        terminal_mark["mark_index"] + 1
                        if terminal_mark is not None
                        else None
                    ),
                    "mtm_terminal_mark_sha256": (
                        terminal_mark["mark_sha256"]
                        if terminal_mark is not None
                        else None
                    ),
                }
            )
        broker._log("SESSION_STOP", stop_payload)
        tmp = session_dir / ".broker_snapshot.json.tmp"
        tmp.write_text(
            json.dumps(
                broker.snapshot(), ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        )
        os.replace(tmp, snap_path)
        if terminal_failure is not None:
            raise terminal_failure
    print(
        json.dumps(
            {
                "status": "SESSION_DONE",
                "account": broker.account() if broker.last_quotes else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
