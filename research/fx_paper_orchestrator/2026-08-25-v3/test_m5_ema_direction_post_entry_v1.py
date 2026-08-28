from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER_PATH = HERE / "run_m5_ema_direction_post_entry_v1.py"
PREREG_PATH = HERE / "M5_EMA_DIRECTION_POST_ENTRY_V1_PREREGISTRATION.json"

SPEC = importlib.util.spec_from_file_location("m5_ema_direction_post_entry_v1", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

D = Decimal


def make_bar(
    pair: str,
    ordinal: int,
    start_ns: int,
    mid_open: str,
    mid_high: str,
    mid_low: str,
    mid_close: str,
    spread: str = "0.0002",
) -> M.Bar:
    half = D(spread) / D(2)
    values = [D(mid_open), D(mid_high), D(mid_low), D(mid_close)]
    bid = [value - half for value in values]
    ask = [value + half for value in values]
    return M.Bar(
        pair=pair,
        ordinal=ordinal,
        start_ns=start_ns,
        time=M.format_epoch_ns(start_ns),
        bid_o=bid[0],
        bid_h=bid[1],
        bid_l=bid[2],
        bid_c=bid[3],
        ask_o=ask[0],
        ask_h=ask[1],
        ask_l=ask[2],
        ask_c=ask[3],
        row_sha256=hashlib.sha256(f"{pair}:{ordinal}:{start_ns}".encode()).hexdigest(),
    )


def constant_bars(
    count: int,
    *,
    pair: str = "EUR_USD",
    start_ns: int = 1_700_000_100_000_000_000,
    gap_after: int | None = None,
) -> list[M.Bar]:
    bars: list[M.Bar] = []
    current = start_ns - (start_ns % M.BAR_NS)
    for ordinal in range(count):
        bars.append(
            make_bar(pair, ordinal, current, "1.1000", "1.1001", "1.0999", "1.1000")
        )
        current += M.BAR_NS
        if gap_after == ordinal:
            current += 7 * M.BAR_NS
    return bars


def manual_signal(bar: M.Bar, direction: int, fill_ordinal: int | None) -> M.Signal:
    decision_ns = bar.end_ns
    signal_id = hashlib.sha256(
        f"{M.CANDIDATE_ID}|{bar.pair}|{decision_ns}|{direction}".encode("ascii")
    ).hexdigest()
    return M.Signal(
        signal_id=signal_id,
        pair=bar.pair,
        direction=direction,
        decision_bar_ordinal=bar.ordinal,
        decision_source_time=bar.time,
        decision_ns=decision_ns,
        fill_bar_ordinal=fill_ordinal,
        fill_source_time=None,
        source_row_sha256=bar.row_sha256,
        ema3=D("1.1"),
        ema12=D("1.0"),
    )


def target_freeze(pair: str = "EUR_USD", target_pips: str = "10") -> dict[str, object]:
    return {
        "strata": {
            pair: {
                direction: {
                    str(age): {"frozen_target_pips": target_pips}
                    for age in M.MAX_AGES
                }
                for direction in ("LONG", "SHORT")
            }
        }
    }


def source_row(pair: str, when: datetime, mid: float, previous: float) -> dict[str, object]:
    pip = 0.01 if pair == "USD_JPY" else 0.0001
    spread = 0.02 if pair == "USD_JPY" else 0.0002
    bid_o = previous - spread / 2
    bid_c = mid - spread / 2
    ask_o = previous + spread / 2
    ask_c = mid + spread / 2
    padding = pip * 0.6
    return {
        "ask": {
            "o": ask_o,
            "h": max(ask_o, ask_c) + padding,
            "l": min(ask_o, ask_c) - padding,
            "c": ask_c,
        },
        "bid": {
            "o": bid_o,
            "h": max(bid_o, bid_c) + padding,
            "l": min(bid_o, bid_c) - padding,
            "c": bid_c,
        },
        "complete": True,
        "granularity": "M5",
        "pair": pair,
        "price": "BA",
        "time": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "volume": 1,
    }


def write_synthetic_source(path: Path, pair: str) -> str:
    base = 145.0 if pair == "USD_JPY" else (0.66 if pair == "AUD_USD" else 1.08)
    unit = 0.01 if pair == "USD_JPY" else 0.0001
    pattern = (0, 1, 3, 6, 9, 7, 4, 1, -2, -5, -8, -6, -3, 0)
    segments = (
        (datetime(2026, 4, 29, tzinfo=timezone.utc), 180),
        (datetime(2026, 5, 1, tzinfo=timezone.utc), 120),
        (datetime(2026, 6, 1, tzinfo=timezone.utc), 120),
    )
    rows: list[bytes] = []
    previous = base
    for segment_start, count in segments:
        for index in range(count):
            current = base + pattern[index % len(pattern)] * unit
            row = source_row(pair, segment_start + timedelta(minutes=5 * index), current, previous)
            rows.append(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            previous = current
    raw = gzip.compress(b"".join(rows), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class EmaDirectionUnitTests(unittest.TestCase):
    def test_sma_seeded_ema_emits_every_eligible_bar_and_carries_equality(self) -> None:
        bars = constant_bars(15)
        signals = M.ema_signals(bars)
        self.assertEqual(len(signals), 4)
        self.assertEqual([signal.decision_bar_ordinal for signal in signals], [11, 12, 13, 14])
        self.assertEqual([signal.direction for signal in signals], [1, 1, 1, 1])
        self.assertTrue(all(signal.ema3 == D("1.1000") for signal in signals))
        self.assertTrue(all(signal.ema12 == D("1.1000") for signal in signals))
        for signal in signals:
            expected = hashlib.sha256(
                f"{M.CANDIDATE_ID}|{signal.pair}|{signal.decision_ns}|{signal.direction}".encode(
                    "ascii"
                )
            ).hexdigest()
            self.assertEqual(signal.signal_id, expected)

    def test_next_record_fill_is_causal_across_a_market_gap(self) -> None:
        bars = constant_bars(13, gap_after=11)
        signal = M.ema_signals(bars)[0]
        self.assertEqual(signal.decision_bar_ordinal, 11)
        self.assertEqual(signal.decision_ns, bars[11].end_ns)
        self.assertEqual(signal.fill_bar_ordinal, 12)
        self.assertEqual(signal.fill_source_time, bars[12].time)
        self.assertGreater(bars[12].start_ns, signal.decision_ns)

    def test_nearest_rank_q40_uses_ceiling_rank(self) -> None:
        self.assertEqual(M.nearest_rank([D("9"), D("1"), D("7"), D("3"), D("5")], D("0.4")), D("3"))
        self.assertEqual(M.nearest_rank([D("1"), D("2"), D("3")], D("0.4")), D("2"))
        with self.assertRaises(M.ChallengerError):
            M.nearest_rank([], D("0.4"))

    def test_one_common_exit_has_componentwise_cost_ordering(self) -> None:
        bars = constant_bars(3)
        signal = manual_signal(bars[0], 1, 1)
        position = M.open_position(signal, bars[1], None)
        pnl, ratios = M.pnl_for_exit(position, bars[2], "MARKET_OPEN", bars[2].start_ns, D("1000"))
        self.assertEqual(set(pnl), set(M.COST_ARMS))
        self.assertEqual(set(ratios), set(M.COST_ARMS))
        self.assertGreaterEqual(pnl["RAW_SIGNAL"], pnl["EXECUTABLE_BASE"])
        self.assertGreaterEqual(pnl["EXECUTABLE_BASE"], pnl["ADVERSE_STRESS"])
        self.assertLess(pnl["ADVERSE_STRESS"], pnl["EXECUTABLE_BASE"])

    def test_finite_max_age_closes_on_next_open_and_terminal_liquidates(self) -> None:
        bars = constant_bars(10)
        signals = [manual_signal(bars[0], 1, 1), manual_signal(bars[8], 1, 9)]
        run = M.simulate_pair(
            bars,
            signals,
            bars[0].start_ns,
            bars[-1].end_ns,
            "A_MAX_AGE_ONLY__H06",
            target_freeze(),
            D("1000"),
            24,
        )
        first = next(trade for trade in run.trades if trade.signal_id == signals[0].signal_id)
        self.assertEqual(first.exit_reason, "FINITE_MAX_AGE")
        self.assertEqual(first.age_completed_bars, 6)
        self.assertEqual(first.exit_bar_ordinal, 7)
        self.assertEqual(run.terminal_liquidation_count, 1)
        self.assertEqual(run.terminal_open_inventory, 0)

    def test_opposite_signal_closes_exactly_the_oldest_opposite_lot(self) -> None:
        bars = constant_bars(8)
        signals = [
            manual_signal(bars[0], 1, 1),
            manual_signal(bars[1], 1, 2),
            manual_signal(bars[2], -1, 3),
        ]
        run = M.simulate_pair(
            bars,
            signals,
            bars[0].start_ns,
            bars[-1].end_ns,
            "B_OPPOSITE_SIGNAL_OLDEST_FIRST__H24",
            target_freeze(),
            D("1000"),
            24,
        )
        oldest = next(trade for trade in run.trades if trade.signal_id == signals[0].signal_id)
        newer = next(trade for trade in run.trades if trade.signal_id == signals[1].signal_id)
        self.assertEqual(oldest.exit_reason, "OPPOSITE_SIGNAL_OLDEST_FIRST")
        self.assertEqual(oldest.exit_bar_ordinal, 3)
        self.assertEqual(newer.exit_reason, "TERMINAL_LIQUIDATION")

    def test_tp_touch_precedes_other_exits_and_giveback_uses_next_open(self) -> None:
        start = 1_700_000_100_000_000_000
        start -= start % M.BAR_NS
        tp_bars = [
            make_bar("EUR_USD", 0, start, "1.0000", "1.0001", "0.9999", "1.0000"),
            make_bar("EUR_USD", 1, start + M.BAR_NS, "1.0000", "1.0002", "0.9999", "1.0001"),
            make_bar("EUR_USD", 2, start + 2 * M.BAR_NS, "1.0001", "1.0007", "1.0000", "1.0005"),
            make_bar("EUR_USD", 3, start + 3 * M.BAR_NS, "1.0005", "1.0006", "1.0003", "1.0004"),
        ]
        signal = manual_signal(tp_bars[0], 1, 1)
        tp_run = M.simulate_pair(
            tp_bars,
            [signal],
            tp_bars[0].start_ns,
            tp_bars[-1].end_ns,
            "D_TP_Q40_PROFIT_GIVEBACK__H06",
            target_freeze(target_pips="2"),
            D("1000"),
            24,
        )
        self.assertEqual(tp_run.trades[0].exit_reason, "FROZEN_TP_Q40")
        self.assertEqual(tp_run.trades[0].exit_bar_ordinal, 2)

        giveback_bars = [
            make_bar("EUR_USD", 0, start, "1.0000", "1.0001", "0.9999", "1.0000"),
            make_bar("EUR_USD", 1, start + M.BAR_NS, "1.0000", "1.0002", "0.9999", "1.0001"),
            make_bar("EUR_USD", 2, start + 2 * M.BAR_NS, "1.0001", "1.0010", "1.0000", "1.0009"),
            make_bar("EUR_USD", 3, start + 3 * M.BAR_NS, "1.0009", "1.0010", "1.0003", "1.0005"),
            make_bar("EUR_USD", 4, start + 4 * M.BAR_NS, "1.0005", "1.0006", "1.0003", "1.0004"),
            make_bar("EUR_USD", 5, start + 5 * M.BAR_NS, "1.0004", "1.0005", "1.0002", "1.0003"),
        ]
        giveback = M.simulate_pair(
            giveback_bars,
            [manual_signal(giveback_bars[0], 1, 1)],
            giveback_bars[0].start_ns,
            giveback_bars[-1].end_ns,
            "D_TP_Q40_PROFIT_GIVEBACK__H24",
            target_freeze(target_pips="10"),
            D("1000"),
            24,
        )
        self.assertEqual(giveback.trades[0].exit_reason, "PROFIT_GIVEBACK_UNWIND")
        self.assertEqual(giveback.trades[0].exit_bar_ordinal, 4)

    def test_exact_twelve_configs_share_signal_identity_and_cost_arm_fanout(self) -> None:
        self.assertEqual(len(M.CONFIG_IDS), 12)
        self.assertEqual(
            set(M.CONFIG_IDS),
            {f"{policy}__H{age:02d}" for policy in M.POLICIES for age in M.MAX_AGES},
        )
        bars = constant_bars(18)
        signals = M.ema_signals(bars)
        expected_ids = M.signal_id_set_sha256(
            signal.signal_id for signal in signals if signal.fill_bar_ordinal is not None
        )
        for config_id in M.CONFIG_IDS:
            run = M.simulate_pair(
                bars,
                signals,
                bars[0].start_ns,
                bars[-1].end_ns,
                config_id,
                target_freeze(target_pips="100"),
                D("1000"),
                24,
            )
            self.assertEqual(run.filled_signal_id_set_sha256, expected_ids)
            for trade in run.trades:
                self.assertEqual(set(trade.pnl_jpy), set(M.COST_ARMS))
                self.assertEqual(set(trade.return_ratio), set(M.COST_ARMS))

    def test_pair_cap_rejections_are_visible_and_terminal_inventory_is_zero(self) -> None:
        bars = constant_bars(7)
        signals = [manual_signal(bars[index], 1, index + 1) for index in range(5)]
        run = M.simulate_pair(
            bars,
            signals,
            bars[0].start_ns,
            bars[-1].end_ns,
            "A_MAX_AGE_ONLY__H24",
            target_freeze(),
            D("1000"),
            1,
        )
        self.assertEqual(run.max_open_lots, 1)
        self.assertEqual(run.cap_rejected_count, 4)
        self.assertEqual(run.terminal_open_inventory, 0)
        self.assertEqual(run.terminal_liquidation_count, 1)

    def test_discovery_contract_has_no_profit_threshold_or_holdout_capability(self) -> None:
        prereg = json.loads(PREREG_PATH.read_text())
        M.validate_prereg(prereg)
        self.assertNotIn("threshold", inspect.signature(M.selection_receipt).parameters)
        fake: dict[str, dict[str, object]] = {}
        for offset, config_id in enumerate(M.CONFIG_IDS):
            multiple = D("1.01") - D(offset) / D("10000")
            fake[config_id] = {
                "summary_sha256": hashlib.sha256(config_id.encode()).hexdigest(),
                "arms": {
                    arm: {
                        "ending_equity_multiple": str(multiple),
                        "mean_net_expectancy_jpy": "1",
                    }
                    for arm in ("EXECUTABLE_BASE", "ADVERSE_STRESS")
                },
            }
        receipt = M.selection_receipt(fake)
        self.assertFalse(receipt["walk_forward_metrics_read_before_selection"])
        self.assertFalse(receipt["holdout_read"])
        self.assertFalse(receipt["two_x_or_three_x_used"])
        leaked = copy.deepcopy(prereg)
        leaked["selection_contract"]["two_x_or_three_x_used_for_selection"] = True
        with self.assertRaises(M.ChallengerError):
            M.validate_prereg(leaked)
        opened = copy.deepcopy(prereg)
        opened["periods"]["holdout"]["may_read"] = True
        with self.assertRaises(M.ChallengerError):
            M.validate_prereg(opened)


class ContentAddressedEndToEndTest(unittest.TestCase):
    def test_compact_synthetic_build_verify_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            prereg = json.loads(PREREG_PATH.read_text())
            descriptors: dict[str, dict[str, str]] = {}
            for pair in ("EUR_USD", "USD_JPY", "AUD_USD"):
                relative = f"{pair}/{pair}.jsonl.gz"
                sha = write_synthetic_source(input_root / relative, pair)
                descriptors[pair] = {"relative_path": relative, "sha256": sha}
            prereg["source_contract"]["root"] = "SYNTHETIC_TEST_ROOT"
            prereg["source_contract"]["files"] = descriptors
            prereg_path = root / M.PREREG_NAME
            prereg_path.write_bytes(M.canonical_json_bytes(prereg))

            built = M.official_build(prereg_path, input_root, output_root)
            self.assertTrue(built["verified"])
            self.assertEqual(built["status"], "UNADMITTED_CHALLENGER")
            self.assertTrue(built["profit_unproven"])
            self.assertTrue(built["holdout_unopened"])
            self.assertEqual(built["external_orders"], 0)
            verified = M.verify_artifacts(prereg_path, input_root, output_root)
            self.assertTrue(verified["verified"])
            manifest = M.strict_json_file(output_root / M.ARTIFACT_MANIFEST_NAME)
            packet_path = output_root / manifest["packet_path"]
            packet = M.strict_json_file(packet_path)
            self.assertEqual(packet_path.name, f"UNADMITTED_CHALLENGER_PACKET_{packet['packet_sha256']}.json")
            self.assertEqual(packet["same_signal_id_cost_arms"], list(M.COST_ARMS))
            self.assertFalse(packet["cost_gate_at_entry"])
            self.assertFalse(packet["authority"]["live_authority"])
            self.assertFalse(packet["authority"]["credential_access"])

            result_path = output_root / M.RESULT_NAME
            result = M.strict_json_file(result_path)
            result["profit_unproven"] = False
            result_path.write_bytes(M.canonical_json_bytes(result))
            with self.assertRaises(M.ChallengerError):
                M.verify_artifacts(prereg_path, input_root, output_root)


if __name__ == "__main__":
    unittest.main()
