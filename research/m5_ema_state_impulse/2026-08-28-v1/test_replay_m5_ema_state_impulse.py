import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "m5_ema_state_impulse_replay", HERE / "replay_m5_ema_state_impulse.py"
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def synthetic_bar(pair, when, mid_o, mid_c=None, spread=None, wing=None):
    mid_c = mid_o if mid_c is None else mid_c
    spread = spread if spread is not None else (0.02 if pair.endswith("_JPY") else 0.0002)
    wing = wing if wing is not None else (0.03 if pair.endswith("_JPY") else 0.0003)
    mid_h = max(mid_o, mid_c) + wing
    mid_l = min(mid_o, mid_c) - wing
    half = spread / 2.0
    bid = (mid_o - half, mid_h - half, mid_l - half, mid_c - half)
    ask = (mid_o + half, mid_h + half, mid_l + half, mid_c + half)
    material = json.dumps([pair, when, bid, ask], separators=(",", ":")).encode()
    return M.Bar(when, bid, ask, 10, hashlib.sha256(material).hexdigest())


def trending_bars(pair, count=40, start=0, step=None):
    if pair.endswith("_JPY"):
        value = 150.0
        step = 0.01 if step is None else step
    else:
        value = 1.0
        step = 0.0001 if step is None else step
    rows = []
    for index in range(count):
        close = value + step
        rows.append(synthetic_bar(pair, start + index * 300, value, close))
        value = close
    return rows


def manual_signal(pair, decision, fill, side="LONG", tp_distance=0.01):
    return M.Signal(
        signal_id=hashlib.sha256(f"{pair}|{decision}|{side}".encode()).hexdigest(),
        pair=pair,
        side=side,
        decision_time=decision,
        decision_bar_time=decision - 300,
        expected_fill_time=fill,
        decision_bar_hash="a" * 64,
        fast_ema=1.1,
        slow_ema=1.0,
        slow_slope=0.01,
        momentum_price=0.01,
        atr_price=0.01,
        observed_spread_price=0.0002,
        tp_distance_price=tp_distance,
        direction_correct_at_six_bars=True,
    )


class PreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg = M.load_preregistration()

    def test_file_and_canonical_seals_are_frozen(self):
        self.assertEqual(M.sha_file(HERE / "PREREGISTRATION.json"), M.EXPECTED_PREREG_FILE_SHA256)
        self.assertEqual(
            M.sha_bytes(M.canonical(self.prereg)), M.EXPECTED_PREREG_CANONICAL_SHA256
        )
        self.assertEqual(len(M.validate_preregistration(self.prereg)), 14)

    def test_exact_runtime_configuration_is_bound(self):
        s = self.prereg["strategy"]
        self.assertEqual(
            (s["fast_ema_bars"], s["slow_ema_bars"], s["momentum_bars"],
             s["atr_bars"], s["tp_atr_multiple"], s["tp_spread_multiple_floor"],
             s["max_age_bars"], s["virtual_units"]),
            (3, 6, 3, 6, 0.5, 1.5, 6, 1000),
        )
        self.assertFalse(s["entry_cost_gate_used"])

    def test_audit_does_not_open_or_decode_candles(self):
        audit = M.audit_local_contracts(self.prereg)
        self.assertEqual(audit["candle_files_opened"], 0)
        self.assertEqual(audit["rows_decoded"], 0)
        self.assertEqual(audit["post_boundary_bytes_read"], 0)
        self.assertEqual(audit["external_orders"], 0)

    def test_independent_review_is_required(self):
        original = M.REVIEW_PATH
        try:
            M.REVIEW_PATH = HERE / "DOES_NOT_EXIST_REVIEW.json"
            with self.assertRaisesRegex(PermissionError, "INDEPENDENT_REVIEW_REQUIRED"):
                M.validate_review_receipt(self.prereg)
        finally:
            M.REVIEW_PATH = original

    def test_one_shot_fails_before_any_prefix_reader_without_review(self):
        original_review = M.REVIEW_PATH
        original_reader = M.read_exact_prefix
        try:
            M.REVIEW_PATH = HERE / "DOES_NOT_EXIST_REVIEW.json"
            M.read_exact_prefix = lambda *args, **kwargs: self.fail(
                "candle prefix reader reached before independent review"
            )
            with self.assertRaisesRegex(PermissionError, "INDEPENDENT_REVIEW_REQUIRED"):
                M.run()
        finally:
            M.REVIEW_PATH = original_review
            M.read_exact_prefix = original_reader

    def test_result_artifacts_do_not_exist_pre_review(self):
        for path in M.OUTPUTS:
            self.assertFalse(path.exists(), path)


class BoundedDecoderTests(unittest.TestCase):
    def test_exact_prefix_does_not_consume_suffix(self):
        prefix = b'{"safe":1}\n'
        suffix = b'{"forbidden_future":999}\n'
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "source.jsonl"
            path.write_bytes(prefix + suffix)
            observed = M.read_exact_prefix(
                path, len(prefix), hashlib.sha256(prefix).hexdigest()
            )
        self.assertEqual(observed, prefix)
        self.assertNotIn(b"forbidden_future", observed)

    def test_semantic_boundary_is_checked_before_price_decode(self):
        row = {
            "schema": "QR_OANDA_HISTORICAL_M5_BA_ROW_V1",
            "instrument": "EUR_USD",
            "granularity": "M5",
            "price_component": "BA",
            "complete": True,
            "time_utc": "2025-08-28T04:05:00.000000Z",
            "volume_semantics": "OANDA_PRICE_COUNT_NOT_TRADED_VOLUME",
            "volume": 1,
            "bid": {"o": "SECRET_BAD", "h": "x", "l": "x", "c": "x"},
            "ask": {"o": "x", "h": "x", "l": "x", "c": "x"},
        }
        boundary = M.parse_time("2025-08-28T04:05:00.000000Z")
        with self.assertRaisesRegex(AssertionError, "outside authorized phase"):
            M.parse_bar(M.canonical(row), "EUR_USD", boundary - 300, boundary)


class CausalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg = M.load_preregistration()

    def test_vectorized_features_match_scalar_runtime_formula(self):
        rows = trending_bars("EUR_USD", 12)
        signals = M.build_signals(
            "EUR_USD", rows, 0, 100000, M.EXPECTED_PREREG_CANONICAL_SHA256
        )
        self.assertTrue(signals)
        signal = signals[0]
        tail = rows[:7]
        closes = [bar.mid("c") for bar in tail]

        def ema(values, period):
            alpha = 2.0 / (period + 1.0)
            result = values[0]
            for value in values[1:]:
                result = alpha * value + (1 - alpha) * result
            return result

        ranges = []
        for index in range(1, 7):
            ranges.append(max(
                tail[index].mid("h") - tail[index].mid("l"),
                abs(tail[index].mid("h") - tail[index - 1].mid("c")),
                abs(tail[index].mid("l") - tail[index - 1].mid("c")),
            ))
        self.assertAlmostEqual(signal.fast_ema, ema(closes, 3))
        self.assertAlmostEqual(signal.slow_ema, ema(closes, 6))
        self.assertAlmostEqual(signal.slow_slope, ema(closes, 6) - ema(closes[:-1], 6))
        self.assertAlmostEqual(signal.momentum_price, closes[-1] - closes[-4])
        self.assertAlmostEqual(signal.atr_price, sum(ranges) / 6)

    def test_signal_is_cost_independent_unique_and_strictly_later(self):
        rows = trending_bars("EUR_USD", 20)
        signals = M.build_signals(
            "EUR_USD", rows, 0, 100000, M.EXPECTED_PREREG_CANONICAL_SHA256
        )
        self.assertEqual(len({signal.signal_id for signal in signals}), len(signals))
        for signal in signals:
            self.assertEqual(signal.decision_time, signal.decision_bar_time + 300)
            self.assertEqual(signal.expected_fill_time, signal.decision_time + 300)
            self.assertGreater(signal.expected_fill_time, signal.decision_time)
            self.assertGreaterEqual(
                signal.tp_distance_price, 1.5 * signal.observed_spread_price
            )

    def test_future_mutation_cannot_change_past_signals(self):
        rows = trending_bars("EUR_USD", 30)
        boundary = 15 * 300
        before = M.build_signals(
            "EUR_USD", rows, 0, boundary, M.EXPECTED_PREREG_CANONICAL_SHA256
        )
        changed = list(rows)
        changed[-1] = synthetic_bar("EUR_USD", changed[-1].time, 9.0, 9.5)
        after = M.build_signals(
            "EUR_USD", changed, 0, boundary, M.EXPECTED_PREREG_CANONICAL_SHA256
        )
        self.assertEqual(before, after)

    def test_gap_breaks_feature_window(self):
        rows = trending_bars("EUR_USD", 7) + trending_bars("EUR_USD", 7, start=8 * 300)
        signals = M.build_signals(
            "EUR_USD", rows, 0, 100000, M.EXPECTED_PREREG_CANONICAL_SHA256
        )
        self.assertFalse(any(signal.decision_bar_time == 8 * 300 for signal in signals))


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg = M.load_preregistration()

    def test_three_arms_receive_identical_signal_ids_and_caps_are_hard(self):
        bars = {pair: trending_bars(pair, 35) for pair in self.prereg["input"]["symbols"]}
        signals = []
        for pair in bars:
            signals.extend(M.build_signals(
                pair, bars[pair], 0, 35 * 300, M.EXPECTED_PREREG_CANONICAL_SHA256
            ))
        states = {
            arm: M.simulate_arm(arm, bars, signals, 0, 35 * 300, self.prereg)
            for arm in M.ARMS
        }
        expected = {signal.signal_id for signal in signals}
        self.assertTrue(expected)
        self.assertTrue(all(set(state.outcomes) == expected for state in states.values()))
        self.assertTrue(any(
            outcome["status"] == "CAPACITY_BLOCKED"
            for outcome in states["RAW_SIGNAL"].outcomes.values()
        ))
        self.assertLessEqual(max(states["RAW_SIGNAL"].inventory_samples), 2)

    def test_bid_ask_and_adverse_entry_costs_are_post_signal(self):
        signal = manual_signal("EUR_USD", 300, 600)
        bar = synthetic_bar("EUR_USD", 600, 1.0, 1.0001)
        raw = M._entry_fill("RAW_SIGNAL", signal, bar)
        base = M._entry_fill("EXECUTABLE_BASE", signal, bar)
        adverse = M._entry_fill("ADVERSE_STRESS", signal, bar)
        self.assertAlmostEqual(raw[0], bar.mid("o"))
        self.assertGreater(base[0], raw[0])
        self.assertGreater(adverse[0], base[0])
        self.assertEqual(adverse[2], 0.3)
        self.assertGreater(adverse[3], 0)

    def test_adverse_fill_bar_cannot_claim_same_bar_tp(self):
        signal = manual_signal("EUR_USD", 300, 600, tp_distance=0.00001)
        bar = synthetic_bar("EUR_USD", 600, 1.0, 1.0001, wing=0.001)
        cross = synthetic_bar("USD_JPY", 600, 150.0, 150.01)
        usd = {600: cross}
        base_fill = M._entry_fill("EXECUTABLE_BASE", signal, bar)
        stress_fill = M._entry_fill("ADVERSE_STRESS", signal, bar)
        base = M.ArmState("EXECUTABLE_BASE")
        stress = M.ArmState("ADVERSE_STRESS")
        for state, values in ((base, base_fill), (stress, stress_fill)):
            state.outcomes[signal.signal_id] = {"status": "OPEN"}
            state.positions["EUR_USD"] = M.Position(
                signal, state.arm, 600, values[0], bar.mid("o"),
                values[0] + signal.tp_distance_price, 2400,
                values[1], values[2], values[3],
            )
        M._update_tp_and_excursion(base, "EUR_USD", bar, usd)
        M._update_tp_and_excursion(stress, "EUR_USD", bar, usd)
        self.assertNotIn("EUR_USD", base.positions)
        self.assertIn("EUR_USD", stress.positions)
        self.assertIsNone(base.trades[0]["gross_close_mark_pips"])
        self.assertEqual(
            base.trades[0]["gross_comparator_basis"],
            "UNAVAILABLE_INTRABAR_TP_MID",
        )

    def test_max_age_open_precedes_due_bar_intrabar_tp(self):
        bars = {
            "EUR_USD": [synthetic_bar("EUR_USD", index * 300, 1.0, 1.0,
                                      wing=0.0001 if index != 8 else 0.05)
                        for index in range(10)],
            "USD_JPY": trending_bars("USD_JPY", 10),
            "AUD_USD": trending_bars("AUD_USD", 10),
        }
        signal = manual_signal("EUR_USD", 300, 600, tp_distance=0.01)
        state = M.simulate_arm(
            "EXECUTABLE_BASE", bars, [signal], 0, 3000, self.prereg
        )
        self.assertEqual(len(state.trades), 1)
        self.assertEqual(state.trades[0]["exit_reason"], "MAX_AGE")
        self.assertEqual(state.trades[0]["exit_source_bar_time"], M.iso_utc(2400))

    def test_gap_liquidates_existing_inventory_without_synthesis(self):
        times = [0, 300, 600, 1200, 1500]
        bars = {
            "EUR_USD": [synthetic_bar("EUR_USD", value, 1.0) for value in times],
            "USD_JPY": [synthetic_bar("USD_JPY", value, 150.0) for value in times],
            "AUD_USD": [synthetic_bar("AUD_USD", value, 1.0) for value in times],
        }
        signal = manual_signal("EUR_USD", 300, 600, tp_distance=1.0)
        state = M.simulate_arm("EXECUTABLE_BASE", bars, [signal], 0, 1800, self.prereg)
        self.assertEqual(state.trades[0]["exit_reason"], "DATA_GAP_TERMINAL_LIQUIDATION")
        self.assertEqual(state.trades[0]["exit_source_bar_time"], M.iso_utc(600))
        self.assertFalse(state.positions)

    def test_pair_specific_gap_frees_global_cap_before_pair_resumes(self):
        full_times = list(range(0, 3300, 300))
        eur_times = [value for value in full_times if value != 900]
        bars = {
            "AUD_USD": [synthetic_bar("AUD_USD", value, 1.0) for value in full_times],
            "EUR_USD": [synthetic_bar("EUR_USD", value, 1.0) for value in eur_times],
            "USD_JPY": [synthetic_bar("USD_JPY", value, 150.0) for value in full_times],
        }
        signals = [
            manual_signal("EUR_USD", 300, 600, tp_distance=1.0),
            manual_signal("USD_JPY", 300, 600, tp_distance=100.0),
            manual_signal("AUD_USD", 900, 1200, tp_distance=1.0),
        ]
        state = M.simulate_arm("EXECUTABLE_BASE", bars, signals, 0, 3300, self.prereg)
        aud = state.outcomes[signals[2].signal_id]
        self.assertNotEqual(aud["status"], "CAPACITY_BLOCKED")
        eur = next(trade for trade in state.trades if trade["pair"] == "EUR_USD")
        self.assertEqual(eur["exit_reason"], "DATA_GAP_TERMINAL_LIQUIDATION")
        self.assertEqual(eur["exit_source_bar_time"], M.iso_utc(600))

    def test_signal_at_first_missing_bar_is_recorded_but_never_fills_on_resume(self):
        full_times = [0, 300, 600, 900, 1200, 1500]
        eur_times = [0, 300, 600, 1200, 1500]
        bars = {
            "AUD_USD": [synthetic_bar("AUD_USD", value, 1.0) for value in full_times],
            "EUR_USD": [synthetic_bar("EUR_USD", value, 1.0) for value in eur_times],
            "USD_JPY": [synthetic_bar("USD_JPY", value, 150.0) for value in full_times],
        }
        signal = manual_signal("EUR_USD", 900, 1200, tp_distance=1.0)
        state = M.simulate_arm("EXECUTABLE_BASE", bars, [signal], 0, 1800, self.prereg)
        self.assertEqual(state.outcomes[signal.signal_id]["status"], "GAP_HALTED_NO_ORDER")
        self.assertFalse(any(trade["signal_id"] == signal.signal_id for trade in state.trades))
        self.assertFalse(state.pending)

    def test_jpy_conversion_uses_sign_sensitive_executable_side(self):
        cross = synthetic_bar("USD_JPY", 600, 150.0, 150.0, spread=0.02)
        positive, pos_rate, pos_source = M._conversion(
            "EUR_USD", 1.0, 600, "o", {600: cross}
        )
        negative, neg_rate, neg_source = M._conversion(
            "EUR_USD", -1.0, 600, "o", {600: cross}
        )
        self.assertAlmostEqual(positive, pos_rate)
        self.assertAlmostEqual(negative, -neg_rate)
        self.assertLess(pos_rate, neg_rate)
        self.assertIn("BID", pos_source)
        self.assertIn("ASK", neg_source)

    def test_block_bootstrap_is_deterministic(self):
        trades = []
        for day in range(12):
            when = M.iso_utc(day * 86400)
            trades.append({"pnl_jpy": 1.0, "net_pips": float(day % 3 - 1),
                           "decision_time": when})
        first = M.block_bootstrap_lcb(trades, 250, 7)
        second = M.block_bootstrap_lcb(trades, 250, 7)
        self.assertEqual(first, second)

    def test_raw_pip_bootstrap_does_not_drop_missing_jpy_conversion(self):
        trades = []
        for day in range(12):
            trades.append({
                "pnl_jpy": None,
                "net_pips": 1.0,
                "decision_time": M.iso_utc(day * 86400),
            })
        self.assertAlmostEqual(M.block_bootstrap_lcb(trades, 250, 7), 1.0)

    def test_zero_trade_calendar_months_cannot_disappear_from_stability_gate(self):
        prereg = json.loads(json.dumps(self.prereg))
        prereg["statistics"]["bootstrap_resamples"] = 100
        state = M.ArmState("ADVERSE_STRESS")
        state.trades.append({
            "pair": "EUR_USD", "net_pips": 1.0,
            "exit_time": "1970-01-15T00:00:00.000000Z",
            "decision_time": "1970-01-14T00:00:00.000000Z",
            "pnl_jpy": 10.0, "exit_reason": "MAX_AGE",
            "terminal_liquidation": False, "mfe_pips": 2.0, "mae_pips": -1.0,
            "age_bars": 6, "entry_spread_pips": 1.0,
            "entry_slippage_pips": 0.3, "exit_slippage_pips": 0.3,
            "latency_proxy_pips": 1.0, "gross_close_mark_pips": 2.0,
        })
        state.realized_jpy = 10.0
        start = M.parse_time("1970-01-01T00:00:00.000000Z")
        end = M.parse_time("1970-04-01T00:00:00.000000Z")
        summary = M.summarize_arm(
            state, [manual_signal("EUR_USD", 300, 600)], prereg, start, end
        )
        self.assertEqual(summary["positive_calendar_month_fraction"], 1 / 3)
        self.assertIsNone(summary["calendar_month_mean_pips"]["1970-02"])
        self.assertEqual(summary["calendar_month_equity_multiples"]["1970-02"], 1.0)

    def test_runner_has_no_external_io_stack(self):
        source = (HERE / "replay_m5_ema_state_impulse.py").read_text(encoding="utf-8")
        for forbidden in (
            "import requests", "import socket", "import urllib", "import subprocess",
            "os.environ", "launchctl", "git push", "stream-fxtrade.oanda.com",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
