import copy
import importlib.util
import json
import math
import pathlib
import statistics
import unittest


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "replay_m5_interaction", HERE / "replay_m5_interaction.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class ReplayInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg = json.loads((HERE / "preregistration.json").read_text())
        cls.prereg_sha = M.sha_file(HERE / "preregistration.json")
        cls.data = M.load_inputs(cls.prereg)
        cls.states = {
            pair: M.compute_states(rows) for pair, rows in cls.data.items()
        }
        cls.calibration_ends = {
            pair: int(len(rows) * 0.35) for pair, rows in cls.data.items()
        }
        cls.thresholds = M.derive_thresholds(
            cls.states,
            cls.calibration_ends,
            cls.prereg["features"]["minimum_calibration_rows_per_pair_session"],
            cls.prereg["features"]["calibration_quantile_low"],
            cls.prereg["features"]["calibration_quantile_high"],
        )

    def test_inputs_completed_chronological_and_v1_immutable(self):
        self.assertEqual(
            M.verify_v1_immutable(self.prereg), self.prereg["v1_immutable_hashes"]
        )
        for pair, rows in self.data.items():
            spec = self.prereg["inputs"]["files"][pair]
            self.assertEqual(M.sha_file(pathlib.Path(spec["path"])), spec["sha256"])
            self.assertTrue(all(row["complete"] for row in rows))
            self.assertTrue(
                all(
                    rows[index]["_time"] < rows[index + 1]["_time"]
                    for index in range(len(rows) - 1)
                )
            )

    def test_thresholds_are_calibration_only_and_future_independent(self):
        pair = "EUR_USD"
        rows = list(self.data[pair])
        mutation_index = self.calibration_ends[pair] + 100
        mutated = copy.deepcopy(rows[mutation_index])
        for side in ("bid", "ask"):
            for field in ("o", "h", "l", "c"):
                mutated[side][field] *= 9.0
        rows[mutation_index] = mutated
        changed_states = M.compute_states(rows)
        changed = M.derive_thresholds(
            {pair: changed_states},
            {pair: self.calibration_ends[pair]},
            self.prereg["features"]["minimum_calibration_rows_per_pair_session"],
            self.prereg["features"]["calibration_quantile_low"],
            self.prereg["features"]["calibration_quantile_high"],
        )
        self.assertEqual(changed[pair], self.thresholds[pair])
        self.assertTrue(
            all(
                cell["rows"] >= 500
                for cell in self.thresholds[pair].values()
            )
        )

    def test_future_mutation_cannot_change_earlier_decisions(self):
        pair = "EUR_USD"
        rows = list(self.data[pair][:220])
        states = M.compute_states(rows)
        original = M.make_source_signals(
            pair, rows, states, 0, len(rows), self.prereg_sha
        )
        changed_rows = list(rows)
        changed_rows[180] = copy.deepcopy(changed_rows[180])
        for side in ("bid", "ask"):
            for field in ("o", "h", "l", "c"):
                changed_rows[180][side][field] += 2.0
        changed_states = M.compute_states(changed_rows)
        changed = M.make_source_signals(
            pair, changed_rows, changed_states, 0, len(changed_rows), self.prereg_sha
        )
        original_prefix = [
            (signal["signal_id"], signal["side"] if "side" in signal else signal["trend_side"])
            for signal in original
            if signal["decision_index"] < 180
        ]
        changed_prefix = [
            (signal["signal_id"], signal["side"] if "side" in signal else signal["trend_side"])
            for signal in changed
            if signal["decision_index"] < 180
        ]
        self.assertEqual(original_prefix, changed_prefix)

    def test_spread_and_cost_do_not_change_source_signals(self):
        pair = "EUR_USD"
        rows = list(self.data[pair][:500])
        widened = copy.deepcopy(rows)
        for row in widened:
            for field in ("o", "h", "l", "c"):
                mid = M.midpoint(row, field)
                row["bid"][field] = mid - 0.005
                row["ask"][field] = mid + 0.005
        first = M.make_source_signals(
            pair, rows, M.compute_states(rows), 0, len(rows), self.prereg_sha
        )
        second = M.make_source_signals(
            pair, widened, M.compute_states(widened), 0, len(widened), self.prereg_sha
        )
        self.assertEqual(
            [(s["decision_time"], s["trend_side"]) for s in first],
            [(s["decision_time"], s["trend_side"]) for s in second],
        )
        self.assertEqual(M.signal_set_hash(first), M.signal_set_hash(second))

    def test_exact_eight_config_gates(self):
        self.assertEqual(sorted(self.prereg["configs"]), [f"C{i}" for i in range(8)])
        threshold = {"pe_q67": 0.7, "rv_q33": 0.2, "rv_q67": 0.6}
        base = {
            "trend_side": "LONG",
            "session": "ASIA",
            "path_efficiency": 0.8,
            "realized_energy": 0.4,
            "break_kind": "REJECTION",
            "break_side": "SHORT",
        }
        self.assertEqual(M.gate_signal("C0", base, threshold), "LONG")
        self.assertEqual(M.gate_signal("C1", base, threshold), "SHORT")
        self.assertEqual(M.gate_signal("C2", base, threshold), "SHORT")
        self.assertEqual(M.gate_signal("C3", base, threshold), "SHORT")
        self.assertEqual(M.gate_signal("C4", base, threshold), "SHORT")
        self.assertIsNone(M.gate_signal("C5", base, threshold))
        self.assertEqual(M.gate_signal("C6", base, threshold), "SHORT")
        accepted = dict(
            base,
            realized_energy=0.8,
            break_kind="ACCEPTANCE",
            break_side="LONG",
        )
        self.assertEqual(M.gate_signal("C7", accepted, threshold), "LONG")

    def test_preregistered_constants_fail_closed(self):
        checks = M.validate_preregistered_contract(self.prereg)
        self.assertTrue(all(checks.values()))
        changed = copy.deepcopy(self.prereg)
        changed["selection"]["family_alpha"] = 0.10
        with self.assertRaisesRegex(ValueError, "family_alpha"):
            M.validate_preregistered_contract(changed)

    def test_exact_family_lcb_daily_cluster_math(self):
        daily = {
            "2026-01-01": [1.0, 3.0],
            "2026-01-02": [2.0, 6.0],
            "2026-01-03": [4.0, 8.0],
        }
        summary = M.cluster_summary(daily, self.prereg["selection"])
        means = [2.0, 4.0, 6.0]
        expected_z = statistics.NormalDist().inv_cdf(1.0 - 0.05 / 8.0)
        expected_se = statistics.stdev(means) / math.sqrt(len(means))
        self.assertEqual(summary["daily_means"], means)
        self.assertAlmostEqual(summary["family_critical_z"], expected_z)
        self.assertAlmostEqual(summary["cluster_mean"], 4.0)
        self.assertAlmostEqual(summary["cluster_standard_error"], expected_se)
        self.assertAlmostEqual(
            summary["family_adjusted_lcb"], 4.0 - expected_z * expected_se
        )

    def _synthetic_metric(
        self,
        lcb=1.0,
        expectancy=1.0,
        median=1.0,
        n_eff=20,
        trades=120,
        pair_count=2,
        positive_pairs=2,
    ):
        return {
            "family_adjusted_lcb_pips": lcb,
            "expectancy_pips": expectancy,
            "utc_day_cluster_median_pips": median,
            "n_eff_utc_day_clusters": n_eff,
            "trades": trades,
            "pairs_with_at_least_30_trades": pair_count,
            "pairs_meeting_per_pair_trade_floor": pair_count,
            "pairs_with_positive_expectancy": positive_pairs,
        }

    def test_density_fallback_raw_tie_chain_and_cost_non_reselection(self):
        configs = {}
        for index in range(8):
            raw = self._synthetic_metric(lcb=float(index), expectancy=float(index))
            configs[f"C{index}"] = {
                "tuning": {
                    "scenario_metrics": {
                        "raw": raw,
                        "base": {"expectancy_pips": 9999.0 - index},
                        "adverse": {"expectancy_pips": -9999.0 + index},
                    }
                }
            }
        selected, dense = M.select_config(
            configs, [f"C{i}" for i in range(8)], self.prereg["selection"]
        )
        self.assertEqual(selected, "C7")
        self.assertEqual(dense, [f"C{i}" for i in range(8)])
        perturbed = copy.deepcopy(configs)
        for index, config in enumerate(perturbed.values()):
            config["tuning"]["scenario_metrics"]["base"]["expectancy_pips"] = -index * 1e9
            config["tuning"]["scenario_metrics"]["adverse"]["expectancy_pips"] = index * 1e9
        self.assertEqual(
            M.select_config(
                perturbed, [f"C{i}" for i in range(8)], self.prereg["selection"]
            )[0],
            selected,
        )

        # If every config fails density, the preregistered diagnostic fallback
        # still ranks RAW; it never substitutes a BASE/ADVERSE winner.
        sparse = copy.deepcopy(configs)
        for config in sparse.values():
            config["tuning"]["scenario_metrics"]["raw"]["trades"] = 119
        fallback, dense = M.select_config(
            sparse, [f"C{i}" for i in range(8)], self.prereg["selection"]
        )
        self.assertEqual(fallback, "C7")
        self.assertEqual(dense, [])

        # Exact tie order: LCB, expectancy, median, N_eff, then lexical ID.
        tied = copy.deepcopy(configs)
        for config in tied.values():
            config["tuning"]["scenario_metrics"]["raw"] = self._synthetic_metric()
        tied["C1"]["tuning"]["scenario_metrics"]["raw"]["expectancy_pips"] = 2.0
        tied["C2"]["tuning"]["scenario_metrics"]["raw"].update(
            expectancy_pips=2.0, utc_day_cluster_median_pips=2.0
        )
        tied["C3"]["tuning"]["scenario_metrics"]["raw"].update(
            expectancy_pips=2.0,
            utc_day_cluster_median_pips=2.0,
            n_eff_utc_day_clusters=21,
        )
        tied["C4"]["tuning"]["scenario_metrics"]["raw"].update(
            expectancy_pips=2.0,
            utc_day_cluster_median_pips=2.0,
            n_eff_utc_day_clusters=21,
        )
        self.assertEqual(
            M.select_config(
                tied, [f"C{i}" for i in range(8)], self.prereg["selection"]
            )[0],
            "C3",
        )

    def test_gross_and_cost_classification_gates(self):
        raw = self._synthetic_metric()
        negative = self._synthetic_metric(lcb=-1.0, expectancy=-1.0)
        positive = self._synthetic_metric(lcb=0.5, expectancy=0.5)
        tuning = {"raw": copy.deepcopy(raw), "base": negative, "adverse": negative}
        development = {
            "raw": copy.deepcopy(raw),
            "base": copy.deepcopy(negative),
            "adverse": copy.deepcopy(negative),
        }
        classified = M.classify_candidate(
            tuning,
            development,
            self.prereg["selection"],
            self.prereg["gross_edge_gate"],
        )
        self.assertEqual(classified["classification"], "GROSS_ONLY_COST_BOUND")
        development["base"] = copy.deepcopy(positive)
        classified = M.classify_candidate(
            tuning,
            development,
            self.prereg["selection"],
            self.prereg["gross_edge_gate"],
        )
        self.assertEqual(classified["classification"], "BASE_EXECUTABLE_CANDIDATE")
        development["adverse"] = copy.deepcopy(positive)
        classified = M.classify_candidate(
            tuning,
            development,
            self.prereg["selection"],
            self.prereg["gross_edge_gate"],
        )
        self.assertEqual(classified["classification"], "STRESS_ROBUST_CANDIDATE")
        development["raw"]["family_adjusted_lcb_pips"] = 0.0
        classified = M.classify_candidate(
            tuning,
            development,
            self.prereg["selection"],
            self.prereg["gross_edge_gate"],
        )
        self.assertEqual(classified["classification"], "REJECTED_NO_GROSS_EDGE")

    def test_next_bar_fixed_close_lineage_units_and_split_boundary(self):
        pair = "EUR_USD"
        end = 900
        signals = M.make_source_signals(
            pair,
            self.data[pair],
            self.states[pair],
            200,
            end,
            self.prereg_sha,
        )
        replay = M.replay_config(
            pair, self.data[pair], signals, self.thresholds, "C3", end
        )
        self.assertTrue(replay["trades"])
        for trade in replay["trades"]:
            self.assertEqual(trade["entry_index"], trade["decision_index"] + 1)
            self.assertEqual(M.parse_time(trade["entry_time"]), M.parse_time(trade["decision_time"]))
            self.assertEqual(
                M.parse_time(trade["exit_time"]),
                M.close_time(self.data[pair][trade["exit_index"]]),
            )
            self.assertLess(trade["exit_index"], end)
            self.assertLessEqual(trade["age_bars"], 48)
            self.assertEqual(trade["units"], 1000)
            self.assertIn(trade["exit_reason"], ("FIXED_H48_CLOSE", "TERMINAL_LIQUIDATION"))
            self.assertGreater(
                trade["entry_executable"], trade["entry_mid"]
            ) if trade["side"] == "LONG" else self.assertLess(
                trade["entry_executable"], trade["entry_mid"]
            )

    def test_close_availability_clock_controls_session_and_h48_cutoff(self):
        pair = "EUR_USD"
        rows = self.data[pair]
        states = self.states[pair]
        boundary = next(
            index
            for index, row in enumerate(rows)
            if row["_time"].hour == 6 and row["_time"].minute == 55 and states[index]
        )
        self.assertEqual(states[boundary]["session"], "LONDON")
        self.assertEqual(
            M.parse_time(states[boundary]["decision_time"]), M.close_time(rows[boundary])
        )
        signals = M.make_source_signals(
            pair, rows, states, 0, 1500, self.prereg_sha
        )
        replay = M.replay_config(
            pair, rows, signals, self.thresholds, "C3", 1500
        )
        nonterminal = [
            trade for trade in replay["trades"] if not trade["terminal_liquidation"]
        ]
        self.assertTrue(nonterminal)
        self.assertTrue(
            all(
                (M.parse_time(trade["exit_time"]).hour,
                 M.parse_time(trade["exit_time"]).minute)
                <= (20, 55)
                for trade in nonterminal
            )
        )

    def test_gap_is_unscorable_and_does_not_create_trade(self):
        pair = "EUR_USD"
        rows = list(self.data[pair][:300])
        signals = M.make_source_signals(
            pair, rows, M.compute_states(rows), 0, len(rows), self.prereg_sha
        )
        signal = signals[50]
        changed = list(rows)
        gap_index = signal["fill_index"] + 3
        changed[gap_index] = copy.deepcopy(changed[gap_index])
        changed[gap_index]["_time"] += M.BAR_STEP
        changed[gap_index]["time"] = changed[gap_index]["_time"].isoformat()
        replay = M.replay_config(
            pair, changed, [signal], self.thresholds, "C0", len(changed)
        )
        self.assertEqual(replay["gap_unscorable"], 1)
        self.assertEqual(replay["gap_signal_ids"], [signal["signal_id"]])
        self.assertEqual(replay["trades"], [])

    def test_jpy_and_non_jpy_pip_units(self):
        converter = M.JpyConverter(self.data["USD_JPY"])
        when = M.iso_utc(M.close_time(self.data["USD_JPY"][100]))
        self.assertAlmostEqual(converter.pnl("USD_JPY", 1.0, when), 10.0)
        rate = M.midpoint(self.data["USD_JPY"][100], "c")
        self.assertAlmostEqual(converter.pnl("EUR_USD", 1.0, when), 0.1 * rate)

    def test_written_result_has_three_arm_lineage_and_zero_inventory(self):
        result = json.loads((HERE / "result.json").read_text())
        self.assertFalse(result["admission"])
        self.assertFalse(result["holdout"])
        self.assertTrue(result["opened_development_only"])
        for config in result["configs"].values():
            for period in ("tuning", "opened_development"):
                payload = config[period]
                self.assertTrue(payload["same_signal_and_trade_path_all_scenarios"])
                self.assertEqual(len(set(payload["lineage_by_scenario"].values())), 1)
                metrics = payload["scenario_metrics"]
                self.assertEqual(
                    len({metrics[arm]["source_signal_set_sha256"] for arm in M.SCENARIOS}),
                    1,
                )
                self.assertEqual(
                    len({metrics[arm]["gated_signal_set_sha256"] for arm in M.SCENARIOS}),
                    1,
                )
                self.assertTrue(
                    all(metrics[arm]["terminal_open_inventory"] == 0 for arm in M.SCENARIOS)
                )

    def test_deterministic_rerun_and_packet_hashes(self):
        written = json.loads((HERE / "result.json").read_text())
        rerun = M.main(False)
        self.assertEqual(written, rerun)
        packet = json.loads((HERE / "evidence_packet.json").read_text())
        self.assertEqual(packet["result_sha256"], M.sha_file(HERE / "result.json"))
        self.assertEqual(packet["script_sha256"], M.sha_file(HERE / "replay_m5_interaction.py"))
        self.assertEqual(packet["test_sha256"], M.sha_file(pathlib.Path(__file__)))
        self.assertEqual(packet["readme_sha256"], M.sha_file(HERE / "README.md"))
        self.assertEqual(packet["selected_config"], written["selected_config"])
        self.assertEqual(
            packet["development_classification"],
            written["development_classification"],
        )
        self.assertEqual(packet["gross_edge_gates"], written["gross_edge_gates"])
        self.assertEqual(
            packet["shadow_challenger_eligible"],
            written["shadow_challenger_eligible"],
        )
        self.assertEqual(packet["selected_tuning"], written["selected_tuning"])
        self.assertEqual(
            packet["selected_opened_development"],
            written["selected_opened_development"],
        )
        self.assertEqual(
            packet["exact_config"],
            self.prereg["configs"][written["selected_config"]],
        )


if __name__ == "__main__":
    unittest.main()
