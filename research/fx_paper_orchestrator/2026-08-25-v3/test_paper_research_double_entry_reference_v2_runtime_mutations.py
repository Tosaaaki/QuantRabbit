from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import itertools
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping

import pytest

import test_paper_research_double_entry_reference_v2 as support


ENGINE_PATH = Path(__file__).with_name(
    "paper_research_double_entry_reference_v2.py"
)
ENGINE_SOURCE = ENGINE_PATH.read_text(encoding="utf-8")
MODULE_SEQUENCE = itertools.count(1)


@dataclass(frozen=True)
class ExpressionPatch:
    function_name: str
    before: str
    after: str
    expected_sites: int = 1


@dataclass(frozen=True)
class MutationSpec:
    mutation_id: str
    name: str
    mode: str
    fixture: Callable[[support.CanonicalFixture], Mapping[str, bytes]]
    assertion: Callable[[Mapping[str, Any]], None]
    patches: tuple[ExpressionPatch, ...]
    allowed_error_text: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentMutationSpec:
    mutation_id: str
    name: str
    mode: str
    reachability_reason: str
    exercise: Callable[[ModuleType, support.CanonicalFixture], Any]
    assertion: Callable[[Any], None]
    patches: tuple[ExpressionPatch, ...]
    allowed_error_text: tuple[str, ...] = ()


class MutationBuildError(AssertionError):
    pass


class _ExpressionMutator(ast.NodeTransformer):
    def __init__(self, mutation_id: str, patches: tuple[ExpressionPatch, ...]):
        self.mutation_id = mutation_id
        self.patches = patches
        self.function_stack: list[str] = []
        self.matches = [0 for _ in patches]
        self.before_shapes = [
            ast.dump(
                ast.parse(patch.before, mode="eval").body,
                include_attributes=False,
            )
            for patch in patches
        ]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.function_stack.append(node.name)
        try:
            return self.generic_visit(node)
        finally:
            self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.function_stack.append(node.name)
        try:
            return self.generic_visit(node)
        finally:
            self.function_stack.pop()

    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return super().visit(node)
        if isinstance(node, ast.expr) and self.function_stack:
            shape = ast.dump(node, include_attributes=False)
            for index, patch in enumerate(self.patches):
                if (
                    self.function_stack[-1] == patch.function_name
                    and shape == self.before_shapes[index]
                ):
                    self.matches[index] += 1
                    replacement = ast.parse(patch.after, mode="eval").body
                    probed = ast.Call(
                        func=ast.Name(id="__mutation_probe__", ctx=ast.Load()),
                        args=[ast.Constant(self.mutation_id), replacement],
                        keywords=[],
                    )
                    return ast.copy_location(probed, node)
        return super().visit(node)

    def validate(self) -> None:
        failures = [
            (patch.function_name, patch.before, actual, patch.expected_sites)
            for patch, actual in zip(self.patches, self.matches)
            if actual != patch.expected_sites
        ]
        if failures:
            raise MutationBuildError(f"wrong AST mutation site count: {failures!r}")


@contextmanager
def _loaded_engine(
    spec: MutationSpec | ComponentMutationSpec | None,
) -> Iterator[tuple[ModuleType, Counter[str]]]:
    tree = ast.parse(ENGINE_SOURCE, filename=str(ENGINE_PATH))
    hits: Counter[str] = Counter()
    if spec is not None:
        mutator = _ExpressionMutator(spec.mutation_id, spec.patches)
        tree = mutator.visit(tree)
        mutator.validate()
        ast.fix_missing_locations(tree)
    module_name = (
        f"_paper_reference_runtime_mutant_"
        f"{spec.mutation_id if spec is not None else 'BASELINE'}_"
        f"{next(MODULE_SEQUENCE)}"
    )
    module = ModuleType(module_name)
    module.__file__ = f"<{module_name}>"

    def probe(mutation_id: str, value: Any) -> Any:
        hits[mutation_id] += 1
        return value

    module.__dict__["__mutation_probe__"] = probe
    sys.modules[module_name] = module
    try:
        exec(compile(tree, module.__file__, "exec"), module.__dict__)
        yield module, hits
    finally:
        sys.modules.pop(module_name, None)


def _short_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._short_artifacts(fixture)


def _canonical_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return fixture.artifacts


def _pip_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._pip_artifacts(fixture)


def _inverse_cad_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._inverse_conversion_fixture(fixture, "CAD").artifacts


def _inverse_cad_short_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    artifacts = support._inverse_conversion_fixture(fixture, "CAD").artifacts
    return support._replace_json_artifact(
        artifacts,
        "proposal",
        "proposal_sha256",
        lambda payload: payload["rows"][0].__setitem__("direction", -1),
    )


def _stale_alternate_path_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    def accounting(payload: dict[str, Any]) -> None:
        payload["max_conversion_staleness_ns"] = 2_000_000_000

    return support._build_matrix_fixture(
        fixture,
        events=(
            {
                "tag": "stale-usd-chf",
                "instrument": "USD_CHF",
                "bid_ticks": 89,
                "ask_ticks": 91,
                "source_ts_ns": support.START_NS,
                "arrival_ts_ns": support.START_NS,
            },
            {
                "tag": "stale-chf-jpy",
                "instrument": "CHF_JPY",
                "bid_ticks": 16_999,
                "ask_ticks": 17_001,
                "source_ts_ns": support.START_NS,
                "arrival_ts_ns": support.START_NS,
            },
            {
                "tag": "fresh-usd-jpy",
                "instrument": "USD_JPY",
                "bid_ticks": 14_999,
                "ask_ticks": 15_001,
                "source_ts_ns": support.START_NS + 2_000_000_000,
                "arrival_ts_ns": support.START_NS + 2_000_000_000,
            },
            {
                "tag": "decision",
                "instrument": "EUR_USD",
                "bid_ticks": 109,
                "ask_ticks": 111,
                "source_ts_ns": support.START_NS + 2_000_000_000,
                "arrival_ts_ns": support.START_NS + 2_000_000_000,
            },
            {
                "tag": "entry",
                "instrument": "EUR_USD",
                "bid_ticks": 109,
                "ask_ticks": 111,
                "source_ts_ns": support.START_NS + 3_000_000_000,
                "arrival_ts_ns": support.START_NS + 3_000_000_000,
            },
            {
                "tag": "exit",
                "instrument": "EUR_USD",
                "bid_ticks": 119,
                "ask_ticks": 121,
                "source_ts_ns": support.START_NS + 4_000_000_000,
                "arrival_ts_ns": support.START_NS + 4_000_000_000,
            },
        ),
        proposals=({
            "decision_tag": "decision",
            "instrument": "EUR_USD",
            "direction": 1,
            "max_age_ns": 1_000_000_000,
        },),
        registry={
            "CHF_JPY": {"price_scale": 100, "pip_ticks": 1},
            "EUR_USD": {"price_scale": 100, "pip_ticks": 1},
            "USD_CHF": {"price_scale": 100, "pip_ticks": 1},
            "USD_JPY": {"price_scale": 100, "pip_ticks": 1},
        },
        mutate_accounting=accounting,
    ).artifacts


def _commission_rounding_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._replace_json_artifact(
        fixture.artifacts,
        "proposal",
        "proposal_sha256",
        lambda payload: payload["rows"][0].__setitem__(
            "notional_jpy_micros", 90_009_000
        ),
    )


def _divergent_financing_clock_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    def execution(payload: dict[str, Any]) -> None:
        payload["max_trade_quote_staleness_ns"] = 2 * support.DAY_NS

    return support._build_matrix_fixture(
        fixture,
        events=(
            support._usd_jpy_event("decision", 0, 10_000),
            {
                **support._usd_jpy_event("entry", 1, 10_000),
                "arrival_ts_ns": support.START_NS + 10_000_000_000,
            },
            {
                **support._usd_jpy_event("exit", 2, 10_100),
                "arrival_ts_ns": (
                    support.START_NS + support.DAY_NS + 10_000_000_000
                ),
            },
        ),
        proposals=({
            "decision_tag": "decision",
            "instrument": "USD_JPY",
            "direction": 1,
            "max_age_ns": support.DAY_NS,
        },),
        registry={"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
        mutate_execution=execution,
    ).artifacts


def _terminal_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._terminal_artifacts(fixture)


def _equal_arrival_release_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._build_matrix_fixture(
        fixture,
        events=(
            support._usd_jpy_event("decision-1", 0, 10_000),
            support._usd_jpy_event("entry-1", 1, 10_000),
            support._usd_jpy_event("decision-2", 2, 10_025),
            support._usd_jpy_event("release-and-entry-2", 3, 10_050),
            support._usd_jpy_event("exit-2", 4, 10_100),
        ),
        proposals=(
            {
                "decision_tag": "decision-1",
                "instrument": "USD_JPY",
                "direction": 1,
                "max_age_ns": 2_000_000_000,
            },
            {
                "decision_tag": "decision-2",
                "instrument": "USD_JPY",
                "direction": 1,
                "max_age_ns": 1_000_000_000,
            },
        ),
        registry={"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
    ).artifacts


def _opposite_collision_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    artifacts = support._collision_fixture(fixture).artifacts
    return support._replace_json_artifact(
        artifacts,
        "inventory_policy",
        "inventory_policy_sha256",
        lambda payload: payload.__setitem__("max_open_positions", 2),
    )


def _short_gross_cap_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._short_gross_cap_artifacts(fixture)


def _ruin_artifacts(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._ruin_fixture(fixture).artifacts


def _multi_mark_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._build_matrix_fixture(
        fixture,
        events=(
            support._usd_jpy_event("decision", 0, 10_000),
            support._usd_jpy_event("entry", 1, 10_000),
            support._usd_jpy_event("intermediate-mark", 2, 10_050),
            support._usd_jpy_event("exit", 3, 10_100),
        ),
        proposals=({
            "decision_tag": "decision",
            "instrument": "USD_JPY",
            "direction": 1,
            "max_age_ns": 2_000_000_000,
        },),
        registry={"USD_JPY": {"price_scale": 100, "pip_ticks": 1}},
    ).artifacts


def _connected_cluster_fixture(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    def inventory(payload: dict[str, Any]) -> None:
        payload["max_open_positions"] = 2
        payload["max_gross_notional_jpy_micros"] = 300_000_000
        payload["max_currency_notional_jpy_micros"] = 300_000_000

    return support._build_matrix_fixture(
        fixture,
        events=(
            support._usd_jpy_event("usd-decision", 0, 10_000),
            {
                "tag": "eur-decision",
                "instrument": "EUR_USD",
                "bid_ticks": 109,
                "ask_ticks": 111,
                "source_ts_ns": support.START_NS,
                "arrival_ts_ns": support.START_NS,
            },
            support._usd_jpy_event("usd-entry", 1, 10_000),
            {
                "tag": "eur-entry",
                "instrument": "EUR_USD",
                "bid_ticks": 109,
                "ask_ticks": 111,
                "source_ts_ns": support.START_NS + 1_000_000_000,
                "arrival_ts_ns": support.START_NS + 1_000_000_000,
            },
            support._usd_jpy_event("usd-exit", 2, 10_100),
            {
                "tag": "eur-exit",
                "instrument": "EUR_USD",
                "bid_ticks": 95,
                "ask_ticks": 97,
                "source_ts_ns": support.START_NS + 2_000_000_000,
                "arrival_ts_ns": support.START_NS + 2_000_000_000,
            },
        ),
        proposals=(
            {
                "decision_tag": "usd-decision",
                "instrument": "USD_JPY",
                "direction": 1,
                "max_age_ns": 1_000_000_000,
            },
            {
                "decision_tag": "eur-decision",
                "instrument": "EUR_USD",
                "direction": 1,
                "max_age_ns": 1_000_000_000,
            },
        ),
        registry={
            "EUR_USD": {"price_scale": 100, "pip_ticks": 1},
            "USD_JPY": {"price_scale": 100, "pip_ticks": 1},
        },
        mutate_inventory=inventory,
    ).artifacts


def _three_cluster_artifacts(
    fixture: support.CanonicalFixture,
) -> Mapping[str, bytes]:
    return support._three_cluster_fixture(fixture).artifacts


def _row(
    result: Mapping[str, Any],
    arm: str,
    proposal_ordinal: int = 1,
) -> Mapping[str, Any]:
    return next(
        row
        for row in support._ledger_rows(result["ledger_bytes"])
        if row["arm"] == arm and row["proposal_ordinal"] == proposal_ordinal
    )


def _assert_long_prices(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["direction"] == 1
    assert (
        executable["entry_price_numerator"],
        executable["entry_price_denominator"],
    ) == (10_001_000_000, 100_000_000)
    assert (
        executable["exit_price_numerator"],
        executable["exit_price_denominator"],
    ) == (10_099_000_000, 100_000_000)


def _assert_short_result(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["direction"] == -1
    assert (
        executable["entry_price_numerator"],
        executable["entry_price_denominator"],
    ) == (9_999_000_000, 100_000_000)
    assert (
        executable["exit_price_numerator"],
        executable["exit_price_denominator"],
    ) == (10_101_000_000, 100_000_000)
    assert executable["units_micros"] == 1_000_100
    assert executable["executable_pnl_before_direct_cost_jpy_micros"] == -1_020_102
    assert executable["net_pnl_jpy_micros"] == -1_022_613


def _assert_pip_scaled_adverse(result: Mapping[str, Any]) -> None:
    adverse = _row(result, "ADVERSE_STRESS")
    assert (
        adverse["entry_price_numerator"],
        adverse["entry_price_denominator"],
    ) == (10_006_000_000, 100_000_000)
    assert (
        adverse["exit_price_numerator"],
        adverse["exit_price_denominator"],
    ) == (10_094_000_000, 100_000_000)
    assert adverse["units_micros"] == 999_400
    assert adverse["latency_spread_slippage_drag_jpy_micros"] == 119_928


def _assert_inverse_cad_result(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["instrument"] == "EUR_CAD"
    assert executable["units_micros"] == 551_802
    assert executable["filled_notional_jpy_micros"] == 99_999_854
    assert executable["marked_or_exit_notional_jpy_micros"] == 104_448_236
    assert executable["executable_pnl_before_direct_cost_jpy_micros"] == 5_255_257
    assert executable["net_pnl_jpy_micros"] == 5_253_211


def _assert_inverse_cad_short_result(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["instrument"] == "EUR_CAD"
    assert executable["direction"] == -1
    assert executable["units_micros"] == 563_758
    assert executable["marked_or_exit_notional_jpy_micros"] == 108_932_569
    assert executable["executable_pnl_before_direct_cost_jpy_micros"] == -8_119_198
    assert executable["net_pnl_jpy_micros"] == -8_121_289


def _assert_stale_alternate_excluded(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["instrument"] == "EUR_USD"
    assert executable["status"] == "FILLED_CLOSED"
    assert executable["units_micros"] == 600_560
    assert executable["filled_notional_jpy_micros"] == 99_999_907
    assert executable["net_pnl_jpy_micros"] == 7_204_166


def _assert_directed_rounding(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["units_micros"] == 999_900
    assert executable["marked_or_exit_notional_jpy_micros"] == 100_979_901
    assert executable["required_margin_after_entry_jpy_micros"] == 4_999_001
    assert executable["commission_jpy_micros"] == 2_010
    assert executable["net_pnl_jpy_micros"] == 977_392


def _assert_commission_side_rounding(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["entry_price_numerator"] == 10_001_000_000
    assert executable["exit_price_numerator"] == 10_099_000_000
    assert executable["units_micros"] == 900_000
    assert executable["commission_jpy_micros"] == 1_810
    assert executable["net_pnl_jpy_micros"] == 879_739


def _assert_divergent_financing_clock(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["elapsed_ns"] == support.DAY_NS
    assert executable["financing_jpy_micros"] == 500
    assert executable["net_pnl_jpy_micros"] == 977_392


def _assert_terminal_financing_clock(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["exit_disposition"] == "TERMINAL_LIQUIDATION"
    assert executable["exit_arrival_ts_ns"] == support.END_NS - 1
    assert executable["elapsed_ns"] == 2_678_398_999_999_999
    assert executable["financing_jpy_micros"] == 15_500
    assert executable["net_pnl_jpy_micros"] == 962_392


def _assert_equal_arrival_release(result: Mapping[str, Any]) -> None:
    second = _row(result, "EXECUTABLE_BASE", 2)
    assert second["status"] == "FILLED_CLOSED"
    assert second["units_micros"] == 994_925
    assert second["entry_arrival_ts_ns"] == support.START_NS + 3_000_000_000
    assert second["net_pnl_jpy_micros"] == 475_558


def _assert_opposite_collision(result: Mapping[str, Any]) -> None:
    first = _row(result, "EXECUTABLE_BASE", 1)
    second = _row(result, "EXECUTABLE_BASE", 2)
    assert first["status"] == "FILLED_CLOSED"
    assert second["direction"] == -1
    assert second["status"] == "SAME_PAIR_COLLISION_REJECTED"
    assert second["units_micros"] == 0


def _assert_short_gross_cap(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["direction"] == -1
    assert executable["status"] == "GROSS_CAP_REJECTED"
    assert executable["units_micros"] == 0


def _assert_currency_incidence(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["signed_currency_exposure_after_entry_jpy_micros"] == {
        "JPY": -99_980_001,
        "USD": 99_980_001,
    }
    assert executable["gross_open_notional_after_entry_jpy_micros"] == 99_980_001


def _assert_terminal_liquidation(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["exit_disposition"] == "TERMINAL_LIQUIDATION"
    assert (
        executable["exit_price_numerator"],
        executable["exit_price_denominator"],
    ) == (10_099_000_000, 100_000_000)
    assert executable["financing_jpy_micros"] == 15_500
    assert executable["net_pnl_jpy_micros"] == 962_392
    metrics = result["oracle_metrics"]["arms"]["EXECUTABLE_BASE"]
    assert metrics["terminal_open_positions"] == 0
    assert metrics["terminal_inventory_mtm_jpy_micros"] == 0


def _assert_ruin_halts_admissions(result: Mapping[str, Any]) -> None:
    first = _row(result, "EXECUTABLE_BASE", 1)
    second = _row(result, "EXECUTABLE_BASE", 2)
    assert first["exit_disposition"] == "MARGIN_CLOSEOUT"
    assert first["net_pnl_jpy_micros"] == -90_012_099
    assert second["status"] == "ACCOUNT_HALTED"
    assert second["units_micros"] == 0


def _assert_multi_mark_result(result: Mapping[str, Any]) -> None:
    executable = _row(result, "EXECUTABLE_BASE")
    assert executable["status"] == "FILLED_CLOSED"
    assert executable["units_micros"] == 999_900
    assert executable["executable_pnl_before_direct_cost_jpy_micros"] == 979_902
    assert executable["net_pnl_jpy_micros"] == 977_891
    assert result["all_transactions_balanced"] is True
    assert result["journal_transaction_count"] == 26


def _assert_connected_cluster(result: Mapping[str, Any]) -> None:
    metrics = result["oracle_metrics"]["arms"]["EXECUTABLE_BASE"]
    assert metrics["currency_time_cluster_n_eff"] == 1
    assert metrics["cluster_cvar_jpy_micros"] == -13_582_503
    assert metrics["cluster_cvar_return"] == "-0.013582502947282029"
    assert metrics["net_pnl_jpy_micros"] == -13_582_506
    assert len(metrics["currency_time_cluster_observations"]) == 1
    observation = metrics["currency_time_cluster_observations"][0]
    assert observation["currency_nodes"] == ["EUR", "JPY", "USD"]
    assert observation["cluster_risk_net_pnl_jpy_micros"] == -13_582_503


def _assert_three_cluster_tail(result: Mapping[str, Any]) -> None:
    metrics = result["oracle_metrics"]["arms"]["EXECUTABLE_BASE"]
    assert metrics["currency_time_cluster_n_eff"] == 3
    assert metrics["cluster_cvar_jpy_micros"] == -2_021_778
    assert metrics["cluster_cvar_return"] == "-0.002021777807787037"
    assert sorted(
        row["cluster_risk_net_pnl_jpy_micros"]
        for row in metrics["currency_time_cluster_observations"]
    ) == [-3_021_668, -1_021_888, 1_977_782]


def _assert_full_drawdown_observation_set(result: Mapping[str, Any]) -> None:
    metrics = result["oracle_metrics"]["arms"]["EXECUTABLE_BASE"]
    assert metrics["max_drawdown_jpy_micros"] == 21_998
    assert metrics["max_drawdown_ratio"] == "0.000021998000000000"
    assert metrics["minimum_marked_equity_jpy_micros"] == 999_978_002


MUTATION_SPECS: tuple[MutationSpec, ...] = (
    MutationSpec(
        mutation_id="MK01A",
        name="LONG_ENTRY_BID_INSTEAD_OF_ASK",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_long_prices,
        patches=(ExpressionPatch(
            function_name="_execution_price",
            before=(
                "opening and proposal.direction > 0 or "
                "(not opening and proposal.direction < 0)"
            ),
            after=(
                "(opening and proposal.direction > 0 and False) or "
                "(not opening and proposal.direction < 0)"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK01B",
        name="LONG_EXIT_ASK_INSTEAD_OF_BID",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_long_prices,
        patches=(ExpressionPatch(
            function_name="_execution_price",
            before=(
                "opening and proposal.direction > 0 or "
                "(not opening and proposal.direction < 0)"
            ),
            after=(
                "opening and proposal.direction > 0 or "
                "(not opening and proposal.direction < 0) or "
                "(not opening and proposal.direction > 0)"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK01C",
        name="SHORT_ENTRY_ASK_INSTEAD_OF_BID",
        mode="PUBLIC_REPLAY",
        fixture=_short_fixture,
        assertion=_assert_short_result,
        patches=(ExpressionPatch(
            function_name="_execution_price",
            before=(
                "opening and proposal.direction > 0 or "
                "(not opening and proposal.direction < 0)"
            ),
            after=(
                "opening and proposal.direction > 0 or "
                "(not opening and proposal.direction < 0) or "
                "(opening and proposal.direction < 0)"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK01D",
        name="SHORT_EXIT_BID_INSTEAD_OF_ASK",
        mode="PUBLIC_REPLAY",
        fixture=_short_fixture,
        assertion=_assert_short_result,
        patches=(ExpressionPatch(
            function_name="_execution_price",
            before=(
                "opening and proposal.direction > 0 or "
                "(not opening and proposal.direction < 0)"
            ),
            after=(
                "opening and proposal.direction > 0 or "
                "(not opening and proposal.direction < 0 and False)"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK02A",
        name="SLIPPAGE_SIGN_REVERSED",
        mode="PUBLIC_REPLAY",
        fixture=_pip_fixture,
        assertion=_assert_pip_scaled_adverse,
        patches=(ExpressionPatch(
            function_name="_execution_price",
            before="slippage_ticks if buys_base else -slippage_ticks",
            after="-slippage_ticks if buys_base else slippage_ticks",
        ),),
    ),
    MutationSpec(
        mutation_id="MK02B",
        name="PIP_NORMALIZATION_OMITTED",
        mode="PUBLIC_REPLAY",
        fixture=_pip_fixture,
        assertion=_assert_pip_scaled_adverse,
        patches=(ExpressionPatch(
            function_name="_execution_price",
            before=(
                "terms.slippage_micropips_per_side * "
                "registry[proposal.instrument]['pip_ticks']"
            ),
            after="terms.slippage_micropips_per_side",
        ),),
    ),
    MutationSpec(
        mutation_id="MK03A",
        name="ASSET_LIABILITY_EXECUTABLE_SIDE_SWAPPED",
        mode="PUBLIC_REPLAY",
        fixture=_inverse_cad_fixture,
        assertion=_assert_inverse_cad_result,
        patches=(
            ExpressionPatch(
                function_name="_currency_node_yen",
                before="_tick_price(tick, 'BID' if value > 0 else 'ASK')",
                after="_tick_price(tick, 'ASK' if value > 0 else 'BID')",
            ),
            ExpressionPatch(
                function_name="_currency_node_yen",
                before="_tick_price(tick, 'ASK' if value > 0 else 'BID')",
                after="_tick_price(tick, 'BID' if value > 0 else 'ASK')",
            ),
        ),
    ),
    MutationSpec(
        mutation_id="MK03B",
        name="INVERSE_CONVERSION_MULTIPLIED_INSTEAD_OF_DIVIDED",
        mode="PUBLIC_REPLAY",
        fixture=_inverse_cad_fixture,
        assertion=_assert_inverse_cad_result,
        patches=(ExpressionPatch(
            function_name="_currency_node_yen",
            before="_tick_price(tick, 'ASK' if value > 0 else 'BID')",
            after="1 / _tick_price(tick, 'ASK' if value > 0 else 'BID')",
        ),),
    ),
    MutationSpec(
        mutation_id="MK03D",
        name="STALE_CONVERSION_QUOTE_ACCEPTED",
        mode="PUBLIC_REPLAY",
        fixture=_stale_alternate_path_fixture,
        assertion=_assert_stale_alternate_excluded,
        patches=(ExpressionPatch(
            function_name="_latest_fresh_tick",
            before=(
                "source_watermark_ns - tick.source_ts_ns > maximum_staleness_ns or "
                "arrival_cutoff_ns - tick.arrival_ts_ns > maximum_staleness_ns or "
                "arrival_cutoff_ns - tick.source_ts_ns > maximum_staleness_ns"
            ),
            after="False",
        ),),
        allowed_error_text=("JPY conversion path must be uniquely causal",),
    ),
    MutationSpec(
        mutation_id="MK04A",
        name="UNITS_CEILED_INSTEAD_OF_FLOORED",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_directed_rounding,
        patches=(ExpressionPatch(
            function_name="_sized_units",
            before="_floor_fraction(exact)",
            after="_ceil_nonnegative(exact)",
        ),),
    ),
    MutationSpec(
        mutation_id="MK04B",
        name="SIGNED_VALUE_TRUNCATED_TOWARD_ZERO",
        mode="PUBLIC_REPLAY",
        fixture=_inverse_cad_short_fixture,
        assertion=_assert_inverse_cad_short_result,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before="_floor_fraction(executable_exact)",
            after="int(executable_exact)",
        ),),
    ),
    MutationSpec(
        mutation_id="MK04C",
        name="RISK_VALUE_FLOORED_BEFORE_REQUIRED_OUTWARD_ROUND",
        mode="PUBLIC_REPLAY",
        fixture=_inverse_cad_fixture,
        assertion=_assert_inverse_cad_result,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before="_ceil_nonnegative(marked_notional_exact)",
            after="_floor_fraction(marked_notional_exact)",
        ),),
    ),
    MutationSpec(
        mutation_id="MK05A",
        name="COMMISSION_ROUNDED_AFTER_COMBINING_SIDES",
        mode="PUBLIC_REPLAY",
        fixture=_commission_rounding_fixture,
        assertion=_assert_commission_side_rounding,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before="entry_commission + exit_commission",
            after=(
                "_ceil_nonnegative(entry_commission_exact + "
                "exit_commission_exact)"
            ),
        ), ExpressionPatch(
            function_name="_position_values",
            before=(
                "executable - entry_commission - exit_commission - financing"
            ),
            after=(
                "executable - _ceil_nonnegative(entry_commission_exact + "
                "exit_commission_exact) - financing"
            ),
        )),
    ),
    MutationSpec(
        mutation_id="MK05B",
        name="ONE_COMMISSION_SIDE_OMITTED",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_directed_rounding,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before="entry_commission + exit_commission",
            after="entry_commission",
        ), ExpressionPatch(
            function_name="_position_values",
            before=(
                "executable - entry_commission - exit_commission - financing"
            ),
            after="executable - entry_commission - financing",
        )),
    ),
    MutationSpec(
        mutation_id="MK05C",
        name="DIRECT_COST_USES_TARGET_NOTIONAL",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_directed_rounding,
        patches=(
            ExpressionPatch(
                function_name="_position_values",
                before=(
                    "position.entry_notional_exact * "
                    "terms.commission_ppm_per_side / 1000000"
                ),
                after=(
                    "Fraction(position.proposal.target_notional_jpy_micros * "
                    "terms.commission_ppm_per_side, 1000000)"
                ),
            ),
            ExpressionPatch(
                function_name="_position_values",
                before=(
                    "marked_notional_exact * terms.commission_ppm_per_side / "
                    "1000000"
                ),
                after=(
                    "Fraction(position.proposal.target_notional_jpy_micros * "
                    "terms.commission_ppm_per_side, 1000000)"
                ),
            ),
            ExpressionPatch(
                function_name="_reduce_arm_events",
                before=(
                    "entry_notional_exact * terms.commission_ppm_per_side / "
                    "1000000"
                ),
                after=(
                    "Fraction(proposal.target_notional_jpy_micros * "
                    "terms.commission_ppm_per_side, 1000000)"
                ),
            ),
        ),
    ),
    MutationSpec(
        mutation_id="MK06A",
        name="FINANCING_USES_SOURCE_TIME",
        mode="PUBLIC_REPLAY",
        fixture=_divergent_financing_clock_fixture,
        assertion=_assert_divergent_financing_clock,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before="arrival_ns - position.entry.arrival_ts_ns",
            after="mark_tick.source_ts_ns - position.entry.source_ts_ns",
        ),),
    ),
    MutationSpec(
        mutation_id="MK06B",
        name="FINANCING_USES_EXIT_NOTIONAL",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_directed_rounding,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before=(
                "position.entry_notional_exact * terms.financing_ppm_per_day * "
                "elapsed_ns / (DAY_NS * 1000000)"
            ),
            after=(
                "marked_notional_exact * terms.financing_ppm_per_day * "
                "elapsed_ns / (DAY_NS * 1000000)"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK06C",
        name="FINANCING_USES_WRONG_TERMINAL_CLOCK",
        mode="PUBLIC_REPLAY",
        fixture=_terminal_fixture,
        assertion=_assert_terminal_financing_clock,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before=(
                "mark_tick.arrival_ts_ns if valuation_arrival_ns is None else "
                "valuation_arrival_ns"
            ),
            after="mark_tick.arrival_ts_ns",
        ),),
    ),
    MutationSpec(
        mutation_id="MK07A",
        name="ENTRY_ADMITTED_BEFORE_DUE_EXIT",
        mode="PUBLIC_REPLAY",
        fixture=_equal_arrival_release_fixture,
        assertion=_assert_equal_arrival_release,
        patches=(ExpressionPatch(
            function_name="close_due",
            before="tick.arrival_ts_ns >= position.due_arrival_ns",
            after="tick.arrival_ts_ns > position.due_arrival_ns",
        ),),
    ),
    MutationSpec(
        mutation_id="MK07B",
        name="OPPOSITE_SAME_PAIR_NETTED_INSTEAD_OF_REJECTED",
        mode="PUBLIC_REPLAY",
        fixture=_opposite_collision_fixture,
        assertion=_assert_opposite_collision,
        patches=(ExpressionPatch(
            function_name="_reduce_arm_events",
            before=(
                "any((position.proposal.instrument == proposal.instrument for "
                "position in active))"
            ),
            after=(
                "any((position.proposal.instrument == proposal.instrument and "
                "position.proposal.direction == proposal.direction for "
                "position in active))"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK08A",
        name="CAP_USES_TARGET_NOTIONAL",
        mode="PUBLIC_REPLAY",
        fixture=_short_gross_cap_fixture,
        assertion=_assert_short_gross_cap,
        patches=(ExpressionPatch(
            function_name="_reduce_arm_events",
            before="tentative_mark.gross_notional_jpy_micros",
            after=(
                "sum((candidate.proposal.target_notional_jpy_micros for "
                "candidate in tentative))"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK08B",
        name="CURRENCY_NODE_INCIDENCE_SIGN_INVERTED",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_currency_incidence,
        patches=(ExpressionPatch(
            function_name="_currency_exposure_postings",
            before="position.proposal.direction * position.units_micros",
            after="-position.proposal.direction * position.units_micros",
        ),),
        allowed_error_text=("currency posting sign invariant failed",),
    ),
    MutationSpec(
        mutation_id="MK08C",
        name="MARGIN_REQUIREMENT_FLOORED",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_directed_rounding,
        patches=(ExpressionPatch(
            function_name="_mark_state",
            before=(
                "_ceil_nonnegative(Fraction(gross * "
                "data.evaluation['margin_rate_bps'], 10000))"
            ),
            after=(
                "_floor_fraction(Fraction(gross * "
                "data.evaluation['margin_rate_bps'], 10000))"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK09A",
        name="TERMINAL_LIQUIDATION_USES_MIDPOINT",
        mode="PUBLIC_REPLAY",
        fixture=_terminal_fixture,
        assertion=_assert_terminal_liquidation,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before=(
                "_execution_price(mark_tick, position.proposal, terms, "
                "data.registry, opening=False)"
            ),
            after=(
                "(_tick_price(mark_tick, 'MID'), "
                "(mark_tick.bid_ticks + mark_tick.ask_ticks) // 2 * "
                "PRICE_SUBPIP_SCALE, mark_tick.tick_scale * "
                "PRICE_SUBPIP_SCALE) if valuation_arrival_ns == "
                "data.evaluation['period_end_ts_ns'] - 1 and "
                "(not terms.raw_mid) else _execution_price(mark_tick, "
                "position.proposal, terms, data.registry, opening=False)"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK09B",
        name="TERMINAL_LIQUIDATION_OMITS_ACCRUED_COSTS",
        mode="PUBLIC_REPLAY",
        fixture=_terminal_fixture,
        assertion=_assert_terminal_liquidation,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before=(
                "position.entry_notional_exact * terms.financing_ppm_per_day * "
                "elapsed_ns / (DAY_NS * 1000000)"
            ),
            after=(
                "0 if valuation_arrival_ns == "
                "data.evaluation['period_end_ts_ns'] - 1 else "
                "position.entry_notional_exact * terms.financing_ppm_per_day * "
                "elapsed_ns / (DAY_NS * 1000000)"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK09C",
        name="REALIZED_AND_UNREALIZED_PNL_DOUBLE_COUNTED",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_directed_rounding,
        patches=(ExpressionPatch(
            function_name="_position_values",
            before=(
                "_to_jpy_yen(data, quote_pnl, quote_currency, watermark, "
                "arrival_ns) * JPY_MICROS_PER_YEN"
            ),
            after=(
                "2 * _to_jpy_yen(data, quote_pnl, quote_currency, watermark, "
                "arrival_ns) * JPY_MICROS_PER_YEN"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK10A",
        name="ADMISSIONS_CONTINUE_AFTER_CLOSEOUT_OR_RUIN",
        mode="PUBLIC_REPLAY",
        fixture=_ruin_artifacts,
        assertion=_assert_ruin_halts_admissions,
        patches=(ExpressionPatch(
            function_name="_reduce_arm_events",
            before="_risk_closeout_reason(data, mark)",
            after="None",
        ),),
    ),
    MutationSpec(
        mutation_id="MK10B",
        name="TERMINAL_LIQUIDATION_OMITTED",
        mode="PUBLIC_REPLAY",
        fixture=_terminal_fixture,
        assertion=_assert_terminal_liquidation,
        patches=(ExpressionPatch(
            function_name="_reduce_arm_events",
            before=(
                "[(position, _terminal_tick(data, "
                "position.proposal.instrument)) for position in "
                "sorted(active, key=lambda item: item.proposal.ordinal)]"
            ),
            after="[]",
        ),),
        allowed_error_text=("terminal inventory not empty",),
    ),
    MutationSpec(
        mutation_id="MK11A",
        name="JOURNAL_POSTING_LEG_DROPPED",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_directed_rounding,
        patches=(ExpressionPatch(
            function_name="post",
            before=(
                "tuple((Posting(account, combined[account]) for account in "
                "JOURNAL_ACCOUNT_ORDER if combined[account]))"
            ),
            after=(
                "tuple((Posting(account, combined[account]) for account in "
                "JOURNAL_ACCOUNT_ORDER if combined[account]))[:-1]"
            ),
        ),),
        allowed_error_text=("journal transaction is not exactly balanced",),
    ),
    MutationSpec(
        mutation_id="MK11C",
        name="CUMULATIVE_BALANCE_POSTED_INSTEAD_OF_DELTA",
        mode="PUBLIC_REPLAY",
        fixture=_multi_mark_fixture,
        assertion=_assert_multi_mark_result,
        patches=(ExpressionPatch(
            function_name="_journal_mark",
            before=(
                "values['executable_exact'] - position.last_mark_pnl_exact"
            ),
            after="values['executable_exact']",
        ),),
        allowed_error_text=("journal terminal position balance is nonzero",),
    ),
    MutationSpec(
        mutation_id="MK12A",
        name="ECONOMIC_LOT_ROUNDED_PER_TICKET",
        mode="PUBLIC_REPLAY",
        fixture=_connected_cluster_fixture,
        assertion=_assert_connected_cluster,
        patches=(ExpressionPatch(
            function_name="_cluster_metrics_from_events",
            before="_floor_fraction(exact_pnl)",
            after=(
                "sum((_floor_fraction(disposition.values['economic_net_exact']) "
                "for disposition in component))"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK12B",
        name="CURRENCY_CLUSTER_GRAPH_BUILT_WITH_WRONG_CONNECTIVITY",
        mode="PUBLIC_REPLAY",
        fixture=_connected_cluster_fixture,
        assertion=_assert_connected_cluster,
        patches=(ExpressionPatch(
            function_name="_cluster_metrics_from_events",
            before="union(*_pair(disposition.position.proposal.instrument))",
            after="None",
        ),),
    ),
    MutationSpec(
        mutation_id="MK12C",
        name="CVAR_TAIL_COUNT_FLOORED",
        mode="PUBLIC_REPLAY",
        fixture=_three_cluster_artifacts,
        assertion=_assert_three_cluster_tail,
        patches=(ExpressionPatch(
            function_name="_cluster_metrics_from_events",
            before=(
                "max(1, (len(ordered) * evaluation['cvar_tail_bps'] + 9999) "
                "// 10000) if ordered else 0"
            ),
            after=(
                "max(1, len(ordered) * evaluation['cvar_tail_bps'] // 10000) "
                "if ordered else 0"
            ),
        ),),
    ),
    MutationSpec(
        mutation_id="MK12D",
        name="DRAWDOWN_USES_WRONG_OBSERVATION_SET",
        mode="PUBLIC_REPLAY",
        fixture=_canonical_fixture,
        assertion=_assert_full_drawdown_observation_set,
        patches=(ExpressionPatch(
            function_name="_derive_arm_metrics",
            before=(
                "[(mark.arrival_ts_ns, index, mark.marked_equity_jpy_micros) "
                "for (index, mark) in enumerate(replay.risk_snapshots)]"
            ),
            after=(
                "[(mark.arrival_ts_ns, index, mark.marked_equity_jpy_micros) "
                "for (index, mark) in enumerate(replay.risk_snapshots[-1:])]"
            ),
        ),),
    ),
)


def _exercise_future_conversion_guard(
    module: ModuleType,
    canonical: support.CanonicalFixture,
) -> tuple[int, int, int]:
    data = module.decode_reference_input(dict(canonical.artifacts))
    tick = module._latest_fresh_tick(
        data,
        "USD_JPY",
        support.START_NS,
        support.START_NS + 1_000_000_000,
        support.DAY_NS,
    )
    return tick.sequence, tick.source_ts_ns, tick.arrival_ts_ns


def _assert_future_conversion_guard(observed: Any) -> None:
    assert observed == (1, support.START_NS, support.START_NS)


def _exercise_duplicate_journal_guard(
    module: ModuleType,
    _canonical: support.CanonicalFixture,
) -> tuple[str, str, int]:
    journal = module._Journal()
    kwargs = {
        "arrival_ts_ns": support.START_NS,
        "arm": "EXECUTABLE_BASE",
        "proposal_ordinal": 1,
        "event_kind": "COMPONENT_GUARD",
        "event_id": "COMPONENT:DUPLICATE",
        "source_event_sha256": None,
        "postings": (
            ("POSITION_BASIS", Fraction(1)),
            ("POSITION_CONTROL", Fraction(-1)),
        ),
    }
    journal.post(**kwargs)
    try:
        journal.post(**kwargs)
    except module.ReferenceError as error:
        return "REJECTED", str(error), len(journal.transactions)
    return "ACCEPTED", "", len(journal.transactions)


def _assert_duplicate_journal_guard(observed: Any) -> None:
    assert observed == ("REJECTED", "duplicate journal event", 1)


COMPONENT_MUTATION_SPECS: tuple[ComponentMutationSpec, ...] = (
    ComponentMutationSpec(
        mutation_id="MK03C",
        name="FUTURE_CONVERSION_QUOTE_ACCEPTED",
        mode="COMPONENT_GUARD",
        reachability_reason=(
            "Public replay derives the source watermark as the maximum source "
            "timestamp among arrival-eligible ticks, so a later-source tick "
            "within the same arrival cutoff cannot be future to that watermark."
        ),
        exercise=_exercise_future_conversion_guard,
        assertion=_assert_future_conversion_guard,
        patches=(ExpressionPatch(
            function_name="_latest_fresh_tick",
            before=(
                "[tick for tick in data.books.get(instrument, ()) if "
                "tick.source_ts_ns <= source_watermark_ns and "
                "tick.arrival_ts_ns <= arrival_cutoff_ns]"
            ),
            after=(
                "[tick for tick in data.books.get(instrument, ()) if "
                "tick.arrival_ts_ns <= arrival_cutoff_ns]"
            ),
        ),),
    ),
    ComponentMutationSpec(
        mutation_id="MK11B",
        name="DUPLICATE_EVENT_ACCEPTED",
        mode="COMPONENT_GUARD",
        reachability_reason=(
            "Valid public replay constructs canonical event IDs from unique "
            "arm, proposal, event-kind, and clock identities, so duplicate "
            "insertion requires a direct journal boundary exercise."
        ),
        exercise=_exercise_duplicate_journal_guard,
        assertion=_assert_duplicate_journal_guard,
        patches=(ExpressionPatch(
            function_name="post",
            before="event_id in self.event_ids",
            after="False",
        ),),
    ),
)


def _run_public_replay_mutation(
    spec: MutationSpec,
    canonical: support.CanonicalFixture,
) -> tuple[str, str]:
    artifacts = dict(spec.fixture(canonical))
    with _loaded_engine(None) as (baseline, _):
        baseline_result = baseline.replay_reference(dict(artifacts))
    spec.assertion(baseline_result)

    try:
        with _loaded_engine(spec) as (mutant, hits):
            mutant.decode_reference_input(dict(artifacts))
            try:
                mutant_result = mutant.replay_reference(dict(artifacts))
            except mutant.ReferenceError as error:
                if hits[spec.mutation_id] <= 0:
                    return "ERROR", f"semantic error before probe: {error}"
                if not any(text in str(error) for text in spec.allowed_error_text):
                    return "ERROR", f"unexpected ReferenceError: {error}"
                return "KILLED", f"semantic ReferenceError: {error}"
            if hits[spec.mutation_id] <= 0:
                return "ERROR", "mutation probe was not reached"
            try:
                spec.assertion(mutant_result)
            except AssertionError as error:
                return "KILLED", f"named economic assertion failed: {error}"
            return "SURVIVED", "named economic assertion still passed"
    except Exception as error:  # compile, exact-site, or harness failures are errors
        return "ERROR", f"{type(error).__name__}: {error}"


def _run_component_mutation(
    spec: ComponentMutationSpec,
    canonical: support.CanonicalFixture,
) -> tuple[str, str]:
    with _loaded_engine(None) as (baseline, _):
        baseline_observed = spec.exercise(baseline, canonical)
    spec.assertion(baseline_observed)

    try:
        with _loaded_engine(spec) as (mutant, hits):
            try:
                mutant_observed = spec.exercise(mutant, canonical)
            except mutant.ReferenceError as error:
                if hits[spec.mutation_id] <= 0:
                    return "ERROR", f"semantic error before probe: {error}"
                if not any(text in str(error) for text in spec.allowed_error_text):
                    return "ERROR", f"unexpected ReferenceError: {error}"
                return "KILLED", f"component ReferenceError: {error}"
            if hits[spec.mutation_id] <= 0:
                return "ERROR", "component mutation probe was not reached"
            try:
                spec.assertion(mutant_observed)
            except AssertionError as error:
                return "KILLED", f"component invariant failed: {error}"
            return "SURVIVED", "component invariant still passed"
    except Exception as error:  # compile, exact-site, or harness failures are errors
        return "ERROR", f"{type(error).__name__}: {error}"


def test_runtime_source_mutation_campaign() -> None:
    canonical = support._build_canonical_fixture()
    public_report: dict[str, dict[str, str]] = {}
    for spec in MUTATION_SPECS:
        if spec.mode != "PUBLIC_REPLAY":
            public_report[spec.mutation_id] = {
                "status": "ERROR",
                "detail": f"unsupported execution mode: {spec.mode}",
            }
            continue
        status, detail = _run_public_replay_mutation(spec, canonical)
        public_report[spec.mutation_id] = {"status": status, "detail": detail}

    component_report: dict[str, dict[str, str]] = {}
    for spec in COMPONENT_MUTATION_SPECS:
        if spec.mode != "COMPONENT_GUARD":
            component_report[spec.mutation_id] = {
                "status": "ERROR",
                "detail": f"unsupported execution mode: {spec.mode}",
            }
            continue
        status, detail = _run_component_mutation(spec, canonical)
        component_report[spec.mutation_id] = {
            "status": status,
            "detail": detail,
            "reachability_reason": spec.reachability_reason,
        }

    public_counts = Counter(item["status"] for item in public_report.values())
    component_counts = Counter(
        item["status"] for item in component_report.values()
    )
    all_results = {**public_report, **component_report}
    counts = Counter(item["status"] for item in all_results.values())
    print(json.dumps({
        "engine_source_sha256": support._digest(ENGINE_SOURCE.encode("utf-8")),
        "counts": dict(sorted(counts.items())),
        "mode_counts": {
            "PUBLIC_REPLAY": dict(sorted(public_counts.items())),
            "COMPONENT_GUARD": dict(sorted(component_counts.items())),
        },
        "results": all_results,
    }, sort_keys=True))
    expected_ids = {
        "MK01A", "MK01B", "MK01C", "MK01D", "MK02A", "MK02B",
        "MK03A", "MK03B", "MK03C", "MK03D", "MK04A", "MK04B",
        "MK04C", "MK05A", "MK05B", "MK05C", "MK06A", "MK06B",
        "MK06C", "MK07A", "MK07B", "MK08A", "MK08B", "MK08C",
        "MK09A", "MK09B", "MK09C", "MK10A", "MK10B", "MK11A",
        "MK11B", "MK11C", "MK12A", "MK12B", "MK12C", "MK12D",
    }
    assert len(MUTATION_SPECS) == 34
    assert len(COMPONENT_MUTATION_SPECS) == 2
    assert set(all_results) == expected_ids
    assert Counter(mutation_id[:4] for mutation_id in all_results) == {
        "MK01": 4,
        "MK02": 2,
        "MK03": 4,
        "MK04": 3,
        "MK05": 3,
        "MK06": 3,
        "MK07": 2,
        "MK08": 3,
        "MK09": 3,
        "MK10": 2,
        "MK11": 3,
        "MK12": 4,
    }
    assert public_counts == {"KILLED": 34}
    assert component_counts == {"KILLED": 2}
    assert counts == {"KILLED": 36}
