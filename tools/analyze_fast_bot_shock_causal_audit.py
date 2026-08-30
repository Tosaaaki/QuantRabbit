#!/usr/bin/env python3
"""Build a no-order incident audit from retained ledgers and replay evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_rabbit.fast_bot_shock_causal_audit import audit_shock_episode  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _metric(row: dict[str, Any]) -> str:
    pf = row.get("profit_factor")
    pf_text = "n/a" if pf is None else f"{float(pf):.6f}"
    return f"{int(row.get('filled_trades') or row.get('trades') or 0)} / {float(row.get('net_pips') or 0.0):.3f} / {pf_text}"


def _markdown(result: dict[str, Any]) -> str:
    incident = result["incident"]
    historical = result["historical_replay"]
    exact = incident.get("exact_event_eurusd") or []
    eu = exact[0] if exact else {}
    usd = incident.get("exact_event_usdjpy") or []
    uj = usd[0] if usd else {}
    classes = historical["episodes"]["causal_classification"]
    horizons = historical["episodes"]["horizon_rates"]
    actual_arms = incident["arms_same_proposal_stream"]
    historical_arms = historical["arms"]
    frontier = result.get("profitability_frontier") or {}
    frontier_rows = (
        frontier.get("evidence_rows")
        or frontier.get("frontiers")
        or frontier.get("trade_eligible_candidates")
        or []
    )
    lines = [
        "# Fast-bot 2026-08-28 shock causal audit — 2026-08-30",
        "",
        "## 結論",
        "",
        "- 今回の損失原因は単なるLONGではなく、下降継続ショック中にも `RANGE_ROTATION` の逆張りLONGを出し続けたレジーム遷移誤認です。実signal ledgerのmethod/strategy_idを使用し、価格から戦略名を捏造していません。",
        f"- 14:03のEUR/USDは `{eu.get('method')}/{eu.get('side')}`、regime_score `{eu.get('regime_score')}`、spread `{eu.get('spread_pips')}` pips、結果 `{eu.get('exit_reason')}` / `{eu.get('realized_pips')}` pipsです。",
        f"- 14:03のUSD/JPYは `{uj.get('method')}/{uj.get('side')}` signal自体は存在しました。非参加理由は `{incident.get('usdjpy_participation_reason')}` で、veto・spread・gap・quarantineではありません。実注文は全proposalがshadow-onlyかつexecution authority NONEのため0件です。",
        f"- 過去同型shockは raw {historical['episodes']['raw_detected_count']:,}件、比較可能 {historical['episodes']['count']:,}件です。分類は continuation {classes['CONTINUATION']['episodes']:,}、V-reversal {classes['V_REVERSAL']['episodes']:,}、whipsaw {classes['WHIPSAW']['episodes']:,}件です。",
        "- 利益化は未達です。PF<1の案を改善とは呼ばず、損失回避と利益創出を分離します。",
        "",
        "## 14:00–14:20 UTC actual proposal stream",
        "",
        "| Arm | Filled / net pips / PF | Loss avoidance vs baseline | Profit creating |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "baseline",
        "shock_freeze_5m",
        "side_relative_regime_transition_veto",
        "trend_aligned_continuation_after_5m_half_size",
        "v_reversal_confirmed_only",
        "whipsaw_freeze",
        "bot_owned_50pct_staged_drain_proxy",
    ):
        row = actual_arms[key]
        lines.append(
            f"| `{key}` | {_metric(row)} | {float(row.get('loss_avoidance_vs_baseline_pips') or 0.0):.3f} | {str(bool(row.get('profit_creating'))).lower()} |"
        )
    lines.extend(
        [
            "",
            "| Pair / method / side | Proposals | Filled | Net pips | PF |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in incident["by_pair_method_side"]:
        pf = row.get("profit_factor")
        pf_text = "n/a" if pf is None else f"{float(pf):.6f}"
        lines.append(
            f"| `{row.get('pair')}/{row.get('method')}/{row.get('side')}` | {int(row['proposals'])} | {int(row['filled_trades'])} | {float(row['net_pips']):.3f} | {pf_text} |"
        )
    lines.extend(
        [
            "",
            "`catastrophic_stop_plus_structure_exit` はretained proposal outcomeだけでは新しいS5退出経路を再採点できないため、同一proposal streamでは未確認とし、価格proxyで埋めていません。historical M1 bid/ask cohortの別表でのみ比較します。",
            "",
            "## Historical EUR/USD M1 bid/ask shock cohort",
            "",
            "| Horizon | Continuation | 50% retrace | Mean MFE | Mean MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for horizon in ("5m", "15m", "30m", "60m"):
        row = horizons[horizon]
        lines.append(
            f"| {horizon} | {float(row['continuation_rate']):.2%} | {float(row['retrace_50pct_rate']):.2%} | {float(row['mean_mfe_pips']):.3f}p | {float(row['mean_mae_pips']):.3f}p |"
        )
    lines.extend(
        [
            "",
            "| Bounded arm | Trades / net pips / PF |",
            "|---|---:|",
        ]
    )
    for key in (
        "baseline_immediate_continuation",
        "new_shock_guard",
        "new_shock_guard_plus_50pct_drain_proxy",
        "trend_aligned_continuation_after_5m_half_size",
        "v_reversal_after_failed_continuation",
        "whipsaw_freeze",
    ):
        lines.append(f"| `{key}` | {_metric(historical_arms[key])} |")
    architecture = historical["exit_architecture_arms"][
        "CONSERVATIVE_CATASTROPHE_PLUS_STRUCTURE_EXIT"
    ]
    lines.extend(
        [
            f"| `catastrophic_stop_plus_structure_exit` | {int(architecture['trades'])} / {float(architecture['net_pips']):.3f} / {float(architecture['profit_factor']):.6f} |",
            "",
            "ATRはonset triggerに使っていません。volatility bandはshock前60分のraw range、cross-pair confirmationはhistorical inputにUSD/JPY truthがないため unavailable です。",
            "",
            "## Profitability frontierとの統合",
            "",
            f"- requested 224-signal corrective snapshot best PF: {result['corrective_challenger_reference']['requested_snapshot_best_profit_factor']}",
            f"- latest retained corrective scorecard: {result['corrective_challenger_reference']['latest_signal_count']} signals / best PF {result['corrective_challenger_reference']['latest_best_profit_factor']}",
            f"- shock continuation validation PF: {result['profitability_reference']['shock_validation_pf']} / net {result['profitability_reference']['shock_validation_net_pips']} pips",
            f"- nonshock hourly PF: {result['profitability_reference']['nonshock_hourly_pf']} / net {result['profitability_reference']['nonshock_hourly_net_pips']} pips",
            f"- profitability frontier trade-eligible candidates: {len(frontier_rows)}",
            "- 採用はzero-authority shadow観測のみ。live昇格条件は独立holdoutでafter-cost PF>1、上下両方向、cost stress、十分な日数/件数を同時に満たすこと。停止条件はstale/gap、seal drift、PF<=1、tail悪化、片方向集中、実行権限の非NONE化です。",
            "",
            "## Authority",
            "",
            "`execution_authority=NONE`, `Gateway invocation=0`, `external_order_attempts=0`, `external_orders=0`, manual/tagless `NO_TOUCH`。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-ledger", type=Path, required=True)
    parser.add_argument("--outcome-ledger", type=Path, required=True)
    parser.add_argument("--historical-replay", type=Path, required=True)
    parser.add_argument("--profitability-frontier", type=Path)
    parser.add_argument("--corrective-scorecard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    incident = audit_shock_episode(
        signals=_jsonl(args.shadow_ledger),
        outcomes=_jsonl(args.outcome_ledger),
        window_start_utc=datetime(2026, 8, 28, 14, 0, tzinfo=timezone.utc),
        window_end_utc=datetime(2026, 8, 28, 14, 20, tzinfo=timezone.utc),
        shock_at_utc=datetime(2026, 8, 28, 14, 3, 15, tzinfo=timezone.utc),
        shock_pair="EUR_USD",
        shock_direction="DOWN",
        historical_episode_class="CONTINUATION",
    )
    corrective = _object(args.corrective_scorecard)
    corrective_rows = corrective.get("comparison") or []
    latest_signal_count = max(
        (int(row.get("signal_count") or 0) for row in corrective_rows), default=0
    )
    latest_best_pf = max(
        (float(row.get("profit_factor") or 0.0) for row in corrective_rows), default=0.0
    )
    result = {
        "contract": "QR_FAST_BOT_SHOCK_CAUSAL_AUDIT_BUNDLE_V1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "incident": incident,
        "historical_replay": _object(args.historical_replay),
        "profitability_frontier": _object(args.profitability_frontier)
        if args.profitability_frontier
        else None,
        "corrective_challenger_reference": {
            "requested_snapshot_signals": 224,
            "requested_snapshot_best_profit_factor": 0.488,
            "latest_signal_count": latest_signal_count,
            "latest_best_profit_factor": round(latest_best_pf, 6),
            "latest_scorecard_path": str(args.corrective_scorecard),
            "latest_scorecard_contract_sha256": corrective.get("contract_sha256"),
            "source": "DIRECT_RETAINED_SCORECARD_PLUS_REQUESTED_REFERENCE",
        },
        "profitability_reference": {
            "shock_validation_pf": 0.531544,
            "shock_validation_net_pips": -346.938839,
            "nonshock_hourly_pf": 0.864926,
            "nonshock_hourly_net_pips": -178.5,
        },
        "execution_authority": "NONE",
        "gateway_invocations": 0,
        "external_order_attempts": 0,
        "external_orders": 0,
        "manual_tagless_policy": "NO_TOUCH",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": "AUDIT_COMPLETE", "output": str(args.output), "report": str(args.report), "execution_authority": "NONE", "gateway_invocations": 0, "external_order_attempts": 0, "external_orders": 0}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
