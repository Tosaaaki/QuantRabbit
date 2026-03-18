from __future__ import annotations

import json
from pathlib import Path

from scripts import improvement_gate


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _artifact(path: Path, *, warnings: list[str] | None = None) -> Path:
    payload = {
        "generated_at": "2026-03-14T13:12:18Z",
        "query": "yesterday_trade_review",
        "warnings": warnings or [],
        "market": {
            "market_open": True,
            "seconds_until_open": 0.0,
            "spread_pips": 0.8,
            "tick_age_sec": 1.0,
            "data_lag_ms": 120.0,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_build_improvement_gate_payload_blocks_same_surface_pending(
    tmp_path: Path,
) -> None:
    findings = _write(
        tmp_path / "TRADE_FINDINGS.md",
        """
## 2026-03-13 22:15 JST / local-v2: `WickReversalBlend` の weak countertrend short を worker-local に遮断
- Hypothesis Key: `wick_short_countertrend_guard_20260313_2215`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `0`
- Why Not Same As Last Time: `flow_regime=range_fade` と `align:countertrend`
- Promotion Gate: `6h で same fingerprint の filled が 0`
- Escalation Trigger: `same fingerprint の filled が残る`
- Verdict: pending
- Status: pending
""".strip() + "\n",
    )
    artifact = _artifact(tmp_path / "change_preflight_latest.json")

    payload = improvement_gate.build_improvement_gate_payload(
        findings_path=findings,
        artifact_path=artifact,
        query="yesterday_trade_review",
        candidates_raw=(
            "WickReversalBlend::short range_fade macro:trend_long align:countertrend::"
            "STOP_LOSS_ORDER::tighten short guard"
        ),
        limit=5,
    )

    candidate = payload["candidates"][0]
    assert payload["market_status"] == "normal"
    assert candidate["action"] == "review_existing_pending"
    assert candidate["same_surface_unresolved"][0]["hypothesis_key"] == (
        "wick_short_countertrend_guard_20260313_2215"
    )


def test_build_improvement_gate_payload_marks_market_hold(tmp_path: Path) -> None:
    findings = _write(
        tmp_path / "TRADE_FINDINGS.md",
        """
## 2026-03-13 16:35 JST / local-v2: `TickImbalance` の trend exhaustion 追随を worker-local に遮断
- Hypothesis Key: `tick_exhaustion_guard`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `0`
- Verdict: pending
- Status: pending
""".strip() + "\n",
    )
    artifact = _artifact(
        tmp_path / "change_preflight_latest.json",
        warnings=["tick_stale:58416.6s", "spread_wide:1.20p"],
    )

    payload = improvement_gate.build_improvement_gate_payload(
        findings_path=findings,
        artifact_path=artifact,
        query="yesterday_trade_review",
        candidates_raw=(
            "TickImbalance::trend_short rsi:ext_oversold::STOP_LOSS_ORDER::cap sizing"
        ),
        limit=5,
    )

    candidate = payload["candidates"][0]
    assert payload["market_status"] == "market_hold"
    assert candidate["action"] == "market_hold_review_only"
    assert "tick_stale:58416.6s" in candidate["reasons"]


def test_build_improvement_gate_payload_marks_market_closed_hold(
    tmp_path: Path,
) -> None:
    findings = _write(
        tmp_path / "TRADE_FINDINGS.md",
        """
## 2026-03-15 23:06 JST / local-v2: `scalp_ping_5s_c_live` reject review
- Hypothesis Key: `ping5s_c_reject_review`
- Primary Loss Driver: `entry_probability_reject`
- Mechanism Fired: `0`
- Verdict: pending
- Status: pending
""".strip() + "\n",
    )
    artifact = _artifact(tmp_path / "change_preflight_latest.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["warnings"] = []
    payload["market"]["market_open"] = False
    payload["market"]["seconds_until_open"] = 28080.0
    artifact.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = improvement_gate.build_improvement_gate_payload(
        findings_path=findings,
        artifact_path=artifact,
        query="weekend_review",
        candidates_raw=(
            "scalp_ping_5s_c_live::entry_probability_gate::entry_probability_reject::revalidate pending"
        ),
        limit=5,
    )

    candidate = result["candidates"][0]
    assert result["market_status"] == "market_hold"
    assert candidate["action"] == "market_hold_review_only"
    assert "market_closed:28080.0s_to_open" in candidate["reasons"]


def test_build_improvement_gate_payload_allows_new_lane_without_overlap(
    tmp_path: Path,
) -> None:
    findings = _write(
        tmp_path / "TRADE_FINDINGS.md",
        """
## 2026-03-13 21:55 JST / local-v2: `DroughtRevert` current loser setup
- Hypothesis Key: `drought_guard`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `1`
- Verdict: good
- Status: done
""".strip() + "\n",
    )
    artifact = _artifact(tmp_path / "change_preflight_latest.json")

    payload = improvement_gate.build_improvement_gate_payload(
        findings_path=findings,
        artifact_path=artifact,
        query="yesterday_trade_review",
        candidates_raw=(
            "TickImbalance::trend_short rsi:ext_oversold::STOP_LOSS_ORDER::cap sizing"
        ),
        limit=5,
    )

    candidate = payload["candidates"][0]
    assert payload["blocked"] is False
    assert candidate["action"] == "allow_new_lane"


def test_build_improvement_gate_payload_blocks_same_strategy_open_lane(
    tmp_path: Path,
) -> None:
    findings = _write(
        tmp_path / "TRADE_FINDINGS.md",
        """
## 2026-03-16 10:18 JST / local-v2: `MicroLevelReactor-bounce-lower` loser lane
- Hypothesis Key: `microlevelreactor_bounce_lower_prefix_bug_20260316`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `1`
- Verdict: pending
- Status: pending_live_prefix_validation
""".strip() + "\n",
    )
    artifact = _artifact(tmp_path / "change_preflight_latest.json")

    payload = improvement_gate.build_improvement_gate_payload(
        findings_path=findings,
        artifact_path=artifact,
        query="live reopen loser-lane triage",
        candidates_raw=(
            "MicroLevelReactor::range_fade long tight_normal gap:down_flat::"
            "countertrend_continuation::add reclaim fail hard block"
        ),
        limit=5,
    )

    candidate = payload["candidates"][0]
    assert payload["blocked"] is True
    assert candidate["action"] == "review_existing_pending"
    assert candidate["same_strategy_open_lanes"][0]["hypothesis_key"] == (
        "microlevelreactor_bounce_lower_prefix_bug_20260316"
    )


def test_build_improvement_gate_payload_escalates_same_strategy_not_fired_repeats(
    tmp_path: Path,
) -> None:
    findings = _write(
        tmp_path / "TRADE_FINDINGS.md",
        """
## 2026-03-13 23:40 JST / local-v2: `PrecisionLowVol` short lane
- Hypothesis Key: `precision_up_lean_countertrend_guard_20260313_2340`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `0`
- Verdict: pending
- Status: pending

## 2026-03-13 23:53 JST / local-v2: `PrecisionLowVol` long lane
- Hypothesis Key: `precision_up_flat_shallow_long_guard_20260313_2353`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `0`
- Verdict: pending
- Status: pending
""".strip() + "\n",
    )
    artifact = _artifact(tmp_path / "change_preflight_latest.json")

    payload = improvement_gate.build_improvement_gate_payload(
        findings_path=findings,
        artifact_path=artifact,
        query="live reopen loser-lane triage",
        candidates_raw=(
            "PrecisionLowVol::range_compression long countertrend volatility_compression::"
            "countertrend_compression_fail::harden setup pressure block"
        ),
        limit=5,
    )

    candidate = payload["candidates"][0]
    assert payload["blocked"] is True
    assert candidate["action"] == "escalate_family_not_tighten"
    assert len(candidate["same_strategy_open_lanes"]) == 2
    assert any("Mechanism Fired=0/none" in reason for reason in candidate["reasons"])


def test_build_improvement_gate_payload_blocks_advanced_idea_without_baseline(
    tmp_path: Path,
) -> None:
    findings = _write(
        tmp_path / "TRADE_FINDINGS.md",
        """
## 2026-03-13 21:55 JST / local-v2: `MomentumBurst` current loser setup
- Hypothesis Key: `momentumburst_guard`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `1`
- Verdict: good
- Status: done
""".strip() + "\n",
    )
    artifact = _artifact(tmp_path / "change_preflight_latest.json")

    payload = improvement_gate.build_improvement_gate_payload(
        findings_path=findings,
        artifact_path=artifact,
        query="yesterday_trade_review",
        candidates_raw=(
            "MomentumBurst::reaccel lane continuation cluster::STOP_LOSS_ORDER::"
            "trial simple Kalman hedge"
        ),
        limit=5,
    )

    candidate = payload["candidates"][0]
    assert payload["blocked"] is True
    assert candidate["action"] == "baseline_before_complexity"
    assert "kalman" in candidate["complexity_signals"]
    assert candidate["has_baseline_evidence"] is False


def test_build_improvement_gate_payload_allows_advanced_idea_after_baseline_evidence(
    tmp_path: Path,
) -> None:
    findings = _write(
        tmp_path / "TRADE_FINDINGS.md",
        """
## 2026-03-13 21:55 JST / local-v2: `MomentumBurst` current loser setup
- Hypothesis Key: `momentumburst_guard`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `1`
- Verdict: good
- Status: done
""".strip() + "\n",
    )
    artifact = _artifact(tmp_path / "change_preflight_latest.json")

    payload = improvement_gate.build_improvement_gate_payload(
        findings_path=findings,
        artifact_path=artifact,
        query="yesterday_trade_review",
        candidates_raw=(
            "MomentumBurst::baseline replay validation split stable::STOP_LOSS_ORDER::"
            "trial simple Kalman after replay validation"
        ),
        limit=5,
    )

    candidate = payload["candidates"][0]
    assert payload["blocked"] is False
    assert candidate["action"] == "allow_new_lane"
    assert "kalman" in candidate["complexity_signals"]
    assert candidate["has_baseline_evidence"] is True
