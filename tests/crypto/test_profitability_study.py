from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from quant_rabbit.crypto.ledger import CryptoLedger
from quant_rabbit.crypto.profitability_study import (
    BitbankProfitabilityStudy,
)


RESEARCH_CONFIG = Path(
    "config/crypto_bitbank_research_candidates_v1.json"
)


def _trade(
    strategy: str,
    index: int,
    *,
    gross: str,
    fees: str,
    opened: str,
    closed: str,
) -> dict[str, object]:
    return {
        "trade_id": f"{strategy}-{index}",
        "operation_id": f"operation-{strategy}-{index}",
        "run_id": f"run-{strategy}",
        "paper_mode": "SPOT",
        "pair": "btc_jpy",
        "side": "LONG",
        "opened_at_utc": opened,
        "closed_at_utc": closed,
        "entry_notional_jpy": "1000",
        "entry_price": "100",
        "exit_price": "100",
        "quantity": "10",
        "gross_pnl_jpy": gross,
        "fees_jpy": fees,
        "spread_cost_jpy": "0",
        "adverse_cost_jpy": "0",
        "funding_interest_jpy": "0",
        "net_pnl_jpy": str(float(gross) - float(fees)),
        "exit_reason": "MAX_HOLD",
        "strategy": strategy,
        "regime": "RANGE",
        "authority": "NONE",
    }


def _write_lane(
    runtime: Path,
    strategy: str,
    rows: list[dict[str, object]],
) -> None:
    root = (
        runtime
        / strategy.lower().replace("_", "-")
        / "spot"
    )
    root.mkdir(parents=True)
    (root / "trade_outbox.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    ledger = CryptoLedger(root / "ledger.db")
    for index, row in enumerate(rows):
        opened = str(row["opened_at_utc"])
        entry_at = (
            datetime.fromisoformat(opened) - timedelta(seconds=1)
        ).isoformat()
        ledger.append(
            "FAST_DECISION",
            "btc_jpy",
            {
                "run_id": row["run_id"],
                "pair": "btc_jpy",
                "action": "ENTER",
                "observed_at_utc": entry_at,
                "imbalance": "0.5",
                "net_edge_bps": "0.5",
                "authority": "NONE",
            },
            dedupe_key=f"entry-{strategy}-{index}",
        )
        ledger.append(
            "FAST_DECISION",
            "btc_jpy",
            {
                "run_id": row["run_id"],
                "pair": "btc_jpy",
                "action": "WAIT",
                "observed_at_utc": opened,
                "held_ms": 7000,
                "position_pnl_bps": "1",
                "authority": "NONE",
            },
            dedupe_key=f"hold-{strategy}-{index}",
        )


def test_study_preserves_baseline_and_never_adopts_retrospective_screen(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "strategy-lab"
    samples = {
        "RANGE_MAKER_REVERSION": ("3", "1"),
        "BREAKOUT_CONFIRMATION": ("-1", "1"),
        "TREND_PULLBACK_MAKER": ("2", "1"),
        "ORDER_BOOK_FADE": ("-9", "1"),
        "ORDER_BOOK_FADE_COOLDOWN_5S": ("-7", "1"),
        "ORDER_BOOK_FADE_MAKER_EXIT": ("0", "1"),
    }
    for index, (strategy, values) in enumerate(samples.items()):
        hour = index
        opened = f"2026-07-28T{hour:02d}:00:07+00:00"
        closed = f"2026-07-28T{hour:02d}:00:10+00:00"
        rows = [
            _trade(
                strategy,
                index,
                gross=values[0],
                fees=values[1],
                opened=opened,
                closed=closed,
            )
        ]
        if strategy == "ORDER_BOOK_FADE":
            rows.append(
                _trade(
                    strategy,
                    99,
                    gross="-3",
                    fees="1",
                    opened="2026-07-28T05:00:07+00:00",
                    closed="2026-07-28T05:00:10+00:00",
                )
            )
        _write_lane(runtime, strategy, rows)

    result = BitbankProfitabilityStudy(
        runtime,
        research_config=RESEARCH_CONFIG,
        output_root=tmp_path / "output",
    ).run_once()

    baseline = result["baseline_contract"]["metrics"]
    assert baseline["completed_trades"] == 6
    assert baseline["net_pnl_jpy"] == "-21.0"
    assert result["safety"]["authority"] == "NONE"
    assert result["safety"]["existing_shadow_changed"] is False

    comparisons = {
        row["category"]: row
        for row in result["isolated_comparisons"]
    }
    fade = comparisons["board_fade_stop_candidate"]
    assert fade["candidate_metrics"]["completed_trades"] == 3
    assert fade["status"] == "STOP_CANDIDATE_NOT_APPLIED"
    maker = comparisons["maker_taker_alignment"]
    assert maker["evidence_class"] == "PROSPECTIVE_PARALLEL_PAPER"
    assert maker["adopted"] is False
    entry = comparisons["entry_direction_and_threshold"]
    assert entry["prospective_unseen_window_count"] == 0
    assert entry["adopted"] is False
    assert result["research_review"]["external_code_executed"] is False
    assert (tmp_path / "output" / "study.json").exists()
    assert (tmp_path / "output" / "study.md").exists()
