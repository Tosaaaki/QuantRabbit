from __future__ import annotations

import json
from pathlib import Path

from quant_rabbit.crypto.ledger import CryptoLedger
from quant_rabbit.crypto.strategy_audit import StrategyLabAudit
from quant_rabbit.crypto.strategies import load_strategy_profiles


CONFIG = Path("config/crypto_strategy_lab_v1.json")


def test_strategy_audit_separates_fills_from_completed_trades(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "strategy-lab"
    for strategy in load_strategy_profiles(CONFIG):
        slug = strategy.lower().replace("_", "-")
        for mode in ("spot", "margin"):
            root = runtime / slug / mode
            root.mkdir(parents=True)
            ledger = CryptoLedger(root / "ledger.db")
            if strategy == "ORDER_BOOK_FADE" and mode == "spot":
                for index in range(4):
                    ledger.append(
                        "PAPER_FILL",
                        f"fill-{index}",
                        {
                            "status": (
                                "FILLED"
                                if index == 3
                                else "PARTIALLY_FILLED"
                            ),
                            "order_style": (
                                "PAPER_TAKER"
                                if index == 3
                                else "PAPER_MAKER_LIMIT"
                            ),
                        },
                        dedupe_key=f"fill-{index}",
                    )
                trade = {
                    "trade_id": "trade-1",
                    "operation_id": "operation-1",
                    "run_id": "run-1",
                    "paper_mode": "SPOT",
                    "pair": "btc_jpy",
                    "side": "LONG",
                    "opened_at_utc": "2026-07-28T00:00:00+00:00",
                    "closed_at_utc": "2026-07-28T00:00:10+00:00",
                    "entry_notional_jpy": "1000",
                    "gross_pnl_jpy": "0.1",
                    "fees_jpy": "1",
                    "spread_cost_jpy": "0",
                    "adverse_cost_jpy": "0",
                    "funding_interest_jpy": "0",
                    "net_pnl_jpy": "-0.9",
                    "exit_reason": "MAX_HOLD",
                    "strategy": strategy,
                    "regime": "RANGE",
                }
                (root / "trade_outbox.jsonl").write_text(
                    json.dumps(trade) + "\n",
                    encoding="utf-8",
                )
            elif (
                strategy == "ORDER_BOOK_FADE_COOLDOWN_5S"
                and mode == "spot"
            ):
                rows = []
                for index in range(2):
                    rows.append(
                        {
                            "trade_id": f"variant-trade-{index}",
                            "operation_id": f"variant-operation-{index}",
                            "run_id": "variant-run",
                            "paper_mode": "SPOT",
                            "pair": "btc_jpy",
                            "side": "LONG",
                            "opened_at_utc": (
                                "2026-07-28T00:00:00+00:00"
                            ),
                            "closed_at_utc": (
                                f"2026-07-28T00:00:1{index}+00:00"
                            ),
                            "entry_notional_jpy": "1000",
                            "gross_pnl_jpy": "0.1",
                            "fees_jpy": "1",
                            "spread_cost_jpy": "0",
                            "adverse_cost_jpy": "0",
                            "funding_interest_jpy": "0",
                            "net_pnl_jpy": "-0.9",
                            "exit_reason": "MAX_HOLD",
                            "strategy": strategy,
                            "regime": "RANGE",
                        }
                    )
                (root / "trade_outbox.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            (root / "state.json").write_text(
                json.dumps(
                    {
                        "status": "RUNNING",
                        "service_pid": 123,
                        "run_id": "run-1",
                        "events_processed": 2,
                        "guardian": {"state": "GREEN"},
                        "metrics": {
                            "trade_count": 4,
                            "round_trip_count": 1,
                            "equity_jpy": "9999.1",
                            "net_pnl_jpy": "-0.9",
                            "max_drawdown_jpy": "0.9",
                            "open_position_count": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )

    baseline = tmp_path / "baseline"
    for mode in ("spot", "margin"):
        root = baseline / mode
        root.mkdir(parents=True)
        (root / "state.json").write_text(
            json.dumps(
                {
                    "status": "RUNNING",
                    "guardian": {"state": "GREEN"},
                    "metrics": {
                        "round_trip_count": 0,
                        "net_pnl_jpy": "0",
                        "equity_jpy": "10000",
                        "max_drawdown_jpy": "0",
                    },
                }
            ),
            encoding="utf-8",
        )

    result = StrategyLabAudit(
        runtime,
        baseline_root=baseline,
        strategy_config=CONFIG,
    ).run_once()
    fade = next(
        lane
        for lane in result["strategy_lanes"]
        if lane["lane_id"] == "ORDER_BOOK_FADE:SPOT"
    )
    assert fade["completed_trade_count"] == 1
    assert fade["fill_audit"]["fill_events"] == 4
    assert fade["fill_audit"]["partial_fill_ratio"] == 0.75
    assert (
        fade["root_causes_top3"][0]["code"]
        == "FEE_DRAG_DOMINATES_GROSS_EDGE"
    )
    assert result["metric_contract"][
        "fill_count_vs_epoch_events_comparable"
    ] is False
    assert result["strategy_totals"]["net_pnl_jpy"] == "-2.7"
    assert result["baseline"][0]["net_pnl_jpy"] == "0"
    experiment = next(
        row
        for row in result["experiments"]
        if row["mode"] == "SPOT"
    )
    assert experiment["status"] == "REJECTED_EARLY"
    assert experiment["reason"] == (
        "COOLDOWN_DID_NOT_REDUCE_PER_TRADE_FEE_DRAG"
    )
