#!/usr/bin/env python3
"""GET-only OANDA account/quote readiness screen.  This tool cannot order."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from quant_rabbit.broker.oanda import OandaReadOnlyClient
from quant_rabbit.trade_readiness import ExplicitRiskLimits, screen_trade_readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--software-ready", action="store_true")
    parser.add_argument("--max-loss-per-order-jpy", type=float)
    parser.add_argument("--stop-drawdown-jpy", type=float)
    parser.add_argument("--minimum-margin-buffer-jpy", type=float)
    parser.add_argument("--max-post-entry-current-mcp", type=float)
    parser.add_argument("--max-post-entry-stress-mcp", type=float)
    parser.add_argument("--max-currency-factor-nav-multiple", type=float)
    parser.add_argument("--max-bot-positions", type=int)
    parser.add_argument("--mode-hysteresis-mcp", type=float)
    parser.add_argument("--forward-proof-sha256")
    parser.add_argument("--risk-contract-sha256")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    client = OandaReadOnlyClient(env_file=args.env_file)
    snapshot = client.snapshot(("EUR_USD", "USD_JPY"))
    raw_account = client.get_json(f"/v3/accounts/{client.account_id}/summary")
    result = screen_trade_readiness(
        snapshot=snapshot,
        raw_account=raw_account,
        limits=ExplicitRiskLimits(
            max_loss_per_order_jpy=args.max_loss_per_order_jpy,
            stop_drawdown_jpy=args.stop_drawdown_jpy,
            minimum_margin_buffer_jpy=args.minimum_margin_buffer_jpy,
            max_post_entry_current_mcp=args.max_post_entry_current_mcp,
            max_post_entry_stress_mcp=args.max_post_entry_stress_mcp,
            max_currency_factor_nav_multiple=args.max_currency_factor_nav_multiple,
            max_bot_positions=args.max_bot_positions,
            mode_hysteresis_mcp=args.mode_hysteresis_mcp,
            forward_proof_sha256=args.forward_proof_sha256,
            risk_contract_sha256=args.risk_contract_sha256,
        ),
        software_ready=args.software_ready,
        now_utc=datetime.now(timezone.utc),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
