from __future__ import annotations

import unittest

from tools.mine_manual_history import analyze, reconstruct_trades


class ManualHistoryMiningTest(unittest.TestCase):
    def test_counts_partial_reductions_and_transfer_adjusted_return(self) -> None:
        transactions = [
            {
                "type": "TRANSFER_FUNDS",
                "time": "2025-05-30T11:09:33.419382372Z",
                "amount": "200000",
                "accountBalance": "200000.0",
            },
            {
                "type": "ORDER_FILL",
                "time": "2025-06-01T00:00:00.000000000Z",
                "instrument": "USD_JPY",
                "price": "150.0",
                "reason": "MARKET_ORDER",
                "tradeOpened": {
                    "tradeID": "1",
                    "price": "150.0",
                    "units": "1000",
                },
            },
            {
                "type": "ORDER_FILL",
                "time": "2025-06-01T00:10:00.000000000Z",
                "instrument": "USD_JPY",
                "price": "150.2",
                "reason": "MARKET_ORDER_TRADE_CLOSE",
                "accountBalance": "200500.0",
                "tradeReduced": {
                    "tradeID": "1",
                    "price": "150.2",
                    "units": "-500",
                    "realizedPL": "500",
                    "financing": "0",
                },
            },
            {
                "type": "TRANSFER_FUNDS",
                "time": "2025-06-02T00:00:00.000000000Z",
                "amount": "100000",
                "accountBalance": "300500.0",
            },
            {
                "type": "DAILY_FINANCING",
                "time": "2025-06-02T21:00:00.000000000Z",
                "financing": "10",
                "accountBalance": "300510.0",
            },
            {
                "type": "ORDER_FILL",
                "time": "2025-06-03T00:00:00.000000000Z",
                "instrument": "USD_JPY",
                "price": "150.8",
                "reason": "TAKE_PROFIT_ORDER",
                "accountBalance": "302010.0",
                "tradesClosed": [
                    {
                        "tradeID": "1",
                        "price": "150.8",
                        "units": "-500",
                        "realizedPL": "1500",
                        "financing": "0",
                    }
                ],
            },
        ]

        exits = reconstruct_trades(transactions)
        analysis = analyze(exits, transactions)

        self.assertEqual([row["exit_kind"] for row in exits], ["REDUCED", "CLOSED"])
        self.assertEqual(analysis["overall"]["trades"], 2)
        self.assertEqual(analysis["overall"]["net"], 2000.0)
        self.assertEqual(
            analysis["realized_pl_components"],
            {
                "daily_financing": {"count": 1, "net": 10.0},
                "tradeReduced": {"count": 1, "net": 500.0},
                "tradesClosed": {"count": 1, "net": 1500.0},
            },
        )
        cash = analysis["cash_flows"]
        self.assertEqual(cash["net_additional_transfers"], 100000.0)
        self.assertEqual(cash["transfer_adjusted_end_balance"], 202010.0)
        self.assertEqual(cash["transfer_adjusted_end_profit"], 2010.0)
        self.assertEqual(cash["transfer_adjusted_end_return_pct"], 1.0)
        self.assertEqual(cash["best_30d_funding_adjusted"]["profit"], 2010.0)
        self.assertEqual(cash["best_30d_funding_adjusted"]["return_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
