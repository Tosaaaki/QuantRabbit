from __future__ import annotations

import unittest

from quant_rabbit.analysis.flow import build_flow_snapshot


class _AuthBlockedBookClient:
    account_id = "account"

    def __init__(self) -> None:
        self.paths: list[str] = []

    def get_json(self, path: str, params: dict | None = None) -> dict:
        self.paths.append(path)
        if path.endswith("/orderBook") or path.endswith("/positionBook"):
            raise RuntimeError("HTTP Error 401: Unauthorized")
        if path.endswith("/pricing"):
            return {
                "prices": [
                    {
                        "instrument": (params or {}).get("instruments", "EUR_USD"),
                        "bids": [{"price": "1.1000"}],
                        "asks": [{"price": "1.1002"}],
                    }
                ]
            }
        if path.endswith("/candles"):
            return {
                "candles": [
                    {"bid": {"c": "1.1000"}, "ask": {"c": "1.1002"}},
                    {"bid": {"c": "1.1001"}, "ask": {"c": "1.1003"}},
                ]
            }
        raise AssertionError(path)


class FlowSnapshotTest(unittest.TestCase):
    def test_default_skips_unentitled_book_endpoints(self) -> None:
        client = _AuthBlockedBookClient()

        snapshot = build_flow_snapshot(
            client=client,
            pairs=("EUR_USD", "GBP_USD"),
            spread_lookback_minutes=2,
        )

        self.assertFalse(snapshot.book_fetch_enabled)
        self.assertEqual(snapshot.book_fetch_reason, "disabled_by_default_enable_with_include_books")
        self.assertEqual(snapshot.issues, tuple())
        self.assertEqual(snapshot.order_books, tuple())
        self.assertEqual(snapshot.position_books, tuple())
        self.assertEqual(len(snapshot.spreads), 2)
        self.assertFalse(any(path.endswith("/orderBook") for path in client.paths))
        self.assertFalse(any(path.endswith("/positionBook") for path in client.paths))

    def test_book_authorization_failure_is_reported_once_per_feed(self) -> None:
        snapshot = build_flow_snapshot(
            client=_AuthBlockedBookClient(),
            pairs=("EUR_USD", "GBP_USD"),
            spread_lookback_minutes=2,
            include_books=True,
        )

        self.assertTrue(snapshot.book_fetch_enabled)
        self.assertEqual(len(snapshot.issues), 2)
        self.assertTrue(snapshot.issues[0].startswith("ORDERBOOK_FEED_UNAUTHORIZED"))
        self.assertTrue(snapshot.issues[1].startswith("POSITIONBOOK_FEED_UNAUTHORIZED"))
        self.assertIn("account/feed entitlement", snapshot.issues[0])
        self.assertNotIn("token scope", snapshot.issues[0].lower())
        self.assertEqual(len(snapshot.order_books), 2)
        self.assertEqual(len(snapshot.position_books), 2)
        self.assertTrue(all(book.issue == snapshot.issues[0] for book in snapshot.order_books))
        self.assertTrue(all(book.issue == snapshot.issues[1] for book in snapshot.position_books))
        self.assertEqual(len(snapshot.spreads), 2)


if __name__ == "__main__":
    unittest.main()
