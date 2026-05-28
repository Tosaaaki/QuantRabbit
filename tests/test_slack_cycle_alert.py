import unittest

from tools import slack_cycle_alert


class SlackCycleAlertTest(unittest.TestCase):
    def test_contaminated_pending_cleanup_stays_quiet(self) -> None:
        self.assertFalse(slack_cycle_alert._is_alert_status("CANCELED_CONTAMINATED_PENDING"))

    def test_blocked_status_still_alerts(self) -> None:
        self.assertTrue(slack_cycle_alert._is_alert_status("STALE_QUOTE_BLOCKED"))


if __name__ == "__main__":
    unittest.main()
