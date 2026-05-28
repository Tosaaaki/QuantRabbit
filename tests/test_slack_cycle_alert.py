import unittest

from tools import slack_cycle_alert


class SlackCycleAlertTest(unittest.TestCase):
    def test_contaminated_pending_cleanup_stays_quiet(self) -> None:
        self.assertFalse(slack_cycle_alert._is_alert_status("CANCELED_CONTAMINATED_PENDING"))

    def test_gpt_non_trade_outcomes_stay_quiet(self) -> None:
        self.assertFalse(slack_cycle_alert._is_alert_status("GPT_REJECTED"))
        self.assertFalse(slack_cycle_alert._is_alert_status("GPT_REQUEST_EVIDENCE"))
        self.assertFalse(slack_cycle_alert._is_alert_status("GPT_PROTECT"))
        self.assertFalse(slack_cycle_alert._is_alert_status("GPT_TIGHTEN_SL"))

    def test_routine_exit_zero_cycle_statuses_stay_quiet(self) -> None:
        for status in (
            "MONITOR_ONLY_EXPOSURE_OPEN",
            "POSITION_ACTION_STAGED",
            "NO_LIVE_READY_INTENT",
            "NO_TRADE",
            "TARGET_REACHED_PROTECT",
            "CANCELED_TARGET_REACHED_PENDING",
        ):
            with self.subTest(status=status):
                self.assertFalse(slack_cycle_alert._is_alert_status(status))

    def test_blocked_status_still_alerts(self) -> None:
        self.assertTrue(slack_cycle_alert._is_alert_status("STALE_QUOTE_BLOCKED"))
        self.assertTrue(slack_cycle_alert._is_alert_status("POSITION_ACTION_BLOCKED"))

    def test_unknown_reject_status_still_alerts(self) -> None:
        self.assertTrue(slack_cycle_alert._is_alert_status("BROKER_REJECTED"))


if __name__ == "__main__":
    unittest.main()
