import os
import unittest
from unittest import mock

from tools import slack_post


class SlackPostTest(unittest.TestCase):
    def test_post_message_disabled_by_default_skips_network(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("tools.slack_post.urllib.request.urlopen") as urlopen:
                resp = slack_post.post_message("hello", "C123", "xoxb-test")

        self.assertEqual(resp["ok"], True)
        self.assertEqual(resp["skipped"], True)
        self.assertEqual(resp["reason"], "slack_disabled")
        urlopen.assert_not_called()

    def test_post_message_requires_explicit_opt_in_for_network(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"ok": true, "ts": "123.456"}'

        with mock.patch.dict(os.environ, {"QR_SLACK_SEND_ENABLE": "1"}, clear=True):
            with mock.patch("tools.slack_post.urllib.request.urlopen", return_value=FakeResponse()) as urlopen:
                resp = slack_post.post_message("hello", "C123", "xoxb-test")

        self.assertEqual(resp["ok"], True)
        self.assertEqual(resp["ts"], "123.456")
        urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
