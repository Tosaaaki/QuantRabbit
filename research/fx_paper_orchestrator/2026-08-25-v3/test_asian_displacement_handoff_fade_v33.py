import inspect
import unittest

import run_asian_displacement_handoff_fade_v32 as frozen_v32
from test_asian_displacement_handoff_fade_v32 import fixture
from run_asian_displacement_handoff_fade_v33 import (
    canonical_utc_nine_digits,
    detect_day_signals,
)


class AsianDisplacementHandoffFadeV33Test(unittest.TestCase):
    def test_seconds_and_fractional_inputs_canonicalize_to_nine_digits(self):
        self.assertEqual(
            canonical_utc_nine_digits("2026-03-11T06:00:00Z"),
            "2026-03-11T06:00:00.000000000Z",
        )
        self.assertEqual(
            canonical_utc_nine_digits("2026-03-11T06:00:00.1234Z"),
            "2026-03-11T06:00:00.123400000Z",
        )

    def test_canonicalization_preserves_integer_epoch_nanoseconds(self):
        samples = [
            "2026-03-11T06:00:00Z",
            "2026-03-11T06:00:00.123456789Z",
            "2026-03-11T06:00:00.000000001Z",
        ]
        for source in samples:
            self.assertEqual(
                frozen_v32.frozen_v31.ns(source),
                frozen_v32.frozen_v31.ns(canonical_utc_nine_digits(source)),
            )

    def test_v33_preserves_v32_signal_ids_directions_and_instants(self):
        signs = {pair: 1 for pair in frozen_v32.UNIVERSE}
        source = fixture(signs)
        v32_rows = frozen_v32.detect_day_signals(source)
        v33_rows = detect_day_signals(source)
        identity = ("signal_id", "pair", "utc_day", "direction")
        self.assertEqual(
            [[row[field] for field in identity] for row in v32_rows],
            [[row[field] for field in identity] for row in v33_rows],
        )
        for old, new in zip(v32_rows, v33_rows):
            for field in ("decision_time", "fill_time", "exit_time"):
                self.assertEqual(frozen_v32.frozen_v31.ns(old[field]), frozen_v32.frozen_v31.ns(new[field]))
                self.assertRegex(new[field], r"\.\d{9}Z$")

    def test_signal_detector_has_no_cost_or_outcome_parameter(self):
        self.assertEqual(set(inspect.signature(detect_day_signals).parameters), {"pair_day_bars"})


if __name__ == "__main__":
    unittest.main()
