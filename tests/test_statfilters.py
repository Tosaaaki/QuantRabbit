"""Tests for ``quant_rabbit.analysis.statfilters``.

Covers each statistical filter against synthetic series with known
properties, then a smoke test of the end-to-end aggregator.
"""

from __future__ import annotations

import math
import random
import unittest

from quant_rabbit.analysis.statfilters import (
    StatFilterReading,
    abs_return_acf,
    bipower_variation,
    compute_stat_filters,
    detrended_fluctuation_hurst,
    lag1_autocorr,
    lee_mykland_test,
    rolling_moment,
    variance_ratio,
)


def _gaussian(mu: float, sigma: float, rng: random.Random) -> float:
    return rng.gauss(mu, sigma)


def _ar1(n: int, phi: float, sigma: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    xs = [0.0]
    for _ in range(n - 1):
        xs.append(phi * xs[-1] + _gaussian(0.0, sigma, rng))
    return xs


def _arch_returns(n: int, omega: float, alpha: float, seed: int) -> list[float]:
    """ARCH(1) with persistent volatility — should show |r| autocorrelation."""
    rng = random.Random(seed)
    out = [0.0]
    sigma2_prev = omega / max(1e-12, 1.0 - alpha)
    for _ in range(n - 1):
        sigma2 = omega + alpha * (out[-1] ** 2)
        sigma2_prev = sigma2
        out.append(_gaussian(0.0, math.sqrt(sigma2), rng))
    return out


def _random_walk_closes(n: int, sigma: float, seed: int, p0: float = 100.0) -> list[float]:
    rng = random.Random(seed)
    closes = [p0]
    for _ in range(n - 1):
        closes.append(closes[-1] + _gaussian(0.0, sigma, rng))
    return closes


def _trending_closes(n: int, drift: float, sigma: float, seed: int, p0: float = 100.0) -> list[float]:
    rng = random.Random(seed)
    closes = [p0]
    for _ in range(n - 1):
        closes.append(closes[-1] + drift + _gaussian(0.0, sigma, rng))
    return closes


class Lag1AutocorrTest(unittest.TestCase):
    def test_ar1_phi_07(self) -> None:
        # AR(1) with phi=0.7: theoretical lag-1 autocorr = 0.7
        rets = _ar1(2000, phi=0.7, sigma=1.0, seed=42)
        ac = lag1_autocorr(rets, window=1500)
        self.assertIsNotNone(ac)
        assert ac is not None
        self.assertAlmostEqual(ac, 0.7, delta=0.08)

    def test_underfilled_returns_none(self) -> None:
        self.assertIsNone(lag1_autocorr([0.1, 0.2, 0.3], window=100))


class AbsReturnAcfTest(unittest.TestCase):
    def test_arch_has_positive_lag1_acf(self) -> None:
        rets = _arch_returns(2000, omega=0.05, alpha=0.85, seed=7)
        acf = abs_return_acf(rets, max_lag=10)
        self.assertEqual(len(acf), 10)
        # Volatility clustering should produce positive ACF at lag 1.
        self.assertGreater(acf[0], 0.05)

    def test_iid_gaussian_acf_near_zero(self) -> None:
        rng = random.Random(1)
        rets = [rng.gauss(0.0, 1.0) for _ in range(2000)]
        acf = abs_return_acf(rets, max_lag=5)
        self.assertEqual(len(acf), 5)
        # IID: lag-1 ACF should be near zero (within noise).
        self.assertLess(abs(acf[0]), 0.1)


class RollingMomentTest(unittest.TestCase):
    def test_gaussian_kurtosis_near_zero(self) -> None:
        rng = random.Random(11)
        rets = [rng.gauss(0.0, 1.0) for _ in range(4000)]
        k = rolling_moment(rets, window=4000, kind="kurtosis")
        self.assertIsNotNone(k)
        assert k is not None
        # Excess kurtosis of Gaussian = 0; allow generous tolerance.
        self.assertLess(abs(k), 0.5)

    def test_heavy_tailed_kurtosis_positive(self) -> None:
        # Mixture: mostly small Gaussians + occasional huge spikes → heavy tail
        rng = random.Random(13)
        rets: list[float] = []
        for _ in range(4000):
            if rng.random() < 0.02:
                rets.append(rng.gauss(0.0, 8.0))
            else:
                rets.append(rng.gauss(0.0, 1.0))
        k = rolling_moment(rets, window=4000, kind="kurtosis")
        self.assertIsNotNone(k)
        assert k is not None
        self.assertGreater(k, 1.0)

    def test_invalid_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            rolling_moment([0.1, 0.2], window=200, kind="bogus")


class BipowerVariationTest(unittest.TestCase):
    def test_pure_diffusion_rv_and_bv_close(self) -> None:
        rng = random.Random(21)
        rets = [rng.gauss(0.0, 0.01) for _ in range(500)]
        rv, bv = bipower_variation(rets, window=200)
        self.assertGreater(rv, 0.0)
        self.assertGreater(bv, 0.0)
        ratio = rv / bv
        # In pure diffusion BV should approximate RV (ratio ~ 1).
        self.assertLess(abs(ratio - 1.0), 0.4)

    def test_jump_makes_rv_exceed_bv(self) -> None:
        rng = random.Random(22)
        rets = [rng.gauss(0.0, 0.01) for _ in range(500)]
        # Inject a big single-bar jump near the end
        rets[-5] = 0.5
        rv, bv = bipower_variation(rets, window=200)
        self.assertGreater(rv, bv)
        share = (rv - bv) / rv
        self.assertGreater(share, 0.5)


class LeeMyklandTest(unittest.TestCase):
    def test_pure_diffusion_no_jumps(self) -> None:
        rng = random.Random(31)
        rets = [rng.gauss(0.0, 0.01) for _ in range(400)]
        jumps = lee_mykland_test(rets, window=100, alpha=0.999)
        # With alpha=0.999 expect zero or very few false positives.
        self.assertLessEqual(len(jumps), 2)

    def test_single_spike_flagged(self) -> None:
        rng = random.Random(32)
        rets = [rng.gauss(0.0, 0.001) for _ in range(400)]
        rets[-3] = 0.2  # huge outlier (~200 sigma)
        jumps = lee_mykland_test(rets, window=100, alpha=0.999)
        self.assertIn(len(rets) - 3, jumps)


class HurstTest(unittest.TestCase):
    def test_random_walk_hurst_near_half(self) -> None:
        rng = random.Random(41)
        rw = [0.0]
        for _ in range(2000):
            rw.append(rw[-1] + rng.gauss(0.0, 1.0))
        h = detrended_fluctuation_hurst(rw)
        self.assertIsNotNone(h)
        assert h is not None
        # Random walk increments → integrated profile DFA ~ 1.5 in some
        # conventions, but DFA on the random walk values themselves
        # gives H ≈ 1.5; here we DFA the increments directly.
        # We instead test on the increments → H should be ~0.5.
        increments = [rw[i + 1] - rw[i] for i in range(len(rw) - 1)]
        h2 = detrended_fluctuation_hurst(increments)
        self.assertIsNotNone(h2)
        assert h2 is not None
        self.assertAlmostEqual(h2, 0.5, delta=0.15)

    def test_trending_series_hurst_above_half(self) -> None:
        rng = random.Random(42)
        # Persistent series: each step is correlated with previous step
        # via large autoregressive component → H > 0.55
        xs = [0.0]
        for _ in range(2000):
            xs.append(0.9 * xs[-1] + rng.gauss(0.0, 1.0))
        h = detrended_fluctuation_hurst(xs)
        self.assertIsNotNone(h)
        assert h is not None
        self.assertGreater(h, 0.55)

    def test_short_series_returns_none(self) -> None:
        self.assertIsNone(detrended_fluctuation_hurst([1.0, 2.0, 3.0]))


class VarianceRatioTest(unittest.TestCase):
    def test_random_walk_vr_near_one(self) -> None:
        rng = random.Random(51)
        rets = [rng.gauss(0.0, 1.0) for _ in range(4000)]
        vr2 = variance_ratio(rets, q=2)
        self.assertIsNotNone(vr2)
        assert vr2 is not None
        self.assertAlmostEqual(vr2, 1.0, delta=0.1)

    def test_trending_vr_above_one(self) -> None:
        # Strongly positively autocorrelated returns
        rng = random.Random(52)
        rets = [0.0]
        for _ in range(4000):
            rets.append(0.6 * rets[-1] + rng.gauss(0.0, 1.0))
        vr2 = variance_ratio(rets, q=2)
        self.assertIsNotNone(vr2)
        assert vr2 is not None
        self.assertGreater(vr2, 1.2)

    def test_q_below_2_returns_none(self) -> None:
        self.assertIsNone(variance_ratio([1.0, 2.0, 3.0, 4.0], q=1))


class ComputeStatFiltersTest(unittest.TestCase):
    def test_end_to_end_random_walk(self) -> None:
        closes = _random_walk_closes(800, sigma=0.05, seed=99)
        reading = compute_stat_filters(
            closes,
            autocorr_window=100,
            moment_window=200,
            jump_window=50,
            jump_alpha=0.999,
            atr_for_flat=0.07,
            dfa_window=400,
        )
        self.assertIsInstance(reading, StatFilterReading)
        # Random walk → autocorr near zero
        self.assertIsNotNone(reading.lag1_autocorr)
        assert reading.lag1_autocorr is not None
        self.assertLess(abs(reading.lag1_autocorr), 0.2)
        # No jumps expected at alpha=0.999 in pure diffusion
        self.assertLessEqual(len(reading.lee_mykland_jumps), 2)
        # Variance ratios computed
        self.assertIsNotNone(reading.variance_ratio_2)
        self.assertIsNotNone(reading.variance_ratio_4)
        # Hurst computed
        self.assertIsNotNone(reading.hurst_returns)
        # Bipower share is between 0 and 1
        self.assertIsNotNone(reading.bipower_jump_share)
        assert reading.bipower_jump_share is not None
        self.assertGreaterEqual(reading.bipower_jump_share, 0.0)
        self.assertLessEqual(reading.bipower_jump_share, 1.0)

    def test_end_to_end_with_injected_jump(self) -> None:
        closes = _random_walk_closes(800, sigma=0.05, seed=100)
        # Inject a price jump
        closes[-10] += 5.0
        reading = compute_stat_filters(
            closes,
            jump_window=50,
            atr_for_flat=0.07,
        )
        self.assertGreater(len(reading.lee_mykland_jumps), 0)
        self.assertIsNotNone(reading.last_jump_bars_ago)

    def test_flat_spot_count_with_zero_atr_falls_through(self) -> None:
        # No ATR → no silent literal → flat_spot_count = 0 per §3.5
        closes = _random_walk_closes(300, sigma=0.05, seed=200)
        reading = compute_stat_filters(closes, atr_for_flat=None)
        self.assertEqual(reading.flat_spot_count, 0)

    def test_flat_spot_count_detects_flat_bars(self) -> None:
        # A frozen quote series → many flat bars
        closes = [100.0] * 200
        reading = compute_stat_filters(
            closes,
            jump_window=50,
            atr_for_flat=1.0,
            flat_spot_epsilon_atr_ratio=0.05,
        )
        # All zero-diff bars should be counted
        self.assertEqual(reading.flat_spot_count, 50)

    def test_short_series_returns_nones(self) -> None:
        reading = compute_stat_filters([100.0, 100.1, 100.2], atr_for_flat=0.1)
        self.assertIsNone(reading.lag1_autocorr)
        self.assertIsNone(reading.rolling_kurtosis)
        self.assertIsNone(reading.hurst_returns)
        self.assertEqual(reading.lee_mykland_jumps, ())


if __name__ == "__main__":
    unittest.main()
