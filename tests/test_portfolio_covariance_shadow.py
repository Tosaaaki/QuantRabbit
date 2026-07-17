from __future__ import annotations

import math

import pytest

from quant_rabbit.portfolio_covariance_shadow import (
    CovarianceRiskError,
    evaluate_covariance_risk,
    portfolio_volatility,
    risk_decomposition,
)

# Two assets, each daily vol 0.02.
SIG = 0.02


def _cov(rho: float) -> dict:
    off = rho * SIG * SIG
    var = SIG * SIG
    return {
        "AUD_USD": {"AUD_USD": var, "NZD_USD": off},
        "NZD_USD": {"NZD_USD": var, "AUD_USD": off},
    }


def test_correlation_drives_portfolio_vol_and_diversification() -> None:
    weights = {"AUD_USD": 0.5, "NZD_USD": 0.5}

    # Perfectly correlated: vol adds linearly, one effective bet.
    corr1 = risk_decomposition(weights, _cov(1.0))
    assert corr1["portfolio_vol"] == pytest.approx(0.5 * SIG + 0.5 * SIG)
    assert corr1["effective_number_of_bets"] == pytest.approx(1.0, abs=1e-6)

    # Uncorrelated: vol is sqrt-reduced, two effective bets.
    corr0 = risk_decomposition(weights, _cov(0.0))
    assert corr0["portfolio_vol"] == pytest.approx(math.sqrt(2) * 0.5 * SIG)
    assert corr0["effective_number_of_bets"] == pytest.approx(2.0, abs=1e-6)


def test_hedge_cancels_risk_when_anticorrelated_long_short() -> None:
    # LONG AUD, SHORT NZD with rho=+1 -> the position is a hedge, vol ~ 0.
    weights = {"AUD_USD": 0.5, "NZD_USD": -0.5}
    assert portfolio_volatility(weights, _cov(1.0)) == pytest.approx(0.0, abs=1e-9)


def test_covariance_guard_catches_correlated_stack_notional_misses() -> None:
    # Two same-direction correlated positions: per-currency netting would see
    # 0.5 USD each side; the covariance guard sees the real stacked vol.
    current = [{"pair": "AUD_USD", "side": "LONG", "nav_exposure_fraction": 0.5}]
    candidate = {"pair": "NZD_USD", "side": "LONG", "nav_exposure_fraction": 0.5}

    # Target set between the diversified (~0.0141) and correlated (~0.0197) vols.
    target = 0.015
    correlated = evaluate_covariance_risk(current, candidate, _cov(0.95), nav_vol_target=target)
    diversified = evaluate_covariance_risk(current, candidate, _cov(0.0), nav_vol_target=target)

    assert correlated["admitted"] is False
    assert correlated["reason"] == "PROJECTED_PORTFOLIO_VOL_EXCEEDS_TARGET"
    assert diversified["admitted"] is True
    assert correlated["projected_portfolio_vol"] > diversified["projected_portfolio_vol"]
    assert correlated["candidate_marginal_vol"] > 0.0


def test_eigensolver_and_enb_hold_on_multi_asset_edge_cases() -> None:
    from quant_rabbit.portfolio_covariance_shadow import _jacobi_eigen, risk_decomposition

    # Known spectra and structural invariants.
    ev, _ = _jacobi_eigen([[2.0, 1.0], [1.0, 2.0]])
    assert sorted(round(x, 9) for x in ev) == [1.0, 3.0]
    A = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]]
    ev, vecs = _jacobi_eigen(A)
    assert round(sum(ev), 9) == round(4.0 + 3.0 + 2.0, 9)  # trace preserved
    n = 3
    assert all(
        abs(sum(vecs[k][i] * vecs[k][j] for k in range(n)) - (1.0 if i == j else 0.0)) < 1e-9
        for i in range(n)
        for j in range(n)
    )

    # ENB bounds: identity => N bets; rank-1 (all perfectly correlated) => 1.
    w = {f"P{i}_USD": 0.33 for i in range(3)}
    identity = {f"P{i}_USD": {f"P{j}_USD": (0.0004 if i == j else 0.0) for j in range(3)} for i in range(3)}
    rank1 = {f"P{i}_USD": {f"P{j}_USD": 0.0004 for j in range(3)} for i in range(3)}
    assert risk_decomposition(w, identity)["effective_number_of_bets"] == pytest.approx(3.0, abs=1e-6)
    assert risk_decomposition(w, rank1)["effective_number_of_bets"] == pytest.approx(1.0, abs=1e-6)


def test_malformed_covariance_fails_closed() -> None:
    weights = {"AUD_USD": 0.5, "NZD_USD": 0.5}
    asymmetric = {
        "AUD_USD": {"AUD_USD": 0.0004, "NZD_USD": 0.0003},
        "NZD_USD": {"NZD_USD": 0.0004, "AUD_USD": 0.0001},
    }
    with pytest.raises(CovarianceRiskError, match="symmetric"):
        portfolio_volatility(weights, asymmetric)
    with pytest.raises(CovarianceRiskError, match="positive"):
        evaluate_covariance_risk(
            [], {"pair": "AUD_USD", "side": "LONG", "nav_exposure_fraction": 0.0},
            _cov(0.0), nav_vol_target=0.02,
        )
