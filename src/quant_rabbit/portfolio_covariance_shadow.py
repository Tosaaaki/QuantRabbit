"""Portfolio covariance risk model (completeness gap #2).

The per-currency notional guard treats a +0.4 AUD and +0.4 NZD book as
diversified, but in a high-vol risk-off shock those returns correlate to
~1 and true portfolio variance is a multiple of the per-currency cap —
precisely the high-vol cell where negative days concentrate.  This module
sizes and vetoes against a NAV volatility target using a real covariance
matrix: projected portfolio vol, each position's marginal risk
contribution, and the effective number of bets (a concentration measure).
Pure functions; no order authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

CONTRACT = "QR_PORTFOLIO_COVARIANCE_RISK_V1"


class CovarianceRiskError(ValueError):
    """Raised when covariance inputs are malformed."""


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _signed_weights(positions: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Net signed NAV-fraction weight per pair (LONG +, SHORT -)."""

    weights: dict[str, float] = {}
    for row in positions:
        pair = str(row.get("pair") or "").upper()
        if len(pair.split("_")) != 2:
            raise CovarianceRiskError(f"pair identity is invalid: {pair!r}")
        side = str(row.get("side") or "").upper()
        if side not in {"LONG", "SHORT"}:
            raise CovarianceRiskError("position side must be LONG or SHORT")
        frac = row.get("nav_exposure_fraction")
        if isinstance(frac, bool) or not isinstance(frac, (int, float)) or frac <= 0:
            raise CovarianceRiskError("nav_exposure_fraction must be positive")
        signed = float(frac) * (1.0 if side == "LONG" else -1.0)
        weights[pair] = weights.get(pair, 0.0) + signed
    return weights


def _validate_cov(
    pairs: Sequence[str], cov_by_pair: Mapping[str, Mapping[str, float]]
) -> None:
    for a in pairs:
        if a not in cov_by_pair:
            raise CovarianceRiskError(f"covariance row missing for {a}")
        diag = cov_by_pair[a].get(a)
        if not isinstance(diag, (int, float)) or diag < 0:
            raise CovarianceRiskError(f"variance for {a} is invalid")
        for b in pairs:
            ab = cov_by_pair.get(a, {}).get(b)
            ba = cov_by_pair.get(b, {}).get(a)
            if ab is None or ba is None:
                raise CovarianceRiskError(f"covariance missing for {a},{b}")
            if not math.isclose(float(ab), float(ba), abs_tol=1e-12):
                raise CovarianceRiskError(f"covariance not symmetric for {a},{b}")


def _cov_times_weights(
    pairs: Sequence[str],
    weights: Mapping[str, float],
    cov_by_pair: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    return {
        a: sum(float(cov_by_pair[a][b]) * weights[b] for b in pairs) for a in pairs
    }


def _jacobi_eigen(
    matrix: list[list[float]], *, max_sweeps: int = 100, tol: float = 1e-15
) -> tuple[list[float], list[list[float]]]:
    """Symmetric-matrix eigenvalues/vectors via cyclic Jacobi rotation.

    Pure Python, robust for the small (<=28) covariance matrices here.
    Returns (eigenvalues, eigenvectors) with eigenvectors as columns.
    """

    n = len(matrix)
    a = [[float(matrix[i][j]) for j in range(n)] for i in range(n)]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        if off <= tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= tol:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    eigenvalues = [a[i][i] for i in range(n)]
    return eigenvalues, v


def _effective_number_of_bets(
    pairs: Sequence[str],
    weights: Mapping[str, float],
    cov_by_pair: Mapping[str, Mapping[str, float]],
    portfolio_variance: float,
) -> float:
    """Meucci effective number of bets via principal-component entropy."""

    if portfolio_variance <= 0.0 or not pairs:
        return 0.0
    matrix = [[float(cov_by_pair[a][b]) for b in pairs] for a in pairs]
    eigenvalues, vectors = _jacobi_eigen(matrix)
    w = [weights[a] for a in pairs]
    contributions: list[float] = []
    for k in range(len(pairs)):
        projection = sum(vectors[i][k] * w[i] for i in range(len(pairs)))
        contributions.append(max(0.0, eigenvalues[k]) * projection * projection)
    total = sum(contributions)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in contributions:
        p = value / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return math.exp(entropy)


def portfolio_volatility(
    weights: Mapping[str, float], cov_by_pair: Mapping[str, Mapping[str, float]]
) -> float:
    pairs = sorted(weights)
    if not pairs:
        return 0.0
    _validate_cov(pairs, cov_by_pair)
    cw = _cov_times_weights(pairs, weights, cov_by_pair)
    variance = sum(weights[a] * cw[a] for a in pairs)
    return math.sqrt(max(0.0, variance))


def risk_decomposition(
    weights: Mapping[str, float], cov_by_pair: Mapping[str, Mapping[str, float]]
) -> dict[str, Any]:
    """Marginal risk contributions and the effective number of bets."""

    pairs = sorted(weights)
    vol = portfolio_volatility(weights, cov_by_pair)
    if vol <= 0.0:
        return {
            "portfolio_vol": 0.0,
            "marginal_contributions": {a: 0.0 for a in pairs},
            "effective_number_of_bets": 0.0,
        }
    cw = _cov_times_weights(pairs, weights, cov_by_pair)
    mcr = {a: weights[a] * cw[a] / vol for a in pairs}  # sums to vol
    enb = _effective_number_of_bets(pairs, weights, cov_by_pair, vol * vol)
    return {
        "portfolio_vol": round(vol, 12),
        "marginal_contributions": {a: round(mcr[a], 12) for a in pairs},
        "effective_number_of_bets": round(enb, 9),
    }


def evaluate_covariance_risk(
    current_positions: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    cov_by_pair: Mapping[str, Mapping[str, float]],
    *,
    nav_vol_target: float,
) -> dict[str, Any]:
    """Admit the candidate only if projected portfolio vol <= the target."""

    if not isinstance(nav_vol_target, (int, float)) or nav_vol_target <= 0:
        raise CovarianceRiskError("nav_vol_target must be positive")
    current_weights = _signed_weights(current_positions)
    projected_weights = _signed_weights([*current_positions, candidate])
    current = risk_decomposition(current_weights, cov_by_pair)
    projected = risk_decomposition(projected_weights, cov_by_pair)
    marginal_vol = round(projected["portfolio_vol"] - current["portfolio_vol"], 12)
    admitted = projected["portfolio_vol"] <= float(nav_vol_target) + 1e-12
    body: dict[str, Any] = {
        "contract": CONTRACT,
        "schema_version": 1,
        "nav_vol_target": float(nav_vol_target),
        "current_portfolio_vol": current["portfolio_vol"],
        "projected_portfolio_vol": projected["portfolio_vol"],
        "candidate_marginal_vol": marginal_vol,
        "projected_effective_number_of_bets": projected[
            "effective_number_of_bets"
        ],
        "admitted": admitted,
        "reason": (
            "WITHIN_NAV_VOL_TARGET"
            if admitted
            else "PROJECTED_PORTFOLIO_VOL_EXCEEDS_TARGET"
        ),
        "replaces_notional_netting": True,
        "order_authority": "NONE",
        "live_permission": False,
    }
    return {**body, "risk_sha256": _canonical_sha(body)}
