"""
signals.pocket_allocator
~~~~~~~~~~~~~~~~~~~~~~~~
GPT が返す weight_macro と、risk_guard が計算した
口座許容 lot を入力し、各 Pocket の lot を算出。
"""

from __future__ import annotations
from typing import Dict

DEFAULT_SCALP_SHARE = 0.3
MIN_SCALP_FRACTION = 0.05  # floor share of total lot reserved for scalping
MAX_SCALP_REALLOC_FRACTION = 0.35  # cap per-pocket reallocation toward the scalp bucket


def alloc(
    total_lot: float, weight_macro: float, scalp_share: float = DEFAULT_SCALP_SHARE
) -> Dict[str, float]:
    """
    Parameters
    ----------
    total_lot   : 口座全体で許容される lot
    weight_macro: 0‑1

    Returns
    -------
    {"micro": 0.02, "macro": 0.04, "scalp": 0.01}
    """
    macro = round(total_lot * weight_macro, 3)
    remainder = round(total_lot - macro, 3)
    remainder = max(remainder, 0.0)
    share = max(min(scalp_share, 0.9), 0.0)
    scalp = round(remainder * share, 3)
    micro = round(remainder - scalp, 3)
    dist = {"micro": micro, "macro": macro, "scalp": scalp}

    if total_lot <= 0:
        return dist

    min_scalp = round(min(total_lot * MIN_SCALP_FRACTION, total_lot), 3)
    if dist["scalp"] + 1e-6 < min_scalp:
        shortfall = round(min_scalp - dist["scalp"], 3)
        for donor in sorted(("micro", "macro"), key=lambda k: dist[k], reverse=True):
            if shortfall <= 0 or dist[donor] <= 0:
                continue
            give_cap = round(dist[donor] * MAX_SCALP_REALLOC_FRACTION, 3)
            give_cap = max(give_cap, 0.0)
            transfer = min(shortfall, give_cap if give_cap > 0 else dist[donor])
            if transfer <= 0:
                continue
            dist[donor] = round(dist[donor] - transfer, 3)
            dist["scalp"] = round(dist["scalp"] + transfer, 3)
            shortfall = round(shortfall - transfer, 3)
        if shortfall > 0:
            donor = "macro" if dist["macro"] >= dist["micro"] else "micro"
            transfer = min(shortfall, dist[donor])
            if transfer > 0:
                dist[donor] = round(dist[donor] - transfer, 3)
                dist["scalp"] = round(dist["scalp"] + transfer, 3)

    total_alloc = round(dist["micro"] + dist["macro"] + dist["scalp"], 3)
    target_total = round(total_lot, 3)
    if total_alloc != target_total:
        diff = round(target_total - total_alloc, 3)
        if diff > 0:
            dist["micro"] = round(dist["micro"] + diff, 3)
        elif diff < 0:
            deficit = -diff
            for key in ("micro", "macro"):
                available = max(dist[key], 0.0)
                if available <= 0:
                    continue
                take = min(deficit, available)
                dist[key] = round(dist[key] - take, 3)
                deficit = round(deficit - take, 3)
                if deficit <= 0:
                    break
            if deficit > 0 and dist["scalp"] > 0:
                take = min(deficit, dist["scalp"])
                dist["scalp"] = round(dist["scalp"] - take, 3)
    return dist


def dynamic_scalp_share(
    snapshot, base_share: float = DEFAULT_SCALP_SHARE, min_buffer: float = 0.1
) -> float:
    """
    口座の証拠金状況を考慮してスキャル pocket の配分を調整する。
    health_buffer: 1.0 - marginCloseoutPercent
    free_margin_ratio: NAV に対する利用可能証拠金の割合
    """
    try:
        buffer = snapshot.health_buffer
        free_ratio = snapshot.free_margin_ratio
    except AttributeError:
        return base_share

    if buffer <= 0.0:
        return 0.0

    if buffer < min_buffer:
        # 閾値を下回った場合はバッファに応じて線形に縮小
        scale_buffer = max(buffer / max(min_buffer, 1e-6), 0.0)
    else:
        scale_buffer = 1.0

    scale_free = min(max(free_ratio / 0.25, 0.0), 1.0)
    scale = min(scale_buffer, scale_free)

    if buffer < 0.03 or free_ratio < 0.05:
        return 0.0

    return round(base_share * scale, 3)
