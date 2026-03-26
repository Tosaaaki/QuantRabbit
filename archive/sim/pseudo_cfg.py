"""
Configuration dataclasses for pseudo tick generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DensityCfg:
    """Target tick density (Poisson) per 5-second window."""

    tpm_5s_london: float = 8.0
    tpm_5s_ny: float = 7.0
    tpm_5s_asia: float = 3.0
    atr_k_high: float = 1.2
    atr_k_low: float = 0.8
    target_tickrate_min: int = 6
    target_window_sec: int = 10
    target_coverage: float = 0.7
    tickrate_checks: Tuple[Tuple[int, int], ...] = ((5, 6), (10, 10))


@dataclass
class ShapeCfg:
    """Controls how consolidation/impulse blocks are inserted."""

    path_variant_p: float = 0.5
    stall_prob: float = 0.35
    stall_range_pips: float = 0.3
    stall_ticks: int = 4
    impulse_prob: float = 0.35
    impulse_atr_k: float = 1.3
    impulse_ticks: int = 3
    noise_pips_sigma: float = 0.08


@dataclass
class SpreadCfg:
    """Simple spread model by session."""

    mean_pips_london: float = 0.4
    mean_pips_ny: float = 0.45
    mean_pips_asia: float = 0.6
    std_pips: float = 0.1
    night_multiplier: float = 1.3


@dataclass
class SimCfg:
    density: DensityCfg = field(default_factory=DensityCfg)
    shape: ShapeCfg = field(default_factory=ShapeCfg)
    spread: SpreadCfg = field(default_factory=SpreadCfg)
    random_seed: int = 42
