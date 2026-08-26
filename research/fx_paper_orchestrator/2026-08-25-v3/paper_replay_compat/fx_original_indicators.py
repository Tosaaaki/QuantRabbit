"""Minimal, paper-only compatibility surface for the frozen V25 replay.

The original helper lived in an untracked sibling research directory and was
not part of the handoff.  V25 imports only ``Bar``, ``load_bars``,
``pip_size`` and ``sha256_file``.  Keeping this bounded copy inside the owned
v3 directory makes the official replay restartable without changing V25.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    pair: str
    time: str
    bid_o: float
    bid_h: float
    bid_l: float
    bid_c: float
    ask_o: float
    ask_h: float
    ask_l: float
    ask_c: float
    volume: float = 0.0

    @property
    def mid_o(self) -> float:
        return (self.bid_o + self.ask_o) / 2.0

    @property
    def mid_h(self) -> float:
        return (self.bid_h + self.ask_h) / 2.0

    @property
    def mid_l(self) -> float:
        return (self.bid_l + self.ask_l) / 2.0

    @property
    def mid_c(self) -> float:
        return (self.bid_c + self.ask_c) / 2.0

    @property
    def spread_c(self) -> float:
        return self.ask_c - self.bid_c


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bars(path: Path) -> list[Bar]:
    opener = gzip.open if path.suffix == ".gz" else open
    result: list[Bar] = []
    previous = ""
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = json.loads(line)
            if raw.get("complete") is not True or raw.get("price") != "BA":
                raise ValueError(f"non-completed or non-BA row at {path}:{line_no}")
            stamp = str(raw["time"])
            if stamp <= previous:
                raise ValueError(f"non-increasing timestamp at {path}:{line_no}")
            previous = stamp
            bid, ask = raw["bid"], raw["ask"]
            values = [float(bid[key]) for key in "ohlc"] + [float(ask[key]) for key in "ohlc"]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError(f"invalid price at {path}:{line_no}")
            if any(float(ask[key]) < float(bid[key]) for key in "ohlc"):
                raise ValueError(f"crossed BID/ASK at {path}:{line_no}")
            result.append(Bar(
                pair=str(raw["pair"]),
                time=stamp,
                bid_o=float(bid["o"]), bid_h=float(bid["h"]),
                bid_l=float(bid["l"]), bid_c=float(bid["c"]),
                ask_o=float(ask["o"]), ask_h=float(ask["h"]),
                ask_l=float(ask["l"]), ask_c=float(ask["c"]),
                volume=float(raw.get("volume", 0.0)),
            ))
    if len(result) < 100:
        raise ValueError(f"dataset unexpectedly short: {path}")
    return result


def pip_size(pair: str) -> float:
    return 0.01 if pair.endswith("_JPY") else 0.0001


def generate_events(*_args: object, **_kwargs: object) -> list[dict]:
    """Reject use outside the frozen V25 import surface.

    ``run_auction_trap_geometry_v7`` imports this name while V25 uses only its
    immutable cost-arm constants.  Silently implementing the older event
    generator here would widen the migration authority.
    """
    raise RuntimeError("legacy generate_events is outside the V25 compatibility contract")
