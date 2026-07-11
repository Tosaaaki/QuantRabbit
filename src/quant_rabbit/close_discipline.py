from __future__ import annotations

import re


_BUFFERED_INVALIDATION_HIT_RE = re.compile(
    r"(?:^|;)\s*invalidation hit:\s*current\s+(?P<label>bid|ask)\s+"
    r"(?P<current>\d+(?:\.\d+)?)\s+(?P<comparator><=|>=)\s+"
    r"buffered invalidation\s+(?P<buffered>\d+(?:\.\d+)?)\s*"
    r"\(raw\s+(?P<raw>\d+(?:\.\d+)?),\s*"
    r"buffer\s+(?P<buffer_pips>\d+(?:\.\d+)?)p\)",
    re.IGNORECASE,
)


def thesis_evolution_reason_has_hard_close_evidence(
    reason: str,
    *,
    expected_side: str,
) -> bool:
    """Allow only the canonical, side-consistent hard-invalidation rationale.

    A free-form ``BROKEN`` label or a rationale that merely contains similar
    words must never become unattended loss-close authority.  The positive
    form below mirrors ``entry_thesis_ledger`` output and therefore also fails
    closed for negated, unbuffered, malformed, or wrong-side text.
    """

    side = str(expected_side or "").strip().upper()
    if side not in {"LONG", "SHORT"}:
        return False
    text = str(reason or "")
    match = _BUFFERED_INVALIDATION_HIT_RE.search(text)
    invalidation_hit = False
    if match is not None:
        label = match.group("label").lower()
        comparator = match.group("comparator")
        current = float(match.group("current"))
        buffered = float(match.group("buffered"))
        raw = float(match.group("raw"))
        buffer_pips = float(match.group("buffer_pips"))
        # Broker truth is executable-side specific: a LONG invalidates on bid
        # below its buffered floor, while a SHORT invalidates on ask above its
        # buffered ceiling. A positive anti-wick buffer and coherent prices are
        # required; if broker formatting changes, replace this prose contract
        # with structured fields from the thesis-evolution report.
        if side == "LONG":
            invalidation_hit = bool(
                label == "bid"
                and comparator == "<="
                and buffer_pips > 0.0
                and current <= buffered < raw
            )
        else:
            invalidation_hit = bool(
                label == "ask"
                and comparator == ">="
                and buffer_pips > 0.0
                and current >= buffered > raw
            )
    technical_confirmation = re.search(
        rf"(?:^|;)\s*technical invalidation confirmed against {side}:\s*\S",
        text,
        flags=re.IGNORECASE,
    ) is not None
    return invalidation_hit and technical_confirmation
