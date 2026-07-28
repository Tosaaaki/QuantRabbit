from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from .bitbank import BitbankAPIError, BitbankPublicClient, utc_from_ms
from .config import CryptoSafetyContract, ScannerConfig

BPS = Decimal("10000")


def _decimal(value: object, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(default)


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class PairAssessment:
    pair: str
    eligible: bool
    candidate: bool
    regime: str
    last: Decimal
    bid: Decimal
    ask: Decimal
    volume_24h_jpy: Decimal
    move_24h_bps: Decimal
    spread_bps: Decimal
    depth_25bps_jpy: Decimal
    estimated_slippage_bps: Decimal
    maker_fee_rate_quote: Decimal
    taker_fee_rate_quote: Decimal
    gross_edge_bps: Decimal
    expected_cost_bps: Decimal
    net_edge_bps: Decimal
    quality_score: Decimal
    freshness_ms: int
    exchange_status: str
    circuit_break_mode: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


class CryptoMarketScanner:
    def __init__(
        self,
        client: BitbankPublicClient,
        config: ScannerConfig | None = None,
        safety: CryptoSafetyContract | None = None,
    ) -> None:
        self.client = client
        self.config = config or client.config
        self.safety = safety or CryptoSafetyContract.from_env()
        self.safety.assert_safe()

    def scan(self, *, now: datetime | None = None) -> dict[str, Any]:
        observed_at = now or datetime.now(timezone.utc)
        settings = self.client.fetch_pair_settings()
        statuses = self.client.fetch_exchange_status()
        tickers = self.client.fetch_tickers_jpy()
        settings_by_pair = {
            str(row.get("name", "")).lower(): row for row in settings
        }
        status_by_pair = {
            str(row.get("pair", "")).lower(): row for row in statuses
        }
        ticker_by_pair = {
            str(row.get("pair", "")).lower(): row for row in tickers
        }
        discovered = sorted(
            pair
            for pair, spec in settings_by_pair.items()
            if str(spec.get("quote_asset", "")).lower() == "jpy"
        )
        prelim = sorted(
            discovered,
            key=lambda pair: self._volume_jpy(ticker_by_pair.get(pair, {})),
            reverse=True,
        )
        detailed = set(prelim[: self.config.detailed_pair_limit])
        assessments: list[PairAssessment] = []
        errors: list[dict[str, str]] = []
        for pair in discovered:
            depth: dict[str, Any] = {}
            circuit: dict[str, Any] = {"mode": "UNKNOWN"}
            if pair in detailed:
                try:
                    depth = self.client.fetch_depth(pair)
                    circuit = self.client.fetch_circuit_break_info(pair)
                except BitbankAPIError as exc:
                    errors.append({"pair": pair, "error": type(exc).__name__})
            assessments.append(
                self._assess(
                    pair,
                    settings_by_pair[pair],
                    status_by_pair.get(pair, {}),
                    ticker_by_pair.get(pair, {}),
                    depth,
                    circuit,
                    observed_at,
                )
            )
        ranked = sorted(
            assessments,
            key=lambda item: (item.candidate, item.net_edge_bps, item.quality_score),
            reverse=True,
        )
        candidates = [item for item in ranked if item.candidate][
            : self.config.shortlist_size
        ]
        guardian = self._guardian(assessments, errors)
        intents = [self._intent(item, observed_at) for item in candidates]
        stats = self.client.stats
        result = {
            "schema": "QR_CRYPTO_MARKET_SCAN_V1",
            "observed_at_utc": observed_at.isoformat(),
            "venue": "bitbank",
            "mode": "READ_ONLY_SHADOW_PAPER",
            "safety": self.safety.as_dict(),
            "guardian": guardian,
            "counts": {
                "discovered_jpy_pairs": len(discovered),
                "eligible_pairs": sum(item.eligible for item in assessments),
                "candidate_pairs": len(candidates),
                "rejected_pairs": sum(not item.eligible for item in assessments),
            },
            "request_stats": {
                "requests": stats.requests,
                "retries": stats.retries,
                "rate_limits": stats.rate_limits,
            },
            "candidates": [item.as_dict() for item in candidates],
            "pairs": [item.as_dict() for item in ranked],
            "virtual_intents": intents,
            "errors": errors,
        }
        return result

    def _assess(
        self,
        pair: str,
        spec: dict[str, Any],
        status: dict[str, Any],
        ticker: dict[str, Any],
        depth: dict[str, Any],
        circuit: dict[str, Any],
        now: datetime,
    ) -> PairAssessment:
        reasons: list[str] = []
        last = _decimal(ticker.get("last"))
        bid = _decimal(ticker.get("buy"))
        ask = _decimal(ticker.get("sell"))
        opened = _decimal(ticker.get("open"))
        exchange_status = str(status.get("status", "UNKNOWN")).upper()
        circuit_mode = str(circuit.get("mode", "UNKNOWN")).upper()
        freshness_ms = 2**31 - 1
        if ticker.get("timestamp"):
            freshness_ms = max(
                0, int((now - utc_from_ms(ticker["timestamp"])).total_seconds() * 1000)
            )
        spread_bps = ((ask - bid) / ((ask + bid) / 2) * BPS) if bid > 0 and ask > 0 else BPS
        move_bps = ((last - opened) / opened * BPS) if opened > 0 else Decimal("0")
        volume_jpy = self._volume_jpy(ticker)
        depth_jpy = self._depth_within(depth, bid, ask, Decimal("25"))
        slippage_bps = (
            self.config.target_notional_jpy / depth_jpy * Decimal("25")
            if depth_jpy > 0
            else Decimal("100")
        )
        maker_fee_rate = _decimal(spec.get("maker_fee_rate_quote"))
        taker_fee_rate = _decimal(spec.get("taker_fee_rate_quote"))
        maker_fee = maker_fee_rate * BPS
        gross_edge = abs(move_bps) * Decimal("0.03")
        expected_cost = (
            spread_bps
            + max(Decimal("0"), maker_fee)
            + slippage_bps
            + self.config.uncertainty_penalty_bps
        )
        net_edge = gross_edge - expected_cost
        enabled = bool(spec.get("is_enabled"))
        stopped = any(
            bool(spec.get(field))
            for field in (
                "stop_order",
                "stop_order_and_cancel",
                "stop_buy_order",
            )
        )
        if not enabled:
            reasons.append("PAIR_DISABLED")
        if stopped:
            reasons.append("BUY_ORDER_STOPPED")
        if exchange_status != "NORMAL":
            reasons.append(f"EXCHANGE_STATUS_{exchange_status}")
        if circuit_mode not in {"NONE"}:
            reasons.append(f"CIRCUIT_{circuit_mode}")
        if freshness_ms > self.config.max_ticker_age_sec * 1000:
            reasons.append("STALE_TICKER")
        if volume_jpy < self.config.min_volume_jpy:
            reasons.append("LOW_VOLUME")
        if spread_bps > self.config.max_spread_bps:
            reasons.append("WIDE_SPREAD")
        if depth and depth_jpy < self.config.min_depth_25bps_jpy:
            reasons.append("THIN_BOOK")
        if not depth:
            reasons.append("DEPTH_NOT_SAMPLED")
        if last <= 0 or bid <= 0 or ask <= 0:
            reasons.append("INVALID_PRICE")
        eligible = not any(
            reason
            for reason in reasons
            if reason
            not in {
                "DEPTH_NOT_SAMPLED",
            }
        )
        candidate = (
            eligible
            and bool(depth)
            and move_bps > 0
            and net_edge > self.config.required_safety_buffer_bps
        )
        if eligible and not candidate:
            if move_bps <= 0:
                reasons.append("SPOT_LONG_MOMENTUM_NOT_POSITIVE")
            if net_edge <= self.config.required_safety_buffer_bps:
                reasons.append("NET_EDGE_BELOW_SAFETY_BUFFER")
        quality = Decimal("100")
        quality -= min(Decimal("80"), spread_bps)
        quality -= min(Decimal("30"), slippage_bps)
        if volume_jpy > 0:
            quality += Decimal(str(max(0.0, math.log10(float(volume_jpy)) - 7))) * 5
        quality = max(Decimal("0"), min(Decimal("100"), quality))
        regime = "TREND_UP" if move_bps > 100 else "TREND_DOWN" if move_bps < -100 else "RANGE"
        if "THIN_BOOK" in reasons:
            regime = "LIQUIDITY_THIN"
        if "STALE_TICKER" in reasons:
            regime = "DATA_UNCERTAIN"
        return PairAssessment(
            pair=pair,
            eligible=eligible,
            candidate=candidate,
            regime=regime,
            last=last,
            bid=bid,
            ask=ask,
            volume_24h_jpy=volume_jpy,
            move_24h_bps=move_bps,
            spread_bps=spread_bps,
            depth_25bps_jpy=depth_jpy,
            estimated_slippage_bps=slippage_bps,
            maker_fee_rate_quote=maker_fee_rate,
            taker_fee_rate_quote=taker_fee_rate,
            gross_edge_bps=gross_edge,
            expected_cost_bps=expected_cost,
            net_edge_bps=net_edge,
            quality_score=quality,
            freshness_ms=freshness_ms,
            exchange_status=exchange_status,
            circuit_break_mode=circuit_mode,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _volume_jpy(ticker: dict[str, Any]) -> Decimal:
        return _decimal(ticker.get("vol")) * _decimal(ticker.get("last"))

    @staticmethod
    def _depth_within(
        depth: dict[str, Any], bid: Decimal, ask: Decimal, width_bps: Decimal
    ) -> Decimal:
        if bid <= 0 or ask <= 0:
            return Decimal("0")
        mid = (bid + ask) / 2
        lower = mid * (Decimal("1") - width_bps / BPS)
        upper = mid * (Decimal("1") + width_bps / BPS)
        notional = Decimal("0")
        for price, amount in depth.get("asks", []):
            price_d = _decimal(price)
            if price_d <= upper:
                notional += price_d * _decimal(amount)
        for price, amount in depth.get("bids", []):
            price_d = _decimal(price)
            if price_d >= lower:
                notional += price_d * _decimal(amount)
        return notional

    def _guardian(
        self, assessments: list[PairAssessment], errors: list[dict[str, str]]
    ) -> dict[str, Any]:
        issues: list[str] = []
        if errors:
            issues.append("PUBLIC_ENDPOINT_ERRORS")
        if self.client.stats.rate_limits:
            issues.append("RATE_LIMIT_OBSERVED")
        if not assessments or all(
            item.freshness_ms > self.config.max_ticker_age_sec * 1000
            for item in assessments
        ):
            issues.append("NO_FRESH_MARKET_DATA")
        state = (
            "HALT"
            if "NO_FRESH_MARKET_DATA" in issues
            else "RESTRICT"
            if issues
            else "GREEN"
        )
        return {
            "state": state,
            "issues": issues,
            "kill_switch": state == "HALT",
            "deterministic": True,
        }

    def _intent(self, item: PairAssessment, now: datetime) -> dict[str, Any]:
        digest = f"{now.isoformat()}|{item.pair}|BUY|{item.bid}|{item.net_edge_bps}"
        import hashlib

        intent_id = hashlib.sha256(digest.encode("utf-8")).hexdigest()[:24]
        amount = (
            self.config.target_notional_jpy / item.bid if item.bid > 0 else Decimal("0")
        )
        return _json_value(
            {
                "intent_id": intent_id,
                "pair": item.pair,
                "side": "BUY",
                "order_style": "PAPER_MAKER_LIMIT",
                "limit_price": item.bid,
                "amount": amount,
                "notional_jpy": self.config.target_notional_jpy,
                "expected_net_edge_bps": item.net_edge_bps,
                "required_safety_buffer_bps": self.config.required_safety_buffer_bps,
                "regime": item.regime,
                "authority": "NONE",
                "live_permission": False,
                "created_at_utc": now.isoformat(),
            }
        )
