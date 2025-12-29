from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from utils.cost_guard import add_tokens
from utils.gpt_monitor import track_gpt_call
from utils.secrets import get_secret


def _bool_flag(name: str, default: bool) -> bool:
    env_val = os.environ.get(name.upper())
    if env_val is not None:
        return env_val.strip().lower() not in {"0", "false", "no"}
    try:
        return str(get_secret(name.lower())).strip().lower() not in {"0", "false", "no"}
    except KeyError:
        return default


def _int_secret(key: str, default: int) -> int:
    try:
        return int(get_secret(key))
    except Exception:
        return default


@dataclass(frozen=True)
class ExitHint:
    max_drawdown_pips: Optional[float]
    min_takeprofit_pips: Optional[float]
    confidence: float
    reason: Optional[str] = None
    model_used: Optional[str] = None


class ExitAdvisor:
    """Provide per-pocket/direction exit hints via GPT."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        cache_ttl_seconds: int = 300,
        max_drawdown_bounds: tuple[float, float] = (1.0, 8.0),
        min_tp_bounds: tuple[float, float] = (0.5, 6.0),
    ) -> None:
        self.enabled = _bool_flag("EXIT_ADVISOR_ENABLED", True) if enabled is None else enabled
        self._ttl = cache_ttl_seconds
        self._max_draw_bounds = max_drawdown_bounds
        self._min_tp_bounds = min_tp_bounds
        self._cache: Dict[str, tuple[str, ExitHint, dt.datetime]] = {}
        self._model = (
            os.environ.get("OPENAI_MODEL_EXIT_ADVISOR")
            or os.environ.get("OPENAI_MODEL")
        )
        if not self._model:
            try:
                self._model = get_secret("openai_model_exit_advisor")
            except KeyError:
                try:
                    self._model = get_secret("openai_model")
                except KeyError:
                    self._model = "gpt-4o-mini"
        try:
            api_key = get_secret("openai_api_key")
        except KeyError:
            api_key = None
        if not api_key:
            logging.warning("[EXIT_ADVISOR] OPENAI_API_KEY missing; advisor disabled.")
            self.enabled = False
            self._client = None  # type: ignore[assignment]
        else:
            self._client = AsyncOpenAI(api_key=api_key)
        self._max_month_tokens = _int_secret("openai_max_month_tokens", 300000)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def build_hints(
        self,
        open_positions: Dict[str, Dict[str, Any]],
        *,
        fac_m1: Dict[str, Any],
        fac_h4: Dict[str, Any],
        fac_h1: Optional[Dict[str, Any]] = None,
        fac_d1: Optional[Dict[str, Any]] = None,
        range_active: bool = False,
        now: Optional[dt.datetime] = None,
    ) -> Dict[str, ExitHint]:
        if not self.enabled or not self._client:
            return {}
        now = now or dt.datetime.utcnow()
        hints: Dict[str, ExitHint] = {}
        base_price = float(fac_m1.get("close") or 0.0)
        for pocket, info in open_positions.items():
            if pocket == "__net__" or pocket not in {"macro", "micro"}:
                continue
            for direction, key_units in (("long", "long_units"), ("short", "short_units")):
                units = int(info.get(key_units, 0) or 0)
                if units <= 0:
                    continue
                avg_price = info.get(f"{direction}_avg_price") or info.get("avg_price")
                if not avg_price or not base_price:
                    continue
                profit_pips = (
                    (base_price - avg_price) / 0.01 if direction == "long"
                    else (avg_price - base_price) / 0.01
                )
                if profit_pips < -6.5 or profit_pips > 8.0:
                    continue
                key = f"{pocket}:{direction}"
                context = self._build_context(
                    pocket,
                    direction,
                    profit_pips,
                    units,
                    avg_price,
                    base_price,
                    fac_m1,
                    fac_h4,
                    fac_h1,
                    fac_d1,
                    range_active,
                )
                hint = await self._advise_once(key, context, now=now)
                if hint:
                    hints[key] = hint
        return hints

    async def _advise_once(
        self,
        key: str,
        context: Dict[str, Any],
        *,
        now: dt.datetime,
    ) -> Optional[ExitHint]:
        signature = self._hash_context(context)
        cached = self._cache.get(key)
        if cached:
            cached_sig, hint, ts = cached
            if cached_sig == signature and (now - ts).total_seconds() <= self._ttl:
                return hint

        if not add_tokens(0, self._max_month_tokens):
            logging.warning("[EXIT_ADVISOR] token budget exceeded; skip.")
            return None

        payload = dict(context)
        try:
            data = await self._call_model(context)
        except Exception as exc:
            logging.warning("[EXIT_ADVISOR] GPT call failed: %s", exc)
            return None

        hint = self._parse_hint(data)
        if hint:
            self._cache[key] = (signature, hint, now)
        return hint

    def _parse_hint(self, data: Dict[str, Any]) -> Optional[ExitHint]:
        if not data:
            return None
        try:
            dd = data.get("max_drawdown_pips")
            tp = data.get("min_takeprofit_pips")
            confidence = float(data.get("confidence") or 0.5)
        except (TypeError, ValueError):
            return None
        dd_val = None
        if dd is not None:
            try:
                dd_val = max(self._max_draw_bounds[0], min(self._max_draw_bounds[1], float(dd)))
            except (TypeError, ValueError):
                dd_val = None
        tp_val = None
        if tp is not None:
            try:
                tp_val = max(self._min_tp_bounds[0], min(self._min_tp_bounds[1], float(tp)))
            except (TypeError, ValueError):
                tp_val = None
        return ExitHint(
            max_drawdown_pips=dd_val,
            min_takeprofit_pips=tp_val,
            confidence=max(0.0, min(1.0, confidence)),
            reason=data.get("reason"),
            model_used=data.get("model_used"),
        )

    def _build_context(
        self,
        pocket: str,
        direction: str,
        profit_pips: float,
        units: int,
        avg_price: float,
        close_price: float,
        fac_m1: Dict[str, Any],
        fac_h4: Dict[str, Any],
        fac_h1: Optional[Dict[str, Any]],
        fac_d1: Optional[Dict[str, Any]],
        range_active: bool,
    ) -> Dict[str, Any]:
        context = {
            "pocket": pocket,
            "direction": direction,
            "profit_pips": round(profit_pips, 3),
            "units": units,
            "avg_price": round(avg_price, 5),
            "close_price": round(close_price, 5),
            "range_active": range_active,
            "factors_m1": self._summarize_factors(fac_m1),
            "factors_h4": self._summarize_factors(fac_h4),
        }
        if fac_h1:
            context["factors_h1"] = self._summarize_factors(fac_h1)
        if fac_d1:
            context["factors_d1"] = self._summarize_factors(fac_d1)
        return context

    def _summarize_factors(self, factors: Dict[str, Any]) -> Dict[str, float]:
        keys = ("adx", "rsi", "ma10", "ma20", "atr_pips", "close")
        summary: Dict[str, float] = {}
        for key in keys:
            val = factors.get(key)
            if val is None:
                continue
            try:
                summary[key] = round(float(val), 6)
            except (TypeError, ValueError):
                continue
        return summary

    def _hash_context(self, context: Dict[str, Any]) -> str:
        serialized = json.dumps(context, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    async def _call_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an FX trade risk monitor. "
                    "Given pocket, direction, profit/loss and multi-timeframe indicators, "
                    "recommend how much adverse move to tolerate (max_drawdown_pips) "
                    "and what profit to secure (min_takeprofit_pips). "
                    "Respond only with JSON containing keys "
                    "\"max_drawdown_pips\", \"min_takeprofit_pips\", \"confidence\", \"reason\"."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload),
            },
        ]
        with track_gpt_call("exit_advisor") as tracker:
            resp = await self._client.responses.create(  # type: ignore[union-attr]
                model=self._model,
                input=messages,
                max_output_tokens=240,
                temperature=0.2,
            )
            usage = getattr(resp, "usage", None)
            used_tokens = 0
            if usage is not None:
                used_tokens = getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
            if used_tokens:
                tracker.add_tag("tokens", used_tokens)
                add_tokens(used_tokens, self._max_month_tokens)
            content_parts: list[str] = []
            for item in resp.output or []:
                if getattr(item, "type", "") != "message":
                    continue
                for block in getattr(item, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        content_parts.append(text)
            content = "".join(content_parts).strip()
            if content.startswith("```"):
                content = content.strip("`")
                content = content.replace("json", "", 1).strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                logging.warning("[EXIT_ADVISOR] invalid JSON response: %s", content)
                tracker.mark_status("invalid_json")
                data = {}
            model_used = getattr(resp, "model", self._model)
            tracker.set_model(model_used)
        data["model_used"] = model_used
        return data
