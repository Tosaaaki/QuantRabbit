from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from execution.strategy_entry import close_trade as _close_trade
from execution.pro_stop import plan_pro_stop_closes
from indicators.factor_cache import all_factors
from utils.metrics_logger import log_metric

try:  # reuse pocket mapping if available
    from execution.position_manager import _STRATEGY_POCKET_MAP as _POCKET_MAP  # type: ignore
except Exception:  # pragma: no cover - optional import
    _POCKET_MAP = {}

LOG = logging.getLogger(__name__)
_LOG_PREFIX = os.getenv('PRO_STOP_EXIT_LOG_PREFIX', '[PRO_STOP_EXIT]')
_FALLBACK_POCKET = os.getenv('PRO_STOP_FALLBACK_POCKET', 'micro').strip().lower() or 'micro'


def _extract_client_id(trade: dict) -> Optional[str]:
    client_id = trade.get('client_id') or trade.get('client_order_id')
    if client_id:
        return str(client_id)
    client_ext = trade.get('clientExtensions')
    if isinstance(client_ext, dict):
        cid = client_ext.get('id')
        if cid:
            return str(cid)
    return None


def _infer_pocket(trade: dict) -> Optional[str]:
    pocket = trade.get('pocket')
    if pocket:
        return str(pocket).strip().lower()
    tag = trade.get('strategy_tag') or trade.get('strategy')
    thesis = trade.get('entry_thesis')
    if not tag and isinstance(thesis, dict):
        tag = thesis.get('strategy_tag') or thesis.get('strategy') or thesis.get('tag')
    if tag:
        tag_text = str(tag)
        base = tag_text.split('-', 1)[0]
        for key in (tag_text, base, tag_text.lower(), base.lower()):
            pocket = _POCKET_MAP.get(key)
            if pocket:
                return str(pocket).strip().lower()
        lower = tag_text.lower()
        if 'fast' in lower and 'scalp' in lower:
            return 'scalp_fast'
        if 'macro' in lower or 'h1' in lower:
            return 'macro'
        if 'scalp' in lower or 's5' in lower or 'm1' in lower:
            return 'scalp'
    return None


async def maybe_close_pro_stop(
    trade: dict,
    *,
    now: Optional[datetime] = None,
    pocket: Optional[str] = None,
) -> bool:
    trade_id = trade.get('trade_id')
    if not trade_id:
        return False
    units = int(trade.get('units') or 0)
    if units == 0:
        return False
    client_id = _extract_client_id(trade)
    if not client_id:
        return False
    pocket_key = (pocket or _infer_pocket(trade) or _FALLBACK_POCKET).strip().lower()
    fac_m1 = all_factors().get('M1') or {}
    fac_h4 = all_factors().get('H4') or {}
    try:
        actions = plan_pro_stop_closes(
            {pocket_key: {'open_trades': [trade]}},
            fac_m1,
            fac_h4,
            now=now,
        )
    except Exception as exc:
        LOG.warning('%s plan failed trade=%s err=%s', _LOG_PREFIX, trade_id, exc)
        return False
    if not actions:
        return False
    action = actions[0]
    reason = action.get('reason') or 'hard_stop'
    ok = await _close_trade(
        str(trade_id),
        -units,
        client_order_id=client_id,
        allow_negative=True,
        exit_reason=reason,
    )
    log_metric('pro_stop_exit_close', 1.0 if ok else 0.0, tags={'reason': str(reason)})
    if ok:
        LOG.info('%s closed trade=%s reason=%s', _LOG_PREFIX, trade_id, reason)
    else:
        LOG.warning('%s close failed trade=%s reason=%s', _LOG_PREFIX, trade_id, reason)
    return ok
