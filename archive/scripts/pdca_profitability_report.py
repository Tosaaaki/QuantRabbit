#!/usr/bin/env python3
"""Local V2 profitability PDCA report (report-only).

What it does (local-only):
- Fetch OANDA account summary + pricing + openTrades snapshot (default USD_JPY)
- Aggregate logs/trades.db (24h / 7d): PF / win_rate / net (JPY + pips)
- Rank by pocket / strategy_tag / (pocket,strategy_tag)
- Aggregate logs/orders.db reject/error signals (status + error_code)

What it writes:
- logs/pdca_profitability_latest.json
- logs/pdca_profitability_history.jsonl (append)
- logs/pdca_profitability_latest.md
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    os.chdir(PROJECT_ROOT)
except Exception:
    pass

from utils.secrets import get_secret

UTC = timezone.utc
JST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class OandaCreds:
    token: str
    account_id: str
    practice: bool
    base_url: str


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def _iso_jst(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(JST).isoformat()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: object) -> str:
    return str(value or "").strip()


def _mask_account_id(account_id: str) -> str:
    acc = (account_id or "").strip()
    if len(acc) <= 6:
        return "****"
    return f"{acc[:2]}****{acc[-4:]}"


def _git_rev() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _load_oanda_creds() -> OandaCreds:
    token = get_secret("oanda_token")
    account_id = get_secret("oanda_account_id")
    try:
        practice = get_secret("oanda_practice").strip().lower() == "true"
    except Exception:
        practice = False
    base_url = (
        "https://api-fxpractice.oanda.com"
        if practice
        else "https://api-fxtrade.oanda.com"
    )
    return OandaCreds(
        token=token, account_id=account_id, practice=practice, base_url=base_url
    )


def _oanda_get_json(
    url: str,
    *,
    headers: dict[str, str],
    params: Optional[dict[str, str]] = None,
    timeout_sec: float = 7.0,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    started = time.time()
    resp: Optional[requests.Response] = None
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout_sec)
        latency_ms = (time.time() - started) * 1000.0
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise ValueError("unexpected_json_shape")
        return payload, {
            "ok": True,
            "status": int(resp.status_code),
            "latency_ms": round(latency_ms, 1),
        }
    except Exception as exc:
        latency_ms = (time.time() - started) * 1000.0
        meta: dict[str, Any] = {
            "ok": False,
            "latency_ms": round(latency_ms, 1),
            "error": str(exc),
        }
        try:
            if resp is not None:
                meta["status"] = int(resp.status_code)
                meta["body"] = (resp.text or "")[:500]
        except Exception:
            pass
        return None, meta


def _fetch_oanda_summary(creds: OandaCreds, *, timeout_sec: float) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {creds.token}"}
    payload, meta = _oanda_get_json(
        f"{creds.base_url}/v3/accounts/{creds.account_id}/summary",
        headers=headers,
        timeout_sec=timeout_sec,
    )
    out: dict[str, Any] = {"meta": meta}
    if payload is None:
        return out
    account = payload.get("account")
    if not isinstance(account, dict):
        return out

    def f(key: str, digits: int = 2) -> float:
        return round(_safe_float(account.get(key), 0.0), digits)

    out.update(
        {
            "currency": _safe_str(account.get("currency")) or None,
            "balance_jpy": f("balance", 2),
            "nav_jpy": f("NAV", 2),
            "unrealized_pl_jpy": f("unrealizedPL", 2),
            "margin_used_jpy": f("marginUsed", 2),
            "margin_available_jpy": f("marginAvailable", 2),
            "margin_rate": round(_safe_float(account.get("marginRate"), 0.0), 6),
            "open_trade_count": int(_safe_float(account.get("openTradeCount"), 0.0)),
            "open_position_count": int(
                _safe_float(account.get("openPositionCount"), 0.0)
            ),
            "pending_order_count": int(
                _safe_float(account.get("pendingOrderCount"), 0.0)
            ),
        }
    )
    return out


def _fetch_oanda_pricing(
    creds: OandaCreds, *, instrument: str, timeout_sec: float
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {creds.token}"}
    payload, meta = _oanda_get_json(
        f"{creds.base_url}/v3/accounts/{creds.account_id}/pricing",
        headers=headers,
        params={"instruments": instrument},
        timeout_sec=timeout_sec,
    )
    out: dict[str, Any] = {"meta": meta}
    if payload is None:
        return out
    prices = payload.get("prices")
    if not isinstance(prices, list) or not prices or not isinstance(prices[0], dict):
        return out
    px = prices[0]
    bids = px.get("bids") if isinstance(px.get("bids"), list) else []
    asks = px.get("asks") if isinstance(px.get("asks"), list) else []
    bid = _safe_float(
        bids[0].get("price") if bids and isinstance(bids[0], dict) else None, 0.0
    )
    ask = _safe_float(
        asks[0].get("price") if asks and isinstance(asks[0], dict) else None, 0.0
    )
    out["time"] = _safe_str(px.get("time")) or None
    out["bid"] = bid if bid > 0 else None
    out["ask"] = ask if ask > 0 else None
    if bid > 0 and ask > 0:
        out["mid"] = round((bid + ask) / 2.0, 6)
        out["spread_pips"] = round((ask - bid) * 100.0, 3)
    else:
        out["mid"] = None
        out["spread_pips"] = None
    return out


def _trade_has_sl(trade: dict[str, Any]) -> bool:
    sl_order_id = _safe_str(trade.get("stopLossOrderID"))
    if sl_order_id:
        return True
    sl_order = trade.get("stopLossOrder")
    if isinstance(sl_order, dict):
        if _safe_str(sl_order.get("id")) or _safe_str(sl_order.get("price")):
            return True
        return True
    return False


def _trade_has_tp(trade: dict[str, Any]) -> bool:
    tp_order_id = _safe_str(trade.get("takeProfitOrderID"))
    if tp_order_id:
        return True
    tp_order = trade.get("takeProfitOrder")
    if isinstance(tp_order, dict):
        if _safe_str(tp_order.get("id")) or _safe_str(tp_order.get("price")):
            return True
        return True
    return False


def _fetch_oanda_open_trades(
    creds: OandaCreds,
    *,
    instrument: str,
    top_n: int,
    timeout_sec: float,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {creds.token}"}
    payload, meta = _oanda_get_json(
        f"{creds.base_url}/v3/accounts/{creds.account_id}/openTrades",
        headers=headers,
        timeout_sec=timeout_sec,
    )
    out: dict[str, Any] = {
        "meta": meta,
        "open_trades_count": 0,
        "open_trades_count_instrument": 0,
        "no_sl_count": 0,
        "no_sl_count_instrument": 0,
        "no_tp_count": 0,
        "no_tp_count_instrument": 0,
        "no_sl_top": [],
    }
    if payload is None:
        return out
    trades = payload.get("trades")
    if not isinstance(trades, list):
        return out

    trade_dicts = [t for t in trades if isinstance(t, dict)]
    out["open_trades_count"] = len(trade_dicts)

    instrument_key = _safe_str(instrument).upper()
    instrument_trades = [
        t
        for t in trade_dicts
        if _safe_str(t.get("instrument")).upper() == instrument_key
    ]
    out["open_trades_count_instrument"] = len(instrument_trades)

    def _count_missing(
        trs: list[dict[str, Any]],
    ) -> tuple[int, int, list[dict[str, Any]]]:
        no_sl: list[dict[str, Any]] = []
        no_tp = 0
        for tr in trs:
            if not _trade_has_sl(tr):
                no_sl.append(tr)
            if not _trade_has_tp(tr):
                no_tp += 1
        return len(no_sl), int(no_tp), no_sl

    no_sl_total, no_tp_total, no_sl_trades_total = _count_missing(trade_dicts)
    no_sl_inst, no_tp_inst, _ = _count_missing(instrument_trades)
    out["no_sl_count"] = int(no_sl_total)
    out["no_sl_count_instrument"] = int(no_sl_inst)
    out["no_tp_count"] = int(no_tp_total)
    out["no_tp_count_instrument"] = int(no_tp_inst)

    # Prioritize "worst" missing-SL positions by unrealizedPL (most negative first).
    no_sl_sorted = sorted(
        no_sl_trades_total,
        key=lambda t: _safe_float(t.get("unrealizedPL"), 0.0),
    )
    limit = max(0, int(top_n))
    for tr in no_sl_sorted[:limit]:
        client = (
            tr.get("clientExtensions")
            if isinstance(tr.get("clientExtensions"), dict)
            else {}
        )
        out["no_sl_top"].append(
            {
                "id": _safe_str(tr.get("id")) or None,
                "instrument": _safe_str(tr.get("instrument")) or None,
                "units": _safe_str(
                    tr.get("currentUnits") or tr.get("initialUnits") or tr.get("units")
                )
                or None,
                "unrealized_pl": _safe_float(tr.get("unrealizedPL"), 0.0),
                "client_id": _safe_str(client.get("id")) or None,
                "client_tag": _safe_str(client.get("tag")) or None,
            }
        )

    return out


def _sqlite_rows(
    path: Path,
    query: str,
    params: tuple[Any, ...],
    *,
    timeout_sec: float,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    if not path.exists():
        return [], f"missing_db:{path}"
    con: Optional[sqlite3.Connection] = None
    try:
        uri = f"file:{path}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=timeout_sec)
        con.row_factory = sqlite3.Row
        con.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
        cur = con.execute(query, params)
        return [dict(row) for row in cur.fetchall()], None
    except Exception as exc:
        return [], f"sqlite_error:{path}:{exc}"
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _calc_trade_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trades = 0
    wins = 0
    losses = 0
    gross_win_pips = 0.0
    gross_loss_pips = 0.0
    gross_win_jpy = 0.0
    gross_loss_jpy = 0.0
    net_pips = 0.0
    net_jpy = 0.0

    for row in rows:
        pips = _safe_float(row.get("pl_pips"), 0.0)
        jpy = _safe_float(row.get("realized_pl"), 0.0)
        trades += 1
        net_pips += pips
        net_jpy += jpy
        if pips > 0:
            wins += 1
            gross_win_pips += pips
        elif pips < 0:
            losses += 1
            gross_loss_pips += abs(pips)
        if jpy > 0:
            gross_win_jpy += jpy
        elif jpy < 0:
            gross_loss_jpy += abs(jpy)

    win_rate = (wins / trades) if trades else 0.0

    if gross_loss_pips > 1e-9:
        pf_pips: float | None = gross_win_pips / gross_loss_pips
    else:
        pf_pips = None if gross_win_pips > 0 else 0.0

    if gross_loss_jpy > 1e-6:
        pf_jpy: float | None = gross_win_jpy / gross_loss_jpy
    else:
        pf_jpy = None if gross_win_jpy > 0 else 0.0

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "pf_pips": round(pf_pips, 4) if isinstance(pf_pips, (int, float)) else pf_pips,
        "pf_jpy": round(pf_jpy, 4) if isinstance(pf_jpy, (int, float)) else pf_jpy,
        "net_pips": round(net_pips, 3),
        "net_jpy": round(net_jpy, 3),
        "gross_win_pips": round(gross_win_pips, 3),
        "gross_loss_pips": round(gross_loss_pips, 3),
        "gross_win_jpy": round(gross_win_jpy, 3),
        "gross_loss_jpy": round(gross_loss_jpy, 3),
    }


def _bucket(value: object, *, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text or default


def _rank_items(
    items: list[dict[str, Any]], *, sort_key: str, top_n: int
) -> dict[str, list[dict[str, Any]]]:
    winners = sorted(
        items, key=lambda r: _safe_float(r.get(sort_key), 0.0), reverse=True
    )[:top_n]
    losers = sorted(items, key=lambda r: _safe_float(r.get(sort_key), 0.0))[:top_n]
    return {"top_winners": winners, "top_losers": losers}


def _build_trade_window(rows: list[dict[str, Any]], *, top_n: int) -> dict[str, Any]:
    overall = _calc_trade_metrics(rows)

    by_pocket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_combo: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pocket = _bucket(row.get("pocket"), default="unknown")
        strategy = _bucket(row.get("strategy_tag"), default="unknown")
        by_pocket[pocket].append(row)
        by_strategy[strategy].append(row)
        by_combo[(pocket, strategy)].append(row)

    pocket_items: list[dict[str, Any]] = [
        {"pocket": p, **_calc_trade_metrics(b)} for p, b in by_pocket.items()
    ]
    strategy_items: list[dict[str, Any]] = [
        {"strategy_tag": s, **_calc_trade_metrics(b)} for s, b in by_strategy.items()
    ]
    combo_items: list[dict[str, Any]] = [
        {"pocket": p, "strategy_tag": s, **_calc_trade_metrics(b)}
        for (p, s), b in by_combo.items()
    ]

    return {
        "overall": overall,
        "rankings": {
            "by_pocket_net_jpy": _rank_items(
                pocket_items, sort_key="net_jpy", top_n=top_n
            ),
            "by_strategy_net_jpy": _rank_items(
                strategy_items, sort_key="net_jpy", top_n=top_n
            ),
            "by_pocket_strategy_net_jpy": _rank_items(
                combo_items, sort_key="net_jpy", top_n=top_n
            ),
        },
    }


def _load_trades_window(
    trades_db: Path,
    *,
    window_expr: str,
    instrument: Optional[str],
    exclude_manual: bool,
    timeout_sec: float,
) -> tuple[list[dict[str, Any]], Optional[str]]:
    where = [
        "state = 'CLOSED'",
        "close_time IS NOT NULL",
        "julianday(close_time) >= julianday('now', ?)",
    ]
    params: list[Any] = [window_expr]
    if instrument:
        where.append("instrument = ?")
        params.append(str(instrument).upper())
    if exclude_manual:
        where.append("COALESCE(LOWER(pocket), '') != 'manual'")

    query = f"""
        SELECT pocket, strategy_tag, pl_pips, realized_pl, close_time
        FROM trades
        WHERE {' AND '.join(where)}
    """
    return _sqlite_rows(trades_db, query, tuple(params), timeout_sec=timeout_sec)


def _summarize_orders_window(
    orders_db: Path,
    *,
    window_expr: str,
    instrument: Optional[str],
    timeout_sec: float,
    top_n: int,
) -> tuple[dict[str, Any], Optional[str]]:
    where = [
        "ts IS NOT NULL",
        "julianday(ts) >= julianday('now', ?)",
    ]
    params: list[Any] = [window_expr]
    if instrument:
        where.append("instrument = ?")
        params.append(str(instrument).upper())

    query = f"""
        SELECT status, error_code, error_message
        FROM orders
        WHERE {' AND '.join(where)}
    """
    rows, err = _sqlite_rows(orders_db, query, tuple(params), timeout_sec=timeout_sec)
    if err is not None:
        return {
            "total_orders": 0,
            "failed_orders": 0,
            "reject_rate": 0.0,
            "status_counts": [],
        }, err

    status_counts: Counter[str] = Counter()
    error_code_counts: Counter[str] = Counter()
    error_code_samples: dict[str, str] = {}
    fail_reason_counts: Counter[str] = Counter()

    for row in rows:
        status = _bucket(row.get("status"), default="unknown")
        status_counts[status] += 1

        code = _safe_str(row.get("error_code"))
        msg = _safe_str(row.get("error_message"))
        if code:
            error_code_counts[code] += 1
            if code not in error_code_samples and msg:
                error_code_samples[code] = msg[:220]

        status_low = status.lower()
        is_fail_status = any(
            tok in status_low
            for tok in (
                "reject",
                "rejected",
                "failed",
                "error",
                "timeout",
                "cancel",
                "quote_unavailable",
            )
        )
        if code:
            fail_reason_counts[code] += 1
        elif is_fail_status:
            fail_reason_counts[status] += 1

    total_orders = len(rows)
    failed_orders = sum(fail_reason_counts.values())
    reject_rate = (failed_orders / total_orders) if total_orders else 0.0

    top_status_counts = [
        {"status": s, "count": n} for s, n in status_counts.most_common(30)
    ]
    top_error_codes = [
        {
            "error_code": code,
            "count": n,
            "sample_message": error_code_samples.get(code) or None,
        }
        for code, n in error_code_counts.most_common(top_n)
    ]
    top_fail_reasons = [
        {"reason": r, "count": n} for r, n in fail_reason_counts.most_common(top_n)
    ]

    return {
        "total_orders": total_orders,
        "failed_orders": failed_orders,
        "reject_rate": round(reject_rate, 4),
        "status_counts": top_status_counts,
        "top_error_codes": top_error_codes,
        "top_fail_reasons": top_fail_reasons,
    }, None


def _atomic_write_text(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")


def _fmt_pf(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "n/a"


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except Exception:
        return "0.0%"


def _render_markdown(report: dict[str, Any], *, top_n: int) -> str:
    instrument = report.get("instrument") or "USD_JPY"
    oanda = report.get("oanda") if isinstance(report.get("oanda"), dict) else {}
    summary = oanda.get("summary") if isinstance(oanda.get("summary"), dict) else {}
    pricing = oanda.get("pricing") if isinstance(oanda.get("pricing"), dict) else {}
    open_trades = (
        oanda.get("open_trades") if isinstance(oanda.get("open_trades"), dict) else {}
    )
    trades = report.get("trades") if isinstance(report.get("trades"), dict) else {}
    t24 = trades.get("24h") if isinstance(trades.get("24h"), dict) else {}
    t7d = trades.get("7d") if isinstance(trades.get("7d"), dict) else {}
    orders = report.get("orders") if isinstance(report.get("orders"), dict) else {}
    o24 = orders.get("24h") if isinstance(orders.get("24h"), dict) else {}
    o7d = orders.get("7d") if isinstance(orders.get("7d"), dict) else {}

    lines: list[str] = []
    lines.append(f"# PDCA Profitability Report ({instrument})")
    lines.append("")
    lines.append(f"- generated_at_utc: {report.get('generated_at_utc')}")
    lines.append(f"- generated_at_jst: {report.get('generated_at_jst')}")
    if report.get("git_rev"):
        lines.append(f"- git_rev: {report.get('git_rev')}")
    lines.append("")

    lines.append("## OANDA Snapshot")
    lines.append(f"- env: {oanda.get('env')}")
    lines.append(f"- account: {oanda.get('account_id_masked')}")
    if isinstance(summary.get("balance_jpy"), (int, float)) and isinstance(
        summary.get("nav_jpy"), (int, float)
    ):
        lines.append(
            f"- balance/nav: {summary.get('balance_jpy'):,.2f} / {summary.get('nav_jpy'):,.2f} JPY "
            f"(uPL {summary.get('unrealized_pl_jpy'):,.2f} JPY)"
        )
    if isinstance(summary.get("margin_used_jpy"), (int, float)) and isinstance(
        summary.get("margin_available_jpy"), (int, float)
    ):
        lines.append(
            f"- margin: used {summary.get('margin_used_jpy'):,.2f} / avail {summary.get('margin_available_jpy'):,.2f} JPY "
            f"rate {summary.get('margin_rate')}"
        )
    if pricing:
        lines.append(
            f"- {instrument} mid={pricing.get('mid')} spread_pips={pricing.get('spread_pips')} (bid={pricing.get('bid')} ask={pricing.get('ask')})"
        )
    if open_trades:
        lines.append(
            f"- open_trades: total={open_trades.get('open_trades_count')} (instrument={open_trades.get('open_trades_count_instrument')})"
        )
        lines.append(
            f"- no_sl: total={open_trades.get('no_sl_count')} (instrument={open_trades.get('no_sl_count_instrument')})"
        )
        lines.append(
            f"- no_tp: total={open_trades.get('no_tp_count')} (instrument={open_trades.get('no_tp_count_instrument')})"
        )
        no_sl_top = (
            open_trades.get("no_sl_top")
            if isinstance(open_trades.get("no_sl_top"), list)
            else []
        )
        if no_sl_top:
            lines.append("- no_sl_top:")
            for row in no_sl_top[: max(0, int(top_n))]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "  - id={id} instrument={instrument} units={units} uPL={u_pl} client_id={client_id} tag={client_tag}".format(
                        id=row.get("id") or "-",
                        instrument=row.get("instrument") or "-",
                        units=row.get("units") or "-",
                        u_pl=row.get("unrealized_pl"),
                        client_id=row.get("client_id") or "-",
                        client_tag=row.get("client_tag") or "-",
                    )
                )
    lines.append("")

    def _trade_line(label: str, section: dict[str, Any]) -> str:
        overall = (
            section.get("overall") if isinstance(section.get("overall"), dict) else {}
        )
        return (
            f"- {label}: trades={overall.get('trades')} win={_fmt_pct(overall.get('win_rate'))} "
            f"PF(pips)={_fmt_pf(overall.get('pf_pips'))} net_pips={overall.get('net_pips')} net_jpy={overall.get('net_jpy')}"
        )

    lines.append("## Trades")
    lines.append(_trade_line("24h", t24))
    lines.append(_trade_line("7d", t7d))
    lines.append("")

    def _rank_block(title: str, block: dict[str, Any]) -> None:
        if not isinstance(block, dict):
            return
        winners = (
            block.get("top_winners")
            if isinstance(block.get("top_winners"), list)
            else []
        )
        losers = (
            block.get("top_losers") if isinstance(block.get("top_losers"), list) else []
        )
        lines.append(f"### {title}")
        if losers:
            lines.append("- top_losers:")
            for row in losers[: min(top_n, len(losers))]:
                label = row.get("strategy_tag") or row.get("pocket") or "unknown"
                if row.get("pocket") and row.get("strategy_tag"):
                    label = f"{row.get('pocket')} / {row.get('strategy_tag')}"
                lines.append(
                    f"  - {label}: net_jpy={row.get('net_jpy')} trades={row.get('trades')} win={_fmt_pct(row.get('win_rate'))} PF={_fmt_pf(row.get('pf_pips'))}"
                )
        if winners:
            lines.append("- top_winners:")
            for row in winners[: min(top_n, len(winners))]:
                label = row.get("strategy_tag") or row.get("pocket") or "unknown"
                if row.get("pocket") and row.get("strategy_tag"):
                    label = f"{row.get('pocket')} / {row.get('strategy_tag')}"
                lines.append(
                    f"  - {label}: net_jpy={row.get('net_jpy')} trades={row.get('trades')} win={_fmt_pct(row.get('win_rate'))} PF={_fmt_pf(row.get('pf_pips'))}"
                )
        lines.append("")

    r24 = t24.get("rankings") if isinstance(t24.get("rankings"), dict) else {}
    r7 = t7d.get("rankings") if isinstance(t7d.get("rankings"), dict) else {}
    if r24.get("by_pocket_strategy_net_jpy"):
        _rank_block(
            "24h pocket/strategy (net_jpy)", r24.get("by_pocket_strategy_net_jpy")
        )
    if r7.get("by_pocket_strategy_net_jpy"):
        _rank_block(
            "7d pocket/strategy (net_jpy)", r7.get("by_pocket_strategy_net_jpy")
        )

    lines.append("## Orders (reject/error)")
    lines.append(
        f"- 24h: total={o24.get('total_orders')} failed={o24.get('failed_orders')} reject_rate={_fmt_pct(o24.get('reject_rate'))}"
    )
    if isinstance(o24.get("top_fail_reasons"), list) and o24.get("top_fail_reasons"):
        lines.append("- 24h top_fail_reasons:")
        for r in o24.get("top_fail_reasons")[:top_n]:
            lines.append(f"  - {r.get('reason')}: {r.get('count')}")
    lines.append(
        f"- 7d: total={o7d.get('total_orders')} failed={o7d.get('failed_orders')} reject_rate={_fmt_pct(o7d.get('reject_rate'))}"
    )
    if isinstance(o7d.get("top_fail_reasons"), list) and o7d.get("top_fail_reasons"):
        lines.append("- 7d top_fail_reasons:")
        for r in o7d.get("top_fail_reasons")[:top_n]:
            lines.append(f"  - {r.get('reason')}: {r.get('count')}")

    errs = report.get("errors") if isinstance(report.get("errors"), list) else []
    if errs:
        lines.append("")
        lines.append("## Errors")
        for e in errs:
            lines.append(f"- {e}")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Local V2 profitability PDCA report (report-only)"
    )
    ap.add_argument("--instrument", default="USD_JPY")
    ap.add_argument("--trades-db", default="logs/trades.db")
    ap.add_argument("--orders-db", default="logs/orders.db")
    ap.add_argument("--out-json", default="logs/pdca_profitability_latest.json")
    ap.add_argument("--history-jsonl", default="logs/pdca_profitability_history.jsonl")
    ap.add_argument("--out-md", default="logs/pdca_profitability_latest.md")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument(
        "--include-manual",
        action="store_true",
        help="Include pocket=manual trades in aggregates (default: excluded)",
    )
    ap.add_argument("--no-oanda", action="store_true", help="Skip OANDA API calls")
    ap.add_argument("--oanda-timeout", type=float, default=7.0)
    ap.add_argument("--db-timeout", type=float, default=0.6)
    return ap.parse_args()


def main() -> int:
    ns = parse_args()
    now = _now_utc()
    errors: list[str] = []

    instrument = _safe_str(ns.instrument).upper()
    trades_db = Path(ns.trades_db)
    orders_db = Path(ns.orders_db)
    out_json = Path(ns.out_json)
    out_md = Path(ns.out_md)
    history_jsonl = Path(ns.history_jsonl)

    report: dict[str, Any] = {
        "generated_at_utc": _iso(now),
        "generated_at_jst": _iso_jst(now),
        "instrument": instrument,
        "git_rev": _git_rev(),
        "errors": errors,
    }

    # OANDA snapshot
    oanda_section: dict[str, Any] = {
        "env": None,
        "account_id_masked": None,
        "summary": {},
        "pricing": {},
        "open_trades": {},
    }
    if not ns.no_oanda:
        try:
            creds = _load_oanda_creds()
            oanda_section["env"] = "practice" if creds.practice else "live"
            oanda_section["account_id_masked"] = _mask_account_id(creds.account_id)
            oanda_section["summary"] = _fetch_oanda_summary(
                creds, timeout_sec=float(ns.oanda_timeout)
            )
            oanda_section["pricing"] = _fetch_oanda_pricing(
                creds, instrument=instrument, timeout_sec=float(ns.oanda_timeout)
            )
            oanda_section["open_trades"] = _fetch_oanda_open_trades(
                creds,
                instrument=instrument,
                top_n=int(ns.top_n),
                timeout_sec=float(ns.oanda_timeout),
            )
        except Exception as exc:
            errors.append(f"oanda_snapshot_failed:{exc}")
    report["oanda"] = oanda_section

    # Trades
    exclude_manual = not bool(ns.include_manual)
    trades_section: dict[str, Any] = {"exclude_manual": exclude_manual}
    for label, window_expr in (("24h", "-24 hours"), ("7d", "-7 days")):
        rows, err = _load_trades_window(
            trades_db,
            window_expr=window_expr,
            instrument=instrument,
            exclude_manual=exclude_manual,
            timeout_sec=float(ns.db_timeout),
        )
        if err is not None:
            errors.append(f"trades_{label}:{err}")
        trades_section[label] = _build_trade_window(rows, top_n=int(ns.top_n))
    report["trades"] = trades_section

    # Orders
    orders_section: dict[str, Any] = {}
    for label, window_expr in (("24h", "-24 hours"), ("7d", "-7 days")):
        summary, err = _summarize_orders_window(
            orders_db,
            window_expr=window_expr,
            instrument=instrument,
            timeout_sec=float(ns.db_timeout),
            top_n=int(ns.top_n),
        )
        if err is not None:
            errors.append(f"orders_{label}:{err}")
        orders_section[label] = summary
    report["orders"] = orders_section

    # Persist outputs
    try:
        _atomic_write_json(out_json, report)
    except Exception:
        return 2
    try:
        _append_jsonl(history_jsonl, report)
    except Exception:
        pass

    md = _render_markdown(report, top_n=int(ns.top_n))
    print(md)
    try:
        _atomic_write_text(out_md, md)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
