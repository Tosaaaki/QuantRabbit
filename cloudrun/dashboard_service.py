import os
import json
import logging
from datetime import datetime, timedelta

from flask import Flask, jsonify, request, Response
import threading
import time
from google.cloud import storage
from google.cloud import firestore


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BUCKET = os.environ.get("BUCKET") or os.environ.get("BUCKET_NEWS") or "quantrabbit-fx-news"

app = Flask(__name__)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET)
fs = firestore.Client()


@app.after_request
def add_no_store(resp: Response):
    """Avoid client/proxy caching for dynamic endpoints and HTML."""
    try:
        resp.headers["Cache-Control"] = "no-store"
    except Exception:
        pass
    return resp


@app.route("/api/stats")
def api_stats():
    try:
        # Backlog: number of raw objects waiting to be summarized (in GCS)
        raw_count = sum(1 for _ in bucket.list_blobs(prefix="raw/"))

        # Prefer Firestore for summary stats; fallback to GCS summary/ when FS empty or unavailable
        now = datetime.utcnow()
        since = now - timedelta(hours=24)
        sum_count = 0
        series = []
        try:
            qn = fs.collection("news").where("ts_utc", ">=", since.isoformat(timespec="seconds"))
            docs = list(qn.stream())
            sum_count = len(docs)
            if sum_count > 0:
                buckets = {}
                for d in docs:
                    data = d.to_dict() or {}
                    ts = data.get("ts_utc")
                    try:
                        t = datetime.fromisoformat(str(ts))
                    except Exception:
                        continue
                    key = t.replace(minute=0, second=0, microsecond=0).isoformat()
                    buckets[key] = buckets.get(key, 0) + 1
                series = sorted([{ "ts": k, "count": v } for k, v in buckets.items()], key=lambda x: x["ts"])
        except Exception:
            sum_count = 0
            series = []

        if sum_count == 0:
            # Fallback: scan GCS summary/ objects for last 24h
            buckets = {}
            sc = 0
            try:
                for b in bucket.list_blobs(prefix="summary/"):
                    if b.name.endswith("/"):
                        continue
                    try:
                        data = json.loads(b.download_as_text())
                        ts = data.get("ts_utc")
                        t = datetime.fromisoformat(str(ts)) if ts else (b.updated or now)
                    except Exception:
                        t = b.updated or now
                    if t < since:
                        continue
                    sc += 1
                    key = t.replace(minute=0, second=0, microsecond=0).isoformat()
                    buckets[key] = buckets.get(key, 0) + 1
                sum_count = sc
                series = sorted([{ "ts": k, "count": v } for k, v in buckets.items()], key=lambda x: x["ts"])
            except Exception:
                pass

        resp = jsonify({
            "bucket": BUCKET,
            "raw_count": raw_count,
            "summary_count": sum_count,
            "series": series,
        })
        try:
            resp.headers["Cache-Control"] = "no-store"
        except Exception:
            pass
        return resp
    except Exception as e:
        logging.error(f"dashboard /api/stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/news")
def api_news():
    try:
        limit = int(request.args.get("limit", 20))
        q = fs.collection("news").order_by("ts_utc", direction=firestore.Query.DESCENDING).limit(limit)
        items = []
        try:
            for d in q.stream():
                data = d.to_dict() or {}
                items.append({
                    "uid": data.get("uid", d.id),
                    "ts_utc": data.get("ts_utc"),
                    "summary": data.get("summary", ""),
                    "sentiment": data.get("sentiment", 0),
                    "impact": data.get("impact", 1),
                })
        except Exception:
            items = []
        # Fallback: when Firestore has no recent items, read latest summaries from GCS
        if not items:
            try:
                blobs = list(bucket.list_blobs(prefix="summary/"))
                # sort by updated time desc if available; else by name
                def _key(b):
                    try:
                        return b.updated or datetime.utcnow()
                    except Exception:
                        return datetime.utcnow()
                blobs.sort(key=_key, reverse=True)
                for b in blobs[:limit]:
                    try:
                        data = json.loads(b.download_as_text())
                        items.append({
                            "uid": data.get("uid", b.name),
                            "ts_utc": data.get("ts_utc"),
                            "summary": data.get("summary", ""),
                            "sentiment": int(data.get("sentiment", 0) or 0),
                            "impact": int(data.get("impact", 1) or 1),
                        })
                    except Exception:
                        continue
            except Exception:
                pass
        return jsonify({"items": items})
    except Exception as e:
        logging.error(f"dashboard /api/news error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/trading-state")
def api_trading_state():
    try:
        doc = fs.collection("status").document("trader").get()
        if not doc.exists:
            return jsonify({"error": "no status"}), 404
        data = doc.to_dict() or {}
        return jsonify(data)
    except Exception as e:
        logging.error(f"dashboard /api/trading-state error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/events")
def api_events():
    try:
        within = int(request.args.get("within", 60))
        now = datetime.utcnow()
        end = now + timedelta(minutes=within)
        q = fs.collection("news").where("event_time", ">=", now.isoformat(timespec="seconds")).limit(50)
        items = []
        for d in q.stream():
            x = d.to_dict() or {}
            ts = x.get("event_time")
            if not ts:
                continue
            try:
                t = datetime.fromisoformat(str(ts))
            except Exception:
                continue
            if t <= end:
                delta = int((t - now).total_seconds())
                items.append({
                    "uid": x.get("uid", d.id),
                    "event_time": ts,
                    "impact": int(x.get("impact", 1) or 1),
                    "title": x.get("title", ""),
                    "currency": x.get("currency", ""),
                    "countdown_sec": max(delta, 0),
                })
        items.sort(key=lambda z: z.get("event_time", ""))
        return jsonify({"items": items})
    except Exception as e:
        logging.error(f"dashboard /api/events error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/perf")
def api_perf():
    try:
        # Prefer status/trader; fallback to OANDA NAV + perf/daily.day_start_nav
        sdoc = fs.collection("status").document("trader").get()
        s = sdoc.to_dict() if sdoc.exists else {}

        equity = float(s.get("equity") or 0.0)
        day_start = float(s.get("day_start_nav") or 0.0)

        # Fallbacks when trader status is not being written
        if equity <= 0.0:
            try:
                import asyncio
                from execution.account_info import get_account_summary
                acc = asyncio.run(get_account_summary())
                equity = float(acc.get("NAV", 0.0))
            except Exception:
                pass
        if day_start <= 0.0:
            try:
                d = fs.collection("perf").document("daily").get()
                if d.exists:
                    day_start = float((d.to_dict() or {}).get("day_start_nav") or 0.0)
            except Exception:
                pass

        progress = None
        if day_start:
            try:
                progress = (equity - day_start) / day_start
            except Exception:
                progress = None
        # Target (default 10%)
        try:
            target_pct = float(os.environ.get("DAILY_TARGET_PCT", "0.10"))
        except Exception:
            target_pct = 0.10
        achieved = (progress is not None and progress >= target_pct)
        remaining = (max(target_pct - progress, 0.0) if progress is not None else None)
        resp = jsonify({
            "equity": equity,
            "day_start_nav": day_start,
            "progress_pct": progress,
            "target_pct": target_pct,
            "achieved": achieved,
            "remaining_pct": remaining,
        })
        try:
            resp.headers["Cache-Control"] = "no-store"
        except Exception:
            pass
        return resp
    except Exception as e:
        logging.error(f"dashboard /api/perf error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/pnl")
def api_pnl():
    """Return today's aggregated PnL if present (perf/daily)."""
    try:
        d = fs.collection("perf").document("daily").get()
        if not d.exists:
            return jsonify({"error": "no daily pnl"}), 404
        data = d.to_dict() or {}
        # enrich with target progress if day_start_nav available
        try:
            target_pct = float(os.environ.get("DAILY_TARGET_PCT", "0.10"))
        except Exception:
            target_pct = 0.10
        day_start_nav = data.get("day_start_nav")
        # Fallback: read from status/trader when missing
        if day_start_nav is None:
            try:
                sdoc = fs.collection("status").document("trader").get()
                if sdoc.exists:
                    sd = sdoc.to_dict() or {}
                    dsn = sd.get("day_start_nav")
                    if dsn is not None:
                        day_start_nav = float(dsn)
                        data["day_start_nav"] = day_start_nav
            except Exception:
                pass
        # 進捗は原則Equityベース、無ければRealizedベース
        progress_equity = data.get("equity_progress_pct")
        progress_realized = data.get("realized_progress_pct")
        if progress_equity is None:
            try:
                net = float(data.get("net_pl", 0.0))
                base = float(day_start_nav) if day_start_nav else 0.0
                # If base is unknown, avoid forcing 0 — keep None so UI can fallback
                progress_equity = (net / base) if base else None
            except Exception:
                progress_equity = None
        # When progress is unknown, keep pct_of_target undefined to avoid UI reset
        pct_of_target = (float(progress_equity) / target_pct) if (target_pct and progress_equity is not None) else None
        data.update({
            "target_pct": target_pct,
            "progress_pct": progress_equity,
            "pct_of_target": pct_of_target,
        })
        resp = jsonify(data)
        try:
            resp.headers["Cache-Control"] = "no-store"
        except Exception:
            pass
        return resp
    except Exception as e:
        logging.error(f"dashboard /api/pnl error: {e}")
        resp = jsonify({"error": str(e)})
        try:
            resp.headers["Cache-Control"] = "no-store"
        except Exception:
            pass
        return resp, 500


@app.route("/api/tx")
def api_tx():
    """Return today's recent OANDA transactions (simplified)."""
    try:
        # Lazy import to avoid hard dependency during unit tests
        from execution.oanda_pnl import fetch_transactions_since, _parse_time
        start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start = start.replace(tzinfo=datetime.utcnow().astimezone().tzinfo)  # ensure tz awareness
        txs = fetch_transactions_since(start)
        # Keep latest 30, map fields
        items = []
        for t in txs[-30:]:
            tt = _parse_time(t.get("time"))
            items.append({
                "id": t.get("id"),
                "time": tt.isoformat() if isinstance(tt, datetime) else t.get("time"),
                "type": t.get("type"),
                "instrument": t.get("instrument"),
                "pl": t.get("pl"),
            })
        return jsonify({"items": items})
    except Exception as e:
        logging.error(f"dashboard /api/tx error: {e}")
        return jsonify({"error": str(e)}), 500


# Optional: background PnL writer (every 5 minutes) when enabled
def _pnl_writer_loop():
    try:
        from execution.oanda_pnl import compute_daily_pnl
        from execution.account_info import get_account_summary
    except Exception as _e:
        logging.error(f"pnl_writer import error: {_e}")
        return
    while True:
        try:
            res = compute_daily_pnl()
            doc_ref = fs.collection("perf").document("daily")
            # set day_start_nav if missing using status.trader or current NAV
            cur = doc_ref.get().to_dict() if doc_ref.get().exists else {}
            if not cur or cur.get("day_start_nav") is None:
                try:
                    st = fs.collection("status").document("trader").get()
                    if st.exists:
                        stv = st.to_dict() or {}
                        dsn = stv.get("day_start_nav")
                        if dsn is not None:
                            res["day_start_nav"] = dsn
                except Exception:
                    pass
            if res.get("day_start_nav") is None:
                try:
                    acc = get_account_summary()
                    res["day_start_nav"] = float(acc.get("NAV", 0.0))
                except Exception:
                    pass
            doc_ref.set(res)
            fs.collection("perf").document("daily_meta").set({"updated": datetime.utcnow().isoformat(timespec="seconds")})
            logging.info("[PNL] daily updated: %s", res)
        except Exception as e:
            logging.error(f"pnl_writer error: {e}")
        time.sleep(int(os.environ.get("PNL_UPDATE_SEC", "300")))


if os.environ.get("ENABLE_PNL_TASK", "false").lower() == "true":
    t = threading.Thread(target=_pnl_writer_loop, daemon=True)
    t.start()


@app.route("/")
def index():
    # Simple dashboard with Chart.js
    html = """
<!doctype html>
<html lang=\"ja\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>QuantRabbit News Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans JP', Arial, sans-serif; margin: 0; }
    header { background:#121826; color:#fff; padding: 12px 16px; }
    header h1 { margin:0; font-size: 18px; }
    .container { padding: 16px; max-width: 1100px; margin: 0 auto; }
    .cards { display:flex; gap:12px; flex-wrap: wrap; }
    .card { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:12px; flex:1; min-width:220px; }
    .muted { color:#6b7280; font-size:12px; }
    .list { margin-top: 12px; }
    .item { padding:8px 0; border-bottom:1px solid #f0f0f0; }
    .sent-pos { color: #16a34a; }
    .sent-neg { color: #dc2626; }
    .row { display:flex; gap:12px; flex-wrap: wrap; }
    .w50 { flex:1; min-width:320px; }
    .cards.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; }
    .card .value { font-size: 18px; margin-top: 2px; }

    /* Badges */
    .badge { display:inline-block; padding:2px 6px; border-radius:6px; font-size: 12px; border:1px solid #e5e7eb; }
    .badge-blue { background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe; }
    .badge-gray { background:#f3f4f6; color:#374151; border-color:#e5e7eb; }
    .badge-green { background:#ecfdf5; color:#059669; border-color:#a7f3d0; }
    .badge-amber { background:#fffbeb; color:#b45309; border-color:#fde68a; }
    .badge-red { background:#fef2f2; color:#b91c1c; border-color:#fecaca; }

    /* Bars */
    .bar { background:#f3f4f6; height:10px; border-radius:6px; overflow:hidden; margin-top:6px; }
    .bar .fill { height:100%; width:0; transition: width .3s ease; }
    .fill-blue { background:#3b82f6; }
    .fill-green { background:#22c55e; }
    .fill-red { background:#ef4444; }
    .fill-amber { background:#f59e0b; }
    .pl-pos { color:#16a34a; }
    .pl-neg { color:#dc2626; }
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <script>
    // Global state for live updates
    let chartSummaries = null;
    const REFRESH_MS = 5000; // 5s refresh cadence
    window.useSSE = false; // switched on when SSE connects
    function setClass(el, base, cls) {
      el.className = base + ' ' + cls;
    }

    function pct(n) {
      if (n === null || n === undefined) return '-';
      return Number(n).toFixed(2);
    }

    async function refreshKPIs() {
      // Trading State
      try {
        const t = await fetch('/api/trading-state', {cache:'no-store'}).then(r => r.json());
        document.getElementById('equity').textContent = pct(t.equity);
        if (t.mode) document.getElementById('mode').textContent = t.mode;
        if (Array.isArray(t.selected_instruments)) {
          document.getElementById('instruments').textContent = t.selected_instruments.join(', ');
        }
        if (t.diversify_slots != null) {
          document.getElementById('divslots').textContent = String(t.diversify_slots);
        }
        if (t.resource_ttl_sec != null) {
          const mins = Math.floor(Number(t.resource_ttl_sec)/60);
          document.getElementById('ttl').textContent = `${mins}m`;
        }
        document.getElementById('lot').textContent = pct(t.lot_total);

        // Regime badges
        const macro = (t.macro_regime||'-').toString().toUpperCase();
        const micro = (t.micro_regime||'-').toString().toUpperCase();
        const macroEl = document.getElementById('badge-reg-macro');
        const microEl = document.getElementById('badge-reg-micro');
        macroEl.textContent = macro;
        microEl.textContent = micro;
        setClass(macroEl, 'badge', macro === 'TREND' ? 'badge-blue' : 'badge-gray');
        setClass(microEl, 'badge', micro === 'TREND' ? 'badge-blue' : 'badge-gray');

        // Focus + Weight bar
        const w = Math.max(0, Math.min(1, Number(t.weight_macro||0)));
        document.getElementById('focus').textContent = `${t.focus_tag||'-'} (w=${pct(w*100)}%)`;
        const wb = document.getElementById('bar-weight');
        wb.style.width = (w*100).toFixed(0) + '%';

        // Event badge
        const ev = !!t.event_soon;
        const evEl = document.getElementById('badge-event');
        evEl.textContent = ev ? 'EVENT SOON' : 'NO EVENT';
        setClass(evEl, 'badge', ev ? 'badge-red' : 'badge-green');
      } catch(e) {}

      // Perf (also drive Equity display from perf.equity for consistency)
      try {
        const p = await fetch('/api/perf', {cache:'no-store'}).then(r => r.json());
        // Update Equity from perf
        if (p.equity !== undefined && p.equity !== null) {
          document.getElementById('equity').textContent = Number(p.equity).toFixed(2);
        }
        // Progress Today (only when progress is present)
        if (p.progress_pct != null) {
          const prog = Number(p.progress_pct);
          const progPct = (prog*100);
          document.getElementById('progress').textContent = progPct.toFixed(2)+'%';
          const pb = document.getElementById('bar-progress');
          const absw = Math.max(0, Math.min(100, Math.abs(progPct)));
          pb.style.width = absw.toFixed(0) + '%';
          setClass(pb, 'fill', prog >= 0 ? 'fill-green' : 'fill-red');
        }
        // Target vs Progress (update only when target and progress are both present)
        const tb = document.getElementById('bar-target');
        const status = document.getElementById('target-status');
        const tval = (p.target_pct != null) ? Number(p.target_pct) : null;
        const tpctEl = document.getElementById('target-pct');
        if (tpctEl && tval != null && !Number.isNaN(tval)) tpctEl.textContent = (tval*100).toFixed(2);
        if (tval != null && p.progress_pct != null) {
          const ratio = Math.max(0, Math.min(1, Number(p.progress_pct)/tval));
          tb.style.width = (ratio*100).toFixed(0) + '%';
          if (p.achieved) {
            status.textContent = '達成';
            setClass(status, '', 'badge badge-green');
            setClass(tb, 'fill', 'fill-green');
          } else {
            const rem = (Number(p.remaining_pct||Math.max(0, tval - Number(p.progress_pct)||0))*100).toFixed(2);
            status.textContent = `未達（残 ${rem}%）`;
            setClass(status, '', 'badge badge-amber');
            setClass(tb, 'fill', 'fill-amber');
          }
        }
      } catch(e) {}

      // PnL Today
      try {
        const d = await fetch('/api/pnl', {cache:'no-store'}).then(r => r.json());
        if (!d.error) {
          const pnl = (d.equity_change !== undefined && d.equity_change !== null)
            ? Number(d.equity_change)
            : Number(d.net_pl||0);
          const bar = document.getElementById('bar-pnl-target');
          document.getElementById('pnl').textContent = pnl.toFixed(2);
          const hasBaseline = d.day_start_nav != null && Number(d.day_start_nav) > 0;
          const ratio = (hasBaseline && d.pct_of_target != null) ? Number(d.pct_of_target) : null;
          if (ratio != null && !Number.isNaN(ratio)) {
            bar.style.width = Math.max(0, Math.min(100, ratio*100)).toFixed(0) + '%';
            setClass(bar, 'fill', pnl >= 0 ? 'fill-green' : 'fill-red');
          }

          // Also drive Daily Target card from PnL progress vs target
          const tb = document.getElementById('bar-target');
          const status = document.getElementById('target-status');
          // Update Daily Target label from pnl response as well
          const tpctEl = document.getElementById('target-pct');
          if (tpctEl && d.target_pct != null) tpctEl.textContent = (Number(d.target_pct)*100).toFixed(2);
          if (hasBaseline && d.pct_of_target != null) {
            const ratio2 = Math.max(0, Math.min(1, Number(d.pct_of_target)));
            tb.style.width = (ratio2*100).toFixed(0) + '%';
            if (ratio2 >= 1) {
              status.textContent = '達成';
              setClass(status, '', 'badge badge-green');
              setClass(tb, 'fill', 'fill-green');
            } else {
              const rem = Math.max(0, (Number(d.target_pct||0) - Number(d.progress_pct||0)) * 100);
              status.textContent = `未達（残 ${rem.toFixed(2)}%）`;
              setClass(status, '', 'badge badge-amber');
              setClass(tb, 'fill', 'fill-amber');
            }
          }

          // Drive Progress Today from PnL-based progress (net_pl / day_start_nav)
          if (d.progress_pct != null) {
            const prog2 = Number(d.progress_pct);
            const progPct2 = prog2 * 100;
            document.getElementById('progress').textContent = progPct2.toFixed(2)+'%';
            const pb2 = document.getElementById('bar-progress');
            const absw2 = Math.max(0, Math.min(100, Math.abs(progPct2)));
            pb2.style.width = absw2.toFixed(0) + '%';
            setClass(pb2, 'fill', prog2 >= 0 ? 'fill-green' : 'fill-red');
          }

          // Optionally show Equity ~= day_start_nav + net_pl for coherence
          if (d.day_start_nav !== undefined && d.day_start_nav !== null) {
            const eqSynth = Number(d.day_start_nav||0) + Number(d.net_pl||0);
            if (!isNaN(eqSynth)) {
              document.getElementById('equity').textContent = eqSynth.toFixed(2);
            }
          }
        }
      } catch(e) {}
    }

    async function refreshStats() {
      try {
        const stats = await fetch('/api/stats', {cache:'no-store'}).then(r => r.json());
        document.getElementById('bucket').textContent = stats.bucket;
        document.getElementById('raw').textContent = stats.raw_count;
        document.getElementById('sum').textContent = stats.summary_count;
        // Chart: create or update
        try {
          const labels = (stats.series||[]).map(s => new Date(s.ts).toLocaleString());
          const data = (stats.series||[]).map(s => s.count);
          const ctx = document.getElementById('chart').getContext('2d');
          if (window.Chart && ctx) {
            if (!chartSummaries) {
              chartSummaries = new Chart(ctx, {
                type: 'line',
                data: { labels, datasets: [{ label: 'Summaries/hour', data, borderColor:'#2563eb', fill:false }] },
                options: { responsive:true, scales: { y: { beginAtZero: true } } }
              });
            } else {
              chartSummaries.data.labels = labels;
              chartSummaries.data.datasets[0].data = data;
              chartSummaries.update();
            }
          }
        } catch(e) { /* no-op */ }
      } catch(e) {
        // keep placeholders on failure
      }
    }

    async function refreshNews() {
      try {
        const news = await fetch('/api/news?limit=20', {cache:'no-store'}).then(r => r.json());
        const list = document.getElementById('news');
        list.innerHTML = '';
        (news.items || []).forEach(n => {
          const div = document.createElement('div');
          const sent = (n.sentiment||0);
          const sentCls = sent > 0 ? 'sent-pos' : (sent < 0 ? 'sent-neg' : '');
          div.className = 'item';
          div.innerHTML = `<div class=\"muted\">${n.ts_utc||''}</div>
                           <div>${n.summary||''}</div>
                           <div class=\"muted\">sentiment: <span class=\"${sentCls}\">${sent}</span>, impact: ${n.impact||1}</div>`;
          list.appendChild(div);
        });
      } catch(e) {}
    }

    async function refreshTx() {
      try {
        const tx = await fetch('/api/tx', {cache:'no-store'}).then(r => r.json());
        const lst = document.getElementById('tx');
        if (lst) {
          lst.innerHTML = '';
          (tx.items || []).slice().reverse().forEach(t => {
            const div = document.createElement('div');
            const v = Number(t.pl||0);
            const cls = v>0 ? 'pl-pos' : (v<0 ? 'pl-neg' : '');
            div.className = 'item';
            div.innerHTML = `<div class=\"muted\">${t.time||''} / ${t.instrument||''}</div>
                             <div>${t.type||''} <span class=\"${cls}\">${v.toFixed(2)}</span></div>`;
            lst.appendChild(div);
          });
        }
      } catch(e) {}
    }

    async function loadAll() {
      await refreshStats();

      // First KPI render
      await refreshKPIs();

      // PnL Today
      try {
        const d = await fetch('/api/pnl', {cache:'no-store'}).then(r => r.json());
        if (!d.error) {
          const pnl = (d.equity_change !== undefined && d.equity_change !== null)
            ? Number(d.equity_change)
            : Number(d.net_pl||0);
          const bar = document.getElementById('bar-pnl-target');
          document.getElementById('pnl').textContent = pnl.toFixed(2);
          const hasBaseline = d.day_start_nav != null && Number(d.day_start_nav) > 0;
          const ratio = (hasBaseline && d.pct_of_target != null) ? Number(d.pct_of_target) : null;
          if (ratio != null && !Number.isNaN(ratio)) {
            bar.style.width = Math.max(0, Math.min(100, ratio*100)).toFixed(0) + '%';
            setClass(bar, 'fill', pnl >= 0 ? 'fill-green' : 'fill-red');
          }
          // Update Daily Target label on initial load if present
          const tpctEl = document.getElementById('target-pct');
          if (tpctEl && d.target_pct != null) tpctEl.textContent = (Number(d.target_pct)*100).toFixed(2);
        }
      } catch(e) {}

      // News list
      const news = await fetch('/api/news?limit=20').then(r => r.json());
      const list = document.getElementById('news');
      list.innerHTML = '';
      (news.items || []).forEach(n => {
        const div = document.createElement('div');
        const sent = (n.sentiment||0);
        const sentCls = sent > 0 ? 'sent-pos' : (sent < 0 ? 'sent-neg' : '');
        div.className = 'item';
        div.innerHTML = `<div class=\"muted\">${n.ts_utc||''}</div>
                         <div>${n.summary||''}</div>
                         <div class=\"muted\">sentiment: <span class=\"${sentCls}\">${sent}</span>, impact: ${n.impact||1}</div>`;
        list.appendChild(div);
      });
      // Start periodic refresh
      setInterval(() => { if (!window.useSSE) refreshKPIs(); }, REFRESH_MS);
      setInterval(() => { if (!window.useSSE) refreshStats(); }, REFRESH_MS);
      setInterval(() => { if (!window.useSSE) refreshNews(); }, REFRESH_MS);
      setInterval(() => { if (!window.useSSE) refreshTx(); }, REFRESH_MS);

      // Try SSE; if connected, push updates will drive the UI
      try {
        const es = new EventSource('/sse');
        es.onopen = () => { window.useSSE = true; };
        es.onerror = () => { window.useSSE = false; };
        es.addEventListener('kpi', (ev) => {
          try {
            const t = JSON.parse(ev.data);
            if (t.equity != null) document.getElementById('equity').textContent = Number(t.equity).toFixed(2);
            if (t.lot_total != null) document.getElementById('lot').textContent = Number(t.lot_total).toFixed(2);
            const macro = (t.macro_regime||'-').toString().toUpperCase();
            const micro = (t.micro_regime||'-').toString().toUpperCase();
            const macroEl = document.getElementById('badge-reg-macro');
            const microEl = document.getElementById('badge-reg-micro');
            if (macroEl) { macroEl.textContent = macro; setClass(macroEl, 'badge', macro === 'TREND' ? 'badge-blue' : 'badge-gray'); }
            if (microEl) { microEl.textContent = micro; setClass(microEl, 'badge', micro === 'TREND' ? 'badge-blue' : 'badge-gray'); }
            const w = Math.max(0, Math.min(1, Number(t.weight_macro||0)));
            document.getElementById('focus').textContent = `${t.focus_tag||'-'} (w=${(w*100).toFixed(2)}%)`;
            const wb = document.getElementById('bar-weight');
            if (wb) wb.style.width = (w*100).toFixed(0) + '%';
            const evSoon = !!t.event_soon;
            const evEl = document.getElementById('badge-event');
            if (evEl) { evEl.textContent = evSoon ? 'EVENT SOON' : 'NO EVENT'; setClass(evEl, 'badge', evSoon ? 'badge-red' : 'badge-green'); }
          } catch(_) {}
        });
        es.addEventListener('perf', (ev) => {
          try {
            const p = JSON.parse(ev.data);
            const prog = Number(p.progress_pct||0);
            const progPct = prog*100;
            document.getElementById('progress').textContent = progPct.toFixed(2)+'%';
            const pb = document.getElementById('bar-progress');
            if (pb) { pb.style.width = Math.max(0, Math.min(100, Math.abs(progPct))).toFixed(0) + '%'; setClass(pb, 'fill', prog >= 0 ? 'fill-green' : 'fill-red'); }
            if (p.equity != null) document.getElementById('equity').textContent = Number(p.equity).toFixed(2);
            const tb = document.getElementById('bar-target');
            const status = document.getElementById('target-status');
            const target = (p.target_pct != null) ? Number(p.target_pct) : null;
            const tpctEl = document.getElementById('target-pct');
            if (tpctEl && target != null && !Number.isNaN(target)) tpctEl.textContent = (target*100).toFixed(2);
            if (target != null && !Number.isNaN(target) && p.progress_pct != null) {
              const ratio = Math.max(0, Math.min(1, Number(p.progress_pct)/target));
              if (tb) tb.style.width = (ratio*100).toFixed(0) + '%';
              if (p.achieved) { status.textContent = '達成'; setClass(status, '', 'badge badge-green'); setClass(tb, 'fill', 'fill-green'); }
              else {
                const rem = (Number(p.remaining_pct||Math.max(0, target - Number(p.progress_pct)||0))*100).toFixed(2);
                status.textContent = `未達（残 ${rem}%）`;
                setClass(status, '', 'badge badge-amber');
                setClass(tb, 'fill', 'fill-amber');
              }
            }
            const pnl = (p.equity_change !== undefined && p.equity_change !== null) ? Number(p.equity_change) : Number(p.net_pl||0);
            document.getElementById('pnl').textContent = pnl.toFixed(2);
            const bp = document.getElementById('bar-pnl-target');
            if (bp && p.pct_of_target != null) { const ratio2 = Number(p.pct_of_target); bp.style.width = Math.max(0, Math.min(100, ratio2*100)).toFixed(0) + '%'; setClass(bp, 'fill', pnl >= 0 ? 'fill-green' : 'fill-red'); }
          } catch(_) {}
        });
        es.addEventListener('stats', (ev) => {
          try {
            const s = JSON.parse(ev.data);
            document.getElementById('bucket').textContent = s.bucket;
            document.getElementById('raw').textContent = s.raw_count;
            document.getElementById('sum').textContent = s.summary_count;
            const labels = (s.series||[]).map(x => new Date(x.ts).toLocaleString());
            const data = (s.series||[]).map(x => x.count);
            const ctx = document.getElementById('chart').getContext('2d');
            if (window.Chart && ctx) {
              if (!chartSummaries) {
                chartSummaries = new Chart(ctx, { type:'line', data:{ labels, datasets:[{ label:'Summaries/hour', data, borderColor:'#2563eb', fill:false }] }, options:{ responsive:true, scales:{ y:{ beginAtZero:true } } } });
              } else {
                chartSummaries.data.labels = labels; chartSummaries.data.datasets[0].data = data; chartSummaries.update();
              }
            }
          } catch(_) {}
        });
        es.addEventListener('news', (ev) => {
          try {
            const n = JSON.parse(ev.data);
            const list = document.getElementById('news');
            if (!list) return;
            list.innerHTML = '';
            (n.items || []).forEach(it => {
              const div = document.createElement('div');
              const sent = (it.sentiment||0);
              const sentCls = sent > 0 ? 'sent-pos' : (sent < 0 ? 'sent-neg' : '');
              div.className = 'item';
              div.innerHTML = `<div class=\\"muted\\">${it.ts_utc||''}</div><div>${it.summary||''}</div><div class=\\"muted\\">sentiment: <span class=\\"${sentCls}\\">${sent}</span>, impact: ${it.impact||1}</div>`;
              list.appendChild(div);
            });
          } catch(_) {}
        });
        es.addEventListener('tx', (ev) => {
          try {
            const t = JSON.parse(ev.data);
            const lst = document.getElementById('tx');
            if (!lst) return;
            lst.innerHTML = '';
            (t.items || []).slice().reverse().forEach(it => {
              const div = document.createElement('div');
              const v = Number(it.pl||0);
              const cls = v>0 ? 'pl-pos' : (v<0 ? 'pl-neg' : '');
              div.className = 'item';
              div.innerHTML = `<div class=\\"muted\\">${it.time||''} / ${it.instrument||''}</div><div>${it.type||''} <span class=\\"${cls}\\">${v.toFixed(2)}</span></div>`;
              lst.appendChild(div);
            });
          } catch(_) {}
        });
      } catch(_) {}
      // Initial lists rendering (news + trades)
      try {
        const news = await fetch('/api/news?limit=20', {cache:'no-store'}).then(r => r.json());
        const list = document.getElementById('news');
        list.innerHTML = '';
        (news.items || []).forEach(n => {
          const div = document.createElement('div');
          const sent = (n.sentiment||0);
          const sentCls = sent > 0 ? 'sent-pos' : (sent < 0 ? 'sent-neg' : '');
          div.className = 'item';
          div.innerHTML = `<div class=\"muted\">${n.ts_utc||''}</div>
                           <div>${n.summary||''}</div>
                           <div class=\"muted\">sentiment: <span class=\"${sentCls}\">${sent}</span>, impact: ${n.impact||1}</div>`;
          list.appendChild(div);
        });
      } catch(e) {}
      try {
        const tx = await fetch('/api/tx', {cache:'no-store'}).then(r => r.json());
        const lst = document.getElementById('tx');
        if (lst) {
          lst.innerHTML = '';
          (tx.items || []).slice().reverse().forEach(t => {
            const div = document.createElement('div');
            const v = Number(t.pl||0);
            const cls = v>0 ? 'pl-pos' : (v<0 ? 'pl-neg' : '');
            div.className = 'item';
            div.innerHTML = `<div class=\"muted\">${t.time||''} / ${t.instrument||''}</div>
                             <div>${t.type||''} <span class=\"${cls}\">${v.toFixed(2)}</span></div>`;
            lst.appendChild(div);
          });
        }
      } catch(e) {}
    }
    document.addEventListener('DOMContentLoaded', loadAll);
  </script>
</head>
<body>
  <header>
    <h1>QuantRabbit News Dashboard</h1>
  </header>
  <div class="container">
    <div class="cards kpis">
      <div class="card"><div class="muted">Bucket</div><div id="bucket" class="value">-</div></div>
      <div class="card"><div class="muted">Raw Backlog</div><div id="raw" class="value">-</div></div>
      <div class="card"><div class="muted">Summary Objects</div><div id="sum" class="value">-</div></div>
      <div class="card"><div class="muted">Equity</div><div id="equity" class="value">-</div></div>
      <div class="card"><div class="muted">Lot Allowed</div><div id="lot" class="value">-</div></div>
      <div class="card"><div class="muted">Mode</div><div id="mode" class="value">-</div></div>
      <div class="card"><div class="muted">Instruments</div><div id="instruments" class="value">-</div></div>
      <div class="card"><div class="muted">Diversify Slots</div><div id="divslots" class="value">-</div></div>
      <div class="card"><div class="muted">Resource TTL</div><div id="ttl" class="value">-</div></div>
      <div class="card"><div class="muted">PnL Today</div><div id="pnl" class="value">-</div><div class="bar"><div id="bar-pnl-target" class="fill"></div></div></div>
      <div class="card"><div class="muted">Progress Today</div><div id="progress" class="value">-</div><div class="bar"><div id="bar-progress" class="fill"></div></div></div>
      <div class="card"><div class="muted">Daily Target (<span id="target-pct">-</span>%)</div><div id="target-status" class="value">-</div><div class="bar"><div id="bar-target" class="fill fill-amber"></div></div></div>
    </div>
    <div class="row" style="margin-top:12px;">
      <div class="card w50">
        <div class="muted">Summaries/hour</div>
        <canvas id="chart" height="120"></canvas>
      </div>
      <div class="card w50">
        <div class="muted">Regime / Focus / Event</div>
        <div>Macro Regime: <span id="badge-reg-macro" class="badge">-</span></div>
        <div>Micro Regime: <span id="badge-reg-micro" class="badge">-</span></div>
        <div style="margin-top:6px;">Focus: <span id="focus">-</span></div>
        <div style="margin-top:6px;">
          <div class="muted">Weight Macro</div>
          <div class="bar"><div id="bar-weight" class="fill fill-blue" style="width:0%"></div></div>
        </div>
        <div style="margin-top:8px;">Event: <span id="badge-event" class="badge">-</span></div>
      </div>
    </div>
    <div class="row" style="margin-top:12px;">
      <div class="card w50">
        <div class="muted">Recent News</div>
        <div id="news" class="list"></div>
      </div>
      <div class="card w50">
        <div class="muted">Recent Trades</div>
        <div id="tx" class="list"></div>
      </div>
    </div>
  </div>
</body>
</html>
"""
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\n" + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


@app.route("/sse")
def sse_stream():
    def gen():
        i = 0
        while True:
            try:
                # KPI
                try:
                    sdoc = fs.collection("status").document("trader").get()
                    if sdoc.exists:
                        yield _sse_format("kpi", sdoc.to_dict() or {}).encode("utf-8")
                except Exception:
                    pass
                # Perf (enriched to include target/progress fields)
                try:
                    pdoc = fs.collection("perf").document("daily").get()
                    if pdoc.exists:
                        pdata = pdoc.to_dict() or {}
                        try:
                            target_pct = float(os.environ.get("DAILY_TARGET_PCT", "0.10"))
                        except Exception:
                            target_pct = 0.10
                        progress = None
                        # Prefer equity-based if available; fallback to realized
                        if pdata.get("equity_progress_pct") is not None:
                            try:
                                progress = float(pdata.get("equity_progress_pct"))
                            except Exception:
                                progress = None
                        if progress is None and pdata.get("realized_progress_pct") is not None:
                            try:
                                progress = float(pdata.get("realized_progress_pct"))
                            except Exception:
                                progress = None
                        pct_of_target = (progress / target_pct) if (progress is not None and target_pct) else None
                        remaining = (max(target_pct - progress, 0.0) if progress is not None else None)
                        achieved = (progress is not None and progress >= target_pct)
                        out = dict(pdata)
                        out.update({
                            "target_pct": target_pct,
                            "progress_pct": progress,
                            "pct_of_target": pct_of_target,
                            "remaining_pct": remaining,
                            "achieved": achieved,
                        })
                        yield _sse_format("perf", out).encode("utf-8")
                except Exception:
                    pass
                if i % 5 == 0:
                    # Stats (Firestore first, then GCS fallback)
                    try:
                        raw_count = sum(1 for _ in bucket.list_blobs(prefix="raw/"))
                        now = datetime.utcnow()
                        since = now - timedelta(hours=24)
                        sum_count = 0
                        series = []
                        try:
                            qn = fs.collection("news").where("ts_utc", ">=", since.isoformat(timespec="seconds"))
                            docs = list(qn.stream())
                            sum_count = len(docs)
                            if sum_count > 0:
                                buckets = {}
                                for d in docs:
                                    data = d.to_dict() or {}
                                    ts = data.get("ts_utc")
                                    try:
                                        t = datetime.fromisoformat(str(ts))
                                    except Exception:
                                        continue
                                    key = t.replace(minute=0, second=0, microsecond=0).isoformat()
                                    buckets[key] = buckets.get(key, 0) + 1
                                series = sorted([{ "ts": k, "count": v } for k, v in buckets.items()], key=lambda x: x["ts"])
                        except Exception:
                            sum_count = 0
                            series = []
                        if sum_count == 0:
                            try:
                                buckets = {}
                                sc = 0
                                for b in bucket.list_blobs(prefix="summary/"):
                                    if b.name.endswith("/"):
                                        continue
                                    try:
                                        data = json.loads(b.download_as_text())
                                        ts = data.get("ts_utc")
                                        t = datetime.fromisoformat(str(ts)) if ts else (b.updated or now)
                                    except Exception:
                                        t = b.updated or now
                                    if t < since:
                                        continue
                                    sc += 1
                                    key = t.replace(minute=0, second=0, microsecond=0).isoformat()
                                    buckets[key] = buckets.get(key, 0) + 1
                                sum_count = sc
                                series = sorted([{ "ts": k, "count": v } for k, v in buckets.items()], key=lambda x: x["ts"])
                            except Exception:
                                pass
                        yield _sse_format("stats", {"bucket": BUCKET, "raw_count": raw_count, "summary_count": sum_count, "series": series}).encode("utf-8")
                    except Exception:
                        pass
                    # News (with GCS fallback when Firestore empty)
                    try:
                        q = fs.collection("news").order_by("ts_utc", direction=firestore.Query.DESCENDING).limit(20)
                        items = []
                        for d in q.stream():
                            x = d.to_dict() or {}
                            items.append({"uid": x.get("uid", d.id), "ts_utc": x.get("ts_utc"), "summary": x.get("summary", ""), "sentiment": x.get("sentiment", 0), "impact": x.get("impact", 1)})
                        if not items:
                            blobs = list(bucket.list_blobs(prefix="summary/"))
                            blobs.sort(key=lambda b: getattr(b, 'updated', datetime.utcnow()), reverse=True)
                            for b in blobs[:20]:
                                try:
                                    data = json.loads(b.download_as_text())
                                    items.append({
                                        "uid": data.get("uid", b.name),
                                        "ts_utc": data.get("ts_utc"),
                                        "summary": data.get("summary", ""),
                                        "sentiment": int(data.get("sentiment", 0) or 0),
                                        "impact": int(data.get("impact", 1) or 1),
                                    })
                                except Exception:
                                    continue
                        yield _sse_format("news", {"items": items}).encode("utf-8")
                    except Exception:
                        pass
                    # Transactions
                    try:
                        from execution.oanda_pnl import fetch_transactions_since, _parse_time
                        start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                        start = start.replace(tzinfo=datetime.utcnow().astimezone().tzinfo)
                        txs = fetch_transactions_since(start)
                        items = []
                        for t in txs[-30:]:
                            tt = _parse_time(t.get("time"))
                            items.append({"id": t.get("id"), "time": tt.isoformat() if isinstance(tt, datetime) else t.get("time"), "type": t.get("type"), "instrument": t.get("instrument"), "pl": t.get("pl")})
                        yield _sse_format("tx", {"items": items}).encode("utf-8")
                    except Exception:
                        pass
                # keepalive
                yield b": keepalive\n\n"
                i += 1
                time.sleep(2)
            except GeneratorExit:
                break
            except Exception:
                time.sleep(2)
                continue
    headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-store", "Connection": "keep-alive"}
    return Response(gen(), headers=headers)


@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
