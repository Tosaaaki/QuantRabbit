import os
import logging
from datetime import datetime, timezone, timedelta

from flask import Flask
from google.cloud import firestore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
fs = firestore.Client()


@app.route('/run', methods=['GET'])
def run_once():
    try:
        # Lazy import to keep container light when not used
        from execution.oanda_pnl import compute_daily_pnl
        import asyncio
        from execution.account_info import get_account_summary

        # セッション開始はJST 06:00（UTC+9 の 06:00）
        JST = timezone(timedelta(hours=9))
        now_jst = datetime.now(JST)
        base_day = now_jst.date()
        if now_jst.hour < 6:
            # 当日06:00前は前日06:00スタート
            base_day = (now_jst - timedelta(days=1)).date()
        start_jst = datetime(base_day.year, base_day.month, base_day.day, 6, 0, 0, tzinfo=JST)
        start_utc = start_jst.astimezone(timezone.utc)

        # 指定セッション開始を与えて集計
        res = compute_daily_pnl(start_utc)
        # 結果のdateはJST基準日に更新
        res['date'] = base_day.isoformat()

        # Attach day_start_nav if present in status or keep existing
        try:
            st = fs.collection('status').document('trader').get()
            if st.exists:
                dsn = (st.to_dict() or {}).get('day_start_nav')
                if dsn is not None:
                    res['day_start_nav'] = float(dsn)
        except Exception:
            pass

        doc_ref = fs.collection('perf').document('daily')
        # Merge with existing day_start_nav if missing
        cur = doc_ref.get().to_dict() if doc_ref.get().exists else {}
        if res.get('day_start_nav') is None and cur:
            dsn = cur.get('day_start_nav')
            if dsn is not None:
                res['day_start_nav'] = float(dsn)
        # If still missing, set to current NAV as baseline for the day (first run of the day)
        if res.get('day_start_nav') is None:
            try:
                acc = asyncio.run(get_account_summary())
                res['day_start_nav'] = float(acc.get('NAV', 0.0))
            except Exception:
                pass
        # Read current NAV for equity-based progress
        try:
            acc = asyncio.run(get_account_summary())
            nav_now = float(acc.get('NAV', 0.0))
            res['nav_now'] = nav_now
        except Exception:
            nav_now = None

        dsn = res.get('day_start_nav')
        if dsn and nav_now is not None and dsn > 0:
            res['equity_change'] = round(nav_now - dsn, 2)
            res['equity_progress_pct'] = (nav_now - dsn) / dsn
        # 参考までに実現ベースの進捗も返す
        if res.get('day_start_nav'):
            res['realized_progress_pct'] = (res.get('net_pl', 0.0) or 0.0) / float(res['day_start_nav'])

        # Write with merge=True so we never wipe existing keys unintentionally
        doc_ref.set(res, merge=True)
        # Mirror latest
        fs.collection('perf').document('daily_meta').set({'updated': datetime.utcnow().isoformat(timespec='seconds')})
        logging.info('PNL updated: %s', res)
        return 'OK', 200
    except Exception as e:
        logging.error('pnl_runner error: %s', e)
        return 'ERROR', 500


@app.route('/healthz', methods=['GET'])
def healthz():
    return 'ok', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
