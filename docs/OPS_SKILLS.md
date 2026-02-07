# Ops Skills

## 1. スキル運用
- 日次運用/調査/デプロイ/リプレイ/リスク監査は専用スキルを優先利用する。
- スキル一覧: `qr-log-triage`, `qr-deploy-ops`, `qr-replay-backtest`, `qr-risk-guard-audit`, `qr-tick-entry-validate`
- 明示的に使う場合は `$qr-log-triage` のようにスキル名を付けて依頼する（自動発火も許容）。
- スキル定義は `~/.codex/skills/<skill>/SKILL.md` を参照する。

## 2. ポジション問い合わせ対応
- 直近ログを優先し最新建玉/サイズ/向き/TP/SL/時刻を即答。
- オープン無しなら「今はフラット」＋直近クローズ理由。
- サイズ異常時は決定した設定（`ORDER_MIN_UNITS_*` など）を明示。

## 3. 損益報告
- `sum(realized_pl)` (JPY) と `sum(pl_pips)` を必ず併記。
- 未実現損益は別枠で提示。
- JPY がマイナスなら勝ち扱いしない。
- UTC/JST を明記する。

## 4. マージン/エクスポージャ
- OANDA snapshot の total を使用し、手動玉を含めて確認する。

## 5. Tick照合（エントリー/EXIT精度検証）
- 目的: 「エントリーが早い/浅い」「ハードSLがタイト」「EXITが早すぎ」を切り分ける。
- 原則: **本番稼働は VM**。`trades.db`/tick は VM から取得し、ローカル `logs/*.db` だけで断定しない。
- 価格サイド: USD/JPY は **long の決済トリガは bid / short は ask**（SL/TP 到達判定も同様）。

手順（例）:
```bash
VM="./scripts/vm.sh -p quantrabbit -z asia-northeast1-a -m fx-trader-vm -t"

# 1) trades 抽出（UTCで指定。JSTは+9hで必ず併記する）
$VM sql -q "SELECT ticket_id,strategy_tag,units,entry_price,pl_pips,close_reason,open_time,close_time FROM trades WHERE instrument='USD_JPY' AND open_time>='2026-02-06T18:15:00+00:00' AND open_time<'2026-02-06T18:48:00+00:00' ORDER BY open_time;"

# 2) VMからDBとtickを取得（tickは bid/ask JSONL）
$VM scp --from-remote ~/QuantRabbit/logs/trades.db tmp/trades_vm.db
$VM scp --from-remote ~/QuantRabbit/logs/replay/USD_JPY/USD_JPY_ticks_YYYYMMDD.jsonl tmp/USD_JPY_ticks_YYYYMMDD.jsonl

# 3) tick照合レポート（SL-hit/TP-touch時刻、MAE/MFE、post-close TP touch）
python3 ~/.codex/skills/qr-tick-entry-validate/scripts/tick_entry_validate.py \
  --trades-db tmp/trades_vm.db \
  --ticks tmp/USD_JPY_ticks_YYYYMMDD.jsonl \
  --instrument USD_JPY \
  --open-from '2026-02-06T18:15:00+00:00' \
  --open-to   '2026-02-06T18:48:00+00:00'
```

解釈の目安:
- SL-hit が数秒〜数十秒で多発し、かつ TP-touch が数分以内に多い: SL 過小/フォロー不足/入りが早い可能性。
- TP-touch がほぼ無い: エントリーの方向/条件が悪い（精度課題）。
- `close_reason=MARKET_ORDER_TRADE_CLOSE` なのに post-close で TP-touch 多発: EXIT が早すぎる可能性。
