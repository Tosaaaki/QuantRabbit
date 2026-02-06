# Ops Skills

## 1. スキル運用
- 日次運用/調査/デプロイ/リプレイ/リスク監査は専用スキルを優先利用する。
- スキル一覧: `qr-log-triage`, `qr-deploy-ops`, `qr-replay-backtest`, `qr-risk-guard-audit`
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
