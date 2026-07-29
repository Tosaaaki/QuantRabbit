# QuantRabbit｜現行4室監査と2025戦略ワーカー A/B Paper

## 実行境界

- Project: `qr-trading` / QuantRabbit
- repository / Git top-level / pwd: `/Users/tossaki/App/QuantRabbit-worktrees/dojo-dual-eval`
- branch: `codex/dojo-dual-eval`
- environment: DOJO Paper / replay only
- authority: `live_permission=false`, external broker mutation forbidden, `order_authority=NONE`
- current four-room experiment: `dojo-range-direction-pair-20260728-v1`
- independent A/B experiment: `dojo-legacy-worker-ai-ab-20260729-v1`
- existing four rooms were not stopped or modified.

## Profit-protection verdict

**現行4室は、まだ利益を守れていない。**

2026-07-29 10:30 JST の観測値は、実現 `+957.43円`、含み
`-2,439.30円`、現在価値 `-1,481.87円`。ledger再構成でも各室3件、
合計12件のopen tradeを確認し、12件すべて `sl=None` だった。ledger eventの
丸め値合計は `+957.41円` で、観測snapshotとの差は `0.02円`。

11:04 JST前後の再確認では、実現 `+1,585.34円`、含み `-2,947.09円`、
現在価値 `-1,361.75円`。実現益は増えたが、含み損がそれを超えており、
全12ポジションが引き続きSLなしだった。

旧PAPER supervisorのcwdは正しかったが、promptのrooms rootだけが停止済み
`episode-s5-outcome` を指していた。現在の
`dojo-dual-eval/research/data/dojo_paper_rooms_v1` へ修正し、automation
inventory syncを dry-run → apply → dry-run で実施、`residual_changes=0`。

## 2025年ワーカー発掘

コード、2025年Git履歴、結果台帳
`backtest_20251001_20251022_full.json` を突合した。

| priority | worker | 旧台帳 net pips | PF | trades | max DD pips | 判定 |
|---:|---|---:|---:|---:|---:|---|
| 1 | TrendMA | +495.25 | 1.4896 | 82 | 173.03 | A/B対象 |
| 2 | PulseBreak | +25.40 | 1.5248 | 26 | 16.60 | A/B対象 |
| 3 | M1Scalper | +323.40 | 1.0811 | 1,560 | 163.70 | 高コスト感応、replay対象 |
| 4 | RangeFader | +1.35 | 1.1500 | 6 | 9.00 | sample不足で保留 |

2025年履歴には、10月のbacktester/worker導入、PulseBreak gate調整、
TrendMA投影統合、M1Scalper ATR連動TP/SL、12月のworker別exit/tech
overlayが残る。アーカイブはread-onlyで、旧scheduler/order helperは再利用していない。

## 同一条件 A/B 高速replay

条件は全戦略共通で、初期資金20万円、2025-10-01〜10-22の同一M1相場、
同一entry cohort、1取引リスク1%、往復コスト0.8 pips。先に機械replayを
完走し、各戦略のworst loss 5件だけをfresh modelが後段評価した。
84-cell queueは使用していない。

AI方針は、15分方向確認、逆行時entry抑制、高ボラ時0.5倍、0.6R建値、
0.8Rで半分決済、1R以降ATR trailing、短期逆行3本で途中離脱。
逆方向へ反転して新規riskを増やす判断は禁止した。

| strategy | arm | net JPY | PF | expectancy JPY | max DD JPY | giveback | trades | AI decisions | AI差（cost前） | 採否 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PulseBreak | A Bot | +2,053.53 | 1.0793 | +78.98 | 10,428.37 | 0.0000 | 26 | 0 | — | 対照継続 |
| PulseBreak | B AI | +2,530.95 | 1.2252 | +97.34 | 5,007.92 | 0.0000 | 26 | 91 | +477.42 | `AI_PAPER_CONTINUE` |
| TrendMA | A Bot | +31,107.84 | 1.3577 | +384.05 | 14,460.00 | 0.2423 | 81 | 0 | — | 対照継続 |
| TrendMA | B AI | -8,585.72 | 0.5278 | -106.00 | 10,047.49 | 11.6436 | 81 | 266 | -39,693.56 | AI方針reject、Paper観察のみ |
| M1Scalper | A Bot | -352,351.19 | 0.8067 | -229.54 | 364,030.90 | 170.4038 | 1,535 | 0 | — | cost後reject |
| M1Scalper | B AI | -57,206.60 | 0.5079 | -37.27 | 57,526.60 | 0.0000 | 1,535 | 1,886 | +295,144.59 | 損失圧縮のみ、採用せず |

AIのplatform costは実測値を取得できないため `未計測`。上表のAI差は
AI cost控除前であり、経済判定全体は
`UNDETERMINED_AI_COST_MISSING` のまま。これを利益確定とは扱わない。

## 継続Paper

適格なTrendMAとPulseBreakを、既存4室と別experimentで4室起動した。

- `trendma-bot-only`
- `trendma-ai-inventory`
- `pulsebreak-bot-only`
- `pulsebreak-ai-inventory`

各AI室にはvirtual broker ledgerとは別のhash-chain
`ai_decisions.jsonl` があり、`ROOM_START` から判断数を記録する。
両armは20万円、同じslippage/financing、同じUSD_JPY read-only quote
sourceで継続し、2026-08-29までのroom windowを設定した。3時間の
replay工程で停止しない。

起動直後はOANDA quote timestampがローカル時刻より約数十秒古い/未来側に
見える既存のfreshnessずれにより
`INCOMPLETE_OR_STALE_QUOTES: refusing actions`。これはfail-closedで、
4室のprocessは稼働中だが、fresh quoteが通るまで新規Paper actionを出さない。

## 成果物

- `high_information_windows.json`: loss前後だけのfresh AI後段評価packet
- `comparison.json`: 戦略別A/B取引、集計、source/policy hash
- `paper_rooms.json`: 独立4室のauthority/cost/config
- `config/dojo_legacy_ai_inventory_policy_v1.json`: frozen Paper-only AI policy
- `bots/legacy_worker_paper_bot.py`: TrendMA/PulseBreakのforward Paper worker
- `scripts/run-dojo-legacy-worker-comparison.py`: 高速replay
- `scripts/run-dojo-legacy-paper-room.py`: 独立room launcher

## 残るリスク

- 現行4室の12ポジションはSLなしで、current valueはまだ負。停止や強制決済は
  行っていない。
- TrendMAに同一AI方針を一律適用すると利益を破壊した。戦略別policyへ分離するまで
  adoptionしない。
- M1Scalperの旧gross edgeは0.8-pip往復コストで消えた。frequencyだけで採用しない。
- 高速replayは昨年の固定entry cohort比較で、AI entry抑制後の代替signalを
  再生成しない保守的counterfactual。継続Paperでforward差を検証する。
- AI cost未計測のため最終経済判定は未確定。
