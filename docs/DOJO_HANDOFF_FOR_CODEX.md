# DOJO 引き継ぎ書（Codex向け・完全自己完結）

作成: 2026-07-19 Claude (Fable)。オペレーター指示「Codexがわかるように引き継いで」。
これ1枚でDOJOの全体・使い方・確定済み結論・次の仕事がわかるように書く。

## 1. DOJOとは何か

**実市場と同一メカニクスの仮想取引環境**。OANDA準拠の仮想ブローカーで、
ボット（戦略ワーカー）とAIエージェントの両方が同じ口座・同じ約定エンジンで取引できる。
目的: 戦略・裁量の候補を、**実弾前に・嘘の入り込めない条件で**審判すること。

- 場所: git worktree `/private/tmp/QuantRabbit-episode-outcome`（branch `codex/episode-s5-outcome`）
- 本体repo（`/Users/tossaki/App/QuantRabbit`）とはブランチが別。混ぜない。

## 2. 構成ファイル

| ファイル | 役割 |
|---|---|
| `src/quant_rabbit/virtual_broker.py` | 仮想ブローカー核（15テスト）。両建てネッティング証拠金・レバ25x・成行/指値/STOP/TP/SL・証拠金不足の注文拒否・100%で強制ロスカット・JPY換算（USD_JPY等の換算フィード必須）・hash-chain台帳。**約定はフィードが供給した実気配のタッチ時のみ**（価格合成なし・ギャップは不利側） |
| `scripts/run-virtual-market-session.py` | セッションデーモン。`--feed replay`（M1/S5史料を時刻順配信・lookahead構造不可）/ `--feed live`（本口座気配5秒ポーリング・閉場/stale拒否）。`--step`でターン制。`--slippage-pips 0.3 --financing-pips-day 0.8`で強化関門。`--bot-module`でボット同居 |
| `bots/lab_bot.py` | パラメータ化ワーカー（11戦略族実装済み）。config は env `DOJO_BOT_CONFIG`(JSON) |
| `bots/combo_bot.py` | 複数の手を同一口座で同居運転（env `DOJO_BOT_COMBO`=configリスト） |
| `scripts/run-dojo-lab.py` | 宣言済みグリッド一括審判（TRAIN→VAL→OLHCブラケット） |
| `scripts/run-pair-adaptation-lab.py` | ペア別幾何適応（screen→専用服→VAL/S5処刑）。換算フィード同席は`feed_pairs()`が担保 |
| `scripts/run-live-shadow-environment.py` | 受動シャドー環境（黄金日MomentumBurst観測用・二重出口簿記） |
| `scripts/oanda_history_fetch_m1.py` | M1取得（凍結fetcherの削除ガード準拠派生。原本 `oanda_history_fetch.py` は編集禁止） |
| `docs/virtual_market_environment.md` | エージェント操作書（inbox JSONプロトコル） |
| `docs/design_weakness_ledger_20260718.md` | **W1〜W55 全審判記録（正本）。必読** |

## 3. データ（読み取り専用・削除禁止）

- M1 bid/ask: `/Users/tossaki/App/QuantRabbit-live/logs/replay/oanda_history_m1_2020_2026`
  （USD_JPY/EUR_USD 2020-2026、他26ペア 2024-2026。シャード名 `<PAIR>_M1_BA_<from>_<to>.jsonl.gz`）
- S5 bid/ask: `.../oanda_history`（28ペア 2026-05-08〜07-17。約定現実性の処刑テスト用）
- M5: `.../oanda_history_m5_2020_2026`（28ペア6.5年）
- Python: pandas必要なスクリプトは `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`

## 4. 実行例

```bash
cd /private/tmp/QuantRabbit-episode-outcome
# ボット入り高速リプレイ（強化関門）
DOJO_BOT_CONFIG='{"signal":"range_fade_limit","pairs":["USD_JPY"],"tp_pips":6,"sl_pips":null,"ceiling_min":480,"max_concurrent":3,"per_pos_lev":4.3,"atr_floor_pips":1.0,"fade_atr":1.2,"eff_max":0.2}' \
PYTHONPATH=src /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  scripts/run-virtual-market-session.py --feed replay --session-dir /tmp/dojo1 \
  --pairs USD_JPY --from 2026-05-10T00:00:00 --to 2026-07-04T00:00:00 \
  --bars-per-second 100000 --state-every 50000 --fast-ledger \
  --slippage-pips 0.3 --financing-pips-day 0.8 \
  --bot-module bots/lab_bot.py:Bot
# ライブペーパー（市場開場中のみ約定。発注権限なし）
QR_OANDA_ENV_FILE=/Users/tossaki/App/QuantRabbit-live/.env.local ... --feed live --minutes 480
```
台帳集計: ledger.jsonl の EXIT_*/CLOSE/MARGIN_* 行の pl_jpy を日次合算（既存ラボスクリプトの report() を流用）。

## 5. 確定済みの結論（W37〜W55、詳細は台帳）

- **床（機械の確定実力）**: USD_JPY 2手併走 = W_FADE（レンジ両面LIMITフェード, TP6, SLなし, 8h天井, 効率比≤0.2ゲート）+ W_SPIKE_FADE（2.5ATR髭LIMIT受け, TP=3ATR, SLなし）。**強化関門後 月+3.6%・worst日-6.3%・死なない**。W_ROUND / W_LADDER（有限1段ナンピン）も全関門通過（仮説級・薄い）
- **通貨宇宙**: 28ペア×専用幾何×全関門で審判済み — **払うのはUSD_JPYだけ**（換算バグで一度誤判定→修理済み・W51訂正参照）
- **サイズ**: どの構成でも月+5-7%に飽和。前線lev16×1玉、lev20は破滅圏。M1約定の複利倍率は楽観幻・引用禁止
- **死んだもの**: 追いかける型（burst系）全滅・伸ばす型（trailing）全滅・圧縮ブレイク死・多ペアfade死・AI構造のみ裁量死（クリーン実験24戦12勝=50%・月x0.93で床に負け。87%/34連勝はpacket lookaheadリークの幻 — W54-55訂正必読）
- **教訓（重大2件）**: ①盲検設計は「答えがコンテキストに物理的に存在しない」ことを構造で保証する ②非JPY/USDクォートのペアはUSD_JPY+クォート通貨JPYペアの換算フィード同席必須（無いと静かにサイズ0）
- **3倍の算数**: 市場供給の実測で「毎月3倍」は25xレバでは神の目でも不可（中央値月x1.69上限）。高ボラ月（供給8-14日）のみx3-4が可能。層C（読み）は機械では未証明 — 前向きのみ

## 6. 次の仕事（優先順）

1. **床のライブペーパー**: 月曜市場開場後、DOJO liveモードで W_FADE+W_SPIKE_FADE combo を回す（`DOJO_BOT_COMBO`）。台帳を毎日集計し、S5実測の月+3.6%が前向きでも出るか確認。※旧watcher（シャドー環境用）はプロセス終了で死んでいる可能性あり — 再起動必要
2. **前向き採点の継続**: 台帳→ prospective registry（`src/quant_rabbit/prospective_registry.py`）へ接続
3. **昇格判定**: 前向き2-4週プラスなら極小実弾（オペレーター承認必須。発注はqr-traderルーチンの管轄 — DOJO自体は永遠に発注しない）
4. コスト注意: DOJOのリプレイ/ボット運転はローカル計算のみでLLMコストゼロ。**LLM読み実験（サブエージェント大量起動）は高コストなので勝手にやらない**

## 7. 禁止事項

- 凍結スクリプト `scripts/oanda_history_fetch.py` の編集禁止（受領chainがsha固定）
- corpus（logs/replay/*）の削除・移動禁止
- DOJOから実ブローカーへの発注経路を作らない（構造的に存在しないことが価値）
- M1約定の複利倍率を成績として報告しない（S5強化関門値のみが公式値）
