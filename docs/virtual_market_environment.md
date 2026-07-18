# DOJO（道場） — 実市場忠実仮想トレード環境

正式名称: **DOJO**。エージェント（裁量）もボット（ワーカー）も同じ実市場メカニクスの土俵で
腕を磨く研究場所。DOJOは仮説を前向き評価へ送れるが、単独で実弾権限や昇段を作れない。
契約名: QR_VIRTUAL_MARKET_SESSION_V1。

オペレーター指示「現実と同じ仮想環境でトレードしまくる。裁量の再現は担当エージェントがやる」の実装。

## 構成

- `src/quant_rabbit/virtual_broker.py` — OANDAメカニクスを近似する仮想ブローカー
  - 約定は**フィードが供給した実気配のタッチ時のみ**。成行=実ask/bid、指値/TP/SL=実気配到達時
    (ギャップ時はトレーダーに不利な側)。価格合成なし
  - 両建てネッティング証拠金 (大きい側のみ)、レバ25x、証拠金使用率100%で全玉強制ロスカット
  - 非JPYペアのPnLは最新USD_JPY midでJPY換算 (宣言済み近似)
  - market/stop entry、SL、手動close、margin closeoutへ宣言済みの不利slippageを適用。LIMIT/TPは
    quoteからの改善を削っても保護価格を越えて悪化約定させず、margin closeoutまでfinancingを課金
  - 換算気配は約定quote sequence以前のwatermarkだけを使用し、fill台帳へrate/source timestamp/phaseを記録
  - 全アクション・全約定を、原因となった気配ごとhash-chain台帳に記録
  - 実ブローカーへの発注経路は構造的に存在しない
- `scripts/run-virtual-market-session.py` — セッションデーモン
  - `--feed live`: 本口座ライブ気配を5秒毎ポーリング (壁時計時間、閉場/stale時は約定・注文処理を拒否して記録)
  - `--feed replay`: 封印M1コーパスを `epoch → intrabar phase → pair` で同期配信。
    simクロック=史実時刻、同時刻の別ペアを含めカーソルより先の情報はstateに存在しない
  - `SESSION_START` はcommit、コード/bot/config/corpus shard SHA、期間、pairs、intrabar、cost、
    granularity、snapshotを封印する。空corpusや不正な非有限costはfail closed
  - V2 snapshotはquote履歴、feed cursor、ledger tipとID sequenceを束縛し、旧schema、ledger不一致、
    時刻巻き戻し、既存ID再利用を拒否する。bot内部stateを永続化していないためbot付き途中resumeは拒否
  - `--step`: リプレイをターン制に (inbox/STEP でバー送り) — 裁量エージェントの熟考用

## 担当エージェントの操作方法

セッションdirの `state.json` を読み、`inbox/` にJSONを置くだけ:

```json
{"action":"MARKET","pair":"USD_JPY","side":"LONG","units":10000,"tp_pips":5,"sl_pips":null}
{"action":"LIMIT","pair":"USD_JPY","side":"SHORT","units":5000,"price":156.80,"tp_pips":10}
{"action":"CLOSE","trade_id":"T000001"}
{"action":"CANCEL","order_id":"O000001"}
{"action":"SET_EXIT","trade_id":"T000001","tp_price":156.5,"sl_price":null}
```

処理済みは `inbox/processed/` へ改名 (削除なし)。拒否は理由つきで台帳へ。

## 起動例

```bash
# 史実リプレイで大量練習 (2026年上半期、20バー/秒)
python3 scripts/run-virtual-market-session.py --feed replay \
  --session-dir <dir> --from 2026-01-05T00:00:00 --to 2026-07-01T00:00:00

# ターン制 (裁量エージェント用)
... --feed replay --step --from 2025-12-09T00:00:00 --to 2025-12-10T00:00:00

# ライブ気配ペーパートレード (8時間)
QR_OANDA_ENV_FILE=... python3 scripts/run-virtual-market-session.py \
  --feed live --session-dir <dir> --minutes 480
```

## 注意

- モデル知識カットオフ後の期間でも、日付・正確な価格系列・後続packetから再識別や答え漏洩が起こる。
  cutoff日は構造的な盲検保証ではなく、一日packet・日付隠蔽・answer key隔離を別途必須とする
- 採点は台帳をそのまま供給 (prospective registry / supervision_outcome_scorer と接続可)
- 受動シャドー環境 (`run-live-shadow-environment.py`) は機械ワーカー観測用として併存

## 前向き評価の操作

workerは候補・コード・依存・期間・市場mechanicsを開始前に固定し、UTC日を順番どおり封印する。
現在のsmokeは既に `precommit` と `start` を完了しているため、同じrun-dirへ再実行して上書きしない。

```bash
PYTHONPATH=src python3 scripts/run-dojo-worker-forward.py status \
  --run-dir research/forward/dojo-worker-forward-smoke-v1

PYTHONPATH=src python3 scripts/run-dojo-worker-forward.py seal-day \
  --run-dir research/forward/dojo-worker-forward-smoke-v1 \
  --ordinal <1..14> --source-manifest <sealed-source-manifest.json>

PYTHONPATH=src python3 scripts/run-dojo-worker-forward.py finalize \
  --run-dir research/forward/dojo-worker-forward-smoke-v1 \
  --result-manifest <fixed-denominator-results.json>
```

`seal-day` は期間内の全sourceをmanifestへ束縛してから使う。欠測日、閉場日、遅延は黙って飛ばさず、
契約どおりreceiptまたはblockerとして残す。period-endは各strategy ownerの注文をcancelしてから同ownerの
建玉だけをcloseし、全candidateをfully-resolved balanceで採点する。

AIは `run-dojo-ai-experiment.py build-manifest` で90セルを応答前に固定し、各responseを
`seal-dojo-ai-response.py` でanswer keyをmountする前に封印する。`score` はexact 90-cell phaseだけを受理し、
欠測やschema failureを分母から落とさない。現在は登録・評価器実装までで、新規responseは0件である。

## ボット搭載 (W44追補)

`--bot golden_burst` でワーカーボットがセッション内で稼働 (エージェントと同一ブローカー・同一台帳)。
`--bot golden_burst_blindspread` はライブ忠実構成 (2025-12のライブ機はspread monitor不在でスプレッド盲目)。

### 12/9 黄金日の環境内実証 (相互検証)

| 構成 | 決済 | 勝ち | net |
|---|---|---|---|
| ライブ実績 | 53 | 53 | +24,542 |
| bot (実スプレッド可視) | 1 | 0 | -291 (戦略自身のゲートが拒否) |
| bot (ライブ忠実・盲目) | **54** | 3 | -6,942 (SL51本) |

取引数54≒ライブ53は、この一日のsignal countを近似再現したに留まる。勝敗が大きく異なるため
環境忠実性や二エンジンの相互検証の証明ではない。差の有力要因は、ライブ53件がタイトSLを
付けず全件TP決済だったこと。新ボットは GoldenBurstBot と同形のクラスを足すだけ。


## どのエージェント/ボットでも実行可能 (W45)

**ボット**: 任意の .py ファイルを `--bot-module <file.py>:<Class>` でロード。契約は
`bots/example_bot.py`（テンプレート）の通り: `__init__(broker)` + `on_bar_closed(pair, bar, epoch)`。
live/replay両フィードで同一に発火。DOJO本体の編集は不要。

**プロンプト駆動エージェント**: `state.json` を読み、`inbox/` にJSONを置くだけ（一時ファイル→
rename のアトミック書き込み推奨。書き込み中ファイルは0.5秒の猶予で保護）。`--step` でターン制。

## 監査で発見・修正した穴 (2026-07-19)

1. バー内順序のロング楽観バイアス → `--intrabar OHLC/OLHC` 両側ブラケットで宣言化
2. ライブモードでボット未発火 → ライブM1組み立て＋コールバック追加
3. inbox書き込み競合で注文消失 → 0.5秒猶予＋アトミック書き込み規約
4. 全史リプレイのメモリ爆発 → 年単位ストリーミング
5. ボットのハードコード → プラグインローダ（実証: 外部ファイルの新作ボットが無改修で29取引）

## 残存する宣言済み制約（正直な限界）

- **バー内経路は合成** — 真のtick経路は不明。両タッチバーはOHLC/OLHCの両走で挟むこと
- **ライブは5秒ポーリング** — ポーリング間の極値は見えない。利確を逃す場合は保守的だが、
  adverse stop/closeoutを見逃す場合は楽観的になり得るため一方向の保守性を主張しない
- EUR_USDのJPY換算はUSD_JPYの気配が必要（ペアに必ずUSD_JPYを含める）
- スリッページは固定の不利幅であり、実弾の状態依存・裾の厚い滑り分布を再現するものではない
- `--step` 中の成行はバーclose気配で約定（熟考時間は価格を止めない現実と異なる点は宣言済み）

## 証拠の状態（2026-07-19再監査）

過去のW46〜W53成績証拠は、position ownership混線、全出口へのcost未適用、VAL/S5窓再利用、
period-end open position未採点、trial上書き可能性が見つかったため、すべて `INVALIDATED` である。
戦略アイデアだけを再試験可能な `HYPOTHESIS` として残し、現在の昇段生存者は0。修理後runnerは
owner分離、append-only trial、非重複TRAIN/VAL/FINAL、
OHLC/OLHC両方、terminal equity/MTM、zero-trade拒否を必須とする。研究プログラムと3倍判定の正本は
`docs/DOJO_PROGRAM.md` を参照する。
