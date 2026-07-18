# 仮想マーケット環境 (QR_VIRTUAL_MARKET_SESSION_V1)

オペレーター指示「現実と同じ仮想環境でトレードしまくる。裁量の再現は担当エージェントがやる」の実装。

## 構成

- `src/quant_rabbit/virtual_broker.py` — OANDAメカニクスの仮想ブローカー (13テスト)
  - 約定は**フィードが供給した実気配のタッチ時のみ**。成行=実ask/bid、指値/TP/SL=実気配到達時
    (ギャップ時はトレーダーに不利な側)。価格合成なし
  - 両建てネッティング証拠金 (大きい側のみ)、レバ25x、証拠金使用率100%で全玉強制ロスカット
  - 非JPYペアのPnLは最新USD_JPY midでJPY換算 (宣言済み近似)
  - 全アクション・全約定を、原因となった気配ごとhash-chain台帳に記録
  - 実ブローカーへの発注経路は構造的に存在しない
- `scripts/run-virtual-market-session.py` — セッションデーモン
  - `--feed live`: 本口座ライブ気配を5秒毎ポーリング (壁時計時間、閉場/stale時は約定・注文処理を拒否して記録)
  - `--feed replay`: 封印M1コーパス (2020〜2026、USD_JPY/EUR_USD) を時刻順に1バー=OHLC 4気配で配信。
    simクロック=史実時刻、カーソルより先の情報はstateに存在しない (lookahead構造的に不可)
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

- 2026年1月以降のリプレイ期間はモデル知識カットオフ後 → 裁量エージェントの後知恵汚染が最小
- 採点は台帳をそのまま供給 (prospective registry / supervision_outcome_scorer と接続可)
- 受動シャドー環境 (`run-live-shadow-environment.py`) は機械ワーカー観測用として併存

## ボット搭載 (W44追補)

`--bot golden_burst` でワーカーボットがセッション内で稼働 (エージェントと同一ブローカー・同一台帳)。
`--bot golden_burst_blindspread` はライブ忠実構成 (2025-12のライブ機はspread monitor不在でスプレッド盲目)。

### 12/9 黄金日の環境内実証 (相互検証)

| 構成 | 決済 | 勝ち | net |
|---|---|---|---|
| ライブ実績 | 53 | 53 | +24,542 |
| bot (実スプレッド可視) | 1 | 0 | -291 (戦略自身のゲートが拒否) |
| bot (ライブ忠実・盲目) | **54** | 3 | -6,942 (SL51本) |

取引数54≒ライブ53で環境忠実性を実証。勝敗の全差分は「ライブはタイトSLを付けていなかった」
(ライブ53件全部TP決済) に帰着 — M1ベクトルリプレイ (58取引14勝) と独立に一致し、
2エンジンの相互検証が成立。新ボットは GoldenBurstBot と同形のクラスを足すだけ。
