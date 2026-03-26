# 記録ルール — チェック→注文→記録は同一動作

**エントリーの流れ: pretrade_check → 注文 → 4点記録。この5ステップは分割不可。**
**決済の流れ: preclose_check → 決済 → 4点記録。この流れも分割不可。**

## STEP 0a: pretrade_check（エントリー前に必ず実行）

```bash
cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 pretrade_check.py {PAIR} {LONG|SHORT}
```

- **HIGH判定** → サイズ半減 + SL必須。それでも入るか再考
- **LOW/MEDIUM** → 通常通り進む
- 結果をtrades.mdのエントリー記録に含める（`pretrade: LOW` 等）

## STEP 0b: preclose_check（決済前に必ず実行）

```bash
cd /Users/tossaki/App/QuantRabbit && python3 tools/preclose_check.py {PAIR} {SIDE} {UNITS} {含み損益円}
```

- 出力はテーゼの再確認と事実の提示。判断を奪わない
- 答えた上で決済するのはOK。答えずに反射で切るのがNG
- **決済理由はlive_trade_logに必ず明記**（`reason=H1_DI+逆転` 等）
- **理由なき決済 = ルール違反**

## STEP 1-4: 注文 → 4点記録（後回し禁止）

| ファイル | 何を書く |
|----------|---------|
| `collab_trade/daily/YYYY-MM-DD/trades.md` | エントリー・決済の詳細テーブル |
| `collab_trade/state.md` | 現在のポジション・テーゼ・確定益（外部記憶） |
| `logs/live_trade_log.txt` | トレード実行ログ（時系列） |
| `#qr-trades` (Slack通知) | エントリー/変更/決済をSlackに投稿 |

## Slack通知（4点目）

注文実行と同時に `slack_trade_notify.py` で `#qr-trades` に投稿する。

```bash
# エントリー
python3 tools/slack_trade_notify.py entry --pair {PAIR} --side {LONG|SHORT} --units {UNITS} --price {PRICE} [--sl {SL}] [--thesis "テーゼ"]

# 変更（半利確、SL移動、ナンピン等）
python3 tools/slack_trade_notify.py modify --pair {PAIR} --action "TP半利確" --units {UNITS} --price {PRICE} --pl "{PL}" [--note "残units, BE移動等"]

# 全決済
python3 tools/slack_trade_notify.py close --pair {PAIR} --side {LONG|SHORT} --units {UNITS} --price {PRICE} --pl "{PL}" [--total_pl "確定益合計"]
```

## state.md はスナップショットじゃない。ストーリーだ

**悪い例:** `USD_JPY: H1上昇トレンド(ADX32)。押し目買い狙い`
**良い例:**
```
## USD_JPY LONG テーゼ
- 読み: 円安方向。158.50→159.00を目指す
- 根拠: Fed hawkish hold + Iran risk-off → USD bid
- 転換条件: DXY 98.5割れ、米国債利回り急落、または158.30明確割れ
- 経過: 158.38→ナンピン158.37→半利確158.41
```

## ユーザー発言の記録

ユーザーが何か言ったら即 `daily/YYYY-MM-DD/notes.md` に**チャート状態込み**で記録。
`ユーザー: 「上がりそう」— M5で3本連続陰線後に長い下ヒゲ、BB下限タッチ、H1上昇中(ADX=32)、RSI=35`

**「ちゃんと記録してる？」と聞かれた時点で負け。**
