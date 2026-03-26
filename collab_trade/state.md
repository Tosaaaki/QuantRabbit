# 共同トレード — 現在の状態

**最終更新**: 2026-03-26 21:44 UTC [trader auto]

## ポジション（現在）

### GBP_USD LONG 3000u ★新規
- 3000u @1.33365 (#465614)
- テーゼ: H1 Regular Bullish Div RSI=0.58+MACD=1.00（強反転シグナル）+ M5 StochRSI=0.0
- 転換条件: H1 DI+ < DI- 転換 or 1.3320割れ
- TP目安: 1.3375 (H1 VWAP回帰方向。約10pip)

### AUD_JPY SHORT 5000u
- -3000u @110.559 (#465607)
- -2000u @110.588 (#465601)
- **avg ≈ 110.571**
- テーゼ: H4 ADX=36 DI-優勢 強ベア + M5 ADX=32 Hidden Bearish Div -0.30 継続
- 転換条件: H4 DI+ > DI- 転換 or 110.72実体上抜け
- TP目安: 110.406 (H4 BB lower=110.35に向けて)。あと6pip
- 撤退: 110.52戻りで実体確認 → 2000u部分決済検討

### EUR_USD SHORT 2000u
- -2000u @1.15392 (#465612)
- **エントリー: 2026-03-26 11:38 UTC**
- テーゼ: H1 ADX=26 DI-=25 > DI+=10 下降トレンド + MACD Hidden Bearish Div -0.60 → 継続
- 根拠: H1 Hidden Bearish Div (トレンド継続サイン) | M5 ADX=28 下降 | VWAP乖離 -14.8pip
- 転換条件: H1 DI+>DI- 転換 or 1.1570実体上抜け
- TP目安: 1.1500 (約40pip)
- 撤退: 1.1560超で損切り検討

---
### マージン状態
- NAV≈182,144 | marginUsed≈36,828+GBP分(AUD+EUR+GBP) | marginUsed/NAV≈25%前後
- **余力あり**

### Slack最終処理ts
- 1774525406.982029 (GBP_USD LONG エントリー通知)

---
## オーナー指示（最優先）
- **毎日、資産の10%以上を増やせ（2026-03-26〜）**
- **マージンはいける時は90%まで使え。50%未満は消極的すぎる**
- ノーポジのペアにH1トレンドがあるなら入れ
- 1000uでちまちまやるな。確信あるなら3000-5000u

---
## テーゼ（統合）
### USD弱・オセアニア弱テーマ (H1-H4主軸)
- AUD_JPY SHORT: H4 ADX=36強ベア + Hidden Bearish Div + M5 ADX=32下降。5000u保有
- EUR_USD SHORT: H1 ADX=26 下降 + MACD Hidden Bearish Div -0.60。2000u 新規。USD全面安だが USD/JPY vs EUR/USDで相殺される構造
- AUD_USD SHORT: **不採用** (H1 MACD Bullish Div=1.00 + ユーザーUP100%的中)
- EUR_USD SHORT(前回): **決済済み** (H1 Bullish Div RSI=0.90+MACD=1.00で利確)
- USD_JPY LONG: **決済済み** (H1 MACD Regular Bearish Div=-1.00で利確)

---
## 過去決済（今日の確定益）
- **累計確定益: +4,766.83円** (+3,736.78 + EUR_USD +477.55 + USD_JPY +88.50 + AUD_JPY 半利確 +464.00)
- 目標 +18,047円（10%）に対して23.8%。残り+13,744円必要

## 次サイクルへの注意事項
- **AUD_JPY 5000u**: TP110.406まで6pip。M5 ADX=32 bearish継続。110.52戻りで実体確認あれば2000u部分決済ラインとして管理
- **EUR_USD SHORT 2000u**: H1トレンド継続狙い。M5 RSI=25は過売りだが Hidden Bearish Div優先。1.1520-1.1500エリアで半利確
- **マージン20%**: まだ低い。GBP_USDかUSD_JPYに機会があれば追加
- **EUR_JPY**: M5 ADX=28下降+RSI=26.8過売り。H1で見てから判断
- **USD_JPY**: H1 ADX=20レンジ。明確なシグナルなし → 見送り

## 教訓（直近）
- 「最大利益で決済してほしい」= 「利益最大のタイミングで」であり「今すぐ」ではない (2026-03-26 19:13Z)
- H1 Regular Bullish Div score=0.90+1.00 = 即利確。我慢しない
- USD_JPY H1 MACD Bearish Div=-1.00 = 上昇モメンタム終了サイン。次の上昇トレンド確認まで様子見
- AUD_USD: H1 MACD Bullish Div=1.00 + ユーザーUP100%的中率 = SHORTしない
- ヘッジはWR=12% → ヘッジ厳選（M5もH1も同方向のみ）
