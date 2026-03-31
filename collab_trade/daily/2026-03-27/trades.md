# 2026-03-27 トレード記録

## 決済済み

| 時刻(UTC) | ペア | サイド | 数量 | エントリー | 決済 | PL | 根拠 |
|-----------|------|--------|------|-----------|------|-----|------|
| 早朝 | GBP_USD | LONG | 2000u | 1.33597 | 1.33602 | +15.93円 | H1 StochRSI=1.0+MACD hidden bearish div age=11+M5 double bearish div(RSI+MACD score=-1.00 age=5) |

| 16:26Z | AUD_JPY | SHORT close | 2000u | 110.380 | 110.194 | +372円 | H1 BB Lower=110.19タッチ+M5 CCI=-191.7 extreme oversold bounce前に半利確 |
| 16:26Z | AUD_JPY | SHORT close | 2000u | 110.362 | 110.194 | +336円 | 同上 #2 |

## 保有中（引き継ぎ）

| ペア | サイド | 数量 | avg | PL目安 | テーゼ |
|------|--------|------|-----|--------|--------|
| EUR_USD | SHORT | 12000u | 1.15350 | -2,720円 | H1 ADX=28 DI-=24>DI+=15 ベア継続。TP 1.1500。撤退: 1.1570実体上抜けorDI+逆転 |
| AUD_JPY | SHORT | 4000u | 110.371avg | -4円 | H1 ADX=23 DI-=21.6>DI+=8.8 ベア継続。M5 DI-=22.4。TP 110.10。撤退: 110.65明確上抜け |
| AUD_JPY | SHORT add-on | +2000u @110.362 | - | - | pretrade: LOW (勝率71%), H1+M5 同方向ベア。15:24Z |
| AUD_JPY | SHORT add-on | +2000u @110.337 | - | - | ~15:30Z add-on #3 |

## 新規エントリー（このセッション）

| 時刻(UTC) | ペア | サイド | 数量 | 価格 | トレードID | 根拠 | pretrade |
|-----------|------|--------|------|------|------------|------|---------|
| 15:39Z | GBP_USD | LONG | 3000u | 1.33434 | #465661 | M5 stoch_rsi=0.0(売られすぎ) + H1 DI+≈DI-(中立) + ユーザーUP予測100%的中率 | LOW (勝率50%) |
| 16:37Z | AUD_JPY | SHORT | 2000u | 110.178 | #465669 | H1 ADX=25.5 DI-=23.4>DI+=7.8 ベア継続。MACD Hidden Bearish age=12。M5 extreme oversold(bounce注意)。73%WR。LAZY alert対応 | LOW |

## 追加エントリー（traderセッション）

| 時刻(UTC) | ペア | サイド | 数量 | 価格 | TradeID | 根拠 | pretrade |
|-----------|------|--------|------|------|---------|------|---------|
| 17:08Z | AUD_JPY | SHORT add-on | 3000u | 110.171 | #465682 | H1 ADX=27 DI-=23>>DI+=8 強bear。M5 ADX=27 DI-=22>>DI+=12 同方向。AUD弱テーマ継続。MARGIN ALERT対応 | LOW WR=73% |

## 決済（このセッション）

| 時刻(UTC) | ペア | サイド | 数量 | 決済価格 | PL | 理由 |
|-----------|------|--------|------|---------|-----|------|
| 16:31Z | AUD_JPY | SHORT | 2000u | 110.192 | +290円 | M5 stoch=0.00/cci=-191.3 extreme oversold + vwap-42pip延長。TP=110.10まで6pip残だがbounce risk > remaining upside |

---
### [Automated] EUR_USD SHORT 全クローズ — H1 MACD Regular Bullish Div score=1.0

| # | Trade ID | Units | Entry | Close | PL |
|---|----------|-------|-------|-------|----|
| 1 | #465641 | -2000u | 1.15398 | 1.15364 | +108.35円 |
| 2 | #465639 | -3000u | 1.15386 | 1.15364 | +105.16円 |
| 3 | #465637 | -2000u | 1.15306 | 1.15364 | -185.57円 |
| 4 | #465622 | -3000u | 1.15284 | 1.15364 | -383.98円 |
| 5 | #465612 | -2000u | 1.15392 | 1.15364 | +89.23円 |

- **合計PL: -266.81円**
- 撤退理由: H1 MACD Regular Bullish Div score=1.0 (過去教訓: この状態で即利確しないと大損→教訓通り実行)
- pretrade: N/A (決済)

### AUD_JPY SHORT 3000u add-on #3 [22:32Z]
| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-27 22:32 UTC |
| エントリー | 110.146 (ask) |
| ID | #465684 |
| pretrade | LOW (73%WR) |
| テーゼ | AUD弱テーマ継続。H1 ADX=27 DI-=23>>DI+=7。AUD_USD H1 ADX=34も同方向でAUD全面安確認 |
| avg | 2000@110.178 + 3000@110.171 + 3000@110.146 = 8000u avg≈110.162 |
| TP | 109.80-110.00 |
| 撤退条件 | 110.65明確上抜け |

## オープンポジション追加

| 時刻(UTC) | ペア | サイド | 数量 | エントリー | tid | 根拠 | pretrade |
|-----------|------|--------|------|-----------|-----|------|---------|
| ~00:10Z | AUD_JPY | SHORT add-on#4 | 3000u | 110.100 | 465688 | H1 ADX=28 DI-=25>>DI+=11 ベア継続 + M5 Hidden Bearish Div + avg改善 | LOW(73%WR) |
| 17:55Z | GBP_USD | LONG | 1000u | 1.33235 | #465690 | M5 StochRSI=0.0 CCI=-228 extreme oversold at swing low。H1 range(ADX=20)。bounce trade。TP 1.3338(M5 BB mid)。SL mental 1.3310 | LOW(48%WR) |
| **18:11Z** | **GBP_USD** | **LONG CLOSE** | **1000u** | **1.33240** | #465696 | **M5 ADX=27 下降継続 + Regular Bearish Div score=-0.46 → バウンス失敗。+7.97円確定** |

### GBP_JPY SHORT 3000u #465694 [18:07Z]
| 項目 | 値 |
|------|-----|
| 時刻 | 18:07 UTC |
| エントリー | 212.819 (ask) |
| ID | #465694 |
| pretrade | LOW (82%WR) |
| テーゼ | M5 ADX=26 DI-=28>DI+=11 下降トレンド。JPY強テーマ継続（AUD_JPY SHORT好調と同方向）。GBP弱+JPY強の2方向圧力 |
| TP目安 | 212.50 (-30pip) |
| SL mental | 213.15 |

### EUR_USD SHORT 3000u #465692 [01:21Z]
| 項目 | 値 |
|------|-----|
| 時刻 | 01:21 UTC |
| エントリー | 1.15268 |
| テーゼ | H1 ADX=30.7 DI-=24.9ベア。前サイクルBullishDiv消滅、RSI Hidden Bearish継続。M5 ADX=27ベア |
| pretrade | MEDIUM (67%WR) |
| TP目安 | 1.1505-1.1510 |
| SL mental | 1.1545 |


### EUR_USD SHORT 2000u #465698 [18:23Z] — ADD-ON
| 項目 | 値 |
|------|-----|
| 時刻 | 18:23 UTC |
| エントリー | 1.15291 (add-on) |
| ID | #465698 |
| pretrade | MEDIUM (67%WR) |
| テーゼ | H1 ADX=31 DI-=24>>DI+=12 強ベア。RSI Hidden Bearish Div -0.60継続。既存3000u @1.15268と合算→5000u avg≈1.15277 |
| TP目安 | 1.1505-1.1510 |
| SL mental | 1.1545 |

---

## GBP_JPY SHORT 3000u — 決済 18:43Z

| 項目 | 内容 |
|------|------|
| 決済時刻 | 2026-03-27 18:43Z |
| 決済価格 | 212.875 (ask) |
| エントリー価格 | 212.819 |
| ユニット | 3000u |
| PL | **-168円** |
| tradeID | #465694→#465700 |
| 決済理由 | M5 Regular Bullish Div score=1.0 (age=8本) + H1 range ADX=15 + H1 Hidden Bullish Div 0.38 → ベアテーゼ崩壊 |
| 教訓 | M5 Regular Bullish Div 1.0 = 即クローズ判断正解。-168円で守れた |


## EUR_USD SHORT 3000u #465702 [add-on #3] [03:55Z]
| 項目 | 値 |
|------|-----|
| 時刻 | 03:55 UTC (12:55 JST) |
| エントリー | 1.15328 |
| avg後 | 1.15296 (8000u total) |
| 根拠 | H1 ADX=27.2 DI-=25.3>>DI+=12.9 強ベア継続。M5 StochRSI=1.0バウンス天井。LAZY alert対応 |
| pretrade | MEDIUM (67%WR) |
| SL_mental | 1.1545 |
| TP | 1.1505-1.1510 |

## GBP_JPY SHORT 2000u #465704 [19:09Z]
| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-27 19:09 UTC |
| エントリー | 212.896 (ask) |
| ID | #465704 |
| pretrade | LOW (75%WR) |
| テーゼ | M5 StRSI=1.00天井(MAX overbought) + H1 ADX=13.6 DI-=20.0>DI+=17.3 レンジ内弱ベア。LAZY alert対応。小サイズ(range) |
| TP目安 | 212.60-212.70 |
| SL mental | 213.20 |

## EUR_JPY SHORT 2000u #465710 [20:07Z]
| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-26 20:07 UTC |
| エントリー | 184.140 (ask) |
| ID | #465710 |
| pretrade | LOW (61%WR) |
| テーゼ | EUR弱テーマ延長。EUR_USD H1 ADX=32 DI-=25強ベア継続。EUR_JPY H1 MACD Hist=-0.013 bearish、EMAスロープ全マイナス。Div=0.00クリーン |
| TP目安 | 184.00-184.02 (H1 BB Lower) |
| SL mental | 184.50 (BB Upper) |
| マージン状況 | 57%→追加エントリー |

## AUD_JPY SEMI-TP 5000u @110.000 [UTC: $(date -u +%H:%M)]
| # | ペア | 方向 | Units | Entry | Close | PL | 根拠 |
|---|------|------|-------|-------|-------|----|------|
| #465688 | AUD_JPY | SHORT | 3000u | 110.100 | 110.000 | +300JPY | bid≤110.00トリガー |
| #465682 | AUD_JPY | SHORT | 2000u (partial) | 110.171 | 110.000 | +342JPY | bid≤110.00トリガー |
- H1 ADX=34 DI-=24>>DI+=6 強ベア継続。M5 Reg Bullish Div=0.63(監視、0.8で撤退検討)
- 残: 6000u (#465669 2000u @110.178 + #465682 1000u @110.171 + #465684 3000u @110.146)

## EUR_USD SHORT CLOSE [$UTCNOW]
| # | ペア | 方向 | Units | Entry | Close | PL |
|---|------|------|-------|-------|-------|----|
| #465734 | EUR_USD | SHORT | 3000u | 1.15360 | 1.15372 | -57.59円 |
- 決済理由: H1 MACD Regular Bullish Div=0.72(age=8本) → ≥0.7即決済ルール発動
- 今朝の-2385円損失と同じパターン。被害最小化

## AUD_JPY SHORT add-on #465738 [$UTCNOW]
| # | ペア | 方向 | Units | Entry | pretrade |
|---|------|------|-------|-------|---------|
| #465738 | AUD_JPY | SHORT | 3000u | 110.085 | LOW(73%WR) |
- M5バウンス(DI+=35)を利用して追加。H1 ADX=34 DI-=22>>DI+=11 強ベア継続。Div=0.00クリーン
- 合計: 8000u #465730(3000u@110.067)+#465732(2000u@110.045)+#465738(3000u@110.085)
- avg≈110.068 | TP=109.80(26.8pip) | pain limit=110.317

### [15:40Z] USD_JPY SHORT -2000u @159.701 (#465743) ★新規
| 項目 | 値 |
|------|-----|
| 方向 | SHORT |
| サイズ | 2000u |
| エントリー | 159.701 |
| pretrade | HIGH (score=10, halved from 4000u) |
| 確度 | A |
| テーゼ | H1 ADX=26 DI-=23>DI+=16 bearish + Regular Bearish Div -0.79 (MACD=-1.00最強) |
| M5根拠 | ADX=35 DI-=36>DI+=16 強下降トレンド |
| 転換条件 | H1 DI+ > DI- 逆転 or 160.20実体上抜け |
| TP目安 | 159.20 (H1 bearish div到達点) |

### [15:41Z] EUR_USD LONG 1000u @1.15337 (#465745) ★新規
| 項目 | 値 |
|------|-----|
| 方向 | LONG |
| サイズ | 1000u |
| エントリー | 1.15337 |
| pretrade | HIGH (score=8, halved) |
| 確度 | B |
| テーゼ | H1 MACD Bullish Div 0.80 (下降トレンド疲弊) + M5 ADX=26 bullish + Hidden Bullish Div 0.52 |
| 転換条件 | H1 DI- 拡大 or 1.1510実体割れ |
| TP目安 | 1.1560 (H1 VWAP回帰方向) |

### [15:41Z] GBP_USD LONG 2000u @1.33335 (#465747) ★新規
| 項目 | 値 |
|------|-----|
| 方向 | LONG |
| サイズ | 2000u |
| エントリー | 1.33335 |
| 確度 | B |
| テーゼ | M5 ADX=34 bullish DI+=30>DI-=16 + Hidden Bullish Div 0.60. H1 range |
| 転換条件 | M5 DI- > DI+ 逆転 or 1.3310実体割れ |
| TP目安 | 1.3365 (M5 resistance zone) |

### 15:44 UTC — AUD_USD SHORT 3000u @0.68897 #465749
- テーゼ: H1 ADX=36.4 bearish + Hidden Bearish Div -0.60。AUD全面安(AUD_JPY H1 ADX=33.6 bearish確認)
- pretrade: MEDIUM (勝率48%だがH1構造明確)
- 転換条件: H1 DI+ > DI- 逆転 or 0.6920実体上抜け
- TP目安: 0.6860 (H1 BB下限0.68677近辺)

### 15:44 UTC — GBP_USD LONG 2000u add-on @1.33397 #465751
- テーゼ: M5 ADX=34.2 bullish DI+=28>DI-=18。前回#465747の追加
- pretrade: LOW
- avg: (1.33335×2000 + 1.33397×2000)/4000 ≈ 1.33366

## AUD_JPY TP半利確 16:14 UTC
| 項目 | 値 |
|------|-----|
| 方向 | SHORT |
| 決済units | 6000u (3000@110.085 + 3000@110.072) |
| 決済価格 | 110.035 |
| 確定PL | +261円 (150+111) |
| 理由 | H1 RSI Regular Bullish Div score=1.00 (age=10本) + M5 bullish転換(ADX=28, DI+=29>DI-=19) |
| 残ポジ | -5000u (2000@110.045 + 3000@110.067) |

### ENTRY GBP_USD LONG 2000u @1.33385 #465759
- 時刻: 00:05Z (09:05 JST)
- テーゼ: H1 Bullish Div score=1.00(RSI+MACD両方。最強シグナル) + M5上昇トレンド(ADX=37, DI+=28)
- pretrade: LOW (score=1, 勝率50%)
- Total: 6000u (2000@1.33335 + 2000@1.33397 + 2000@1.33385) avg≈1.33372
- TP目安: 1.3370→1.3390
- 転換条件: H1 DI-逆転 or 1.3310実体割れ

## 01:20 UTC AUD_JPY SHORT 2000u 半利確
- CLOSE @109.788 | PL: **+514円**
- 理由: H1 RSI bullish div=0.88 + RSI=32.1(売られすぎ) + VWAP-75pip(極端乖離)
- 残: 3000u @110.067 (UPL≈+873)
- 累計確定益: +5,542円

## AUD_JPY SHORT 3000u → 全決済 +657円
| 項目 | 値 |
|------|-----|
| 方向 | SHORT |
| エントリー | @110.067 |
| 決済 | @109.848 |
| Units | 3000 |
| PL | +657円 |
| 理由 | H1 RSI=28.2<30 全撤退トリガー + div=0.88警告 |
| 備考 | 半利確(+514) + 全決済(+657) = AUD_JPY合計+1,171円 |

## USD_JPY SHORT -2000u 決済 (01:39 UTC)
| 項目 | 値 |
|------|-----|
| Entry | @159.701 |
| Close | @159.560 |
| PL | +282円 (実質+258 ※誤hedgeで-24) |
| 理由 | H1 StRSI=0.00(極限oversold)+Div=0.8(bullish div)。ATR比88%到達。反発前に確定 |

### [01:43Z] AUD_JPY SHORT -3000u @109.878 (新規)
| 項目 | 値 |
|------|-----|
| 方向 | SHORT |
| サイズ | 3000u |
| エントリー | 109.878 |
| pretrade | LOW (score=1, 勝率74%) |
| テーゼ | H1 ADX=35.5 strong bear (DI-=25.6>DI+=8.1) + M5 ADX=36.5 bear. AUD弱テーマ |
| 転換条件 | H1 DI+逆転 or 110.20上抜け |
| TP目安 | 109.50 |

### [01:44Z] GBP_USD LONG +2000u @1.33413 (add-on#4)
| 項目 | 値 |
|------|-----|
| 方向 | LONG add-on |
| サイズ | 2000u (合計8000u, 4本) |
| エントリー | 1.33413 |
| avg | ≈1.33382 |
| pretrade | LOW |
| 根拠 | M5 ADX=24.9 DI+=31.1>DI-=13.6 bull |

## AUD撤退 (03:28 UTC)
| Action | Pair | Side | Units | Price | PL | Reason |
|--------|------|------|-------|-------|----|--------|
| CLOSE | AUD_JPY | SHORT | 3000 | 110.142 | -792 | -500円超絶対痛みルール + H1 Bullish Div(1.0) + M5強上昇(ADX=33) |
| CLOSE | AUD_USD | SHORT | 3000 | 0.69021 | -595 | -500円超絶対痛みルール + H1 Bullish Div(1.0) + M5 RSI=70 AUD全面高 |

確定益累計: 6,457 - 792 - 595 = **+5,070円**

## AUD_USD LONG 2000u @0.69011
- 時刻: 03:50 UTC
- 根拠: H1 Regular Bullish Div=1.0 (ADX=46 bear中だがDiv確認) + M5 ADX=49強bull + StochRSI=0.4(pullback済)
- pretrade: MEDIUM (50%勝率, 平均PL=-182円)
- 確度: B (H1 Div + M5トレンド)
- TP目安: 0.6920 (+19pip) BB upper付近
- 転換条件: M5 DI-逆転 + 0.6885実体割れ
- 教訓適用: 「AUD H1 Bullish Div=1.0 → 結局AUDが反転上昇。Divが正しかった」

### ENTRY AUD_JPY LONG 3000u @110.227
- 時刻: 04:03Z
- 根拠: AUD最強(M5 ADX=45 AUD_USD, ADX=35 AUD_JPY) + JPY最弱(クロス円全面bull)
- M5: ADX=35.3, DI+=40.6>>DI-=10.9, RSI=74.7, StochRSI=0.97
- H1: ADX=32.6 bearish(DI-=22.4>DI+=17.9), StochRSI=1.0(反発中), div_score=0.4
- pretrade: LOW (score=3)
- TP目安: 110.50(+27pip)
- 転換条件: M5 DI-逆転 + 110.10実体割れ

### AUD_USD LONG 全撤退 07:16 UTC
- **CLOSE**: 2000u @0.68984 | PL=-86円
- Entry: @0.69011 | 保持時間: ~50min
- 理由: 0.6895撤退ライン割れ(bid=0.68944→0.68984で決済)。MTF一致BEAR+macro AUD最弱+H1 DI->DI+。Div反発力なし
- preclose_check答え: H1 bearish構造、転換条件0.6895該当、分析判断(パニックではない)

### EUR_USD LONG 2000u @1.15224 id=465789 [07:42 UTC]
- Add-on 3/5本。pretrade=HIGH(37%WR)
- 根拠: M5 StochRSI=0.08極限oversold + H1 ADX=14 range構造不変 + ユーザー「ユーロドル追加してもいいかも」
- avg: 1.15370→1.15311に改善
- macro: JPY最強、USD微プラス。EUR弱(-0.05)だがM5extreme oversoldでbounce期待

### [07:53Z] AUD_USD LONG 1500u @0.68906 (id=465791)
| 項目 | 値 |
|------|-----|
| 方向 | LONG |
| サイズ | 1500u |
| エントリー | 0.68906 |
| pretrade | MEDIUM (score=4, WR=50%) |
| テーゼ | H1 RegBull Div MACD=1.0 + M5 StochRSI=0.0極限oversold。H4 ADX=21ニュートラル。bounce狙い |
| 転換条件 | 0.6870実体割れ → 撤退 |
| 確度 | B（H1 Div + M5極限。macro AUD最弱が逆風） |
| ユーザー | 「追加はオージー系でもいいよ」(16:42 JST) |

### AUD_USD LONG 1500u — 決済 -19円
| 項目 | 値 |
|------|------|
| ID | 465791 |
| エントリー | 0.68906 (07:53Z) |
| 決済 | 0.68898 (15:36Z) |
| PL | -19円 (-0.8pip) |
| 保持時間 | ~7h43m |
| 決済理由 | macro逆風(AUD弱-0.22/USD強+0.20)+MTF BEAR一致+H1 HidBear Div+M5 StRSI=1.0バウンス過熱。profit_check=TAKE_PROFIT |
