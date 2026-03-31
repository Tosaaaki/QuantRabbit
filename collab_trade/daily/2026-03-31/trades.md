# 2026-03-31 トレード記録

## EUR_JPY 半利確（CPI前リスク管理）

| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-31 07:00 UTC |
| アクション | EUR_JPY SHORT 部分決済 5500u |
| 内訳 | id=465839 500u @182.673 +155.5円 / id=465851 3000u @182.671 +756円 / id=465847 2000u @182.673 +468円 |
| 合計確定益 | +1,379.5円 |
| 理由 | 09:00 UTC ユーロ圏CPI速報前のリスク管理。M5 StochRSI=0.02 極端売られすぎ→反発リスク。ATR1.0〜1.6x達成。デフォルト利確ルール適用。 |
| 残存 | 7000u avg≈182.852 (id=465875 1500u / id=465853 1500u / id=465859 2000u / id=465861 2000u) |
| pretrade | N/A (利確) |

---

## AUD_JPY 全撤退

| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-31 07:02 UTC |
| アクション | AUD_JPY SHORT 3000u 全撤退 |
| 内訳 | id=465849 2000u @109.392 -376円 / id=465855 1000u @109.392 -298円 |
| 確定損失 | -674円 |
| 理由 | M5 DI+優位転換（設定した撤退条件達成）+ 109.418まで2.6pip + H1 BullDiv=1.0継続 + マージン57.8%→資金解放 |

---

## GBP_JPY SHORT 2000u add-on

| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-31 07:03 UTC |
| エントリー | GBP_JPY SHORT 2000u @210.386 |
| id | 465887 |
| 根拠 | H1 ADX=42 DI-=43 BEAR + GBP弱(-0.67)+JPY強(+0.80) + AUD_JPY撤退後マージン回復 |
| pretrade | LOW (score=3) |

---

## 残存ポジション（07:03 UTC時点）
- EUR_JPY SHORT 7000u avg≈182.852 → 09:00 UTC CPI待ち
- GBP_JPY SHORT 6500u avg≈210.490 → H1 ADX=42 BEAR継続
- **本日確定損益: +705.5円** (EUR_JPY +1,379.5 / AUD_JPY -674)

### GBP_JPY SHORT 750u 半利確 [2026-03-31]
| 項目 | 値 |
|------|-----|
| 時刻 | ~10:30 UTC |
| 決済価格 | 210.382 |
| P/L | +237円 |
| 残ポジ | 750u @210.698 id=465841 |
| 根拠 | ATR比1.2x(+32.8pip)トリガー。M5 StochRSI=1.00プルバック中。四半期末ポジ調整リスク |
| 転換条件 | H1 DI-優位継続 → 残750uで210.000目標 |

### AUD_JPY SHORT 1000u [add-on] 11:38Z
| 項目 | 値 |
|------|-----|
| 時刻 | 11:38 UTC |
| エントリー | 109.308 (bid) |
| id | 465895 |
| 根拠 | H4 ADX=46 DI-=29 strong bear。M5 StRSI=0.19冷却確認→押し目SHORT |
| pretrade | LOW (score=1, 65%勝率) |
| 目標 | 109.000 (-30pip) |
| 撤退 | H4 DI+逆転 |

---

### EUR_JPY SHORT 1500u [add-on] ~13:10Z
| 項目 | 値 |
|------|-----|
| 時刻 | ~13:10 UTC |
| エントリー | 182.917 (bid) |
| id | 465899 |
| 根拠 | H1 ADX=49 DI-=38 MONSTER BEAR。**M5 StochRSI=1.00 + BEAR DIV(RSI=0.6/MACD=0.6)** = M5バウンスピークでの反転シグナル。JPY+0.70最強/EUR-0.68最弱テーマ継続。四半期末JPY還流。 |
| pretrade | LOW (score=1, 64%勝率) |
| 目標 | 182.300 (-62pip) |
| 転換条件 | H1 DI+優位転換 or 183.100超え |

### EUR_JPY SHORT 全決済 (invalidation hit) — 2026-03-31 19:47 UTC

| 項目 | 値 |
|------|-----|
| 決済時刻 | 19:47 UTC |
| 決済価格 | 183.002-183.008 (ask) |
| 決済数量 | 10500u (6ポジ) |
| PL合計 | -2,253円 |
| reason | M5 19:45足 ask高値≈183.014 → invalidation 183.000 突破確認 |
| 根拠 | H4 CCI=-274/StRSI=0.0 極端 + M5 25分bull継続 + ask高値183.014 = テーゼ崩壊 |
| H1構造 | ADX=49 DI-=35 MONSTER BEAR (変化なし、但しinvalidation優先) |
| IDs | 465853(-210), 465859(-240), 465861(-276), 465875(-568.5), 465891(-828), 465899(-130.5) |

## GBP_JPY SHORT 2000u add-on [20:00 UTC]

| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-31 20:00 UTC |
| エントリー | 210.548 (ask) |
| ID | 465913 |
| サイズ | -2000u |
| 根拠 | H1 ADX=50 DI-=34>DI+=9 MONSTER BEAR。M5 StRSI=1.0 bounce peak + CCI=160。年度末フロー終了直前の反発を売り増し |
| pretrade | LOW (score=3) |
## AUD_JPY SHORT 2000u add-on @109.401 id=465917
| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-30 20:46 UTC |
| エントリー | 109.401 (ask) |
| 方向 | SHORT |
| サイズ | 2000u (add-on) |
| avg | 109.327 (合計5000u) |
| 根拠 | H4 ADX=46 DI-=29 strong bear intact。H1 ADX=38 DI-=28 StRSI=1.0 = pullback exhaustion。M5 StRSI=1.0 bounce peak |
| pretrade | LOW (score=1, 勝率63%) |
| TP目標 | 109.000 |
| 転換条件 | H4 DI+逆転 or H1 ADX急低下 |


### EUR_JPY SHORT 2000u ADD-ON #466003 [00:12Z 4/1]
| 項目 | 値 |
|------|-----|
| 時刻 | 2026-04-01 00:12 UTC |
| エントリー | 183.115 (ask) |
| id | 466003 |
| テーゼ | H1 ADX=44 DI-=30 monster bear + H1/M5 StRSI=1.0 bounce exhaustion |
| 根拠 | EUR最弱(-0.47) + JPY最強(+0.38) + M5 StRSI=1.0過熱 = bounce peak |
| pretrade | S (score=9) |
| SL | 183.312 (ATR×1.1) |
| TP | 182.880 (ATR×1.3 from entry) |
| スプレッド | ~1.9pip |

---

## EUR_JPY SHORT 8000u [新規] id=466026

| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-31 00:53 UTC |
| エントリー | 183.311 (bid) |
| id | 466026 |
| サイズ | 8000u |
| TP | 182.900 (41pip) |
| SL | 183.540 (23pip) |
| 根拠 | H1 ADX=44 DI-=30 StRSI=0.99 (H1レベルオーバーバウト=最強SHORTシグナル). M5 CCI=247 StRSI=1.0 極端. Fib223%過伸長. マクロ: EUR=-0.47最弱+JPY=+0.38最強. pretrade S(8) LOW |
| pretrade | S (score=8) LOW |

---

## AUD_JPY SHORT 8000u [新規] id=466030

| 項目 | 値 |
|------|-----|
| 時刻 | 2026-03-31 00:53 UTC |
| エントリー | 109.586 (bid) |
| id | 466030 |
| サイズ | 8000u |
| TP | 109.160 (43pip) |
| SL | 109.780 (19pip) |
| 根拠 | H4 ADX=47 DI-=27 + H1 ADX=36 DI-=25 (W bear). M5 CCI=208 StRSI=1.0 極端オーバーバウト. Fib154%過伸長. マクロ: AUD=-0.31+JPY=+0.38. pretrade S(8) LOW |
| pretrade | S (score=8) LOW |
