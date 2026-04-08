# Strategy Memory — プロトレーダーの経験知

**daily-reviewが毎日更新。traderが毎セッション冒頭で読む。**

最終更新: 2026-04-08 06:00 UTC (daily-review)

---

## Confirmed Patterns（3回以上検証済み — ルールとして扱え）

### ✦ 勝ちパターン（これが金を生む — 恐怖より先に読め）

- **EUR_USD LONGはシステム最強エッジ**: 全期間53%WR, avg+160JPY, total+11,980JPY。迷ったらEUR_USD LONG。H1 ADX>25+EUR strongest+M5 StRSI=0.0は鉄板 (3/31-4/1: 6回連続利確, 4/8: 2/2 wins +4,575JPY)
- **AUD_JPY両方向で利益**: LONG 64%WR avg+26JPY, SHORT 59%WR avg+54JPY。両方向プラスの安定ペアだが、チャーン(close→re-enter繰返し)で利益を食い潰すリスクあり
- **M5 StochRSI=0.0 + H1 BULL構造不変 → 高確率反発** (3/23 GBP V字+653円, 3/23 EUR BB Mid, 3/26-27複数回, 4/1 EUR_USD 5回連続成功)
- **H1+M5方向一致 + ADX>25 = 高確度エントリー** (3/23 GBP H1+H4 BULL, 3/31-4/1 EUR_USD ADX=46-53で連勝)
- **3/31の回転トレード = 理想形**: 19エントリー, avg 4,737u, +4,591JPY。LIMITでFib pullback待ち→TP→即再エントリー。サイズを恐れず、回転を止めなかった
- **LIMIT + TP/SLで寝ている間に利確**: 3/31 GBP_JPY TP+906JPY, EUR_USD TP+1,622JPY — LIMITが刺さってTPまで到達。離席中の利益はシステムの強み
- **Counter-trade at H4 extreme → bounce capture**: H4 CCI±200/StRSI 0or1 → 反対方向に小ロットで利益 (4/1 USD_JPY SHORT +448JPY, bounce catch)
- **EUR_JPY LONG in trending conditions = highest WR in system**: 69%WR, avg+140JPY, total+1,824JPY. 4/7: 6/6 wins +1,876JPY ALL at pretrade=LOW. When H1 ADX>35 + EUR strongest + JPY weakest, pretrade LOW is wrong — the trend overpowers historical stats. (3/31, 4/6, 4/7: 3+ sessions confirmed)
- **Rotation SHORT within LONG thesis (M5 pullback trading)**: Holding LONG + M5 turns bearish (StRSI↓from 1.0, MACD hist shrinking, lower highs, upper wicks, CCI falling) → SHORT 2000-3000u, TP=M5 support (ATR×0.5-1.0), 15-30min hold. OANDA hedge = LONG stays open. NOT counter-trading — standard pullback trading. 4/8-4/9: 13 entries all LONG, 0 rotation SHORTs. Drawdown periods (EUR_JPY -849JPY, AUD_JPY 111.667→111.550=11.7pip drop, GBP_USD 44.5pip drop to SL) were all capturable as SHORT profits. If M5 data convinced you to tighten TP, that same data was a SHORT entry signal

### ✦ 負けパターン（認識した上で恐れすぎるな）

- **フロー逆行エントリーは必ず負ける**: テクニカルだけでフロー逆方向に入ると勝てない (3/20 USD_JPY 4/5負け, 3/23 GBP SHORT全敗)
- **スパイク慌て損切り = 最大損失源**: 天井/底で成行は最悪。スパイクは戻す確率が高い (3/23 GBP -3,832円, 3/26 パニック-8,041円)
- **小さすぎる利確が利益を食い潰す**: ATR50%未満の利確はスプレッド+手間に見合わない (3/26 勝率65%でNet-583円, 勝ち平均+84円)
- **マージン極限で新規ペア追加 → クローズアウト** (3/26 EUR_JPY追加→全ポジ強制決済)
- **薄商い(祝日/GoodFriday)のタイトSL = 全部狩られる** (4/3: EUR_USD trail 11pip, GBP_USD trail 15pip, AUD_USD SL 10pip → 全滅 -984円。テーゼは全部正解だった)
- **S-convictionをB-sizeで入る = 最大の機会損失**: 3/20-4/3で7回。6,740-13,140JPY失った。pretrade WR数字でビビってsized_down。二重割引するな
- **連敗サーキットブレーカーは同方向のみ**: AUD_JPY SHORT 4連敗 ≠ AUD_JPY LONG禁止。SHORTの連敗はLONGに無関係。方向ごとに判断しろ
- **Bearish M5 signals used defensively only, never as entries (4/8-4/9)**: "M5 overbought+div" "sellers dominant" "lower lows+bodies growing" — all noted in logs → only TP tightened / SL added. Never entered SHORT. Same data, different action: if it's good enough to change TP, it's good enough to enter the opposite direction. 13 entries, 0 SHORTs in 2 days
- **With-trend SHORT ignored for macro thesis (4/8 USD_JPY)**: H4 DI-=31 + H1 ADX=51 BEAR = clear SHORT chart. Tried LONG 3x as "counter-bounce" → all lost (-113JPY, SL, expired). Macro "risk-on" ≠ LONG every pair. Trade WITH the chart, not against it
- **Shallow indicator scan = one-direction bias (4/8-4/9)**: Checking ADX+StRSI+CS (3 indicators) → sees "BULL" → LONG only. Adding MACD hist + CCI + BB position + Fib + wick patterns → pullback SHORTs become visible within the same bullish context. Need 5+ indicators per direction to see both sides
- **GBP_USD repeated entries after failure = compounding losses**: 4/7: 4 entries (3 LONG, 1 SHORT), 1 win +93 JPY, 3 losses totaling -1,099 JPY. 5000u LONG @1.32383 lost -480 JPY in 17 min. After 2 consecutive GBP_USD losses, move to another pair. (4/7: verified 1x, but pattern matches 4/1 GBP_JPY repeated entries)

## Active Observations（検証中 — daily-reviewが追記する）

<!-- 各観察: [初出日] 内容。検証: N回確認/N回反証 -->

- [3/26] GBP_JPYのM5割れ即撤退は早すぎる。30-50pipスウィングが普通。5分待てフェイクブレイク判定 → 検証: 1回確認(3/26 -237円→戻り)
- [3/23] M5 RSI Regular Bullish Div score=1.0 → 底打ちの決定的シグナル (+25pip実績) → 検証: 2回確認(3/23 GBP +25pip, 3/27 EUR_USD Div=1.0でクローズ判断正解-168円で守れた)
- [3/23] NY session(13:00-17:00Z)は反転が多い。ロンドンの流れが続くか見極めが必要 → 検証: 複数回確認傾向
- [3/27] **pretrade LOWを全件無視して勝率43%**: 今日のpretrade注記エントリー7件全てLOW。決済7件中勝ち3/負け4。LOWの警告は概ね正しい — LOWで入るなら少なくともサイズを落とせ
- [3/27] **AUD_JPY LONGは勝ち(+514,+657)、SHORTは負け(-792)**: 全期間でもAUD_JPY LONGは67%WR、SHORTは66%WR。だが今日はLONGの勝ちが大きく、SHORTの負けが痛い。方向転換(LONG→SHORT)のタイミングが悪い？
- [3/27] **負けの平均保持76分 vs 勝ちは短時間**: 負けトレードを長く持ちすぎている。テーゼが崩れたらもっと早く切るべきか？ → 要検証（保持時間と損益の相関）
- [3/27] **利確プロトコルの空白**: エントリー=pretrade_check、損切り=preclose_check。利確には何もなかった。GBP 含み益+3,000円超がAUD急変中に消滅。→ 「利確を問うトリガー」として risk-management.md に追記済み。**別ポジで急変が起きた瞬間、全保有ポジの含み益を即確認しろ**が最重要。
- [3/27] **AUD_USD SHORT -595円**: AUD_USD SHORTは全期間42%WR。得意ではないペア×方向の組み合わせ
- [4/3] **Tokyo session trailing stop trap**: EUR_USD trail=11pip(ATR×0.7), GBP_USD trail=15pip(ATR×0.7) → both clipped at 02:22Z by Tokyo noise. Trail was set as "NFP protection" the night before but ATR×0.7 is too tight for 00:00-06:00Z low-vol session. Tokyo noise eats sub-ATR×1.0 trails. **Rule candidate: In Tokyo session (00:00-06:00Z), trailing stop minimum = ATR×1.2 or don't set trail — set fixed SL instead.** Verified: 1x
- [4/3] **Pre-NFP trail = double trap**: Trail added at 00:37Z as "NFP protection." Tokyo session noise clipped BOTH EUR_USD(-170 JPY) and GBP_USD(-146 JPY) at 02:22Z — 10 hours BEFORE NFP fired. Thesis was correct (USD weak), direction was correct. Trail calibration at ATR×0.6-0.7 in Tokyo low-vol killed two correct trades. **Hard rule: When holding overnight into NFP, use fixed SL only. No trailing stops until AFTER the NFP candle closes.** Verified: 1x (both positions, same session)
- [4/3] **Binary position management = missed the best option**: EUR_USD/GBP_USD were in profit before trails clipped them. Opus only considered "trail or hold" — never "take profit now and re-enter post-NFP." On Good Friday thin market with NFP 10h away, cutting in profit + waiting for direction was the correct play. **Always evaluate 3 options: (A) adjust SL/TP for new conditions, (B) cut and re-enter at better setup, (C) hold as-is.** Verified: 3x (17,844 JPY thrown away across 3 incidents):
  - 4/3 Good Friday: Only "trail or hold" → -316 JPY. Option B (take profit +110 JPY, re-enter post-NFP) = **426 JPY thrown away**
  - 3/27 GBP_USD: +3,000 JPY unrealized → HOLD bias → -4,796 JPY. Option B (take profit at +3,000) = **7,796 JPY thrown away**
  - 4/1 all-SHORT wipeout: GBP_JPY/AUD_JPY/EUR_JPY all SHORT → -4,438 JPY. Option A (directional diversification + take EUR_USD +536) = **9,622 JPY thrown away**
- [4/3] **SL placement must be structural, not ATR×N**: ATR×0.6 trail in thin market = noise width. SL at swing low / Fib 78.6% / DI reversal = meaningful. If you can't name the market structure at the SL price, the SL is bot-like. User couldn't understand the SL rationale because there wasn't one beyond "ATR×0.6." Verified: 1x
- [4/7] **Pretrade LOW massively wrong for trending EUR_JPY LONG**: 4/7 EUR_JPY LONG had 6 entries all pretrade=LOW. Result: 6/6 wins, +1,876 JPY. LOW-scored entries across all pairs today: 16 trades, 69% WR, +680 JPY. **Pretrade's historical WR for the pair+direction penalizes trending pairs whose stats are diluted by range-bound sessions.** When H1 ADX>35 + macro CS alignment (EUR strongest, JPY weakest), pretrade LOW should be overridden to at least MEDIUM. Verified: 1x (4/7 EUR_JPY 6/6)
- [4/7] **HIGH pretrade SHORT against macro trend = false confidence**: GBP_USD SHORT pretrade=HIGH(8) → -317 JPY. EUR_USD SHORT pretrade=HIGH(8) → -288 JPY. Both were USD-LONG bets in a TACO tariff week where USD was structurally weak. Pretrade doesn't factor macro regime direction. **HIGH score on a position that fights the macro flow is a trap.** Verified: 1x (2 trades same day)
- [4/7] **AUD_JPY churn: 3 round-trips = -778 JPY + 9.6pip spread burned**: LONG 3500u→-217(6min), LONG 5000u→+40(170min BE), SHORT 3500u→-556, SHORT 4500u→-4. Close-and-re-enter on same thesis is spread destruction. First SL should be the final decision. Don't re-enter same pair within 30 min unless thesis genuinely changes. Verified: 1x (matches 4/6 AUD_JPY pattern = 2x total)
- [4/7] **Winners held 127 min avg, losers held 13 min avg**: Patience = profit. Quick cuts on positions that needed time to develop (GBP_USD LONG held 15-17 min before cutting) vs EUR_JPY LONG held 36-225 min for wins. Don't cut within 15 min unless structural invalidation. Verified: 2x (4/7 + 4/8: winners 334min avg vs losers 131min avg)
- [4/8] **Quality audit "S-candidate fix" pressure → oversized loss**: AUD_USD LONG 8000u entered as "quality audit miss fix" → -2,540 JPY. Quality audit flagged AUD_USD as missed S-candidate, trader entered 8000u (~S-size) on a pair with 50%WR LONG all-time. Audit flags are data, not orders. Don't let audit pressure override your own pair-level conviction. AUD_USD LONG avg=-78JPY all-time — this is NOT an S-conviction pair. Verified: 1x
- [4/8] **Counter-trades against H1 ADX>35 BULL = money pit**: AUD_JPY SHORT 1200u (-79JPY) + SHORT 1000u (-121JPY). Both entered on "H4+H1 StRSI=1.0 extreme" counter-thesis. H1 ADX=37-38 DI+ dominant invalidated both within hours. **H4 StRSI=1.0 does NOT override H1 ADX>35 BULL.** The stronger signal is the trend, not the oscillator extreme. Verified: 2x (both AUD_JPY SHORT same day). **Clarification**: This applies to SWING-SIZE counter-trades (4000u+, hours hold). M5 rotation SHORTs (2000-3000u, TP=ATR×0.5, 15-30min hold) within a LONG thesis are standard pullback trading — different concept, different risk
- [4/8] **Margin crisis repeat (97%) → forced close**: EUR_JPY LONG 4500u forced close at -319 JPY. 3 positions stacked + pending AUD_USD LIMIT (7000u×margin) pushed to 97%. Same pattern as 4/1. **Pending LIMITs must be in worst-case calculation. Above 85% = STOP ADDING.** Verified: 2x (4/1 + 4/8)
- [4/9] **Indicator depth determines direction quality**: Checking 3 indicators (ADX+StRSI+CS) → LONG bias locked, SHORT invisible. Checking 6+ indicators (add MACD hist, CCI, BB position, Fib, wick patterns) → rotation SHORTs become visible within bullish context. Shallow scan = one-direction. Deep scan = both directions. 4/8-4/9: 13 LONG, 0 SHORT with shallow scans. Quality audit flagged "全ポジションLONG" repeatedly but trader escaped with "no H4 extreme" — M5 rotation opportunities were there
- [4/9] **USD_JPY: 3 LONG attempts against clear BEAR trend**: H4 DI-=31, H1 ADX=51 DI-=28. Technical downtrend on 2 timeframes. Trader tried LONG 3x (counter-bounce thesis) → all lost. Never once entered WITH-trend SHORT. Macro thesis ("risk-on") overrode 2-TF technical read. This is exactly what trading-philosophy.md prohibits: "Judge based on market conditions, not thesis"
- [4/4] **Conviction-S undersizing = biggest silent profit killer**: 3/20-4/3 data: 7 trades met conviction-S conditions. 5 of 7 were entered at B-size (1,000-2,000u) instead of S-size (8,000-10,000u). Actual P&L: +2,360 JPY. At S-sizing: +9,100-15,500 JPY. **6,740-13,140 JPY thrown away.** Root cause: trader checked 2-3 familiar indicators, rated B, and stopped looking. Deeper analysis (different indicator categories) would have revealed S. Fix: pre-entry format now requires FOR (multi-category) + **Different lens** (check indicator from unused category) + AGAINST + If I'm wrong. Different lens moves conviction BOTH directions — B→S upgrade when deeper analysis reveals alignment, S→C downgrade when alternative perspective reveals contradiction. The B→S upgrade is where the money is. Verified: 5 undersized trades + 4/1 S→C downgrade would have prevented -4,438 JPY

## Deprecated（反証済み — 参考のみ）

_(まだなし — daily-reviewが反証されたパターンをここに移動する)_

---

## Per-Pair Learnings（ペア別）

### USD_JPY
- 東京セッション + JPY最弱時にショートは死亡パターン（3/20: 5本中4本負け）
- フローがLONG方向なのにテクニカルだけでSHORT → フローが勝つ
- SL=4.6pipは止狩りに弱い。スプレッド×3以上を確保
- 介入リスクあり。+20pipで利確検討

### GBP_USD
- ホルムズ系ヘッドラインで+40-90pipの巨大スパイク (3/23)
- N波動targetは信頼できる利確目標
- H1 ADX=26でH4も同方向BULL = 最もクリーンなトレンドペア
- **All-time SHORT: 26% WR avg -1,044 JPY total -24,013 JPY** — worst combination in the system. GBP_USD SHORT is a money pit. Avoid or size down to absolute minimum.
- **All-time LONG: 56% WR avg +25 JPY total +966 JPY** — modestly profitable but requires patience (avg winner hold >100 min)
- Tokyo session trailing stop trap same as EUR_USD: 15pip trail = ATR×0.7 got clipped (4/3 lesson)
- **4/7: repeated entries after failures compound losses.** 4 entries in one day, -1,006 JPY net. 5000u entry after 2 prior losses = chasing. After 2 GBP_USD losses, move to another pair

### EUR_USD
- スプレッド最小。スキャルプに最適
- VWAP+35pip乖離での追撃 → 過熱からの反転で負け
- ロンドンクローズ(16:00 UTC)で方向変わりやすい
- **All-time LONG 53%WR avg +160 JPY total +11,980 JPY** — best total P&L of any pair/direction by far. EUR_USD LONG is the strongest edge in the system. 4/8: +4,575 JPY (2/2 wins) further confirms.
- **All-time SHORT 51%WR avg -156 JPY total -7,015 JPY** — decent WR but heavy average loss. EUR_USD SHORT hurts when it loses; size down on SHORT side.
- Trailing stop in Tokyo session (00:00-06:00Z): ATR×0.7 clips positions. Use ATR×1.2 minimum or hard SL only (4/3 lesson)

### AUD_USD
- フロー分析が特に効く。STRONG_SHORTで+889円実績 (3/20)
- SHORT全期間39%WR avg-14JPY — 得意ではない。LONGの方が50%WR avg-78JPYで「まし」だが両方マイナス
- **All-time: LONG 50%WR avg-78JPY total-1,557JPY | SHORT 39%WR avg-14JPY total-379JPY** — both directions are net negative. This pair is NOT an edge. Enter only when AUD is clearly strongest CS + macro alignment. Never S-size based on quality audit alone
- **4/8: 8000u LONG "quality audit S-candidate fix" → -2,540 JPY** — largest single loss. Audit pressure pushed oversized entry on a pair with no edge. Lesson: audit flags ≠ S-conviction

### AUD_JPY
- スプレッド広い（1.6-1.9pip）。SLは広め必須
- SL=4.6pipで0.2pip差の止狩り→即反発。SL=9pipなら生存
- **LONG 64%WR avg +26 JPY total +859 JPY | SHORT 59%WR avg +54 JPY total +3,297 JPY** — both directions profitable but margins thinning
- SHORT has higher avg P&L per trade (+54 vs +26). When SHORT thesis is clear, size up.
- 方向転換(LONG↔SHORT)のタイミングに注意。転換直後は負けやすい (3/27: LONG+514→SHORT-792)
- **4/7 churn = profit killer**: 4 trades, 3 round-trips, net -738 JPY + 9.6pip spread. LONG 3500u cut in 6 min (-217), then 5000u held 170 min for +40 (BE SL). Close→re-enter same thesis burns spread. First exit should be final for 30 min

### GBP_JPY
- 30-50pipスウィングが普通。痛み上限-40pip
- 25pipの逆行で即切りは早すぎる

### EUR_JPY
- ボラ大きい。利確遅れやすいので注意
- **LONG 69%WR avg +140 JPY total +1,824 JPY** — highest WR and avg in the system. When EUR strongest + JPY weakest (macro alignment), this pair prints money
- **4/7: 6/6 LONG wins, +1,876 JPY** in one day. All pretrade=LOW yet all won. H1 ADX=35-43 trending strongly. Pretrade undervalues this pair in trending conditions
- Spread 1.7-2.3pip is high. Need 20+ pip target to justify entry. Sub-10pip scalps don't work here

---

## Pretrade Feedback（daily-reviewが自動更新）

### LOWが正しかった場面（避けるべきエントリー）

- [3/27] AUD_JPY SHORT pretrade=LOW → -792円(108min保持)。H1構造は合ってたがM5 extreme oversoldでbounceに巻き込まれた
- [3/27] AUD_USD SHORT pretrade=LOW → -595円。AUD_USD SHORTは全期間42%WR。得意でない組み合わせ
- [3/27] AUD_JPY LONG pretrade=LOW → -417円(119min保持)。方向転換後のLONG、テーゼ鮮度が劣化

### LOWが間違っていた場面（pretradeが保守的すぎるケース）

- [3/27] AUD_JPY LONG pretrade=LOW → +514円, +657円。AUD_JPY LONGは全期間67%WR。LOWでも勝てる組み合わせ
- [3/27] GBP_USD LONG pretrade=LOW(WR=50%) → まだ保有中、判定保留
- [4/7] **EUR_JPY LONG pretrade=LOW(1) → 6/6 wins, +1,876 JPY** — biggest single-day pretrade miss. All 6 EUR_JPY LONG entries scored LOW. Every one won. H1 ADX=35-43 trending, EUR strongest(+0.42-0.60), JPY weakest(-0.22 to -0.45). **Pretrade LOW in a trending pair with macro alignment = systematically wrong.** This is the strongest evidence yet that pretrade_check needs a "trending override" when H1 ADX>35 + CS alignment
- [4/7] GBP_USD LONG pretrade=LOW(1) → +93 JPY (1/3 LOW entries won). LOW was partly right here — pair was choppy. Not all LOWs are wrong, but EUR_JPY LOW in trend is clearly miscalibrated

### HIGHが外れた場面（スコア過信に注意）

- [4/3] EUR_USD LONG pretrade=HIGH(score=8) → -170 JPY. Score itself was reasonable (post-NFP USD weak thesis, H1 ADX=59 bull). **Cause was not the entry thesis but the trailing stop calibration**: 11pip trail = ATR×0.7, clipped by Tokyo session (02:22Z) noise before London even opened. Entry thesis was correct — execution/protection was wrong. **Lesson: trail width caused the loss, not the pretrade score. Don't attribute this to score inflation.**
- [4/7] GBP_USD SHORT pretrade=HIGH(8) → -317 JPY. Macro was TACO week, USD structurally weak. HIGH score didn't account for macro regime. **Lesson: HIGH on a direction that fights macro flow = false confidence**
- [4/7] EUR_USD SHORT pretrade=HIGH(8) → -288 JPY. Same macro environment. Both USD-long bets in USD-weak week. **Pretrade needs macro direction filter**

### MEDIUM scores — mixed bag (4/8)

- [4/8] AUD_USD LONG pretrade=MEDIUM(4) → -2,540 JPY. Score was MEDIUM but pair has no edge (50%WR, avg -78JPY). MEDIUM on a historically losing pair = should have been treated as LOW. **Pair-level all-time stats matter more than session-level pretrade score.**
- [4/8] USD_JPY LONG pretrade=MEDIUM(5) → -113 JPY. Counter-trade, small size (1000u), acceptable loss.
- [4/8] EUR_USD LONG pretrade=MEDIUM(6) → +2,200 JPY, +2,376 JPY (2 trades). MEDIUM on EUR_USD LONG = reliable. This pair elevates MEDIUM scores into wins.

### B-score losses worth noting

- [4/3] GBP_USD LONG pretrade=B(no formal check) → -146 JPY. Same Tokyo trail trap as EUR_USD. GBP_USD 15pip trail = ATR×0.7, clipped 02:22Z. **B-score entries in Tokyo session are especially dangerous: low conviction + thin liquidity + aggressive trail = high noise-kill probability. Tokyo overnight entries should be sized to survive without trails.**

---

## 指標組み合わせの学び

### 効いた組み合わせ

| 日付 | ペア | 組み合わせ | TF | 結果 |
|------|------|-----------|-----|------|
| 3/23 | GBP_USD | M5 StochRSI=0.0 + H1 ADX>25 BULL | M5+H1 | +653円 V字回復 |
| 3/23 | GBP_USD | M5 RSI Bullish Div(score=1.0) + H1構造BULL | M5+H1 | +25pip |
| 3/23 | EUR_USD | M5 BB Mid + StochRSI=0 | M5 | 勝ち |
| 3/23 | GBP_USD | H1 ADX=26 + H4同方向BULL | H1+H4 | クリーントレンド |

### 効かなかった組み合わせ

| 日付 | ペア | 組み合わせ | 結果 | なぜ |
|------|------|-----------|------|------|
| 3/20 | USD_JPY | M1 StochRSI=1.0のみ | 4/5負け | M1極限だけでは根拠不足 |
| 3/23 | EUR_USD | VWAP+35pip乖離追撃 | 負け | 乖離大=逆張り指標 |

| 4/7 | EUR_JPY | H1 ADX>35 + M5 StRSI=0.0 + CS EUR strongest + JPY weakest | M5+H1+Macro | 6/6 wins +1,876 JPY. Trending pair + macro CS alignment = highest edge found |
| 4/7 | EUR_JPY | S-conviction scanner (M5 StRSI=0.0 + H1 HidBullDiv + ADX>35) | M5+H1 | Add-on 4000u at 184.547 → TP hit. S-conv scanner correctly identified high-probability add-on |

### 効かなかった組み合わせ

| 日付 | ペア | 組み合わせ | 結果 | なぜ |
|------|------|-----------|------|------|
| 3/20 | USD_JPY | M1 StochRSI=1.0のみ | 4/5負け | M1極限だけでは根拠不足 |
| 3/23 | EUR_USD | VWAP+35pip乖離追撃 | 負け | 乖離大=逆張り指標 |
| 4/7 | GBP_USD | M5 StRSI=0.03 pullback in H1 bull | LONG 5000u -480 JPY(17min) | H1 bull but M1 strong downtrend (ADX37 DI-=34). M1 counter-signal ignored = premature entry |

| 4/8 | EUR_USD | H1 ADX>30 + M5 StRSI dip + CS gap>1.0 + Iran ceasefire risk-ON | M5+H1+Macro | 2/2 wins +4,575 JPY. Strong USD weakness macro + EUR bid = dip-buy at M5 oversold in H1 trend. Best 1-day EUR_USD result |
| 4/8 | AUD_JPY | H4+H1 StRSI=1.0 counter SHORT vs H1 ADX=37 DI+ BULL | Counter | 0/2 losses -200 JPY. H4 oscillator extreme doesn't override H1 trend direction. Counter-trades need H1 structure to weaken first |

### 未検証（今後試す）
- Keltner幅 + BB幅: 同時squeeze=ブレイク予測精度UP？
- Chaikin Vol + ATR変化率: ボラ急変の早期検知？
- cluster_gap + Ichimoku雲: サポレジの二重確認

---

## 両建て戦術メモ

- OANDA v20ヘッジ口座: 大きい方にフルマージン。反対側は追加0
- H1テーゼLONG維持 + M5 StochRSI=1.0でSHORT回転 → プルバックで利確 → LONGだけ残る
- ショート側は同量以下。超えたらマージン増加

---

## Event Day Experience (NFP, CPI, FOMC etc.)

**What we've learned about event days:**
- The market doesn't stop moving 8 hours before an event. Small waves, squeezes, and setups still happen
- "Pre-NFP" became a 10-hour standby on 4/3. The real danger window is the last hour, not the whole day
- Thin liquidity (Good Friday, holidays) amplifies spikes on the event candle. The 5-10 min after release are the most dangerous
- Trailing stops get clipped by pre-event noise. Fixed SL survives better close to events

**The thinking**: Before writing "no entries pre-event" in state.md, ask: "How many hours until the event? Is that really a reason to stop trading, or am I just nervous?" Nervousness is not analysis.

---

## Small Wave Patterns (BB Squeeze Resolution)

**Observation**: When all pairs show M5 BB squeeze simultaneously, the squeezes resolve into small moves — usually 5-15 pips. These are tradeable.

**What works**: M5 squeeze + M1 shows 3+ consecutive directional bodies → enter small (1000-2000u). TP at opposite BB band. Fixed SL (no trailing — trail kills small moves before they develop). Typically resolves in 15-60 minutes.

**What doesn't work**: Trailing stops on squeeze trades (4/3 EUR_USD, GBP_USD both clipped by trail). Holding for big targets on a squeeze play (it's a rhythm trade, not a conviction trade).

**When to use**: When margin is underutilized and you're tempted to write "standby." The market is always moving. The question is whether you're watching.

---

## Thin Market / Holiday Rules (HARD RULES)

**4/3 Good Friday: -984 JPY. Every single loss was noise stop hunting. Every thesis was correct.**

- **Holiday/thin liquidity → NO SL**. ATR×1.2 gets hunted when spreads are 2-3× normal. Either ATR×2.5+ or discretionary only
- **User says "SLいらない" → that's a direct order.** Don't re-add. Don't close on your own judgment. The user is managing risk
- **Pre-event trailing stops are a trap.** Trail added at 00:37Z "for NFP protection" killed EUR_USD and GBP_USD at 02:22Z — 10 hours before NFP. Use fixed SL or nothing
- **Spread > 2× normal = the market is telling you SLs don't work.** Listen to the market

## メンタル・行動

### ✦ 攻めの原則（恐怖より先に読め）

- **Conviction=Size。二重割引するな。** pretrade_checkのWR%でサイズを落とすな。WRはすでにスコアに織り込み済み。Sと判定したなら30%NAV。pretrade=S(8)→sized_down は金をドブに捨てる行為（過去7回で6,740-13,140JPY損失）
- **最低2,000u。** 500u/700uは勝ってもスプレッドに食われる。入る価値がないなら入るな
- **S/A conviction = マーケットオーダー。** LIMIT「3pip節約したい」→ 刺さらず20-50pip逃す。これが4/7で0%マージンの原因
- **0%マージンはありえない。** 7ペア×4TF=28の視点がある。全部で何もない日はない。ない＝見てない
- **BがSに昇格するパターンが最重要**: 最初の2指標でB判定→止まるな。別カテゴリの指標を見たらSだった、が過去5回起きてる
- **同一ペアのclose→re-enterを繰り返すな（チャーン）。** 4/7 AUD_JPY: 3回往復でスプレッド9.6pip燃やして-738JPY。最初のSLで決着をつけろ。**4/7 GBP_USDも同パターン: 4エントリー3敗-1,006JPY。2連敗したら別ペアへ**
- **3/31を基準にしろ**: 19エントリー, avg 4,737u, 65%WR, +4,591JPY。これが目指す姿
- **Quality audit flags ≠ trade orders**: 4/8 AUD_USD 8000u "S-candidate fix" → -2,540 JPY. Audit says "you missed this opportunity." That's information, not an instruction to enter. Your own pair-level analysis (AUD_USD 50%WR, avg-78JPY = no edge) should override audit pressure. Audit-driven S-sizing on a losing pair = worst of both worlds

### ✦ 守りの原則（知っていればいい。恐れるな）

- 慌てて損切りが最大の敵。テーゼで判断しろ、金額で判断するな
- 「今フラットだったらここで入るか？」→ Noなら閉じろ
- 1ペアに固執するな。7ペアある。2連敗したら別ペアを探せ
- ヘッドライン相場では追加LONG禁止。30分で局面が変わる
- [4/1] 全ポジ同方向=信者。方向分散しろ
- [4/1] 指標は過去、チャートは今。チャートの形を見ずに指標の数字で判断するのはボット
- [4/1] 含み益は取れ。市場がくれたものは受け取れ
- [4/1] 「動き切った後」に同方向で入るな。次はバウンス
- [4/3] ユーザー指示無視→パニッククローズ→慌てて入り直し = 最悪パターン。**ユーザーがHOLDと言ったらHOLD**
