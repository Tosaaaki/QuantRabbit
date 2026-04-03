# Strategy Memory — プロトレーダーの経験知

**daily-reviewが毎日更新。traderが毎セッション冒頭で読む。**

最終更新: 2026-04-03 06:10 UTC (daily-review)

---

## Confirmed Patterns（3回以上検証済み — ルールとして扱え）

- **M5 StochRSI=0.0 + H1 BULL構造不変 → 高確率反発** (3/23 GBP V字+653円, 3/23 EUR BB Mid, 3/26-27複数回)
- **フロー逆行エントリーは必ず負ける**: テクニカルだけでフロー逆方向に入ると勝てない (3/20 USD_JPY 4/5負け, 3/23 GBP SHORT全敗)
- **スパイク慌て損切り = 最大損失源**: 天井/底で成行は最悪。スパイクは戻す確率が高い (3/23 GBP -3,832円, 3/26 パニック-8,041円)
- **H1+M5方向一致 + ADX>25 = 高確度エントリー** (3/23 GBP H1+H4 BULL, 複数回検証)
- **小さすぎる利確が利益を食い潰す**: ATR50%未満の利確はスプレッド+手間に見合わない (3/26 勝率65%でNet-583円, 勝ち平均+84円)
- **マージン極限で新規ペア追加 → クローズアウト** (3/26 EUR_JPY追加→全ポジ強制決済)

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
- **All-time SHORT: 27% WR avg -1,077 JPY total -23,696 JPY** — worst combination in the system. GBP_USD SHORT is a money pit. Avoid or size down to absolute minimum.
- **All-time LONG: 53% WR avg -3 JPY total -93 JPY** — near breakeven overall, but strong when H1 structure is confirmed.
- Tokyo session trailing stop trap same as EUR_USD: 15pip trail = ATR×0.7 got clipped (4/3 lesson)

### EUR_USD
- スプレッド最小。スキャルプに最適
- VWAP+35pip乖離での追撃 → 過熱からの反転で負け
- ロンドンクローズ(16:00 UTC)で方向変わりやすい
- **All-time LONG 46%WR avg +85 JPY total +5,520 JPY** — best total P&L of any pair/direction. EUR_USD LONG is the strongest edge in the system.
- **All-time SHORT 51%WR avg -156 JPY total -7,015 JPY** — decent WR but heavy average loss. EUR_USD SHORT hurts when it loses; size down on SHORT side.
- Trailing stop in Tokyo session (00:00-06:00Z): ATR×0.7 clips positions. Use ATR×1.2 minimum or hard SL only (4/3 lesson)

### AUD_USD
- フロー分析が特に効く。STRONG_SHORTで+889円実績 (3/20)
- SHORT全期間42%WR — 得意ではない。LONGの方が54%WRで良い (3/27振り返り)

### AUD_JPY
- スプレッド広い（1.6-1.9pip）。SLは広め必須
- SL=4.6pipで0.2pip差の止狩り→即反発。SL=9pipなら生存
- **LONG 65%WR avg +33 JPY total +867 JPY | SHORT 62%WR avg +70 JPY total +4,061 JPY** — system's best performing pair. Both directions profitable.
- SHORT has higher avg P&L per trade (+70 vs +33). When SHORT thesis is clear, size up.
- 方向転換(LONG↔SHORT)のタイミングに注意。転換直後は負けやすい (3/27: LONG+514→SHORT-792)

### GBP_JPY
- 30-50pipスウィングが普通。痛み上限-40pip
- 25pipの逆行で即切りは早すぎる

### EUR_JPY
- ボラ大きい。利確遅れやすいので注意

---

## Pretrade Feedback（daily-reviewが自動更新）

### LOWが正しかった場面（避けるべきエントリー）

- [3/27] AUD_JPY SHORT pretrade=LOW → -792円(108min保持)。H1構造は合ってたがM5 extreme oversoldでbounceに巻き込まれた
- [3/27] AUD_USD SHORT pretrade=LOW → -595円。AUD_USD SHORTは全期間42%WR。得意でない組み合わせ
- [3/27] AUD_JPY LONG pretrade=LOW → -417円(119min保持)。方向転換後のLONG、テーゼ鮮度が劣化

### LOWが間違っていた場面（pretradeが保守的すぎるケース）

- [3/27] AUD_JPY LONG pretrade=LOW → +514円, +657円。AUD_JPY LONGは全期間67%WR。LOWでも勝てる組み合わせ
- [3/27] GBP_USD LONG pretrade=LOW(WR=50%) → まだ保有中、判定保留

### HIGHが外れた場面（スコア過信に注意）

- [4/3] EUR_USD LONG pretrade=HIGH(score=8) → -170 JPY. Score itself was reasonable (post-NFP USD weak thesis, H1 ADX=59 bull). **Cause was not the entry thesis but the trailing stop calibration**: 11pip trail = ATR×0.7, clipped by Tokyo session (02:22Z) noise before London even opened. Entry thesis was correct — execution/protection was wrong. **Lesson: trail width caused the loss, not the pretrade score. Don't attribute this to score inflation.**

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

## メンタル・行動

- 慌てて損切りが最大の敵。テーゼで判断しろ、金額で判断するな
- 確度でサイズを決めろ: S(鉄板)=5000-8000u, A(高確度)=3000-5000u, B(普通)=1000-2000u
- 「今フラットだったらここで入るか？」→ Noなら閉じろ
- 1ペアに固執するな。7ペアある。2連敗したら別ペアを探せ
- ヘッドライン相場では追加LONG禁止。30分で局面が変わる
- **[4/1] 全ポジ同方向=信者。方向分散しろ。** GBP_JPY/AUD_JPY/EUR_JPY全SHORT→バウンスで全SL hit。片方向集中は分散ではなく賭け
- **[4/1] 指標は過去、チャートは今。** ADX=50「MONSTER BEAR」を30セッション繰り返しSHORTを積んだが、実際のチャートはバウンス中。チャートの形を見ずに指標の数字で判断するのはボット
- **[4/1] 含み益は取れ。** EUR_USD+536円、GBP_JPY+60円 → 「テーゼ生きてる」でHOLD → SL hit。市場がくれたものは受け取れ
- **[4/1] 「動き切った後」に同方向で入るな。** H4 CCI=-274 RSI=29で新規SHORT = 200pip落ちた後にさらに売るということ。次はバウンス
