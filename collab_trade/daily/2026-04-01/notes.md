# User Notes — 2026-04-01

## User Remarks (from Slack #qr-trades)

### 17:21 UTC — "慌てるな。状況教えて"
- Context: GBP_JPY SHORT -895 SL hit (05:33), AUD_JPY SHORT -1,190 SL hit (08:04), EUR_USD SHORT -484 SL hit (11:26), EUR_JPY SHORT -930 SL hit (11:27), GBP_JPY SHORT -939 SL hit (12:09). 5 consecutive SL hits. Margin was high. Trader was stressed
- Lesson: **Don't panic after consecutive SL hits. Assess the situation first**

### 17:26 UTC — "慌てるな。まだ保持"
- Context: Same stress period. User says hold, don't panic-close remaining positions
- Lesson: **User reads: positions will recover. Don't panic-dump on drawdown**

### 17:28 UTC — "ポンドはボラでかいから、待ってれば良い。空いてる資金で稼いで"
- Context: GBP_JPY SHORT underwater. User says GBP volatility is large — just wait for it to come back. Use free margin to find new entries
- Lesson: **GBP_JPY swings are wide (30-50pip normal). Don't cut on noise. Meanwhile, use idle margin on other pairs to generate income**

### 17:50 UTC — "ヒゲに狼狽えないように"
- Context: Multiple wick-driven SL hits earlier in the day
- Lesson: **Wicks are noise. Don't let them drive decisions. Confirmed in risk-management.md failure patterns**

### 18:10 UTC — "全ポジプラス方向にいくと思う"
- Context: GBP_JPY SHORT @210.352, USD_JPY SHORT pending, EUR_USD LONG pending. Market in Asian consolidation after volatile London/NY
- Chart state: H4 bear trends intact on GBP_JPY and USD_JPY. EUR_USD H1 ADX=46 monster bull. CS: EUR(+0.47) strongest, USD(-0.29) weakest
- Lesson: **User's macro read was correct — all positions eventually moved into profit (GBP_JPY TP'd +906, EUR_USD TP'd +1,622, EUR_JPY TP'd +1,100/+1,300, AUD_JPY TP'd +1,256)**

### 18:15 UTC — "マージンクローズアウトはやばいから、99%いったら、95%まで軽くして。軽傷で済むポジ選んで、部分決済とか"
- Context: Multiple positions open, margin usage was elevated
- Lesson: **Margin closeout prevention rule: at 99% margin, immediately reduce to 95% by partial-closing positions that minimize realized loss. This is now codified in risk-management.md (95% = forced half-close)**
