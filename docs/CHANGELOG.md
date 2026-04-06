# Changelog

## 2026-04-06 — Session extended to 15 minutes + STALE_LOCK auto-ingest

**Problem**: Sessions dying without reaching SESSION_END. ingest.py never runs → memory.db stale. Root cause: session_data.py output is massive (7 pairs × M5 20 candles + full technicals + news), model spends all 10 minutes analyzing without emitting Next Cycle Bash.

**Fix (3 changes)**:
1. Lock timeout: 600s → 900s (15 min hard limit before cron kills session)
2. SESSION_END threshold: 600s (10 min — gives 5 min buffer before kill)
3. STALE_LOCK detection: now runs `ingest.py` automatically before starting new session (guaranteed cleanup even if previous session died)

**Effect**: SESSION_END triggers at 10 min, cron kills at 15 min. 5-min buffer for ingest to complete. If session still dies, next session's STALE_LOCK path runs ingest as insurance.

## 2026-04-06 — Session extended to 10 minutes (lock threshold fix)

**Problem**: Earlier 10-min attempt failed because Bash① lock check (`AGE -lt 300`) and Next Cycle Bash (`ELAPSED -ge 300`) were out of sync — one was changed but the other wasn't. New cron killed running sessions at 5 min (STALE_LOCK), causing 30-second zombie sessions (PID 3292 incident).

**Fix**: Both thresholds changed to 600 (10 min) simultaneously:
- Bash① lock check: `AGE -lt 300` → `AGE -lt 600`
- Next Cycle Bash: `ELAPSED -ge 300` → `ELAPSED -ge 600`
- Updated: SKILL_trader.md, schedule.json description, CLAUDE.md

**Rationale**: Average hold time is long enough that 11-min max monitoring gap is acceptable. 10 min gives time for proper chart reading, Different lens, cross-pair analysis, and Fib — all of which were being skipped under 5-min pressure.

## 2026-04-06 — Trader: chart-first time allocation + strategy_memory lessons

**Problem**: Trader pattern-matched indicators (H1 StRSI=1.0 → "overbought → SHORT") instead of reading chart shape. Skipped pretrade_check, conviction block, and Different lens. AUD_JPY SHORT -203 JPY — H4 was BULL (N-wave q=0.65), pullback bodies shrinking (4.9→2.7→1.7→0.5), limit filled into rising market.

**Attempted 10-min fix → reverted**: Extended session to 10 min, but Claude Code kills processes at ~5 min. Relay mechanism added complexity without adding thinking time. Reverted to 5-min sessions.

**Actual fix**: Restructured 5-minute time allocation to prioritize chart reading over indicator transcription:
- 0-1min: data fetch + profit/protection check
- **1-3min: Read chart FIRST → 3 questions → hypothesis → confirm with indicators → conviction block** (was previously 1 min)
- 3-4min: execute trades
- 4-5min: state.md update
- Added: "No entry without Different lens" as explicit time allocation instruction
- strategy_memory: StRSI context-dependence (breakout vs range) + limit fill direction lessons

**Files changed**: `~/.claude/scheduled-tasks/trader/SKILL.md`, `collab_trade/strategy_memory.md`

## 2026-04-06 — Sizing table: hardcoded units removed, formula-only

**Problem**: Conviction sizing table showed hardcoded unit counts (10,000u / 5,000u / 1,667u / 667u) calibrated for NAV 200k. Current NAV is 104k. Trader was copying these numbers instead of recalculating from actual NAV → B entries at ~10% NAV instead of 5%.

**Fix**: Replaced all hardcoded unit examples in SKILL.md (3 locations) with:
- Formula: `Units = (NAV × margin%) / (price / 25)`
- Concrete examples using current NAV (104k) to anchor intuition
- Explicit note: "Never reuse yesterday's unit count"

**Files changed**: `~/.claude/scheduled-tasks/trader/SKILL.md`, `docs/SKILL_trader.md`

## 2026-04-06 — Slack ts tracking moved from Claude to code

**Problem**: Claude (especially Sonnet) forgets to update `Slack最終処理ts` in state.md → next session reads the same user messages → replies again → duplicate/triplicate responses. Dedup catches identical posts but not different wordings of the same reply.

**Root cause**: Relying on Claude to write a ts value to state.md is unreliable. The ts tracking must be in code, not in prompts.

**Fix**:
- `tools/slack_read.py` now auto-writes latest user message ts to `logs/.slack_last_read_ts` after every read
- `tools/session_data.py` reads from this file instead of parsing state.md for `Slack最終処理ts`
- SKILL_trader.md Bash② and Next Cycle Bash simplified — no more `grep Slack最終処理ts` in the shell command
- CLI `--state-ts` override still works if needed

**Result**: Once a user message is read by any session, no subsequent session will see it again. Zero Claude dependency.

## 2026-04-06 — M5 candle data integrated into session_data.py

**Problem**: Trader SKILL instructed Claude to fetch M5 candles via inline python one-liner. Sonnet gets stuck generating this one-liner ("Processing..." hang for 10+ min). Repeated issue.

**Fix**: Added M5 PRICE ACTION section to `tools/session_data.py` — fetches last 20 M5 candles for held pairs + major 4 pairs automatically. Updated SKILL_trader.md to reference session_data output instead of requiring a separate fetch. No quality loss — same data, zero model-generated code needed.

## 2026-04-06 — Slack duplicate reply fix: code-level dedup enforcement

**Context**: User reported duplicate Slack replies to the same message, repeatedly. Previous "fix" was prompt-level instruction only (`Slack最終処理ts` in state.md) — Claude sessions could race past it or skip the check entirely.

**Root cause**: Multiple 1-minute cron trader sessions read the same user message. Each independently decided to reply. No code prevented the second reply.

**Changes**:
- Added `tools/slack_dedup.py` — file-based dedup with `fcntl` lock. Records replied-to message ts in `logs/.slack_replied_ts`. Auto-cleans entries >48h
- Modified `tools/slack_post.py` — new `--reply-to {ts}` flag. When provided, checks dedup before posting. If already replied → silently skips (exit 0). After posting → atomically marks ts as replied
- Updated trader SKILL.md — all user message replies now require `--reply-to {USER_MESSAGE_TS}`. Dedup is enforced in code, not by prompt instruction. Removed the manual `Slack最終処理ts` checking requirement

**How it works**: `slack_post.py "reply" --channel C0APAELAQDN --reply-to 1712345678.123456` → if ts is in dedup file → `SKIP_DEDUP` and exit. If not → post → mark ts. File lock prevents race conditions between concurrent sessions.

## 2026-04-05 — News flow logging: narrative evolution tracking

**Context**: news_digest.md was overwritten hourly with no history. Impossible to see whether a macro theme (e.g. "USD strength") was fresh or exhausted. Even for scalps/momentum, knowing "this theme built for 3 hours vs just appeared" changes conviction.

**Changes**:
- Added `tools/news_flow_append.py` — reads current news_digest.md, appends a compact HOT/THEME/WATCH snapshot to `logs/news_flow_log.md`. Keeps 48 entries (48h). Deduplicates by timestamp.
- Added Cowork scheduled task `qr-news-flow-append` — runs at :15 every hour, after qr-news-digest (:00) finishes
- Updated `docs/SKILL_daily-review.md` — Step 1 now reads news_flow_log.md; Step 2 adds question 7 (did macro narrative shift today, and did the trader adapt?)
- Updated CLAUDE.md architecture section to document the new pipeline

## 2026-04-04 — Conviction framework: FOR / Different lens / AGAINST / If I'm wrong

**Context**: Retroactive analysis found 7 conviction-S trades undersized by 70% avg (6,740-13,140 JPY lost). Root cause: trader checked 2-3 familiar indicators, rated B, stopped. Deeper analysis with different indicator categories would have revealed S. Also: 4/1 all-SHORT wipeout (-4,438 JPY) would have been prevented if CCI/Fib (different lens) had been checked — they showed exhaustion.

**Core change**: Conviction is no longer "how many indicators agree" but "how deeply have you looked, and does the whole picture cohere?" New pre-entry format:
```
Thesis → Type → FOR (multi-category) → Different lens (unused category) → AGAINST → If I'm wrong → Conviction + Size
```

**"Different lens" is the key innovation.** Forces checking indicators from categories NOT already used in FOR. Moves conviction BOTH directions:
- B→S upgrade: initial 2 indicators look like B, but Fib + Ichimoku + cluster all support → actually S. This is where the money is
- S→C downgrade: ADX says BEAR, but CCI=-274 and Fib 78.6% say exhausted → abort. This prevents wipeouts

**6 indicator categories defined**: Direction, Timing, Momentum, Structure, Cross-pair, Macro. Categories serve as a checklist of what to look at, not a scoring rubric. Conviction is the trader's judgment of story coherence.

**Files changed**: risk-management.md (full conviction framework + 6 categories + pre-entry block + sizing table), SKILL_trader.md (pre-entry format + conviction guide + sizing), collab_trade/CLAUDE.md (Japanese version of entry format), strategy_memory.md (evidence + updated sizing guidance)

## 2026-04-04 — 3-option position management + structural SL enforcement

**Context**: 4/3 post-mortem with user. Key insight: Opus read charts correctly but managed positions in binary (trail or hold). Missed "cut in profit and re-enter post-NFP." SL placement was ATR×N mechanical, not structural. User couldn't understand SL rationale because there was none beyond a formula.

**SKILL_trader.md**: Added "Position management — 3 options, always" section. For each position when conditions change, trader must write 3 options (A: hold+adjust, B: cut-and-re-enter, C: hold-as-is) then pick one with reasoning. Output format forces evaluation of all options — prevents binary thinking. Added structural SL placement requirement.

**risk-management.md**: Renamed SL section to "Structural placement. No ATR-only." Added structural SL examples (swing low, Fib, DI reversal vs. ATR×N). Added 3-option position management framework. Added 2 new failure patterns (ATR mechanical SL, binary position management).

**protection_check.py**: Added 3-option prompt to output. After listing all positions, prints A/B/C blanks for each position that the trader must fill in. Forces structured thinking at point of output.

**strategy_memory.md**: Added 2 Active Observations — binary position management lesson and structural SL lesson from 4/3.

## 2026-04-03 — Root cause fix: Stop mechanical SL placement

**SKILL.md (trader task)**: Rewrote protection management section. protection_check output is now "data, not orders." Removed "Trailing=NONE is abnormal" rule. Trailing stops are now "for strong trends only, not default." Added hard rules for when NOT to set SL. Trail minimum raised to ATR×1.0 (was ATR×0.6-0.7).

**protection_check.py**: Added `detect_thin_market()` — detects Good Friday, holidays, weekend proximity, low-liquidity hours. During thin market: suppresses Fix Commands, changes "NO PROTECTION" message from warning to "this is correct."

**Root cause**: SKILL.md had rules that forced trader to mechanically attach SL/trail to every position regardless of market conditions. This caused -984 JPY on 4/3 Good Friday when every thesis was correct but every SL got noise-hunted.

## 2026-04-03 — Hard rule: No tight SL on thin markets / holidays

**risk-management.md**: Added "Thin Market / Holiday SL Rule" section. Holiday/Good Friday = no SL or ATR×2.5+ minimum. Spread > 2× normal = discretionary management only. User "SLいらない" = direct order, don't override. Added two new failure patterns.

**strategy_memory.md**: Added to Confirmed Patterns (薄商いのタイトSL=全滅). Added "Thin Market / Holiday Rules" hard rules section.

**Cause**: 4/3 Good Friday — EUR_USD trail 11pip, GBP_USD trail 15pip, AUD_USD SL 10pip all hunted. -984 JPY total. Every thesis was correct. Also Claude closed AUD_JPY after user explicitly removed SL.

## 2026-04-03 — Display all news times in JST

**news_fetcher.py**: All times in `print_summary()` now displayed in JST (`04/04 21:30 JST`) instead of raw UTC ISO strings. Calendar events, headlines, and upcoming events all converted. User preference: JST is easier to read.

## 2026-04-03 — Add event countdown to news summary

**news_fetcher.py**: Added `_event_countdown()` — calculates remaining time to economic events (NFP etc.) and appends `[in 30min]`, `[in 1h01m]`, `[RELEASED]` etc. to calendar output in `print_summary()`. Prevents Claude from miscalculating event countdown by mental arithmetic (20:29 posted "NFPまで約30分" when it was actually ~61 min away).

## 2026-04-03 — Prompt design principle: "Think at the Point of Output"

**CLAUDE.md**: Added core prompt design principle — all prompts must work equally on Opus and Sonnet. The method: embed thinking into output format, not rules or self-questions. Output format forces thinking; rules and preambles don't.

**change-protocol.md**: Added "Prompt Editing Rule" — when editing any prompt, don't add rules or self-questions. Change the output format so thinking is required to produce it.

## 2026-04-03 — Fix Slack notification calculation errors

**trade_performance.py / slack_daily_summary.py — P/L= format fix**:
- Log entries using `P/L=` (with slash) were silently dropped by parsers that only matched `PL=`
- 8 entries affected, including large losses (-17,521 / -3,719 / -2,196 JPY)
- Fixed regex: `PL=` → `P/?L=` (slash optional)

**intraday_pl_update.py — New dedicated script**:
- `intraday-pl-update` task previously had Claude Code generate OANDA API code on-the-fly each session → unreliable calculations (showed 0 closes when there were 4)
- New `tools/intraday_pl_update.py` script fetches from OANDA transactions API with proper page pagination
- Supports `--dry-run` for testing
- SKILL.md updated to use the script instead of inline code generation

## 2026-04-03 — From rules to thinking: trader prompt philosophy rewrite

**Core change**: Replaced rule-based guardrails with self-questioning thinking habits. Works for both Opus and Sonnet.

**SKILL_trader.md — "The Trader's Inner Dialogue"** (replaced Passivity Trap Detection):
- "Am I reading the market or reading my own notes?"
- "If I had zero positions, what would I do?"
- "What changed in the last 30 minutes?"
- "Am I waiting, or hiding behind waiting?"
→ Not a checklist. A thinking habit that prompts genuine market reading.

**SKILL_trader.md — "Before you pull the trigger"** (replaced Anti-repetition hard block):
- "Am I seeing something new, or the same thing again?"
- "Why THIS pair, not the other six?"
- "If this loses, will I understand why?"
- "Am I trading the market or my bias?"
→ No more BLOCKED. Context of EUR_USD 8× repetition preserved as a lesson, not a rule.

**strategy_memory.md — Event Day / Small Wave sections**:
- Rewritten from prescriptive time windows to experience-based observations
- "Before writing 'no entries pre-event', ask how many hours until the event"
- Small wave guide preserved as pattern observation, not entry checklist

**Daily-review set to Opus**: Opus as coach, Sonnet as player.

## 2026-04-03 — Trader anti-repetition check + daily-review enforcement + task re-enable

**Trader SKILL (anti-repetition gate)**:
- Added 3-question check before every entry: same pair×direction×thesis 3+ = blocked
- Added trailing stop width rules: ATR×0.6 minimum, ATR×1.0 for GBP/JPY crosses, ATR×1.2 pre-event

**Daily-review SKILL (strategy_memory enforcement)**:
- Made strategy_memory.md update mandatory with date verification step
- Added pretrade score inflation tracking, R/R analysis, repetitive behavior detection
- "No changes needed" is no longer acceptable output

**Scheduled tasks re-enabled**:
- daily-review (was disabled since ~3/27 → strategy_memory.md stale)
- daily-performance-report, intraday-pl-update, daily-slack-summary

## 2026-04-03 — Slack anti-spam rules: no unsolicited standby messages, duplicate reply prevention

- SKILL_trader.md + scheduled-tasks/trader/SKILL.md: Added "When NOT to post to Slack" section
- Rule: Never post unsolicited "watching/waiting" status messages
- Rule: Only post on trade action, user message reply (once per ts), or critical alert
- Rule: Duplicate reply prevention — check Slack最終処理ts before replying; skip if already replied

## 2026-04-03 — Doc integrity audit: CLAUDE.md / change-protocol / task table

- CLAUDE.md: Split task table into Claude Code tasks + Cowork tasks. qr-news-digest is a Cowork task, not in scheduled-tasks/
- CLAUDE.md: Skills count 36 → 37
- CLAUDE.md + change-protocol.md: Deprecated bilingual sync rule (Japanese reference copies no longer maintained)
- change-protocol.md: Added news_digest.md must-be-English rule
- change-protocol.md: Removed rules-ja/CLAUDE_ja.md/SKILL_ja.md references

## 2026-04-03 — サイジング更新 + CLAUDE.md v8.1同期

**v8.1サイジング反映（risk-management.md）**
- Conviction S: 5000-8000u → **8000-10000u**（v8.1で引き上げ済みだったのにrisk-management.mdが未更新だった）
- Conviction A: 3000-5000u → **5000-8000u**
- Conviction B: 1000-2000u → **2000-3000u**
- Conviction C: 500-1000u → **1000u**
- pretradeスコア(0-10)との対応を明記: S=8+, A=6-7, B=4-5, C=0-3
- rules-ja/risk-management.mdにも同期

**CLAUDE.md修正**
- バージョン: "v8" → "v8.1"
- Self-Improvement Loop: `pretrade_check`が毎セッション実行に見えていた誤解を修正
  → `profit_check + protection_check`（毎セッション冒頭）と `pretrade_check`（エントリー前のみ）を正確に区別
  → 「相場を読む（M5チャート形状）」ステップを追加
  → SESSION_END に `trade_performance.py` が先行することを明記

## 2026-04-03 — CLAUDE.md全面同期修正

**Round 1（誤記・欠落）**
- 誤記修正: 自己改善ループ「毎7分」→「毎1分」
- 矛盾修正: news_digest.md「15分間隔」→「毎時」
- Required Rules on Changes に #6バイリンガル同期・#7スモークテストを追加（change-protocol.mdには既存、CLAUDE.mdに欠落していた）
- メモリシステムUsage・Rulesサブセクションをスリム化（skills/・rules/と重複していた部分を削除）
- skills一覧を更新（2個→主要4個+「全36スキル」表記）

**Round 2（深い精査）**
- アーキテクチャ表を拡張: trader/daily-review/qr-news-digestの3タスクのみ → 実在する6タスク全部記載（daily-performance-report/daily-slack-summary/intraday-pl-update追加）
- タスク定義パス: `~/.claude/scheduled-tasks/trader/SKILL.md` → `~/.claude/scheduled-tasks/`（正本）+ `docs/SKILL_*.md`（参照コピー）に修正
- Scripts表に重要ツール追加: profit_check.py / protection_check.py / preclose_check.py / fib_wave.py（recording.md・technical-analysis.mdで参照されているのに欠落していた）
- 運用ドキュメントから `docs/TRADE_LOG_*.md` を削除（旧形式。現在は collab_trade/daily/ を使用）
- ランタイムファイルに `collab_trade/summary.md` 追加（collab-tradeスキルで参照）
- `logs/trade_registry.json` 削除（不使用）
- Key Directories を整理: `indicators/`（低レベルエンジン）と `collab_trade/indicators/`（quick_calc）を区別して明記
- ユーザーコマンド「トレード開始」に「traderはスケジュールタスク」旨を明記。秘書・共同トレードのスキルトリガーを正確に記述
- CLAUDE_ja.mdに全変更を同期

## 2026-04-02 — SLルール修正 + 証拠金警告追加

問題: SKILL.mdの「エントリー時SL必須」ルールが4/1の実績（SLなし監視→BE/Trail）と矛盾。session_data.pyが証拠金98%でも無警告のため、traderが90%超で新規エントリーするルール違反を起こした。

### SKILL.md修正
- `NO PROTECTION` → 「5分ごと監視中はSLなしOK。ATR×0.8でBE、ATR×1.0でTrailing」に変更。3/31失敗（12時間放置）と4/1成功（5分監視）は別問題だった
- エントリー時のSLをオプション化: TP必須、SL=監視できない時のみ（夜間・離席・低確度）

### tools/session_data.py修正
- 証拠金90%超で `🚨 DANGER — no new entries` 警告追加
- 証拠金95%超で `🚨 CRITICAL — force half-close now` 警告追加
- 背景: 98.23%でも無警告のためtraderが新規エントリーを実行していた

## 2026-03-31 — 全プロンプト英語化（トークンコスト削減）

日本語プロンプトは英語の約2-3倍のトークンを消費する。1分cronのtraderセッションで積算コストが大きいため、全プロンプトを英語化。

### 変更内容
- `.claude/rules/` 6ファイル → 英語版に置換。日本語版は `.claude/rules-ja/` に保存
- `CLAUDE.md` → 英語版に置換。日本語版は `CLAUDE_ja.md` に保存
- `scheduled-tasks/*/SKILL.md` (7タスク) → 英語版に置換。日本語版は各ディレクトリに `SKILL_ja.md` として保存
- `change-protocol.md` にルール#6「日英同時編集」追加: プロンプト変更時は英語版と日本語版を必ず同時更新

### ファイル構成
```
.claude/rules/           ← 英語版（運用。自動ロード）
.claude/rules-ja/        ← 日本語版（確認用。ロードされない）
CLAUDE.md                ← 英語版（運用）
CLAUDE_ja.md             ← 日本語版（確認用）
scheduled-tasks/*/SKILL.md    ← 英語版（運用）
scheduled-tasks/*/SKILL_ja.md ← 日本語版（確認用）
```

## 2026-04-01 (7) — ボット思考からプロトレーダー思考への根本転換

問題: 4/1 全5ポジSHORT（GBP_JPY/AUD_JPY/EUR_JPY、全JPYクロス）→ バウンスで全SL hit。「H1 ADX=50 MONSTER BEAR」を30セッション繰り返し同じ結論を出すボット思考。指標は過去の事実を語るだけなのに、未来の保証として扱っていた。含み益（EUR_USD+536円、GBP_JPY+60円）も「テーゼ生きてる」でHOLD→吐き出し。

### SKILL_trader.md大幅改修
1. **判断の起点を逆転**: 指標→行動 を チャートの形→仮説→指標で確認→行動 に変更
2. **Bash②c全面書き直し**: 「値動き確認」→「市場を読め」。3つの問い（勢い/波の位置/味方か敵か）を指標の前に答えさせる
3. **方向バイアスチェック新設**: 全ポジ同方向=危険信号。「なぜ逆方向が1つもないか」を説明させる。LONG/SHORT両方持つのが正常
4. **STEP 1改修**: デフォルトを「切る」に変更。含み益→利確がデフォルト、含み損→「今から入るか？」がNOなら切れ
5. **STEP 3改修**: 「市場の空気を1文で語れ」を強制。指標の羅列ではなく物語を語らせる
6. **失敗パターン5件追加**: 全ポジ同方向全滅、指標転記=分析と錯覚、含み益見殺し、動き切った後に追加、ボット思考ループ
7. **時間配分に「市場を読む」ステップ追加**: 1-2分を値動き観察+バイアスチェックに割り当て

### risk-management.md改修
- 方向バイアスチェックセクション新設（確度ベースサイジングの上に）
- 失敗パターン4件追加（全ポジ同方向全滅、指標転記錯覚、含み益見殺し、動き切った後追加）

### strategy_memory.md追記
- メンタル・行動セクションに4/1の教訓4件追加

### state.md更新
- SL hitされたポジションの事実と反省を記録

## 2026-03-31 (6) — TP推奨を構造的レベルベースに全面改修

問題: protection_check.pyのTP推奨がATR×1.0固定（距離だけの無意味な価格）。swing/cluster/BB/Ichimoku等の構造的レベル（市場が実際に反応する価格）を使っていなかった。M5の構造的データも未活用。

### protection_check.py全面改修
- **find_structural_levels()新設**: H1+M5の全構造的レベルを収集し距離順にソート
  - H1: swing high/low, cluster, BB upper/mid/lower, Ichimoku雲SpanA/B
  - M5: swing high/low, cluster, BB upper/mid/lower
  - LONG→上方向、SHORT→下方向のみ返す
- **TP推奨**: ATR×1.0固定 → 構造的レベルのメニュー表示（最大5候補）。最寄りに「← 推奨」マーカー
- **修正コマンド出力**: `=== 修正コマンド (N件) ===` セクションにコピペで即実行可能なPUTコマンドを表示。SL広すぎ修正・TP修正・Trailing設定のコマンド
- 結果例: GBP_JPY SHORT TP=210.000(ATR×2.5)→候補5つ(M5 BB mid/lower, M5 swing low, M5 cluster, H1 swing low)をATR比付きで表示

## 2026-03-31 (5) — 回転数不足+TP/SL放置+1ペア集中の根本対策

問題: 24時間で4エントリーしかしていない。全9ポジがSL広すぎ(ATR×2.5-3.2)+TP広すぎ(ATR×2.3-5.0)+Trailing=NONE。protection_checkの警告を12時間以上放置。GBP_JPYに5ポジ7375u集中（ナンピン地獄）。ボラ的に7,000-12,000円/日取れるのに+834円。

### SKILL.md改善
1. **protection_check警告→即修正**: 「読むだけで次に行くな」を強調。`SL広すぎ`→即PUT修正。放置した実績（3/31 12時間放置→回転不能）を記載
2. **Trailing=NONEは異常**: 含み益ATR×1.0以上でTrailingないなら即設定。全ポジTrailing=NONEだった事実を明記
3. **回転数の目標値追加**: 3,000円=3回転（最低）、7,000円=3-4ペア×3回転（保守的に取れる）、15,000円=5ペア×3回転
4. **1ペア集中禁止**: 1ペア最大3ポジ推奨、含み損合計-500円超えたら他ペアで稼げ
5. **判断の罠に3パターン追加**: protection_check放置、ナンピン地獄、HOLD=仕事の錯覚
6. **時間配分にprotection_check対応を明記**: 0-1分にTP/SL/Trail修正を含める
7. **「1セッション最低1トレード」削除**: スプ広い時は見送りが正解

## 2026-03-31 (4) — スプレッドガード実装

問題: スプレッドに関するガードレールが一切なかった。bid/askは取得しているのにスプレッドを計算すらしていない。スプ3pipで5pip狙いのスキャルプに入ってRR崩壊。

### session_data.py — スプレッド表示+警告
- PRICES表示にスプレッドpip計算を追加: `USD_JPY bid=158.598 ask=158.606 Sp=0.8pip`
- 2.0pip超で `⚠️ スプ広い` 警告表示

### pretrade_check.py — スプレッドペナルティ(第6軸)
- エントリー前にOANDA APIからリアルタイムスプレッド取得
- 波の大きさ別の利幅目標に対するスプレッド比率を計算
  - 大波(20pip目標), 中波(12pip), 小波(7pip)
  - 30%超 = -2点（RR崩壊。見送れ）、20%超 = -1点（サイズ控えめに）
- 確度スコアに直接影響 → サイジングが自動で下がる

### SKILL_trader.md — スプレッド意識セクション追加
- スプレッドと利幅の関係表（大波/中波/小波 × スプ0.8/1.5/3.0pip）
- スプレッドが広がるタイミング（早朝、指標前後、GBP_JPY常時広い）
- live_trade_logにスプレッド記録: `Sp=1.2pip`

## 2026-03-31 (3) — TP/SL幅の根本修正 + 波サイズ≠ポジサイズ

問題: 全TPが「テーゼ夢ターゲット」(round number)でATR×2.4〜5.1先。SLもATR×2.0〜3.2。つまりTP到達不能、SL hit時は-6,000円級。また、波サイズがポジサイズを制限しており小波=小サイズだった。

### TP/SLの正しい付け方
- **TP**: テーゼ目標(round number)→最寄り構造的レベル(swing/cluster/Fib)に変更。ATR×1.0付近を半TP→残りtrailing
- **SL**: ATR×2-3→ATR×1.2に修正。hit時の損失額を明記して妥当性を確認
- **protection_check.py更新**: TP残距離>ATR×2.0で「TP広すぎ」警告、SL>ATR×2.5で「SL広すぎ」警告。構造的レベル(swing_dist, cluster_gap)ベースのTP推奨に変更
- SKILL.md: 「TP/SLの正しい付け方」セクション追加（❌❌✅✅の対比例付き）

### 波サイズ≠ポジサイズ
- **旧**: 小波=2000-3000u、中波=5000-8000u、大波=8000-10000u
- **新**: 確度がサイジングを決める。波サイズはpip目標と保有時間を決めるだけ
- 小波でも確度Sなら8000u。M5でタイミング見れてれば5-10pipでも+400-800円
- pretrade_check.py: サイジング表を確度一本に統一（S=8000-10000u regardless of wave）

### MTF評価の波サイズ対応
- 大波(H4/H1): H4+H1一致で+3点。M5未一致でもペナルティなし（M5はタイミング、セットアップ品質ではない）
- 中波(H1/M5): H1+M5一致で+4点
- 小波(M5/M1): M5+H1背景一致で+3点

## 2026-03-31 (2) — 確度評価の根本修正 + TP/SL/BE保護チェック

問題: pretrade_checkが過去WRしか見ず全部LOW判定(25/30件がLOW)。確度S/A/B/Cがどこにも実装されていない。全7ポジションがTP/SL/Trailなしの裸ポジ。

### pretrade_check.py根本改修
- **セットアップ品質評価を追加(前向き)**: 既存のリスク警告(後ろ向き)に加え、今のテクニカルセットアップの質を0-10で数値化
  - MTF方向一致(0-4点): H4+H1+M5全一致=4, H1+M5=3, H4+H1=2
  - ADXトレンド強度(0-2点): H1 ADX>30で+2
  - マクロ通貨強弱一致(0-2点): 7ペアテクニカルから通貨強弱を自動計算
  - テクニカル複合(0-2点): ダイバージェンス、StochRSI極限、BB位置
  - 波の位置ペナルティ(-2〜+1点): H4極端(CCI±200/RSI極端)で同方向エントリー=-2
- **確度→サイジング直結**: S(8+)=8000-10000u / A(6-7)=5000-8000u / B(4-5)=2000-3000u / C(0-3)=1000u以下
- **実際のテスト結果**: GBP_JPY SHORT→S(8), EUR_JPY SHORT→A(6), USD_JPY LONG→C(0)。今まで全部LOWだったものが正しく差別化された
- 背景: 今まで全エントリーが `pretrade=LOW` でサイズ2000u。LOWで入ってサイズだけ膨らませて-2,253円

### tools/protection_check.py新規作成
- 全ポジのTP/SL/Trailing有無をATRベースで評価
- SL推奨: ATR×1.2(ノイズ耐性)。構造的レベル(cluster)との併記
- TP推奨: 最寄り構造的レベル(ATR×1.0付近) → 半TP + trailing
- BE推奨: 含み益ATR×0.8→BE検討、ATR×1.5→Trailing強く推奨
- SL too tight警告: ATR×0.7未満は「ノイズで刈られるリスク」を警告
- TP広すぎ警告: 残距離>ATR×2.0で警告（ATR何本分かを表示）
- SL広すぎ警告: >ATR×2.5で警告
- session_data.pyのTRADE PROTECTIONS表示と連携

### session flow更新
- Bash②b: `profit_check --all` + `protection_check` を並列実行
- SKILL.md: エントリー前チェックに確度→サイジング表を追加
- recording.md: protection_checkをSTEP 0b-2に組み込み

## 2026-03-31 — 「5分で稼げ」+ サイジング逆転修正

問題: NAV 187kで1日-1,284円。勝ちトレード2000uで+300円、負けトレード10500uで-2,253円。勝つ時に小さく負ける時に大きい。5分セッションの大半を分析テキスト書きに消費。

### SKILL.md改善
1. **「5分で稼げ」時間配分**: 0-1分=データ+判断、1-4分=トレード実行、4-5分=記録。分析テキスト書く時間=稼いでいない時間
2. **サイジング鉄則追加**: 確度S=8000-10000u、確度A=5000-8000u、確度B=2000-3000u、確度C=1000u。自信がある時に大きく張れ
3. **STEP 0簡素化**: fib_wave --all + adaptive_technicalsの毎サイクル実行を廃止。session_data.pyで十分。必要時のみ
4. **波サイズテーブル拡大**: 大波8000-10000u(+1500-3000円/trade)、中波5000-8000u、小波2000-3000u
5. **テーゼポジ以外でスキャルプ**: ホールド中に他ペアのM5/M1チャンスを並行で取れ。2ペアしか触らないのはAIの無駄遣い
6. **risk-management.md整合性修正**: マージン管理をSKILL.md哲学と統一
7. **CLAUDE.md整合性修正**: 同上

6. **指値・TP・SL・トレーリングストップ活用**: 成行のみ→LIMIT/TP/SL/Trailing全活用。セッション間も自動で稼ぐ/守る。コード例付き
7. **session_data.pyにPENDING ORDERS + TRADE PROTECTIONS追加**: 毎セッション冒頭で指値の状態と全ポジのTP/SL有無を表示。「⚠️ NO PROTECTION」で裸ポジを警告
8. **oanda-api.md更新**: 注文タイプ一覧（MARKET/LIMIT/TP/SL/Trailing/Cancel）追加

- 背景: 「おれだったらこの資産で今日中に3万円稼げる」。15pip×20回転×10000u=30,000円。同じ相場読みでサイズだけ変えれば今日の利確合計+3,000→+8,000円だった。さらに全7ポジションがTP/SL/Trail全てなし=セッション間は完全無防備だった

## 2026-03-30 (3) — 回転思考の根本改善 + 「波のどこにいるか」

問題: 方向は当たっている(JPY強テーゼ正解)のに稼げない。利確+3,047円→同方向に10500u再エントリー→-2,253円吐き出し。H4 CCI=-274(動き切った後)にSHORT新規。

### SKILL.md改善
1. **「動き切った後は逆を取れ」**: H4 CCI±200超/RSI極端の時、利確後に同方向再エントリー禁止。バウンス方向で小さく取り、バウンス天井でテーゼ方向に再エントリー = 本当の回転
2. **セッション内で値動きを「観る」**: M1キャンドルを判断前後で2回見る。指標(過去)ではなくM1(今)で勢いを感じる
3. **確定利益を守れ**: 利確直後に前回以上のサイズで同方向エントリー = 倍賭け。再エントリーは同サイズ以下
4. **マージン圧力ルール修正**: 「60%=怠慢→入れ」→「60%未満ならチャンスを見逃してないか自問。ただしマージン自体はエントリー理由にならない」
5. **アクション強制ルール撤去**: 「5回連続HOLDで赤信号→何かしろ」→ 撤去。チャンスがなければ待て。行動の強制がオーバートレードを生んだ
6. **回転の定義変更**: 「TP→同方向に再エントリー」→「TP→バウンス取り→テーゼ方向に再エントリー = 波の上下で稼ぐ」

7. **波の大きさに合わせたサイジング**: 大波(H4/H1)3000-5000u / 中波(M5)2000-3000u / 小波(M1)1000-2000u。H1/H4合致しなくてもM1で明らかなバウンスが見えたら小さく取れ
8. **risk-management.md整合性修正**: マージン管理セクションの「常時80-90%で回せ。60%未満=怠慢」をSKILL.md改善と整合するよう修正。「margin_boostはエントリー理由にならない」を明記

- 背景: EUR_JPY +1,379円利確後に10500u積んで-2,253円。GBP_JPY H4 CCI=-241でSHORT新規。方向の正しさ≠エントリータイミングの正しさ
- SKILL.mdはgit管理に移行済み(docs/SKILL_trader.md → symlink)

## 2026-03-30 (2) — traderタスク判断品質改善

問題: traderタスクが30セッション連続「全ポジHOLD」のレポーターと化していた。分析は書くが行動しない。含み益+20pipを-9pipの損切りにしてしまう（テーゼ目標に固執して市場がくれたものを逃す）。

### SKILL.md改善（~/.claude/scheduled-tasks/trader/SKILL.md）
1. **「市場がくれるものを取れ」マインドセット追加**: テーゼ目標への固執を禁止。利確→押し目再エントリーの回転思考を最上位に配置
2. **値動き確認ステップ(Bash②c)追加**: 指標より先にM5キャンドルで勢いと形を確認。ピーク記録をstate.mdに残す
3. **Devil's Advocate**: 含み損-5k超ポジにprofit_checkがHOLDを出した場合、「今すぐ切るべき理由」を3つ挙げて反論する義務
4. **アクション自己監視**: 連続HOLDセッションカウンター。3回連続で黄色、5回連続で赤（何かアクションを取れ）
5. **state.md肥大化防止**: サイクルログは上書き（積み上げ禁止）。目標100行以内
6. **レポーター化・ユーザー指示免罪符の明示的禁止**: 自分の見解を必ず併記、構造変化時はSlackで提案

### schema.py修正
- `get_conn()`に`busy_timeout=5000ms`追加。traderとingest.pyの並行アクセスでpretrade_checkがBusyErrorスキップされていた問題を修正

- 背景: 2026-03-30 USD_JPY +20pip→-9pip損切り。state.md 290行30エントリー中30回「HOLD継続」。pretrade_checkがapsw errorでスキップ

## 2026-03-30 — ニュースパイプライン追加（Cowork → Claude Code）
- **Cowork定期タスク `qr-news-digest`**: 15分間隔でWebSearch×3 + APIパーサでFXニュースを収集し、トレーダー目線の要約を `logs/news_digest.md` に書き出す
- **tools/news_fetcher.py 新規作成**: 3ソース対応（Finnhub経済カレンダー+ヘッドライン、Alpha Vantageセンチメント、Forex Factoryカレンダー）。APIキー未設定でもFF fallbackで動作
- **session_data.py 更新**: NEWS DIGESTセクション追加。Coworkが作成した `news_digest.md` を読んでtraderセッションに提供。鮮度チェック付き
- **設計思想**: テクニカルだけでは「なぜ動いているか」が分からない。マクロ・地政学・要人発言がテーゼの土台。Coworkの強み（WebSearch+LLM要約）を活かし、Claude Codeのtraderは読むだけ
- **APIキー設定（任意）**: `config/env.toml` に `finnhub_token`, `alphavantage_token` を追加するとセンチメント分析が有効に
- 更新ファイル: `tools/news_fetcher.py`(新規), `tools/session_data.py`, `CLAUDE.md`, `docs/CHANGELOG.md`

## 2026-03-27 (5) — デフォルト逆転 + profit_check.py + 1分cron
- **利確デフォルト逆転**: 「なぜ切るか」→「なぜ持つか」に反転。持つ側が根拠を示す設計に
- **profit_check.py新設**: 6軸評価（ATR比・M5モメンタム・H1構造・7ペア相関・S/R距離・ピーク比較）で利確判定
- **cronを7分→1分に短縮**: ロック機構で多重起動防止。セッション終了→最大1分で次が起動。APIコスト変化なし
- 更新ファイル: `tools/profit_check.py`(新規), `risk-management.md`, `recording.md`, `SKILL.md`, `CLAUDE.md`
- 背景: GBP含み益+3,000円→-4,796円の教訓。HOLDバイアスが利確を阻害していた

## 2026-03-27 (4)
- **利確プロトコルの空白を埋めた** — 「利確を問うトリガー」を策定:
  - `risk-management.md`: 「利確を問うトリガー」セクション追加。5つの状況（別ポジ急変・レンジBB mid・M5モメンタム低下・セッション跨ぎ含み益減・300円超）を定義
  - `recording.md`: STEP 0b-2「profit_check」追加。各セッション開始時に含み益ポジを照合する習慣化
  - `strategy_memory.md`: 今日の失敗（GBP含み益消滅）を Active Observations に追記
  - 設計思想: 命令ではなく「問いを強制するトリガー」。HOLD OK、ただし根拠を言語化しろ
  - 背景: 2026-03-27 GBP LONG 含み益+3,000円超がAUD急変中に誰も見ず消滅した教訓

## 2026-03-27 (3)
- **セッション生存率改善** — 3分セッションが短すぎてトレードに辿り着けない問題を解決:
  1. `tools/session_data.py` 新規作成: Bash②③④（テクニカル更新・OANDA・macro_view・adaptive_technicals・Slack・memory recall・performance）を1スクリプトに統合。4回のBash呼び出しが1回に
  2. trader SKILL.md: 309行→約90行に圧縮。ルールは`.claude/rules/`に委譲し重複削除
  3. セッション時間: 3分→5分、cron間隔: 5分→7分
  4. `tools/adaptive_technicals.py`: ROOTパスバグ修正（parents[2]→parent.parent）

## 2026-03-27 (2)
- **自律学習ループ構築** — データが溜まっても行動が変わらない問題を根本解決:
  1. `ingest.py`: OANDA/trades.mdパス統合。OANDAレコードにtrades.mdの質的データ(テーゼ・教訓・regime)をUPDATE。UNKNOWNペア問題修正。live_trade_logからも補完
  2. `parse_structured.py`: regime検出強化(ADX値判定・英語対応)、lesson抽出拡張(plain text対応)、user_call検出拡張(「」なし対応)
  3. `schema.py`: pretrade_outcomesテーブル追加（pretrade_checkの予測 vs 実際のP&L追跡）
  4. `pretrade_check.py`: チェック結果をpretrade_outcomesに自動記録 + 過去の同条件エントリー結末を表示
  5. `tools/daily_review.py` 新規作成: 日次データ収集エンジン。OANDA決済トレード・pretrade結果マッチング・パターン分析
  6. `daily-review` scheduled task 新規作成: 毎日06:00 UTC。Claudeが自分のトレードを振り返り、strategy_memory.mdを進化させる
  7. `strategy_memory.md` 構造リニューアル: Confirmed Patterns / Active Observations / Deprecated / Pretrade Feedback のセクション分割
  8. trader SKILL.md: strategy_memory.mdの読み方を明確化（Confirmed=ルール、Active=参考）
  9. CLAUDE.md: アーキテクチャにdaily-review記載
  - 設計思想: ボット的自動化ではなく、プロトレーダーが毎日振り返って強くなるプロセスの自動化

## 2026-03-27
- **金額トリガー全廃 + マクロ導線接続 + MTF統合** — ユーザー指示で3点同時改修:
  1. risk-management.md: 金額ベース損切り(-500円, -1000円閾値)を全廃。H1構造→テーゼ根拠→反対シグナルの3段階市況判断フローに置換
  2. SKILL.md: 撤退ルールの金額トリガー(-30pip/-500円/ペア別pip上限)を削除。macro_view参照の市況判断に置換。判断フローにmacro_view読みをStep 0として追加
  3. tools/macro_view.py 新規作成: 7ペアtechnicalsから通貨強弱スコア・テーマ判定・MTF一致ペア検出・H1 Div一覧を4行で出力。Bash②に組込み
  - 背景: traderがM5テクニカルだけでボット的判断→低確度トレード乱発→利益を損失で相殺。マクロ視点(通貨強弱・テーマ)と金額に頼らない市況判断で改善
- **メモリ学習ループ修復** — SKILL.md Bash③を改修: 汎用クエリ1本→保有ペアごとのrecall検索に変更。6,260トレードの記憶がトレード判断に活かされるように
- **collab_trade/CLAUDE.md 死参照掃除** — v6で廃止済みのanalyst/secretary/shared_state.json/quality_alert参照を全削除。macro_view.py参照に置換。品質監視は自己監視に変更
- **close_trade.py追加** — ヘッジ口座でPOST /ordersに反対unitsを送ると新規ポジが開くバグ対策。決済は必ずPUT /trades/{id}/closeを使うラッパースクリプト。SKILL.md・oanda-api.mdに決済ルール追記
- **資金効率改善** — マージン目標を90%→70-80%に変更。50%未満=怠慢ルール追加。日次10%には80%水準が必要（計算根拠: NAV18万×25倍×80%=名目363万、7ペア分散で1ペア平均7pipで達成）
- **ボット的撤退ルール改善** — SKILL.mdの段階的撤退テーブル（固定時間・固定pip）をテーゼベース判断に改善。preclose_check組込

## 2026-03-26
- **v8 — traderを正のシステムとして昇格** — リポジトリ全面整理。旧遺産を全てarchive/に統合。ディレクトリをCLAUDE.md, collab_trade/, tools/, indicators/, logs/, config/, docs/, archive/の8個に整理。21GB削減。staleワークツリー30個+、ブランチ130個+削除。パス変更: scripts/trader_tools/ → tools/
- **trade_performance.py v4** — v6ログ形式対応。日別/ペア別/セッション別集計追加
- **v7 — マージン安全ルール** — marginUsed/NAV ≥ 0.9で新規禁止、≥ 0.95で強制半利確。1ペア最大5本。Sonnet化
- **段階的撤退ルール追加** — M5割れ→5分待つ→10分で半分切り→20分+全撤退。-30pip/-500円超は即全撤退。H1テーゼは「すぐ切らない」理由にはなるが「ずっと持つ」理由にはならない。GBP_JPY -237円の教訓 (risk-management.md, SKILL.md, strategy_memory.md)
- **リスク管理ルール全面改訂** — ユーザーレビューに基づき根本見直し:
  - 固定値(+5pip半利確等)全廃止 → ATR対比・テーゼ射程・モメンタム変化の状況判断に変更
  - 「1トレード+300円目標」明記。+40円利確は時間の無駄(実績: 勝率65%でNet-583円、勝ち平均+84円)
  - 損切り判断を金額→テーゼベースに変更。損切り後に戻るパターン対策
  - add-onルール: ピラ/ナンピン両方OK、ただし「新しい根拠を言えるか」が条件。同じ根拠の繰り返しNG
  - ポジション本数制限(最大2本)撤回。本数ではなく根拠の質が問題
  - 確度ベースサイジング(S/A/B/Cランク)導入

## 2026-03-25
- 両建て（ヘッジ）回転戦術をtraderに組込
- メモリシステム恒久改善 — OANDA APIバックフィル6,123件

## 2026-03-24
- Slack通知統合（4点記録セット）
- v6〜v6.5 — trader一本化、Cowork全廃止、2分短命セッション+1分cronリレー

## 2026-03-23
- v5〜v5.1 — 連続セッション、strategy_memory自律学習、ナラティブレイヤー
- live_monitor完全削除

## v1-v4 (2026-03-17〜22)
詳細は `archive/docs_legacy/CHANGELOG_full.md` を参照。
ボットworker体制 → マルチエージェント → trader一本化への進化の記録。
