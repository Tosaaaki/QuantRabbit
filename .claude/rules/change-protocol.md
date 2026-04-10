# Required Protocol for Changes

When changing code, config, or architecture, execute all of the following:

1. **Update CLAUDE.md**: On architecture changes, update the root CLAUDE.md
2. **Update memory**: Update the relevant memory files
3. **Append to changelog**: Add an entry to `docs/CHANGELOG.md`
4. **Merge to main**: When editing in a worktree, always merge to main
5. **Deploy immediately**: Reflect changes at once. Don't ask.
6. **English only**: All prompt files (.claude/rules/, CLAUDE.md, SKILL.md) are English-only. No Japanese reference copies maintained (deprecated).
7. **Scheduled task SKILLs are symlinks**: `~/.claude/scheduled-tasks/{task}/SKILL.md` → `docs/SKILL_{task}.md`. Edit the docs/ version only. Never create a plain copy — it will silently diverge. (2026-04-10 incident: quality-audit SKILL was a copy, docs version was updated but scheduled task ran the old version. Chart reading never executed.)
8. **Smoke test after every code change** (2026-04-02 incident: entire memory DB was dead for days because nobody ran the scripts):
   - Run the script and verify it produces actual output. `python3 the_script.py` — if it crashes, you're not done
   - Test in **both** `python3` AND `.venv/bin/python` — two environments exist, both must work
   - New pip dependency → install in both environments immediately
   - Path calculations (`dirname`, `Path.parent`) → print and verify the resolved path
   - "Syntax OK" ≠ "works". Import must pass, processing must run, output must appear
   - When claiming "all fixed", re-run **every** item that was reported broken, one by one. Don't assume

## Prompt Editing Rule — Think at the Point of Output

**All prompts must work equally well on Opus and Sonnet.** This is a hard constraint.

When editing SKILL.md, rules, or strategy_memory:
- **Don't add rules** ("if X then BLOCKED") → Models follow rules mechanically or ignore them. Either way, no thinking.
- **Don't add self-questions** ("ask yourself why") → Read and forgotten by the time the model writes output.
- **Change the output format** so thinking is required to produce it → "I would enter if..." forces a trade plan. "Skip" allows copy-paste.
- **Test**: "Could a model produce this output by copying from last session?" If yes → format is too loose.

See CLAUDE.md "Prompt Design Principle" for the full rationale.

## Language Rule — Token Cost Optimization

**Everything that enters the context window MUST be in English.** Japanese text consumes 2-3x more tokens than equivalent English.

| What | Language | Why |
|------|----------|-----|
| Prompt files (.claude/rules/, CLAUDE.md, SKILL.md) | **English** | Auto-loaded every session = high token cost |
| Tool script output (print/f-strings in .py) | **English** | Read by Claude = token cost |
| state.md, trades.md, internal notes | **English** | Read/written every session = token cost |
| Slack messages to user | **Japanese** | User reads Slack. Not in context window |
**When creating or modifying:**
- New tool scripts → all output strings in English
- New prompt files → English only
- Slack notifications → Japanese (only exception)
- `logs/news_digest.md` → **must be English** (read by trader session every cycle)
