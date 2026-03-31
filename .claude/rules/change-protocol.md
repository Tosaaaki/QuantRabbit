# Required Protocol for Changes

When changing code, config, or architecture, execute all of the following:

1. **Update CLAUDE.md**: On architecture changes, update the root CLAUDE.md
2. **Update memory**: Update the relevant memory files
3. **Append to changelog**: Add an entry to `docs/CHANGELOG.md`
4. **Merge to main**: When editing in a worktree, always merge to main
5. **Deploy immediately**: Reflect changes at once. Don't ask.
6. **Bilingual sync**: When editing any prompt file (.claude/rules/, CLAUDE.md, scheduled-tasks/*/SKILL.md), always update BOTH the English version (operational) and the Japanese version (reference) simultaneously. They must stay in sync.
