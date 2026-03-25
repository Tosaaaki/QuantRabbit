---
name: memory-save
description: "共同トレードのセッション記録をmemory.dbに取り込む。日次trades.md/notes.md/state.mdをQAチャンクに分割→Ruri v3で埋め込み→ベクトルDB保存。"
trigger: "Use when user says '/memory-save', 'メモリ保存', or at the end of a collab trade session."
---

# メモリ保存

共同トレードのセッション記録を memory.db（SQLite + sqlite-vec）に取り込む。

## 実行

```bash
cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 ingest.py $(date -u +%Y-%m-%d) --force
```

成功したら stats も表示:

```bash
cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 recall.py stats
```

## 全日付を再取り込み

```bash
cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 ingest.py all --force
```
