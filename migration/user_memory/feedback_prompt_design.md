---
name: feedback_prompt_design
description: scalp-traderプロンプト改善で効果があったパターン（Direction Matrix、MTF表、制限語禁止、小ロット許可）
type: feedback
---

scalp-traderプロンプトの改善で効果があったパターンと失敗したパターン。

**効果があった改善:**

1. **Direction Matrix強制出力** — 7ペア×2方向=14候補をスコア1-5で評価。Matrix無しでPASSできない構造にした。LONGを自然に検討するようになった。

2. **Self-Improvement Logの肥大化を止めた** — macro-intelが毎サイクル「DO NOT」「FORBIDDEN」を追加し続けて110行超の制限塊になっていた。40行のMacro Contextセクションに圧縮、Direction Opportunitiesテーブル（機会ベース）に書き換え。macro-intelに「制限語禁止」ルール追加。

3. **MTFシナリオ表** — H1×M5の4パターン組み合わせ表。「H1強トレンド+M5逆行=押し目エントリー（ベスト）」を明記。

4. **小ロット許可** — 「margin insufficientは言い訳にならない。300u,100uでも入れろ」。実際に300uエントリーが出た。

5. **Tokyo session IS active** — 「Asian dead zone」という言葉を禁止。

**失敗した改善:**
- 「M5に逆らうな」（単純ルール）→ MTFで判断できなくなる。すぐにMTFシナリオ表に置換した。

**Why:** プロンプトに「DO NOT」系の制限を書くと、LLMは全部をハードルールとして読んで累積的にブロックする。「機会」として書き直すと行動が変わる。

**How to apply:** プロンプト改善時は常に「制限ではなく機会」フレームで書く。Self-Improvement Logは定期的に圧縮する。macro-intelの書き込みルールも制御する。
