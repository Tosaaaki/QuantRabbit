---
name: translate-summary
description: "英語記事URLを渡すと日本語で要約。FX関連ニュースの素早い把握に。"
trigger: "Use when the user provides a URL and says '要約', 'summary', '翻訳', 'translate', or gives a URL and asks what it says."
---

# 翻訳要約スキル

## 使い方

- 「https://... 要約して」
- 「この記事なに？ [URL]」
- 「翻訳: [URL]」

## 実行手順

### Step 1: コンテンツ取得

WebFetchツールで記事内容を取得。

### Step 2: 要約生成

以下の構造で日本語要約:

```
## 📰 記事要約

**タイトル**: [元のタイトル]
**ソース**: [サイト名] | **日付**: [発行日]

### 要点 (3-5行)
- xxx
- xxx
- xxx

### FXへの影響
| 通貨 | 影響 | 理由 |
|------|------|------|
| USD | 🔴弱含み | Fed利下げ期待 |
| JPY | 🟢強含み | リスクオフ |

### トレードへの示唆
- 現在のポジション(AUD_USD SHORT)への影響: xxx
- 新規トレードアイデア: xxx
```

### FX関連記事でない場合

FXと無関係な記事は通常の要約のみ（影響分析は省略）。

## ルール

- 著作権を尊重。原文のコピペは15語以内の引用のみ。
- 要約は原文より大幅に短くする。
- FX影響は推測を明記（「〜と見られる」等）。
