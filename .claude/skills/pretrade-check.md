---
name: pretrade-check
description: "エントリー直前に3層メモリ照合でリスクチェック。過去勝率・ヘッドラインリスク・ユーザーコール・類似記憶を総合判定。"
trigger: "Use BEFORE every trade entry. Also when user says '/pretrade-check', 'リスクチェック', or 'チェック'. This should be called automatically before placing any order."
---

# エントリー前リスクチェック

**全てのエントリー前に実行。** 3層の記憶（SQL統計 + マーケットイベント + ベクトル類似）を照合。

## 実行

```bash
cd /Users/tossaki/App/QuantRabbit/collab_trade/memory && python3 pretrade_check.py <PAIR> <LONG|SHORT> [options]
```

### オプション
- `--adx N` — 現在のADX値
- `--headline TEXT` — アクティブなヘッドライン（"Iran", "FOMC"等）
- `--regime TYPE` — 現在のレジーム（quiet/trending/headline/thin_liquidity）

### 例

```bash
# GBP SHORT + イランヘッドライン
python3 pretrade_check.py GBP_USD SHORT --headline Iran --adx 38

# EUR LONG 通常
python3 pretrade_check.py EUR_USD LONG --regime trending
```

## 判定結果

| レベル | 意味 | 行動 |
|--------|------|------|
| LOW | リスク低 | そのまま実行 |
| MEDIUM | 注意 | サイズ縮小 or SL必須を検討 |
| HIGH | 危険 | サイズ半減 + SL必須、または見送り |

## チェック内容（3層）

1. **SQL: トレード統計** — ペア×方向の勝率、平均PL、最大損失
2. **SQL: マーケットイベント** — 過去のスパイク、ヘッドライン影響
3. **SQL: ユーザーコール** — 直近の読みと方向の一致/不一致、的中率
4. **Vector: 類似記憶** — 似た状況での過去の経験とナラティブ
