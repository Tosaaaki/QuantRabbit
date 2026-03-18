# QuantRabbit システム診断レポート (2026-03-17)

## 結論

30日間で **-21,112円**、PF 0.53x、勝率42.2%。
損失の **85%** はコードレベルの3つのバグ/設計欠陥に起因する。
市況を読む以前に、**儲かるトレードを自ら殺している構造** になっている。

---

## 根本原因（3つ + 新規必要な仕組み2つ）

### 原因1: TP圧縮ロジックが勝ちトレードを殺している（-11,217円の直接原因）

**場所**: `execution/risk_guard.py` 行456-503 `_normalize_sl_tp_rr()`

**事象**: TP hit率が54%未満のとき、TPを最大25%圧縮し、SLを拡大する。
```python
# risk_guard.py:499-500
tp_shrink = min(tp_shrink_cap, pressure * _RR_ADAPT_GAIN)
tp_pips *= max(0.2, 1.0 - tp_shrink)  # TPが最小20%まで縮む
```

**なぜ致命的か**:
- M1Scalper: 勝率59%あるのに PL=-5,759円 → **勝ちが小さすぎる**
- RR1-1.5のトレード3,349件で -11,217円（全損失の53%）
- TPが縮む → 勝ち幅が減る → PF低下 → さらにTPが縮む（死のスパイラル）

**データ根拠**:
| RR帯 | トレード数 | 勝率 | 損益 |
|---|---|---|---|
| RR0.8-1 | 694 | 51% | -880 |
| RR1-1.5 | 3,349 | 46% | -11,217 |
| RR1.5-2 | 519 | 41% | -1,675 |

RR1-1.5帯に集中しすぎ。TP圧縮でRRが1付近に潰れている。

---

### 原因2: グローバルDD制御が未接続（57連敗・-7,696円マージンクローズアウトの原因）

**場所**: `execution/risk_guard.py` 行930-935 `can_trade()`

**事象**: `check_global_drawdown()` 関数は実装済みだが `can_trade()` から呼ばれていない。
```python
# risk_guard.py:930-935
def can_trade(pocket: str) -> bool:
    if _trading_paused():
        return False
    # DD ガードは撤廃したまま  ← ここ
    return True
```

**なぜ致命的か**:
- 最大57連敗が放置されている
- 3/2に手動ポジション -8,500units が margin closeout → **-7,696円の単発損失**
- ポケットDD（micro 5%, scalp 3%）もモニタリングのみで enforce なし

---

### 原因3: SLが狭すぎてノイズで刈られる（-7,265円の直接原因）

**場所**: 各worker内のSL計算 + `risk_guard.py` のSL/TP正規化

**事象**: SL < 1.5pips のトレードが 1,643件（27%）あり、SLヒット率 **73%**。
```
SL距離     | トレード数 | 勝率 | 損益    | SLヒット率
SL<1.5 pip | 1,643     | 22%  | -7,265  | 73%
SL1.5-2    | 260       | 33%  | -829    | 53%
SL2-3      | 445       | 27%  | -1,157  | 38%
SL3-5      | 1,214     | 51%  | -817    | 12%  ← ここからまとも
SL>5       | 1,998     | 56%  | -4,933  | 5%
```

**なぜ致命的か**:
- M1のスプレッド（0.3-0.6pip）+ ノイズ（1-2pip）で SL<1.5pip は即刈り
- scalp_extrema_reversal: ATR×0.85 → ATR=1.8pipなら SL=1.53pip
- scalp_ping_5s: さらに狭い固定SL → 228回SLヒットで -6,935円

---

### 保有時間分析（追加根拠）

```
保有時間  | トレード数 | 勝率 | 損益
<30秒    | 1,973     | 17%  | -6,830  ← 即死トレード33%
30-60秒  | 774       | 50%  | -1,485
1-3分    | 1,545     | 60%  | -2,345
3-10分   | 1,334     | 53%  | -4,720
10-60分  | 344       | 38%  | -1,924
>60分    | 43        | 88%  | -5,185  ← マージンクローズアウト含む
```

30秒未満の即死トレード1,973件（全体の33%）が -6,830円。これはSLが狭すぎて即刈りされている証拠。

---

## 具体的修正（コードレベル）

### 修正1: TP圧縮の無効化（最優先、即効性最大）

**ファイル**: `execution/risk_guard.py`
**変更**: `RR_TP_SHRINK_MAX` を環境変数で `0.0` に設定

```bash
# ops/env/quant-order-manager.env に追加
RR_TP_SHRINK_MAX=0.0
RR_SL_EXPAND_MAX=0.0
```

**理由**: TP圧縮は「TP hit率を上げる」ために導入されたが、実際には「勝ち幅を潰す」効果しかない。
M1Scalperの勝率59%がRR1.0で利益を出すには勝率>50%で十分なはずだが、
TP圧縮で勝ち幅がSL幅以下になるとPF<1になる。

**期待効果**: -11,217円（RR1-1.5帯の損失）の50%以上を回復

---

### 修正2: DDガードの接続（安全装置として必須）

**ファイル**: `execution/risk_guard.py`
**変更**: `can_trade()` に `check_global_drawdown()` とポケットDD呼び出しを追加

```python
def can_trade(pocket: str) -> bool:
    if _trading_paused():
        return False
    # グローバルDD: equity drawdown がリミット超えたら entry 拒否
    dd_result = check_global_drawdown()
    if dd_result and dd_result.get("blocked"):
        return False
    # ポケットDD: 各ポケットの累積損失がリミット超えたら entry 拒否
    pocket_dd = _check_pocket_drawdown(pocket)
    if pocket_dd and pocket_dd.get("blocked"):
        return False
    return True
```

**注意**: AGENTS.mdの「後付けの一律EXIT判定は作らない」に従い、**entry拒否のみ**。exitには触らない。

**期待効果**: 57連敗 → 最大10-15連敗に制限、マージンクローズアウト防止

---

### 修正3: SL下限の設定（ノイズ刈り防止）

**ファイル**: 各strategy workerのSL計算箇所
**変更**: SL最低値を 2.0pip に設定（現行は0.85-1.5pip帯に集中）

scalp系: `workers/scalp_extrema_reversal/worker.py`, `workers/scalp_wick_reversal_blend/worker.py`
```python
# 現行: sl_pips = atr_pips * 0.85
# 修正: sl_pips = max(2.0, atr_pips * 1.0)
# units は SL 連動で自動縮小（リスク額一定）
```

ping_5s系: `workers/scalp_ping_5s/worker.py`
```python
# 現行: 固定SL（多くが1.0-1.5pip）
# 修正: sl_pips = max(2.0, current_sl)
```

**理由**: SL 3-5pip帯の勝率51%・SLヒット率12%が「まともなSL距離」の証拠。
SL<1.5pipは73%がSLヒット → ノイズで刈られているだけ。

**期待効果**: -7,265円（SL<1.5pip損失）の70%以上を回復

---

## 新規に必要な仕組み（2つ）

### 新仕組み1: リアルタイム市況レジーム判定の統合（「市況を読む」の実装）

**現状の問題**: 全ストラテジーが同じ閾値で、トレンド相場もレンジ相場も区別せずエントリーしている。
ADXやRSIは個別に見ているが、「今の相場はどのタイプか」を統合判定する仕組みがない。

**提案**: `analysis/market_regime_classifier.py` を新設

```python
class MarketRegime:
    """リアルタイム市況レジーム判定"""

    TRENDING_UP = "trending_up"      # ADX>25, DI+ > DI-, 上昇トレンド
    TRENDING_DOWN = "trending_down"  # ADX>25, DI- > DI+, 下降トレンド
    RANGE_TIGHT = "range_tight"      # ADX<20, BBW<0.001, 狭いレンジ
    RANGE_WIDE = "range_wide"        # ADX<20, BBW>0.002, 広いレンジ
    VOLATILE = "volatile"            # ATR急上昇, BBW拡大, ボラ急変
    CHOPPY = "choppy"                # ADX<15, 方向性なし, 最も危険

def classify_regime(factor_cache) -> MarketRegime:
    """M1/M5/M15の複数タイムフレームから統合判定"""
    adx = factor_cache.get("adx_m1")
    bbw = factor_cache.get("bbw_m1")
    atr = factor_cache.get("atr_pips")
    di_gap = abs(factor_cache.get("di_plus", 0) - factor_cache.get("di_minus", 0))

    # チョッピー（ADX<15 + 方向性なし）= エントリー禁止帯
    if adx < 15 and di_gap < 8:
        return MarketRegime.CHOPPY

    # トレンド判定
    if adx > 25 and di_gap > 15:
        return MarketRegime.TRENDING_UP if ... else MarketRegime.TRENDING_DOWN

    # ボラティリティ急変
    if atr > atr_rolling_mean * 1.5:
        return MarketRegime.VOLATILE

    # レンジ判定
    if bbw < 0.001:
        return MarketRegime.RANGE_TIGHT
    return MarketRegime.RANGE_WIDE
```

**各ストラテジーとの対応表**:
| レジーム | scalp_extrema | M1Scalper | ping_5s | DroughtRevert | MomentumBurst |
|---|---|---|---|---|---|
| trending_up | short拒否 | long優先 | long優先 | 拒否 | long積極 |
| trending_down | long拒否 | short優先 | short優先 | 拒否 | short積極 |
| range_tight | 両方向OK | 拒否 | 拒否 | OK | 拒否 |
| range_wide | OK（SL広め） | OK | OK | OK | 拒否 |
| volatile | 拒否 | 縮小 | 拒否 | 拒否 | OK（SL広め） |
| choppy | **全面拒否** | **全面拒否** | **全面拒否** | **全面拒否** | 拒否 |

**AGENTS.md整合性**: 各workerのstrategy-localで参照する形にすれば、
「共通レイヤは強制的に戦略を選別しない」に適合。shared gateではなく、
各workerが `classify_regime()` を呼んで自分で判断する。

---

### 新仕組み2: アダプティブSL/TP（市況連動の動的リスク管理）

**現状の問題**: SL/TPがATRの固定倍率で決まる。低ボラでSLが狭くなりすぎ、
高ボラでTPが遠すぎて到達しない。

**提案**: `analysis/adaptive_sl_tp.py` を新設

```python
def calculate_adaptive_sl_tp(
    regime: MarketRegime,
    atr_pips: float,
    spread_pips: float,
    strategy_tag: str,
) -> dict:
    """市況レジームに応じてSL/TPを動的決定"""

    # 基本: SLはスプレッドの3倍以上、かつATR×1.0以上
    sl_floor = max(2.0, spread_pips * 3.0, atr_pips * 1.0)

    if regime == MarketRegime.RANGE_TIGHT:
        sl = max(sl_floor, atr_pips * 1.2)
        tp = atr_pips * 1.8  # レンジならTP近め

    elif regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
        sl = max(sl_floor, atr_pips * 1.5)
        tp = atr_pips * 2.5  # トレンドならTP遠め（トレンドに乗る）

    elif regime == MarketRegime.VOLATILE:
        sl = max(sl_floor, atr_pips * 2.0)
        tp = atr_pips * 3.0  # ボラ高ならSL/TP共に広め

    else:  # choppy
        return None  # エントリー禁止

    return {"sl_pips": sl, "tp_pips": tp, "rr": tp / sl}
```

**AGENTS.md整合性**: 各workerのstrategy-localで呼び出す形。
「トレード判断・ロット・利確/損切り・保有調整は固定値運用を避け、
市場状態に応じて常時動的に更新する」に完全適合。

---

## 実装優先順位

| 順位 | 修正内容 | 所要時間 | 期待回復額/月 |
|---|---|---|---|
| **1** | TP圧縮無効化（env変更のみ） | 5分 | +5,000〜8,000円 |
| **2** | SL下限2.0pip設定 | 30分 | +4,000〜5,000円 |
| **3** | DDガード接続 | 30分 | +7,696円（再発防止） |
| **4** | 市況レジーム判定の新設 | 2-3時間 | +3,000〜5,000円 |
| **5** | アダプティブSL/TP | 1-2時間 | 修正2と合算 |

修正1〜3で **直近の損失-21,112円の約80%をカバー**。
新仕組み4-5で **どの相場にも対応できる基盤** が整う。

---

## データソース

- trades.db: 2026-02-15〜2026-03-17、6,014トレード
- entry_thesis JSON: 各トレードの指標スナップショット
- risk_guard.py / order_manager.py: 実行コード精査
