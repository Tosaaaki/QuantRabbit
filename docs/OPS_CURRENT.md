# Ops Current (2026-02-05 JST)

## 1. 運用モード（2025-12 攻め設定）
- マージン活用を 85–92% 目安に引き上げ。
- ロット上限を拡大（`RISK_MAX_LOT` 既定 10.0 lot）。
- 手動ポジションを含めた総エクスポージャでガード。
- PF/勝率の悪い戦略は自動ブロック。
- 必要に応じて `PERF_GUARD_GLOBAL_ENABLED=0` で解除。
- 2026-02-09 以降、`env_prefix` を渡す worker の設定解決は「`<PREFIX>_UNIT_*` → `<PREFIX>_*` のみ」。グローバル `*` へのフォールバックは無効。

## 2. 2026-02-05 JST に `fx-trader-vm` で mask 済みユニット
```
quant-scalp-impulseretrace.service
quant-scalp-impulseretrace-exit.service
quant-m1scalper.service
quant-m1scalper-exit.service
quant-scalp-multi.service
quant-scalp-multi-exit.service
quant-pullback-s5.service
quant-pullback-s5-exit.service
quant-pullback-runner-s5.service
quant-pullback-runner-s5-exit.service
quant-range-comp-break.service
quant-range-comp-break-exit.service
quant-scalp-reversal-nwave.service
quant-scalp-reversal-nwave-exit.service
quant-vol-spike-rider.service
quant-vol-spike-rider-exit.service
quant-tech-fusion.service
quant-tech-fusion-exit.service
quant-macro-tech-fusion.service
quant-macro-tech-fusion-exit.service
quant-micro-pullback-fib.service
quant-micro-pullback-fib-exit.service
quant-manual-swing.service
quant-manual-swing-exit.service
```
