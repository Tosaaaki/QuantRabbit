# QuantRabbit 履歴から見える反復テーマ

- この文書は [REPO_HISTORY_MINUTES.md](./REPO_HISTORY_MINUTES.md) と各 annex を読んだうえで、「この repo が何を何度も悩んでいたか」を整理したメモです。
- current live lane と長期履歴を直接つなぐ索引は [REPO_HISTORY_LANE_INDEX.md](./REPO_HISTORY_LANE_INDEX.md) を参照してください。
- 数値は commit subject の簡易キーワード走査による近似です。厳密な分類ではなく、設計上の重心を見るための補助指標として使ってください。
- `docs/REPO_HISTORY_*` と `scripts/generate_repo_history_minutes.py` だけを触る履歴メンテ commit はノイズになるため、annex 生成時の集計からは外しています。

## 1. 先に結論

- 一番大きいテーマは「entry 数を落とさずに loser lane だけを削ること」です。
- repo は新戦略の追加より、`ping5s`、`M1Scalper`、`reentry`、`negative exit`、`forecast`、`feedback/alloc` の調整に多くの時間を使っています。
- 設計面では `main/cloud/GPT` から `worker-only/V2/local-v2` へかなり明確に収束しています。
- 2026-03 の運用は「setup-scoped 改善」と「anti-loop 規律」が本体で、新機能追加は従です。

## 2. テーマ一覧

| Theme | Approx Commits | Peak Window | 何を意味するか |
| --- | ---: | --- | --- |
| `ping5s` | 208 | `2026-02: 171` | throughput と protection の綱引きが最大の反復論点 |
| `worker-only / V2 / local-v2` | 162 | `2026-02: 96`, `2026-03: 48` | 設計の中心がサービス分離と local 導線へ移った |
| `forecast` | 95 | `2026-02: 89` | 予測は主判断ではなく補助ゲートとして厚くなった |
| `M1Scalper` | 73 | `2026-02: 32`, `2026-01: 20`, `2026-03: 14` | chronic tuning surface の 1 つ |
| `feedback / dynamic alloc / participation` | 53 | `2026-03: 29` | 後段の学習ループが setup-scoped に進化した |
| `reentry` | 39 | `2026-01: 19`, `2026-02: 16` | 再入管理は長期テーマで、勝ち残しと過剰再入の両方が課題 |
| `brain / ollama` | 36 | `2026-03: 34` | 主判定ではなく optional canary として位置づけ直された |
| `negative exit` | 31 | `2026-01: 17` | EXIT の厳格化と救済条件の設計が recurring issue |
| `MomentumBurst` | 21 | `2026-03: 18` | 3 月は micro winner / loser lane の切り分け対象として集中 |
| `RangeFader` | 21 | `2026-03: 12` | blanket trim を避けつつ shallow loser lane を削る題材になった |

## 3. 反復テーマごとの読み取り

### A. `ping5s` は「最大の問題児」ではなく「最大の圧力計」

- `ping5s` 関連は約 `208` commit。
- 特に `2026-02` に `171` commit が集中。
- ここで繰り返しているのは単純なバグ修正ではなく、
  - entry gate が厳しすぎて建たない
  - 緩めると loser lane が増える
  - min-units / perf guard / side filter / force-exit が衝突する
  という live throughput 問題です。
- つまり `ping5s` は 1 戦略というより、「この repo が throughput と risk の均衡をどこに置くか」を一番露骨に示す圧力計として扱うべきです。

### B. 設計の本流は `worker-only -> V2 -> local-v2`

- `worker-only / V2 / local-v2` 系は約 `162` commit。
- `2025-12` に worker-only への転換が始まり、`2026-02` に V2 split が本格化、`2026-03` に local-v2 専用運用へ収束しています。
- ここから見えるのは、この repo の設計上の勝ち筋が
  - monolithic main に機能を足すこと
  ではなく
  - strategy worker / order_manager / position_manager / strategy_control に責務を割ること
  にある、という点です。

### C. `forecast` は主判定にならず、補助ゲートとして定着した

- `forecast` 関連は約 `95` commit、その大半の `89` commit が `2026-02`。
- 予測を強く入れたあとも、最終的な思想は
  - ローカル判定が本体
  - forecast は allow/reduce/block の補助
  に留まっています。
- これは履歴としてかなり一貫していて、予測モデルを「頭脳」にせず「品質フィルタ」にした、という判断です。

### D. `M1Scalper` は chronic tuning surface

- `M1Scalper` は約 `73` commit。
- `2026-01`、`2026-02`、`2026-03` にまたがって出現していて、短期的な hotfix では終わっていません。
- 読み取りとしては、
  - 期待される役割が大きい
  - しかし entry/exit/guard/reentry のどこでも過敏に崩れやすい
  ということです。
- つまり「個別戦略」でもありつつ、「scalp pocket 全体の設計負債が出やすい観測点」でもあります。

### E. `feedback / dynamic alloc / participation` は late-stage optimization の中心

- このテーマは約 `53` commit、うち `2026-03` が `29`。
- 初期は strategy-wide に押し引きしていたものが、履歴後半では
  - setup override
  - live setup match
  - stale artifact no-op
  - active winner only boost
  へ細分化されています。
- これは repo が「広域 multiplier で雑に触る」段階を抜けて、「今の型」にだけ介入する段階へ進んだ証拠です。

### F. `reentry` と `negative exit` は長期未解決テーマ

- `reentry` は約 `39` commit、`negative exit` は約 `31` commit。
- 特に `2026-01` に集中していて、entry と exit を別々に最適化しても解けなかった問題がここに集約されています。
- 要するに、
  - どこで切るか
  - 切ったあとどこで戻るか
  が長く揉めていた。
- この 2 つは将来も再燃しやすいので、「一度解決した」とみなさない方がいいです。

### G. 2026-03 は `MomentumBurst` と `RangeFader` の loser lane 切り分け月

- `MomentumBurst` は `2026-03` に `18` commit。
- `RangeFader` は `2026-03` に `12` commit。
- 3 月に起きていることは broad stop ではなく、
  - shallow loser lane
  - transition / continuation headwind
  - exact setup fingerprint
  を削る、という setup-local 改善です。
- この流れは [TRADE_FINDINGS.md](./TRADE_FINDINGS.md) の anti-loop 規律と整合しています。

## 4. ここから考えるべきこと

### いま優先すべきこと

- 新しい戦略追加より、`setup-scoped trim/boost` と `anti-loop` の定着を優先した方が良い。
- `ping5s` と `M1Scalper` は strategy 固有の話としてではなく、「throughput / guard / exit / reentry が衝突したときの代表ケース」として扱うべき。
- `forecast`、`brain`、`feedback/alloc` は主判定を奪わない位置に置き続けた方が履歴と整合する。

### やらない方がいいこと

- 以前の履歴を見る限り、broad gate や strategy-wide blanket trim は戻りやすい。
- same lane を threshold だけ変えて何度も触ると、`2026-03` に導入した anti-loop 規律と衝突する。
- `cloud/news/GPT` を再び主導線へ戻すのは、履歴全体の収束方向と逆です。

## 5. 一言でまとめると

- この repo は「新しいものを増やして勝つ」より、「entry 数を守りながら loser lane だけを削る」方向へ進化してきた。
- だから次の改善も、広げるより切り分ける方が筋がいいです。
