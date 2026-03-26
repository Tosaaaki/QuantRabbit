# Winner Lane Review Daily Prompt

この prompt は、QuantRabbit の日次 `winner lane review` を同じ観点で回すための固定テンプレートです。毎日の review でそのまま参照し、必要なら query や evaluation window だけ差し替えて使ってください。

## 使い方

1. 先に `scripts/change_preflight.sh "<query>" 3` を実行する。
2. 次の artifact を最新化しておく。
   - `logs/change_preflight_latest.json`
   - `logs/trade_findings_index_latest.json`
   - `logs/lane_scoreboard_latest.json`
   - `logs/entry_path_summary_latest.json`
   - `config/participation_alloc.json`
   - `docs/REPO_HISTORY_LANE_INDEX.md`
   - `docs/TRADE_FINDINGS.md`
3. 下の prompt をそのまま agent へ渡す。

## Prompt

```text
QuantRabbit の daily winner lane review をしてください。

前提:
- local-v2 only。VM/GCP/Cloud Run は見ない。
- 「戦略を増やす」ではなく「winner lane を strategy 化する」前提で判断する。
- review の主眼は strategy 名ではなく `setup_fingerprint / flow_regime / microstructure_bucket` 単位。
- 新しい strategy を提案するのは、十分な sample を持つ winner lane だけ。
- market close / stale window は `market_hold` として扱い、良し悪しを断定しない。

必ず参照するもの:
- `logs/change_preflight_latest.json`
- `logs/trade_findings_index_latest.json`
- `logs/lane_scoreboard_latest.json`
- `logs/entry_path_summary_latest.json`
- `config/participation_alloc.json`
- `docs/REPO_HISTORY_LANE_INDEX.md`
- `docs/TRADE_FINDINGS.md`

やること:
1. current open lane と single-focus lane を確認する。
2. `lane_scoreboard` から winner lane / loser lane / hold lane を抽出する。
3. `REPO_HISTORY_LANE_INDEX` の repeat-risk を重ねて、同じ loser family をまた触ろうとしていないか確認する。
4. `participation_alloc` の setup override が lane decision と整合しているか確認する。
5. 次の 4 区分で判定する。
   - `promote`: winner lane として participation / sizing を太らせてよい
   - `hold`: sample 不足か market_hold でまだ裁けない
   - `quarantine`: loser lane として trim / probability_offset / block 継続
   - `graduate_to_strategy`: 十分な sample を持つ winner lane なので strategy 化候補
6. 必ず `single next action` を 1 本だけ出す。他 family を同時に広げない。

出力形式:
- `Market`
  - market_open / market_hold
  - そう判断した根拠
- `Single Focus`
  - いま最優先で見る lane を 1 本
- `Promote`
  - 昇格してよい lane
  - 根拠: fills, realized_jpy, win_rate, PF, stop_loss_rate, repeat_risk
- `Hold`
  - まだ裁けない lane
  - 根拠: sample 不足 / stale / evaluation window 不足
- `Quarantine`
  - 薄くするか止めるべき lane
  - 根拠: stop_loss_rate, negative realized, repeat-risk, same loser family
- `Graduate Candidate`
  - strategy 化候補があれば 0-2 本
  - 候補が無ければ `none`
- `Single Next Action`
  - 次にやるべき 1 手だけ

禁止:
- winner 抽出前に新しい strategy を増やす提案
- same family の loser lane を複数同時に触る提案
- `long/short` だけで説明する RCA
- `market_hold` 窓での断定的な promote / quarantine 判定
```

## 期待する使い方

- 毎日同じ prompt を使い、評価軸のブレを減らす。
- review の結果は、そのまま `docs/TRADE_FINDINGS.md` の `Next Action` と `Promotion Gate` に接続する。
- `Graduate Candidate` が複数出ても、実装は 1 lane ずつに絞る。
