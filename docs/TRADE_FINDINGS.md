# Trade Findings Ledger (Single Source of Truth)

## GCP/VM運用の廃止方針（現行運用）
- GCP/VMを前提とする本番運用は廃止。VMの起動・デプロイ・停止手順は履歴参照または障害時のみ利用する。
- 実務の実行フローはローカルV2導線（`scripts/local_v2_stack.sh`）を最優先とする。
- 旧VM/GCP資料は過去ログ・移行検証用途に限定し、日次運用はローカル導線の実データを優先する。


このファイルは、QuantRabbit の「改善記録」と「敗因記録」の単一台帳兼 change diary です。
以後、同種の記録は必ずここに追記し、他の分散ファイルは作らないこと。

## Rules (Read First)
- 記録先はこのファイルのみ（`docs/TRADE_FINDINGS.md` 固定）。
- 新規の改善/敗因分析を行ったら、必ず 1 エントリ以上追記する。
- 追記順は「新しいものを上」に統一する。
- 事実は VM/OANDA 実測を優先し、日時は UTC と JST を明記する。
- 推測は `Hypothesis` と明示し、事実 (`Fact`) と混在させない。
- 各エントリは「何を変え、何を期待し、結果が良かったか悪かったか、次にどうするか」が追える change diary として書く。
- `logs/trade_findings_draft_latest.json` / `logs/trade_findings_draft_history.jsonl` / `logs/trade_findings_draft_latest.md` は review-only の自動下書きであり、このファイルへ自動追記してはいけない。正式記録はレビュー後に手動で反映する。
- 最低限の記載項目:
  - `Change`（何を変えたか）
  - `Why`（なぜ今それをやるか）
  - `Hypothesis`（どう効く想定か）
  - `Why Not Same As Last Time`（前回と何が違うか。parameter差分ではなく `setup_fingerprint / flow_regime / market regime / evaluation window` の差を書く）
  - `Expected Good`（期待した改善）
  - `Expected Bad`（想定した副作用/悪化条件）
  - `Promotion Gate`（どの条件なら改善を積み上げてよいか）
  - `Escalation Trigger`（どの条件なら次の微調整をやめるか）
  - `Period`（集計期間）
  - `Fact`（数値）
  - `Failure Cause`（敗因）
  - `Improvement`（改善施策）
  - `Verification`（確認方法/判定基準）
  - `Verdict`（`good/bad/mixed/pending`）
  - `Next Action`（維持/戻す/追加調整/再検証条件）
  - `Status`（`open/in_progress/done`）

## Short Template

```md
## YYYY-MM-DD HH:MM JST / short title
- Hypothesis Key:
- Primary Loss Driver:
- Mechanism Fired:
- Do Not Repeat Unless:
- Change:
- Why:
- Hypothesis:
- Why Not Same As Last Time:
- Expected Good:
- Expected Bad:
- Promotion Gate:
- Escalation Trigger:
- Period:
- Fact:
- Failure Cause:
- Improvement:
- Verification:
- Verdict:
- Next Action:
- Status:
```

## Improvement Memory Protocol
- 収益/リスク/ENTRY/EXIT 改善の前に必ず `scripts/change_preflight.sh "<strategy_tag or hypothesis_key or close_reason>"` を実行する。wrapper は local health refresh / USD/JPY 市況確認 / `TRADE_FINDINGS` review を 1 コマンドにまとめる。
- runtime / risk / env 変更の commit 前には `.githooks/pre-commit` が `logs/change_preflight_latest.json` の freshness と staged `docs/TRADE_FINDINGS.md` を確認する。新しい clone / 端末では `scripts/install_git_hooks.sh` を 1 回実行する。
- `scripts/trade_findings_lint.py` は `2026-03-13 20:00` 以降の entry に required fields と `Hypothesis Key` format を要求する。`scripts/trade_findings_index.py` は `logs/trade_findings_index_latest.{json,md}` に latest key / unresolved / dominant loss driver の derived index を出す。
- `Hypothesis Key` は stable な `snake_case` を使い、同じ仮説で別名を増やさない。新しい名前を作る前に既存 key を review/index で確認する。
- 新しい改善エントリには `Hypothesis Key` / `Primary Loss Driver` / `Mechanism Fired` / `Do Not Repeat Unless` を必須で残す。
- `2026-03-14 10:00 JST` 以降の新規 entry には `Why Not Same As Last Time` / `Promotion Gate` / `Escalation Trigger` も必須。`Why Not Same As Last Time` は threshold 差分ではなく、前回と違う decision surface（`setup_fingerprint / flow_regime / market regime / evaluation window`）を書く。これが具体化できない変更は、同じ改善の焼き直しとして実装しない。
- `Mechanism Fired` は `fired=0` や `none` も含めて明記する。発火していない仕組みを、主損失ドライバ不変のまま繰り返さない。
- `same parameter` は禁止対象ではない。過去と同じ値に戻す変更でも、前回と異なる decision surface か評価窓を説明できるなら許容する。説明できないまま同じ値へ戻す変更だけを loop とみなす。
- 直近の同系改善で `Verdict=bad|pending|mixed` かつ `Primary Loss Driver` が同じで、decision surface も同じなら、何を変えるのかを `Why` に書かずに同じ改善を再実施しない。
- close reason が主因なら、`STOP_LOSS_ORDER` / `MARKET_ORDER_TRADE_CLOSE` / `TAKE_PROFIT_ORDER` など dominant reason を `Primary Loss Driver` にそのまま書く。
- 同じ `Hypothesis Key / setup_fingerprint / flow_regime / Primary Loss Driver` では `pending` entry を 1 本だけ持つ。次の tweak は前回 entry の `Promotion Gate` か `Escalation Trigger` を判定してから入れる。
- `tighten -> reopen -> tighten` を同日反復しない。新しい実測か、新しい fingerprint 分離が無い限り threshold の往復を禁止する。
- stale / close / abnormal market window は `market_hold` として扱い、改善 verdict を出さない。reopen 後の評価窓を別に切る。

## 2026-03-16 10:18 JST / local-v2: `MicroLevelReactor-bounce-lower` は pending guard の threshold ではなく env-prefix 解決漏れで `leading_profile` が skip されていた

- Hypothesis Key:
  - `microlevelreactor_bounce_lower_prefix_bug_20260316`
- Primary Loss Driver:
  - `STOP_LOSS_ORDER`
- Mechanism Fired:
  - `entry_leading_profile_skip_due_strategy_tag_prefix_miss`
  - `2026-03-16 09:03 JST`
    の
    `MicroLevelReactor-bounce-lower`
    loser cluster の
    `orders.db.request_json`
    では
    `forecast.allowed=false`,
    `forecast.reason=style_mismatch_range`,
    `forecast.p_up=0.291769`,
    `forecast.expected_pips=-1.0563`
    なのに、
    `entry_path_attribution.leading_profile`
    は
    `status=skip`
    だった。
  - live service env には
    `MICROLEVELREACTOR_ENTRY_LEADING_PROFILE_*`
    と
    `STRATEGY_ENTRY_LEADING_PROFILE_ENABLED=0`
    が同時に入っていた一方、
    旧
    `_strategy_env_prefix_candidates()`
    は
    `MICRO_MULTI`
    と
    `MICROLEVELREACTOR_BOUNCE_LOWER`
    までしか見ず、
    base family
    `MICROLEVELREACTOR`
    を候補に入れていなかった。
- Do Not Repeat Unless:
  - post-deploy の new
    `MicroLevelReactor-bounce-lower`
    sample で
    `leading_profile`
    が
    `skip`
    ではなく
    `reject/pass`
    になったことを確認した後も、
    同じ
    `style_mismatch_range`
    / negative forecast lane が
    `STOP_LOSS_ORDER`
    を作るときだけ、
    threshold 追加 tightening へ進む。

- Change:
  - `execution/strategy_entry.py`
    の
    `_strategy_env_prefix_candidates()`
    を更新し、
    hyphenated setup tag で
    exact prefix だけでなく
    family fallback
    まで順に見るようにした。
    例:
    `MicroLevelReactor-bounce-lower`
    は
    `MICRO_MULTI -> MICROLEVELREACTOR_BOUNCE_LOWER -> MICROLEVELREACTOR_BOUNCE -> MICROLEVELREACTOR`
    を候補にする。
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    に、
    `STRATEGY_ENTRY_LEADING_PROFILE_ENABLED=0`
    でも
    `MICROLEVELREACTOR_ENTRY_LEADING_PROFILE_*`
    が
    `MicroLevelReactor-bounce-lower`
    に効く回帰を追加した。

- Why:
  - `MicroLevelReactor-bounce-lower`
    は
    2026-03-12
    から pending lane として negative-forecast long を止める方針だったのに、
    2026-03-16 09:03 JST
    の live loser burst でも
    同 lane が 5 本まとめて filled された。
  - threshold をまた積む前に、
    既存 pending guard が本当に発火していたかを確認すると、
    mechanism 自体が skip されていた。

- Hypothesis:
  - hyphenated setup tag でも base strategy family の env prefix を拾えば、
    `MicroLevelReactor-bounce-lower`
    は
    `MICROLEVELREACTOR_ENTRY_LEADING_PROFILE_REJECT_BELOW_LONG=0.44`
    を使えるようになり、
    negative forecast long を
    `leading_profile`
    で reject できる。

- Why Not Same As Last Time:
  - 2026-03-12 16:05 JST
    の
    `MicroLevelReactor-bounce-lower`
    entry は、
    同じ loser surface に対して
    threshold
    `0.44`
    を追加した改善だった。
  - 今回はその threshold をさらに触るのではなく、
    exact same decision surface で
    mechanism が
    `skip`
    されていた実装不備を直している。
  - つまり
    `same lane の新 tweak`
    ではなく、
    前回 pending 改善の
    `Mechanism Fired`
    が実質
    `0`
    だったことの修正である。

- Expected Good:
  - `MicroLevelReactor-bounce-lower`
    の
    `leading_profile`
    が
    `skip`
    ではなく
    `reject`
    になり、
    negative forecast long burst を entry 前に止められる。
  - `MomentumBurst-open_long-reaccel`
    など hyphenated setup tag も、
    必要なら base family env を拾える。

- Expected Bad:
  - hyphenated setup tag が base family env を継承するため、
    これまで generic fallback に逃げていた一部 setup が
    想定より tighter になる可能性がある。
  - family fallback を広げることで、
    将来 lane-specific env を置いたときは precedence を監査する必要がある。

- Promotion Gate:
  - post-deploy の new
    `MicroLevelReactor-bounce-lower`
    order で
    `entry_path_attribution.leading_profile.status`
    が
    `skip`
    ではなく、
    `reason`
    と
    `env_prefixes`
    に
    `MICROLEVELREACTOR`
    が含まれること。
  - `tests/execution/test_strategy_entry_forecast_fusion.py -k entry_leading_profile`
    が通ること。

- Escalation Trigger:
  - post-deploy でも
    `MicroLevelReactor-bounce-lower`
    が
    `leading_profile=skip`
    のまま filled されるなら、
    env-prefix ではなく
    runtime process env / entry path serialization
    の別バグとして切り直す。
  - `leading_profile=reject/pass`
    になった後も同 lane の
    `STOP_LOSS_ORDER`
    burst が続くなら、
    次は threshold ではなく
    worker-local signal guard
    / forecast fusion
    の mechanism audit へ進む。

- Period:
  - 調査:
    `2026-03-16 09:44-10:06 JST`
  - 実装/検証:
    `2026-03-16 10:06-10:18 JST`

- Fact:
  - `orders.db`
    の
    `2026-03-16T00:03:12.932984+00:00`
    filled order は
    `strategy_tag=MicroLevelReactor-bounce-lower`
    で、
    `forecast.allowed=false`,
    `forecast.reason=style_mismatch_range`,
    `forecast.expected_pips=-1.0563`,
    `forecast_fusion.entry_probability_after=0.524769`
    だった。
  - 同 order の
    `entry_path_attribution`
    は
    `leading_profile`
    を
    `status=skip`
    で記録していた。
  - running
    `quant-micro-levelreactor`
    process env には
    `MICROLEVELREACTOR_ENTRY_LEADING_PROFILE_ENABLED=1`,
    `MICROLEVELREACTOR_ENTRY_LEADING_PROFILE_REJECT_BELOW_LONG=0.44`,
    `STRATEGY_ENTRY_LEADING_PROFILE_ENABLED=0`
    が共存していた。
  - patch 後の helper は
    `MicroLevelReactor-bounce-lower`
    で
    `['MICRO_MULTI', 'MICROLEVELREACTOR_BOUNCE_LOWER', 'MICROLEVELREACTOR_BOUNCE', 'MICROLEVELREACTOR']`
    を返す。
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/execution/test_strategy_entry_forecast_fusion.py -k 'entry_leading_profile'`
    は
    `6 passed`
    だった。

- Failure Cause:
  - hyphenated setup tag の env lookup が
    base strategy family
    へ fallback せず、
    shared generic
    `STRATEGY_ENTRY_LEADING_PROFILE_ENABLED=0`
    に落ちた。
  - その結果、
    pending だった
    `MicroLevelReactor-bounce-lower`
    の
    `leading_profile`
    threshold は実運用で一度も発火していなかった。

- Improvement:
  - hyphenated setup tag の env prefix 解決を family fallback まで広げ、
    setup-specific tag でも base strategy の dedicated env guard を発火できるようにした。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/execution/test_strategy_entry_forecast_fusion.py -k 'entry_leading_profile'`
  - `ps ewwp <quant-micro-levelreactor pid>`
    で
    `MICROLEVELREACTOR_ENTRY_LEADING_PROFILE_*`
    と
    `STRATEGY_ENTRY_LEADING_PROFILE_ENABLED=0`
    の coexist を確認
  - post-deploy の new
    `orders.db.request_json.entry_path_attribution`
    で
    `leading_profile`
    stage を確認

- Verdict:
  - pending

- Next Action:
  - `quant-micro-levelreactor`
    を含む active worker を反映後、
    new
    `MicroLevelReactor-bounce-lower`
    sample の
    `leading_profile`
    stage を確認する。
  - 同時に
    `MomentumBurst-open_long-reaccel`
    など hyphenated setup tag でも
    unwanted broad tighten が出ていないかを spot check する。

- Status:
  - `pending_live_prefix_validation`

## 2026-03-14 10:55 JST / trade_findings: anti-loop を same parameter 禁止ではなく decision surface 規律へ補正

- Hypothesis Key:
  - `anti_loop_decision_surface_refine_20260314`
- Primary Loss Driver:
  - 同じ数値へ戻ること自体を loop と誤認し、
    本来別 regime の改善まで硬直的に止めてしまうこと
- Mechanism Fired:
  - `scripts/change_preflight.sh`
    review で、
    直前の anti-loop entry が
    `same-lane`
    基準で読める状態を確認した。
  - 現行 lint は field presence を強制しているが、
    `same parameter`
    と
    `same hypothesis/regime/fingerprint`
    の違いまでは文面でしか案内していなかった。
- Do Not Repeat Unless:
  - `Why Not Same As Last Time`
    が decision surface 差分ではなく
    threshold 差分だけを書いた entry が
    review で still 通ると確認できるまでは、
    新しい anti-loop field を増やさず、
    既存 field の定義と checklist を磨く。

- Change:
  - `AGENTS.md`
    /
    `docs/AGENT_COLLAB_HUB.md`
    /
    `docs/OPS_LOCAL_RUNBOOK.md`
    /
    `docs/TRADE_FINDINGS.md`
    の anti-loop 文言を、
    `same parameter`
    禁止ではなく
    `same hypothesis / same regime / same fingerprint`
    の焼き直し禁止へ補正した。
  - `scripts/trade_findings_review.py`
    の checklist も
    decision surface 基準へ更新した。

- Why:
  - live 改善では、
    同じ数値が別 regime で再登場すること自体はある。
  - 問題は数値の一致ではなく、
    前回と同じ仮説面を、
    窓を切り取った理由だけで繰り返すことにある。

- Hypothesis:
  - anti-loop の禁止対象を
    `same parameter`
    から
    `same decision surface`
    へ寄せれば、
    本当に止めるべき loop だけを止めつつ、
    regime/fingerprint が違う改善は残せる。

- Why Not Same As Last Time:
  - 10:15 JST の anti-loop 導入は
    `pending`
    の積み増し防止を主眼にしていたが、
    `同じ数値に戻ること自体は必ずしも悪ではない`
    という整理が明文化されていなかった。
  - 今回は field を増やさず、
    既存の
    `Why Not Same As Last Time`
    を
    decision surface 差分
    として定義し直した。

- Expected Good:
  - agent が
    `same parameter`
    と
    `same hypothesis/regime`
    を混同しにくくなる。
  - 別 regime の再最適化まで不必要に止めずに済む。
  - loop 判定が
    `同じ数値`
    ではなく
    `同じ仮説面`
    に寄る。

- Expected Bad:
  - entry の記述粒度が少し上がる。
  - `decision surface`
    を雑に書くと、
    逆に何でも別改善に見せかけられる。

- Promotion Gate:
  - 次の改善 entry で
    `Why Not Same As Last Time`
    が
    `setup_fingerprint / flow_regime / market regime / evaluation window`
    の差を実際に書く運用になること。
  - review checklist が
    `same parameter`
    ではなく
    `same decision surface`
    を確認対象として表示すること。

- Escalation Trigger:
  - agent が still
    `同じ数値だから禁止`
    または
    `数値が違うから別改善`
    という浅い判断を続けるなら、
    `decision surface`
    を structured field 化する。

- Period:
  - `2026-03-14 10:55 JST` 以降の process rule

- Fact:
  - `2026-03-14 10:54-10:55 JST`
    の preflight は
    `spread=1.20p`,
    `tick_stale=17711.5s`,
    `fills_30m=0`
    で
    `warn`
    だったため、
    runtime verdict 窓ではなく
    process doc 更新として扱った。
  - `trade_findings_review`
    の checklist は、
    anti-loop entry を current rule として表示できた。

- Failure Cause:
  - anti-loop を硬くしすぎると、
    「同じ数値に見えるが別の改善」
    まで止めてしまう。

- Improvement:
  - 禁止対象を
    `same hypothesis / same regime / same fingerprint`
    に再定義し、
    `Why Not Same As Last Time`
    の意味を
    decision surface 差分
    に固定した。

- Verification:
  - `scripts/trade_findings_review.py --query 'decision surface anti_loop same parameter'`
    で checklist と latest entry の文面を確認する。
  - `scripts/trade_findings_lint.py`
    が field presence を維持したまま通ることを確認する。

- Verdict:
  - pending

- Next Action:
  - 次の live 改善 task で、
    同じ数値でも
    decision surface が違う変更を
    `Why Not Same As Last Time`
    で説明できるかを見る。
  - 逆に、
    数値だけ違って
    decision surface が同じ変更が通ろうとしたら却下する。

- Status:
  - done

## 2026-03-14 10:15 JST / trade_findings: 同じ改善を回し続けない anti-loop 規律を導入

- Hypothesis Key:
  - `anti_loop_improvement_protocol_20260314`
- Primary Loss Driver:
  - 同じ lane で `pending` の微調整を積み続け、
    改善が改善にならず平行線になること
- Mechanism Fired:
  - `scripts/trade_findings_review.py`
    の checklist に
    anti-loop 項目を追加。
  - `scripts/trade_findings_lint.py`
    は
    `2026-03-14 10:00 JST`
    以降の entry に
    `Why Not Same As Last Time / Promotion Gate / Escalation Trigger`
    を要求する。
  - `AGENTS.md`
    /
    `docs/AGENT_COLLAB_HUB.md`
    /
    `docs/OPS_LOCAL_RUNBOOK.md`
    の運用規律を更新した。
- Do Not Repeat Unless:
  - 同じ lane の焼き直しが
    `Why Not Same As Last Time`
    を書いても still 通ると確認できるまでは、
    別の台帳や別ルールを増やさず、
    preflight/lint/review を拡張する。

- Change:
  - `AGENTS.md`
    に
    `pending 1本制限`,
    `tighten->reopen->tighten 禁止`,
    `2回連続 bad/pending なら微調整から昇格`
    を追加した。
  - `docs/TRADE_FINDINGS.md`
    の template / protocol に
    `Why Not Same As Last Time / Promotion Gate / Escalation Trigger`
    を追加した。
  - `scripts/trade_findings_review.py`
    と
    `scripts/trade_findings_lint.py`
    を anti-loop 前提へ更新した。

- Why:
  - 直近の運用では、
    同じ
    `Primary Loss Driver`
    のまま
    narrow tweak を積み、
    `pending`
    が積み上がる一方で
    何が前回と違うのか、
    いつ改善と判定するのか、
    いつ微調整をやめるのかが弱かった。
  - これでは改善ではなく
    「同じ場所を回る管理」
    になりやすい。

- Hypothesis:
  - 新規改善 entry に
    `前回と何が違うか`
    と
    `改善/失敗の判定線`
    を強制すれば、
    same-lane の threshold churn が減り、
    微調整から rollback / redesign への昇格が早くなる。

- Why Not Same As Last Time:
  - 既存ルールは
    `Hypothesis Key / Primary Loss Driver / Mechanism Fired / Do Not Repeat Unless`
    までで、
    `今回どこが前回と違うか`
    と
    `成功/失敗の gate`
    を entry 自体へ必須化していなかった。
  - 今回は
    `AGENTS`
    だけでなく
    `lint/review`
    まで変えて、
    書き忘れではなく commit 前に止める形へ変える。

- Expected Good:
  - 同じ lane の改善を
    `pending`
    のまま積み増ししにくくなる。
  - `Promotion Gate`
    が無い tweak と
    `Escalation Trigger`
    が無い tweak が entry 時点で減る。
  - rollback / redesign / lane分離へ昇格すべき局面が早く見える。

- Expected Bad:
  - entry 記述の負荷は増える。
  - 旧 entry は backfill しない限り、
    新形式ほどの検索精度は出ない。

- Promotion Gate:
  - `2026-03-14 10:00 JST`
    以降の新規 entry で
    `Why Not Same As Last Time / Promotion Gate / Escalation Trigger`
    が lint 必須になること。
  - preflight review が新項目を表示し、
    次の改善着手時に agent が読み返せること。

- Escalation Trigger:
  - same-lane の tweak が
    新項目を書いても still 平行線になる、
    または agent が `pending`
    の前回 entry を無視して次の tweak を入れるなら、
    `review/index/guard`
    で duplicate pending lane をさらに強く検出する。

- Period:
  - `2026-03-14 10:00 JST` 以降の運用ルール

- Fact:
  - この変更は runtime ではなく process change であり、
    直接の PnL はまだ持たない。
  - `scripts/trade_findings_lint.py`
    と
    `scripts/trade_findings_review.py`
    の更新で、
    新形式 entry の必須項目を commit 前 review / lint に載せる。

- Failure Cause:
  - 改善 entry が
    `何を試したか`
    は残していても、
    `何が前回と違うか`
    と
    `どの条件なら次の微調整を止めるか`
    を強制していなかった。

- Improvement:
  - anti-loop field の mandatory 化と、
    AGENTS / review / lint の同時更新。

- Verification:
  - `python3 scripts/trade_findings_lint.py`
    が通ること。
  - `python3 scripts/trade_findings_review.py --query 'anti_loop pending' --limit 3`
    で新項目が見えること。

- Verdict:
  - pending

- Next Action:
  - 次の改善 task から、
    新項目を書けない tweak を実際に却下できるかを確認する。
  - same-lane の duplicate pending が still 出るなら、
    次は review/index 側で
    `repeat-risk family`
    の集計を追加する。

- Status:
  - done

## 2026-03-14 09:55 JST / local-v2: `MomentumBurst` の overbought transition long chase を worker-local pullback 条件に戻す

- Hypothesis Key:
  - `momentumburst_transition_pullback_guard_20260314`
- Primary Loss Driver:
  - `STOP_LOSS_ORDER`
- Mechanism Fired:
  - `0`
- Do Not Repeat Unless:
  - reopen 後の active window で
    `MomentumBurst-open_long|long|transition|...|macro:trend_long`
    が再び dominant loser のまま残り、
    今回の pullback guard が live path で発火しても
    `filled`
    と
    `STOP_LOSS_ORDER`
    寄与が縮まらない時だけ次の tightening を検討する。

- Change:
  - `strategies/micro/momentum_burst.py`
    に
    `_transition_long_pullback_ok()`
    を追加し、
    non-reaccel の
    `MomentumBurst-open_long`
    で
    `range_mode=transition`
    /
    `rsi>=66`
    /
    `atr_pips<=3.4`
    /
    `trend_snapshot.direction=long`
    /
    `trend_snapshot gap/adx` が十分
    のときだけ、
    current candle の
    `close_pos<=0.72`
    と
    `upper_wick<=max(1.0p, atr*0.28)`
    を必須化した。
  - `tests/strategies/test_momentum_burst.py`
    に helper block/keep と signal block/keep の回帰を追加した。

- Why:
  - `2026-03-13 22:18 JST`
    反映後の active window
    （`2026-03-13 22:18 JST - 2026-03-14 05:28 JST`）
    では
    `MomentumBurst 14 trades / -49.06 JPY`
    で、
    `STOP_LOSS_ORDER 10 trades / -435.8 JPY`
    が
    `TAKE_PROFIT_ORDER 4 trades / +386.74 JPY`
    を食っていた。
  - loser の本丸は
    `MomentumBurst-open_long|long|transition|...|macro:trend_long`
    で、
    この subset だけで
    `6 trades / -346.744 JPY`
    を削っていた。
  - 対して winner は
    `trend_long`
    が中心で、
    `transition`
    でも surviving winner は
    controlled pullback candle
    （`spin_dn / balanced`）
    だった。
  - つまり broad cadence 不足ではなく、
    `transition long`
    の overbought chase を high-confidence で通している narrow lane が current 主因だった。

- Hypothesis:
  - `transition long`
    の overbought / ultra-low-ATR lane では、
    fresh chase candle を通すより
    pullback/absorption 型の current candle に限定した方が、
    winner の `macro:trend_long`
    を残したまま
    `STOP_LOSS_ORDER`
    を減らせる。

- Expected Good:
  - `MomentumBurst-open_long`
    の
    `transition`
    loser lane
    （`overbought + macro:trend_long + ultra_low ATR + upper-wick chase`）
    だけを strategy-local に薄くできる。
  - `trend_long`
    winner や
    controlled pullback 型の
    `transition long`
    は維持できる。

- Expected Bad:
  - `transition`
    の early continuation winner を一部取りこぼす可能性がある。
  - そのため
    reaccel は対象外にし、
    `range_mode=transition`
    かつ
    `rsi>=66`
    /
    `atr<=3.4`
    /
    strong higher-TF support
    の narrow lane に限定した。

- Period:
  - RCA window:
    `2026-03-13 13:18 UTC - 2026-03-13 20:28 UTC`
    （`2026-03-13 22:18 JST - 2026-03-14 05:28 JST`）
  - market hold check:
    `2026-03-14 00:42 UTC`
    （`2026-03-14 09:42 JST`）

- Fact:
  - active window の
    `MomentumBurst`
    は
    `14 trades / -49.06 JPY`
    で、
    close reason 内訳は
    `STOP_LOSS_ORDER 10 / -435.8 JPY`,
    `TAKE_PROFIT_ORDER 4 / +386.74 JPY`
    だった。
  - worst fingerprints は
    `MomentumBurst-open_long|long|transition|tight_normal|...|macro:trend_long`
    で、
    `entry_probability=0.886-0.959`,
    `trend_snapshot gap=49.865p`,
    `trend_snapshot adx=21.59`
    の high-confidence long が
    `-127.77 / -78.327 / -44.34 / -39.825 / -30.15 / -26.332 JPY`
    を出していた。
  - 一方で
    `2026-03-14 09:42 JST`
    の preflight は
    `USD/JPY 159.727 / 159.739`,
    `spread=1.2p`,
    `tick_stale=13398.2s`,
    `data_lag_ms=15263.3`,
    `fills_30m=0`
    で
    `warn`
    だった。
    そのため live verdict は market reopen 後まで保留する。
  - 実装/検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/strategies/test_momentum_burst.py`
      -> `43 passed`
    - `python3 -m py_compile strategies/micro/momentum_burst.py tests/strategies/test_momentum_burst.py`
      -> 成功

- Failure Cause:
  - 2026-03-12 に loosen した
    `MomentumBurst-open_long`
    の
    `transition`
    lane が、
    strong higher-TF support を根拠に
    overbought chase candle まで high confidence で通していた。
  - non-reaccel long に
    `current candle must be controlled pullback`
    という条件が無く、
    `upper wick`
    を残した shallow continuation を区別できていなかった。

- Improvement:
  - `transition long`
    の overbought macro-trend lane にだけ
    pullback candle guard を追加し、
    `upper-wick chase`
    を worker-local に落とす。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/strategies/test_momentum_burst.py`
    -> `43 passed`
  - `python3 -m py_compile strategies/micro/momentum_burst.py tests/strategies/test_momentum_burst.py`
    -> 成功
  - reopen 後の次
    `30-120m`
    で
    `MomentumBurst-open_long|long|transition|...|macro:trend_long`
    の
    `filled`
    と
    `STOP_LOSS_ORDER`
    が減るかを確認する。

- Verdict:
  - pending

- Next Action:
  - market reopen 後、
    `MomentumBurst-open_long`
    の
    `transition`
    lane が
    `filled`
    から減るか、
    逆に
    `trend_long`
    winner lane を削りすぎていないかを
    `30-120m`
    で確認する。
  - それでも
    `MomentumBurst`
    が loser のままなら、
    次は
    `range_fade`
    側の
    `up_flat`
    loser lane を separate に切る。

- Status:
  - pending

## 2026-03-13 21:35 JST / trade_findings: lint と derived index で台帳の欠損を commit 前に可視化する

- Hypothesis Key:
  - `trade_findings_lint_index`
- Primary Loss Driver:
  - 改善記録の形式崩れや仮説 key の欠損があっても、
    review だけでは見落として同じ改善を繰り返せること
- Mechanism Fired:
  - `scripts/change_preflight.sh`
    が
    `trade_findings_review.py`
    に加えて
    `trade_findings_lint.py`
    と
    `trade_findings_index.py`
    を実行する。
  - `.githooks/pre-commit`
    の
    `scripts/preflight_guard.py`
    は fresh artifact 確認後に
    `trade_findings_lint.py`
    を再実行し、
    台帳が壊れていれば runtime commit を block する。
  - derived index は
    `logs/trade_findings_index_latest.json`
    と
    `logs/trade_findings_index_latest.md`
    に出力する。
- Do Not Repeat Unless:
  - lint が拾えない欠損か、
    index で拾えない重複仮説が明確に確認できるまでは、
    別の記録台帳を増やさず
    `trade_findings_lint.py`
    /
    `trade_findings_index.py`
    を拡張する。
- Change:
  - `scripts/trade_findings_lint.py`
    を追加し、
    `2026-03-13 20:00`
    以降の entry に required field と
    `Hypothesis Key`
    の `snake_case` format を必須化した。
  - `scripts/trade_findings_index.py`
    を追加し、
    latest hypothesis key、
    unresolved entry、
    dominant primary loss driver、
    missing key を
    `logs/`
    に要約するようにした。
  - `scripts/change_preflight.sh`,
    `scripts/preflight_guard.py`,
    `scripts/install_git_hooks.sh`
    と運用 docs を、
    review-only から lint/index 含みの記録導線へ更新した。
- Why:
  - preflight wrapper と commit guard だけでは、
    台帳の field 欠損や
    `Hypothesis Key`
    の揺れまでは防げない。
  - 「記録する」だけでは再利用しにくいので、
    latest key と unresolved をすぐ見返せる derived index が必要だった。
- Hypothesis:
  - entry 形式を lint し、
    hypothesis key / unresolved / loss driver を index 化すれば、
    同じ改善の再実施前に
    「前回の key / verdict / 主因」
    を短時間で確認できる。
- Expected Good:
  - field 欠損や key 揺れのある
    `TRADE_FINDINGS`
    を commit 前に止められる。
  - agent が full ledger を生読みしなくても、
    unresolved と dominant loss driver を
    `logs/trade_findings_index_latest.*`
    から素早く引ける。
- Expected Bad:
  - strict-since 以降の entry を雑に追記すると lint で commit が止まる。
  - derived index は summary なので、
    詳細判断は元の
    `docs/TRADE_FINDINGS.md`
    を読む必要がある。
- Period:
  - 2026-03-13 21:20-21:35 JST
- Fact:
  - `scripts/change_preflight.sh 'entry_probability_reject close_reject_profit_buffer winner_lane_exact_sizing' 4`
    実行で
    `trade_findings_lint.py`
    と
    `trade_findings_index.py`
    が通り、
    `preflight_status=warn`
    /
    `fills_15m=0`
    /
    `fills_30m=0`
    /
    `rejects_30m=45`
    を伴う artifact が更新された。
  - `python3 scripts/preflight_guard.py --paths execution/order_manager.py`
    は block、
    `python3 scripts/preflight_guard.py --paths execution/order_manager.py docs/TRADE_FINDINGS.md`
    は pass した。
  - 旧 entry の
    `close_reject_profit_buffer`,
    `winner_lane_exact_sizing`,
    `ping5s_c_entry_probability_reject`
    は new field 形式へ backfill し、
    review/index で引ける状態にした。
- Failure Cause:
  - これまでは
    `TRADE_FINDINGS`
    の形式保証と一覧性が弱く、
    review を実行しても
    「同じ仮説の別名」
    や
    「required field 欠損」
    を見落とせた。
- Improvement:
  - strict lint + derived index + preflight/hook 統合。
- Verification:
  - `python3 scripts/trade_findings_lint.py`
    が success すること。
  - `python3 scripts/trade_findings_index.py`
    が
    `logs/trade_findings_index_latest.{json,md}`
    を更新すること。
  - `scripts/preflight_guard.py`
    が lint failure を block すること。
- Verdict:
  - pending
- Next Action:
  - 直近の重要 entry を new field 形式へ順次 backfill し、
    `Hypothesis Key`
    の重複や別名を index で減らす。
- Status:
  - done

## 2026-03-13 21:20 JST / trade_findings: commit 前 guard で preflight 未実施の runtime 変更を止める

- Hypothesis Key:
  - `preflight_commit_guard`
- Primary Loss Driver:
  - preflight を実行せずに runtime / risk / env 変更を commit できてしまうこと
- Mechanism Fired:
  - `.githooks/pre-commit`
    に
    `scripts/preflight_guard.py`
    を追加。
  - `scripts/change_preflight.sh`
    は
    `logs/change_preflight_latest.json`
    を書き出すようにした。
  - hook は protected な staged path があるとき、
    fresh artifact と staged
    `docs/TRADE_FINDINGS.md`
    が無い commit を block する。
- Do Not Repeat Unless:
  - commit-time guard で拾えない抜けが確認できるまでは、
    別の enforcement を足さず、
    `scripts/preflight_guard.py`
    の対象 path と条件を拡張する。
- Change:
  - `scripts/preflight_guard.py`
    を追加した。
  - `.githooks/pre-commit`
    と
    `scripts/install_git_hooks.sh`
    を追加した。
  - `scripts/change_preflight.sh`
    は
    review + market summary を
    `logs/change_preflight_latest.json`
    に残すようにした。
  - 過去の重要 entry として
    `close_reject_profit_buffer`,
    `winner_lane_exact_sizing`,
    `ping5s_c_entry_probability_reject`
    を new field 形式で backfill した。
- Why:
  - review と market check の wrapper は作ったが、
    それを通らずに
    `execution/`, `workers/`, `ops/env`
    を commit すること自体はまだ可能だった。
  - 「見返すべき」と書くだけでは弱いので、
    commit 時に最低限の強制力を持たせる必要がある。
- Hypothesis:
  - protected path だけを対象にした軽い pre-commit guard なら、
    docs-only / process-only commit を邪魔せず、
    runtime 変更だけ preflight 実施と
    `TRADE_FINDINGS`
    更新を強制できる。
- Expected Good:
  - `change_preflight.sh`
    を通していない runtime 変更が commit されにくくなる。
  - `docs/TRADE_FINDINGS.md`
    を書かずに risk/runtime 変更だけが積み上がるパターンを減らせる。
- Expected Bad:
  - 新しい clone / 端末では
    `scripts/install_git_hooks.sh`
    を 1 回実行しないと guard が有効にならない。
  - 長時間作業では artifact age が stale になり、
    commit 前に preflight 再実行が必要になる。
- Period:
  - 2026-03-13
- Fact:
  - as-of
    `2026-03-13 21:04 JST`
    の preflight 実行では
    `fills_15m=0 / fills_30m=3 / rejects_30m=31`
    で
    `preflight_status=warn`
    だった。
  - この程度の事実でも commit 前に毎回見えていないと、
    「低稼働 + high reject pressure」のまま次の変更へ進みやすい。
- Failure Cause:
  - preflight 導線はあっても、
    それを通さない commit を止める仕組みが無かった。
- Improvement:
  - commit-time guard + preflight artifact。
- Verification:
  - `python3 scripts/preflight_guard.py --paths execution/order_manager.py`
    は block される。
  - `python3 scripts/preflight_guard.py --paths execution/order_manager.py docs/TRADE_FINDINGS.md`
    は fresh artifact があれば pass する。
  - `scripts/install_git_hooks.sh`
    実行後、
    `git config --get core.hooksPath`
    が
    `.githooks`
    になる。
- Verdict:
  - pending
- Next Action:
  - この clone で
    `scripts/install_git_hooks.sh`
    を実行し、
    次の runtime commit から hook を実働させる。
- Status:
  - done

## 2026-03-13 21:00 JST / trade_findings: change_preflight wrapper で市況確認と review を一体化

- Hypothesis Key:
  - `change_preflight_wrapper`
- Primary Loss Driver:
  - preflight 手順が分散していて、
    市況確認と `TRADE_FINDINGS` review のどちらかが抜けやすいこと
- Mechanism Fired:
  - `scripts/change_preflight.sh`
    を追加。
  - `collect_local_health.sh`
    で local health を更新し、
    `tick_cache / factor_cache / health_snapshot / orders.db`
    から USD/JPY 市況と直近 fills/rejects をまとめた上で、
    `trade_findings_review.py`
    を自動実行する。
- Do Not Repeat Unless:
  - `change_preflight.sh`
    で必要情報が足りないと確認できるまでは、
    別の preflight wrapper を増やさず、
    この script を拡張する。
- Change:
  - `scripts/change_preflight.sh`
    を追加し、
    変更前 preflight を 1 コマンド化した。
  - `AGENTS.md`,
    `docs/AGENT_COLLAB_HUB.md`,
    `docs/OPS_LOCAL_RUNBOOK.md`,
    `docs/CURRENT_MECHANISMS.md`
    の運用手順を、
    raw
    `python3 scripts/trade_findings_review.py ...`
    から wrapper 実行へ更新した。
- Why:
  - raw review command だけだと、
    AGENTS が要求する USD/JPY 市況確認と別手順になり、
    どちらかが飛びやすい。
  - 改善前に毎回やるべきことは
    `local health refresh -> market summary -> TRADE_FINDINGS review`
    の 3 点なので、
    1 コマンド化した方が agent が再現しやすい。
- Hypothesis:
  - review と市況確認を 1 導線に束ねれば、
    「記録は見たが market を見ていない」
    または
    「market は見たが同系改善を見返していない」
    という抜けが減る。
- Expected Good:
  - 収益/リスク/ENTRY/EXIT 改善の前に、
    現在の USD/JPY 市況と同系改善の過去 verdict を同時に確認できる。
  - agent が preflight 実施をログで残しやすくなる。
- Expected Bad:
  - preflight 出力が長くなり、
    query を雑にすると review 側のノイズも増える。
  - `tick_cache / factor_cache / health_snapshot`
    のどれかが欠けると、
    一部の市場項目は
    `n/a`
    になる。
- Period:
  - 2026-03-13
- Fact:
  - as-of
    `2026-03-13 21:00 JST`
    の local 実測では、
    `tick_cache`
    から
    `bid=159.464 / ask=159.472 / spread=0.8p`
    を取れ、
    `factor_cache`
    から
    `M1 atr_pips`
    を取れた。
  - `health_snapshot`
    から
    `mechanism_integrity`,
    `data_lag_ms`,
    `decision_latency_ms`,
    `orders_status_1h`
    を取得でき、
    `orders.db`
    から
    `fills_15m / fills_30m / rejects_30m`
    を追加集計できた。
- Failure Cause:
  - 必須 preflight は前回追加したが、
    market check と review の実行導線がまだ別だった。
- Improvement:
  - one-command preflight wrapper。
- Verification:
  - `bash -n scripts/change_preflight.sh`
  - `scripts/change_preflight.sh "inventory_stress STOP_LOSS_ORDER" 3`
- Verdict:
  - pending
- Next Action:
  - 次の profitability 系タスクでは、
    raw review command ではなく
    `scripts/change_preflight.sh`
    の出力を先に残してから着手する。
- Status:
  - done

## 2026-03-13 20:00 JST / trade_findings: 改善前 review を必須 preflight に昇格

- Hypothesis Key:
  - `improvement_memory_preflight`
- Primary Loss Driver:
  - 直近の dominant loss driver を見ずに同系改善を繰り返すこと自体
- Mechanism Fired:
  - `trade_findings_review.py` を新設。
  - future task では実行前にこの review を必ず走らせる。
- Do Not Repeat Unless:
  - `Hypothesis Key / Primary Loss Driver / Mechanism Fired / Do Not Repeat Unless`
    の 4 欄で再発防止できないと判断できるまでは、
    さらに別の review 導線を増やさない。
- Change:
  - `scripts/trade_findings_review.py`
    を追加し、
    `docs/TRADE_FINDINGS.md`
    から同系改善の
    `Verdict / Primary Loss Driver / Mechanism Fired / Do Not Repeat Unless`
    を agent が着手前に抜き出せるようにした。
  - `AGENTS.md`
    と
    `docs/AGENT_COLLAB_HUB.md`
    に、
    収益/リスク/ENTRY/EXIT 改善の前に上記スクリプトを必ず実行するルールを追加した。
  - `docs/TRADE_FINDINGS.md`
    のテンプレートに
    `Hypothesis Key / Primary Loss Driver / Mechanism Fired / Do Not Repeat Unless`
    を追加した。
- Why:
  - 「細かく網羅的に残し、改善のたびに見返す」を agent が実行可能な手順に落とさないと、
    記録があっても次の変更前に参照されない。
  - 特に
    `inventory_stress_cleanup`
    のように
    `Mechanism Fired=0`
    かつ dominant loss driver 不変のケースは、
    review 手順が無いと同系改善を再発させやすい。
- Hypothesis:
  - 記録の量だけでなく、
    「変更前に読む導線」と
    「同じ改善を再実施してはいけない条件」
    を固定すれば、
    commit 数先行の PDCA を抑えられる。
- Expected Good:
  - 次の改善前に、
    同じ
    `Hypothesis Key`
    の過去 verdict と
    `Primary Loss Driver`
    を必ず確認できる。
  - `Mechanism Fired=0`
    の改善を、
    dominant loss driver 不変のまま焼き直す回数が減る。
- Expected Bad:
  - 改善前の review 手順が 1 ステップ増えるので、
    緊急時を除く通常 PDCA の着手は少し遅くなる。
  - entry が雑なまま query を広くしすぎると、
    review 出力がノイズになる。
- Period:
  - 2026-03-13
- Fact:
  - as-of
    `2026-03-13 19:54 JST`
    では
    `inventory_stress_exit=0`
    で、
    target lane の dominant
    `Primary Loss Driver`
    はまだ
    `STOP_LOSS_ORDER`
    だった。
  - つまり「新しい保険を入れた」事実だけでは足りず、
    「その保険が実際に使われたか」を改善前に見返す必要があると確認できた。
- Failure Cause:
  - change diary があっても、
    次の改善前に同系 entry を強制的に review する手順が弱かった。
- Improvement:
  - review script + 必須欄 + AGENTS/HUB 強制手順。
- Verification:
  - future の収益/リスク/ENTRY/EXIT 改善タスクで、
    `python3 scripts/trade_findings_review.py --query ...`
    の実行を先に残してから変更しているかを確認する。
- Verdict:
  - pending
- Next Action:
  - 次回の profitability 系タスクで、
    実際に query を使って過去 entry を読んでから着手する運用を 1 回通す。
- Status:
  - done

## 2026-03-13 18:25 JST / local-v2: `positive -> negative close` の残り本丸を 3 本同時に補修

- Hypothesis Key:
  - `close_reject_profit_buffer`
- Primary Loss Driver:
  - before:
    `close_reject_profit_buffer + STOP_LOSS_ORDER`
  - after:
    pending
- Mechanism Fired:
  - `PrecisionLowVol / DroughtRevert / WickReversalBlend`
    の
    `min_profit_pips`
    と
    protection trigger を前倒しした。
  - `session_open_breakout`
    に positive PnL 中だけ broker protection を早める
    early protection path を追加した。
- Do Not Repeat Unless:
  - `close_reject_profit_buffer`
    か
    `MFE>0 -> STOP_LOSS_ORDER`
    が dominant のまま残り、
    今回の early protection / relaxed min-profit path が実際に発火していると確認できた時だけ再調整する。

- Change:
  - `config/strategy_exit_protections.yaml`
    で
    `PrecisionLowVol`
    /
    `DroughtRevert`
    の
    `min_profit_pips`
    を
    `0.35/0.40 -> 0.10`
    へ下げ、
    `be_profile`
    /
    `tp_move`
    を早めた。
  - 同ファイルで
    `WickReversalBlend`
    の
    `min_profit_pips`
    を
    `0.45 -> 0.20`
    に下げ、
    `be_profile`
    /
    `tp_move`
    /
    `exit_profile.min_hold_sec`
    を前倒しした。
  - `session_open_breakout`
    に
    strategy-level
    `be_profile`
    /
    `tp_move`
    /
    `start_delay_sec`
    を追加した。
  - `workers/session_open/exit_worker.py`
    に、
    既存
    `min_hold_sec=300`
    の negative/candle gate とは別に、
    positive PnL 中だけ broker
    `SL/TP`
    を前倒し更新する
    early protection path を追加した。
  - 回帰として
    `tests/workers/test_session_open_exit_worker.py`
    と
    `tests/execution/test_order_manager_exit_policy.py`
    を追加・更新した。
- Why:
  - `close_reject_profit_buffer`
    の直近事実として、
    `DroughtRevert ticket=460199`
    は
    `min_profit_pips=0.4 / est_pips=0.2 / exit_reason=lock_floor`
    で reject、
    `PrecisionLowVol ticket=460323`
    は
    `min_profit_pips=0.35 / est_pips=0.2 / exit_reason=take_profit`
    で reject されていた。
  - tick 照合では
    `session_open_breakout ticket=459853`
    が
    `MFE=3.5p`
    を見た後、
    `296s`
    で
    `STOP_LOSS_ORDER -4.4p`
    に戻していた。
    一方で
    worker 実装は
    `min_hold_sec=300`
    の前に profit protection を持っていなかった。
  - `WickReversalBlend ticket=460529/460533`
    も
    `MFE=1.7p / 1.3p`
    を見た後に
    `STOP_LOSS_ORDER`
    へ落ちており、
    current protection trigger が遅かった。
- Hypothesis:
  - scalp reversion 系は
    `+0.2p ~ +0.6p`
    の protective close を通せば
    `+を見てから-へ戻す`
    churn をかなり減らせる。
  - `session_open_breakout`
    は
    `loss-cut を遅らせる`
    方針自体は維持しつつ、
    positive 側だけ
    broker protection を早く動かせば、
    macro lane の勝ちを守れる。
- Expected Good:
  - `PrecisionLowVol`
    /
    `DroughtRevert`
    の
    `close_reject_profit_buffer`
    が消える。
  - `session_open_breakout`
    の
    `MFE>0`
    からの
    `STOP_LOSS_ORDER`
    が減る。
  - `WickReversalBlend`
    の短い positive excursion が
    broker protection へ変わる。
- Expected Bad:
  - early protection を前倒しし過ぎると、
    micro/scalp の winner が
    near-BE
    で刈られやすくなる。
  - `session_open_breakout`
    で broker SL/TP 更新頻度が増える。
- Period:
  - 直近
    `2026-03-12 00:00 UTC`
    から
    `2026-03-13 09:20 UTC`
    の trades / orders / local-v2 logs。
- Fact:
  - `459853`
    は
    `tp_pips=7.114 / sl_pips=4.446 / MFE=3.5p / SL hit=296s`
    だった。
  - `460529`
    は
    `MFE=1.7p -> STOP_LOSS_ORDER -2.5p`、
    `460533`
    は
    `MFE=1.3p -> STOP_LOSS_ORDER -2.4p`
    だった。
  - `460199`
    /
    `460323`
    の close reject reason は
    どちらも
    `profit_buffer`
    で、
    est positive の protective close が共通 gate に止められていた。
- Failure Cause:
  - scalp reversion lane では
    strategy-local protective intent より
    shared
    `min_profit_pips`
    が勝っていた。
  - `session_open_breakout`
    は
    `negative exit を遅らせる`
    ための
    `min_hold_sec`
    が、
    positive protection まで一緒に遅らせていた。
- Improvement:
  - scalp 3 strategy は
    near-BE close を通すよう
    `min_profit_pips`
    と
    `be/tp move`
    を strategy-local に緩和した。
  - `session_open_breakout`
    は
    `close_trade`
    ではなく
    broker
    `set_trade_protections`
    で先に守る path を worker-local に入れた。
- Verification:
  - `python3 -m py_compile utils/strategy_protection.py workers/session_open/exit_worker.py tests/workers/test_session_open_exit_worker.py tests/execution/test_order_manager_exit_policy.py`
  - `.venv/bin/pytest tests/workers/test_session_open_exit_worker.py -q`
  - `.venv/bin/pytest tests/workers/test_scalp_wick_reversal_blend_exit_worker.py -q`
  - `.venv/bin/pytest tests/execution/test_order_manager_exit_policy.py -q`
- Verdict:
  - pending
- Next Action:
  - 反映後
    `close_reject_profit_buffer`
    の
    `DroughtRevert / PrecisionLowVol`
    再発有無、
    `session_open_breakout`
    の
    `PROTECT][legacy_set_trade_protections`
    実ログ、
    `WickReversalBlend`
    の
    `STOP_LOSS_ORDER`
    比率を
    `30-120m`
    追う。
- Status:
  - open

## 2026-03-13 17:45 JST / local-v2: `scalp_extrema_reversal_live` の `lock_floor` を near-BE で通す

- Change:
  - `config/strategy_exit_protections.yaml`
    の
    `scalp_extrema_reversal_live`
    で
    `min_profit_pips`
    を
    `0.6 -> 0.1`
    へ下げた。
  - 同 strategy の
    `min_profit_ratio_reasons`
    から
    `lock_floor`
    を外し、
    `take_profit / range_timeout / candle_*`
    だけを
    TP ratio guard 対象にした。
  - `tests/execution/test_order_manager_exit_policy.py`
    に、
    `lock_floor`
    near-BE close が通る回帰を追加した。
- Why:
  - `含み益を見てからマイナス決済`
    の direct count を
    `logs/replay/USD_JPY/USD_JPY_ticks_20260312-20260313.jsonl`
    で取ると、
    `scalp_extrema_reversal_live`
    は
    `5 trades`
    が
    `MFE>=0.5p`
    を見た後に負けで閉じていた。
  - その内
    `460187`
    と
    `459735`
    は
    `quant-order-manager.log`
    で
    `close_reject_profit_buffer`
    を踏んだ後に
    later close されており、
    `lock_floor`
    由来の保護 close が
    min-profit gate に止められていた。
- Hypothesis:
  - `lock_floor`
    は
    TP 手前での early profit-take ではなく、
    既に見えた利益を守る protective close なので、
    `candle/take_profit`
    と同じ
    TP ratio guard
    に入れるべきではない。
  - `min_profit_pips=0.1`
    まで下げれば、
    near-BE の保護 close を通しつつ、
    `candle_*`
    は既存の
    `min_profit_ratio=0.60`
    で止められる。
- Expected Good:
  - `scalp_extrema_reversal_live`
    の
    `positive -> negative`
    giveback を減らせる。
  - `460187`
    型の
    `lock_floor`
    protective close が
    negative close になる前に通る。
- Expected Bad:
  - `min_profit_pips`
    を下げることで、
    too-small profit での close が増える可能性がある。
  - ただし
    `candle_*`
    と
    `take_profit`
    は
    ratio guard を残すので、
    broad な早利確にはならない想定。
- Period:
  - local-v2 recent
    `2026-03-12 08:00 UTC - 2026-03-13 08:15 UTC`
    の
    `scalp_extrema_reversal_live`
    negative closes と
    `logs/local_v2_stack/quant-order-manager.log`
    を照合。
- Fact:
  - negative close with
    `in-trade MFE>=0.5p`
    は
    `scalp_extrema_reversal_live=5`,
    `session_open_breakout=2`,
    `WickReversalBlend=2`,
    `DroughtRevert=2`
    だった。
  - `460187`
    は
    `MFE=1.0p`
    を見た後、
    `close_reject_profit_buffer`
    を挟んで
    `-0.4p`
    で閉じていた。
  - `459735`
    も
    `MFE=0.9p`
    後に
    `close_reject_profit_buffer`
    を踏んで
    `-0.3p`
    close だった。
- Failure Cause:
  - `lock_floor`
    protective close が、
    strategy-level
    `min_profit_pips`
    と
    TP ratio guard
    に巻き込まれていた。
- Improvement:
  - `lock_floor`
    だけは
    protective close
    として near-BE で通し、
    early profit-take を抑えるのは
    `candle/take_profit/range_timeout`
    に限定した。
- Verification:
  - `python3 -m py_compile tests/execution/test_order_manager_exit_policy.py`
  - `.venv/bin/pytest tests/execution/test_order_manager_exit_policy.py -k "close_trade_blocks_extrema_candle_exit_until_tp_ratio or close_trade_allows_extrema_lock_floor_near_be or close_trade_uses_explicit_flow_context_for_negative_close" -q`
- Verdict:
  - pending
- Next Action:
  - 反映後、
    `scalp_extrema_reversal_live`
    の
    `close_reject_profit_buffer`
    と
    negative
    `MARKET_ORDER_TRADE_CLOSE`
    が減るかを見る。
  - まだ
    `positive -> negative`
    が多ければ、
    次は
    `WickReversalBlend`
    family
    と
    `session_open_breakout`
    を同じ手順で切る。
- Status:
  - in_progress

## 2026-03-13 17:25 JST / local-v2: `scalp_extrema_reversal_live` の soft TP を broker TP 近傍まで引き上げ

- Change:
  - `config/strategy_exit_protections.yaml`
    の
    `scalp_extrema_reversal_live`
    で
    `min_profit_ratio`
    を
    `0.20 -> 0.60`
    へ引き上げ、
    `min_profit_ratio_min_tp_pips=2.0`
    と
    `min_profit_ratio_reasons=[take_profit, lock_floor, range_timeout, candle_*]`
    を追加した。
  - `tests/execution/test_order_manager_exit_policy.py`
    に、
    `candle_*`
    soft close が
    TP 比率未達なら
    `close_reject_profit_ratio`
    になる回帰を追加した。
- Why:
  - user 指摘どおり、
    `scalp_extrema_reversal_live`
    は
    broker TP を置いているのに、
    soft market close が早すぎた。
  - `ticket 460291`
    は
    `MARKET_ORDER_TRADE_CLOSE`
    で
    `+1.2p`
    で閉じた
    `14s`
    後に
    broker TP へ到達していた。
  - 追加で sampled した
    `459861, 459913`
    も
    market close 後に
    `tp_touch_s=144s, 46s`
    を記録しており、
    `Extrema`
    の positive exits は
    TP 手前で降り過ぎていた。
  - 一方で
    `460254`
    は早逃げ後に
    `sl_hit_s=27`
    を踏んでおり、
    broad に hold を伸ばすのは危険だった。
- Hypothesis:
  - `scalp_extrema_reversal_live`
    の soft exit 全体を緩めるのではなく、
    `take_profit/lock_floor/range_timeout/candle_*`
    の positive market close だけを
    broker TP の
    `60%`
    以上まで待たせれば、
    early profit cut を減らしつつ
    `460254`
    のような早逃げ必要ケースは残せる。
- Expected Good:
  - `Extrema`
    の
    `+0.8p ~ +1.2p`
    での早利確が減り、
    `broker TP`
    へ近い利幅を取りやすくなる。
  - `MARKET_ORDER_TRADE_CLOSE`
    の小利幅偏重を減らせる。
- Expected Bad:
  - 反転前に market close できず、
    小幅利確が減る。
  - `candle_*`
    soft close の一部が blocked されて、
    giveback が増える可能性がある。
- Period:
  - local-v2 recent
    `2026-03-12 16:00 UTC - 2026-03-13 04:30 UTC`
    の
    `scalp_extrema_reversal_live`
    MARKET_ORDER_TRADE_CLOSE を
    `logs/trades.db`
    と
    `logs/replay/USD_JPY/USD_JPY_ticks_20260312-20260313.jsonl`
    で照合。
- Fact:
  - sampled
    `scalp_extrema_reversal_live`
    market close
    `5` 本のうち、
    `TP_touch<=300s: 2/5`,
    `TP_touch<=600s: 4/5`
    だった。
  - recent 24h の
    `scalp_extrema_reversal_live`
    は
    `STOP_LOSS_ORDER 33 trades / -109.648 JPY`
    に対し、
    `MARKET_ORDER_TRADE_CLOSE 17 trades / +19.32 JPY`
    で、
    positive exits の利幅が薄かった。
- Failure Cause:
  - `scalp_extrema_reversal_live`
    は
    broker TP を持っていても、
    existing soft close guard が
    `TP 20%`
    相当で通っていたため、
    take-profit / candle reversal / lock-floor が
    TP 手前で成立していた。
- Improvement:
  - shared layer を広く変えず、
    `scalp_extrema_reversal_live`
    だけ
    TP ratio guard を引き上げた。
- Verification:
  - `python3 /Users/tossaki/.codex/skills/qr-tick-entry-validate/scripts/tick_entry_validate.py --trades-db logs/trades.db --ticks logs/replay/USD_JPY/USD_JPY_ticks_20260312.jsonl --ticks logs/replay/USD_JPY/USD_JPY_ticks_20260313.jsonl --instrument USD_JPY --ticket 459875 --ticket 459861 --ticket 459845 --ticket 459913 --ticket 460187 --post-close-sec 120`
  - `python3 -m py_compile tests/execution/test_order_manager_exit_policy.py`
  - `.venv/bin/pytest tests/execution/test_order_manager_exit_policy.py -k "close_trade_blocks_extrema_candle_exit_until_tp_ratio or close_trade_uses_explicit_flow_context_for_negative_close" -q`
  - `.venv/bin/pytest tests/workers/test_scalp_level_reject_exit_worker.py -q`
- Verdict:
  - pending
- Next Action:
  - 反映後、
    `scalp_extrema_reversal_live`
    の
    `MARKET_ORDER_TRADE_CLOSE`
    平均利幅と
    `TAKE_PROFIT_ORDER`
    件数が改善するかを見る。
  - giveback が増えるなら、
    next は strategy 全体ではなく
    `short countertrend + volatility_compression`
    の reason 別に再分解する。
- Status:
  - in_progress

## 2026-03-13 16:55 JST / local-v2: `DroughtRevert` の current bad probe を worker-local に遮断

- Change:
  - `workers/scalp_wick_reversal_blend/worker.py`
    の
    `DroughtRevert`
    long pathに、
    `volatility_compression + macro trend_long`
    の中でも
    `projection が深くマイナス`
    かつ
    `di support が弱い`
    `down_flat` reclaim probe を落とす exact guard を追加した。
  - `workers/scalp_wick_reversal_blend/config.py`
    に
    `DROUGHT_WEAK_TREND_LONG_PROBE_*`
    の dedicated threshold を追加した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に、
    current loser mirror の block と
    `06:44 UTC` winner mirror の keep を追加した。
- Why:
  - `f9bbcc3d`
    反映後の recent trades は
    `session_open_breakout -8.36 JPY`
    に加えて
    `DroughtRevert`
    が
    `2026-03-13 16:34:01 JST`
    と
    `16:34:44 JST`
    に
    43 秒差で同じ long setup を 2 本建て、
    `16:35:32 JST`
    に同時
    `STOP_LOSS_ORDER`
    で
    `-2.73 / -2.575 JPY`
    を出していた。
  - この 2 本は
    同一
    `setup_fingerprint=DroughtRevert|long|range_fade|tight_normal|rsi:oversold|atr:mid|gap:down_flat|volatility_compression|macro:trend_long`
    で、
    `projection_score=-0.265 / continuation_pressure=0.431 / setup_quality=0.382 / di_gap=8.4 / price_gap=5.198p / ma_gap=-0.72p`
    に揃っていた。
  - 一方で直前
    `06:44 UTC`
    の winner は
    `tight_thin`
    で、
    `projection_score=-0.14 / di_gap=15.611 / price_gap=4.805p / ma_gap=-0.51p`
    だった。
- Hypothesis:
  - current loser は
    `macro trend_long`
    に against ではなく、
    同 trend の押し戻りを拾う lane の中でも
    `projection が弱く、di support も薄い bad probe`
    だった。
  - `projection<=-0.18`
    /
    `di_gap<=10`
    /
    `price_gap>=5p`
    /
    `|ma_gap|>=0.65p`
    を exact に切れば、
    current bad probe を落としつつ
    `tight_thin`
    の winner は残せる。
- Expected Good:
  - `DroughtRevert`
    の
    `tight_normal + gap:down_flat + macro:trend_long`
    loser probe を減らせる。
  - duplicate fill が来ても、
    そもそもの bad lane を signal 段階で止めやすくなる。
- Expected Bad:
  - 押し目の深い reclaim long を切り過ぎると、
    `DroughtRevert`
    の件数が減る。
  - `di_gap`
    や
    `projection`
    の閾値が厳しすぎると、
    later recovery の winner まで削る可能性がある。
- Period:
  - local-v2 recent trades / orders:
    主に
    `2026-03-13 06:44-07:35 UTC`
    の
    `DroughtRevert`
    fills / closes を確認。
- Fact:
  - post-`f9bbcc3d`
    の recent closed trades は
    `DroughtRevert -5.305 JPY`,
    `session_open_breakout -8.36 JPY`,
    `scalp_ping_5s_c_live -0.07 JPY`
    で、
    `DroughtRevert`
    が current repeat loser だった。
  - 直近24hの
    `DroughtRevert`
    exact setup 集計では、
    `tight_normal + gap:down_flat + atr:mid + macro:trend_long`
    が
    `3 trades / net -16.13 JPY`
    だった一方、
    `tight_thin`
    同系 setup は
    `1 trade / +10.692 JPY`
    だった。
- Failure Cause:
  - 既存の
    `flow_guard`
    と
    `setup_pressure`
    だけでは、
    current loser の
    `projection deeply negative + weak di support`
    probe を block できていなかった。
  - shared trim は効いていたが、
    `103-105 units`
    の repeated loser を防ぐには不十分だった。
- Improvement:
  - broad shared gate や time block ではなく、
    `DroughtRevert`
    worker 内に
    exact weak-trend-long-probe guard を追加した。
- Verification:
  - `python3 -m py_compile workers/scalp_wick_reversal_blend/config.py workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - `.venv/bin/pytest tests/workers/test_scalp_wick_reversal_blend_signal_flow.py -k "drought_revert_blocks_current_down_flat_weak_trend_long_probe or drought_revert_keeps_down_flat_trend_long_when_projection_and_di_support_recover or drought_revert_blocks_weak_long_under_recent_setup_pressure or drought_revert_keeps_strong_long_under_recent_setup_pressure" -q`
- Verdict:
  - pending
- Next Action:
  - 反映後、
    `logs/orders.db`
    で
    `DroughtRevert`
    の
    `tight_normal + gap:down_flat + projection<=-0.18`
    fill が消えるかを確認する。
  - まだ同 lane の repeated fill が残るなら、
    次は
    open-trade aware の exact duplicate guard を
    `DroughtRevert`
    に足す。
- Status:
  - in_progress

## 2026-03-13 16:35 JST / local-v2: `TickImbalance` の trend exhaustion 追随を worker-local に遮断

- Change:
  - `workers/scalp_tick_imbalance/worker.py`
    に
    `TickImbalance` /
    `TickImbalanceRRPlus`
    共通の
    `trend exhaustion`
    guard を追加し、
    `TREND`
    文脈で
    side-aligned extreme
    `RSI + ADX + VWAP gap + ema_slope + MACD hist`
    が揃った伸び切り entry を reject するようにした。
  - 同 worker の
    `_build_entry_thesis()`
    で
    `side`
    と
    `tick_imbalance.mode/direction/exhaustion_guard`
    を監査 payload へ保存するようにした。
  - `tests/workers/test_scalp_tick_imbalance_worker.py`
    を追加し、
    stretched short/long block と
    non-exhausted reclaim long keep を固定した。
- Why:
  - 2026-03-13 16:12 JST 時点の local-v2 市況は
    `USD/JPY bid=159.386 / ask=159.394 / spread=0.8p`
    で、
    `M1 ATR=2.399p / M5 ATR=5.678p / H1 ATR=15.047p`
    と通常帯だった。
  - 直近6hの赤字寄与は
    `scalp_ping_5s_d_live=-23.605 JPY`,
    `scalp_extrema_reversal_live=-13.487 JPY`,
    `TickImbalance=-10.183 JPY`
    だったが、
    `TickImbalance`
    は sample が薄いまま
    1 発の loser がサイズ尾を引く構図だった。
  - 実際の recent loser は
    `2026-03-13 09:05 JST`
    short
    `units=-3834 / -69.012 JPY / dynamic_alloc=None / participation_alloc=None`
    で、
    `RSI=20.57 / ADX=48.25 / range_mode=TREND / vwap_gap=-5.37 / ema_slope_10=-0.913p / macd_hist=-1.840p`
    の底売り追随だった。
  - さらに
    `2026-03-09 07:16 JST`
    long loser
    `+1171 units / -93.68 JPY`
    も
    `RSI=98.24 / ADX=72.77 / vwap_gap=+65.19 / ema_slope_10=+1.488p / macd_hist=+2.513p`
    の天井追いで、side対称な exhaustion pattern が見えた。
- Hypothesis:
  - `TickImbalance`
    の本来の edge は
    reclaim / imbalance continuation
    であり、
    side-aligned extreme
    指標が全部揃った「伸び切り追随」を切っても
    winner cadence を壊しにくい。
  - shared
    `dynamic_alloc`
    が warm-up 前や欠落時でも、
    worker local の exhaustion guard があれば
    no-profile full-size loser を減らせる。
- Expected Good:
  - `TickImbalance`
    の cold-start / low-sample 窓で
    oversized loser を直接減らせる。
  - `entry_thesis`
    に
    `side`
    と
    `tick_imbalance`
    diagnostics が残るため、
    今後の setup-scoped feedback / RCA でも型を追いやすくなる。
- Expected Bad:
  - 強い continuation の late winner まで切って、
    `TickImbalance`
    の件数が少し減る可能性がある。
  - 阈値がタイトすぎると、
    trend follow-through の最後の伸びを取り損ねる。
- Period:
  - local-v2 recent trades / orders:
    主に
    `2026-03-13 00:05-04:37 UTC`
    と
    `2026-03-08-2026-03-13`
    の
    `TickImbalance`
    closed trades を確認。
- Fact:
  - `trades.db`
    直近30dの
    `TickImbalance`
    は
    `4 trades / net -171.575 JPY`
    で、
    loser 3 件のうち 2 件が
    `RSI 20/98`
    帯かつ
    `ADX 48/72`
    の extreme trend follow だった。
  - 一方で同30dの唯一の winner は
    `2026-03-10 19:11 JST`
    long
    `+1.3 JPY`
    で、
    `RSI=62.46 / ADX=25.65 / vwap_gap=-16.38`
    と extreme exhaustion 条件には当たっていなかった。
- Failure Cause:
  - `TickImbalance`
    は
    `range_mode=TREND`
    でも
    side-aligned overextension を strategy-local に弾いておらず、
    shared trim が無い/弱い窓では
    底売り・天井買いを full-size 近くで通していた。
- Improvement:
  - broad shared gate は増やさず、
    `workers/scalp_tick_imbalance/worker.py`
    内だけで
    `trend exhaustion`
    guard を適用。
  - 監査 payload へ
    `tick_imbalance.exhaustion_guard`
    を残し、
    live order/trade の RCA と setup 学習を容易にした。
- Verification:
  - `python3 -m py_compile workers/scalp_tick_imbalance/worker.py tests/workers/test_scalp_tick_imbalance_worker.py`
  - `.venv/bin/pytest tests/workers/test_scalp_tick_imbalance_worker.py -q`
- Verdict:
  - pending
- Next Action:
  - local-v2 反映後、
    `logs/orders.db`
    で
    `TickImbalance`
    の
    `RSI extreme + TREND`
    entry が消えるかを確認する。
  - それでも loser が続くなら、
    次は
    `TickImbalance`
    の recent side-pressure を
    worker-local に足して、
    same-side burst 中の weak entry を追加で薄くする。
- Status:
  - in_progress

## 2026-03-12 23:35 JST / local-v2: `session_open_breakout` の addon-live 経路で technical context を `entry_thesis` へ明示伝搬

- Change:
  - `workers/common/addon_live.py`
    に、
    `technical_context_tfs/fields/ticks/candle_counts`
    などの explicit context 要求を
    `order/intention`
    から
    `entry_thesis`
    へ引き継ぐ passthrough を追加した。
  - `workers/session_open/worker.py`
    の
    `_mk_order()`
    で、
    `M1/M5/H1`
    の technical context と
    `tick_rate`
    を explicit に要求するようにした。
  - `tests/addons/test_addon_live_broker_strategy_tag.py`
    と
    `tests/addons/test_session_open_worker.py`
    に回帰を追加した。
- Why:
  - 2026-03-12 23:17 JST 時点の
    `python3 scripts/prepare_local_brain_canary.py --dry-run`
    では、
    `USD/JPY bid=159.074 / ask=159.082 / spread=0.8p / atr_proxy=3.325p / recent_range_6m=4.4p`
    で
    `market_ready=true`
    だった。
  - 一方で
    `logs/orders.db`
    の直近
    `session_open_breakout`
    fill を見ると、
    `entry_path_attribution`
    に
    `technical_context`
    stage はあるのに、
    `entry_thesis.technical_context.indicators.M1.*`
    と
    `live_setup_context.atr_pips/rsi/adx`
    は空で、
    `setup_fingerprint`
    が
    `rsi:unknown|atr:unknown|gap:unknown`
    に固定されていた。
  - `session_open_breakout`
    を Brain allowlist に入れても、
    addon-live 経路で explicit context 要求が落ちている限り、
    LLM に渡す setup identity の質が上がりきらなかった。
- Hypothesis:
  - addon-live 経路で strategy-local の
    `technical_context_*`
    要求を preserving すれば、
    `session_open_breakout`
    の
    `live_setup_context`
    に
    `atr/rsi/gap/microstructure`
    が入り、
    Brain cache / decision の setup fingerprint が少しまともになる。
  - これは gate の閾値や sizing を変えず、
    Brain が読む文脈だけを増やす変更なので、
    participation 悪化のリスクは小さい。
- Expected Good:
  - 次の
    `session_open_breakout`
    order で
    `technical_context.indicators`
    と
    `live_setup_context`
    の unknown が減る。
  - Brain safe canary の
    `session_open_breakout`
    decision が、
    より current setup に紐づいた fingerprint で記録される。
- Expected Bad:
  - addon-live 共通経路の変更なので、
    他の addon-live worker でも explicit context 要求があれば
    `entry_thesis`
    が少し太くなる。
  - 次の session-open window までは live 実データでの確認ができない。
- Period:
  - `scripts/prepare_local_brain_canary.py --dry-run`: 現時点
  - `logs/orders.db`: 直近24h の `session_open_breakout` fill
- Fact:
  - 直近 fill の
    `session_open_breakout`
    は
    `entry_probability=0.62395`
    /
    `flow_regime=transition`
    /
    `microstructure_bucket=unknown`
    /
    `setup_fingerprint=...|rsi:unknown|atr:unknown|gap:unknown`
    だった。
  - `sqlite3 logs/orders.db`
    の抽出では、
    `entry_thesis.technical_context.indicators.M1.atr_pips/rsi/adx`
    が
    null
    で、
    `technical_context`
    の explicit 要求が載っていなかった。
  - 追加した regression は
    `pytest -q tests/addons/test_addon_live_broker_strategy_tag.py`
    と
    `pytest -q tests/addons/test_session_open_worker.py`
    で
    `2 passed`
    / `2 passed`
    を確認した。
- Failure Cause:
  - Brain canary の拡張自体は進んだが、
    addon-live worker が explicit technical context 要求を
    `entry_thesis`
    へ渡していなかったため、
    session-open lane の setup fingerprint が荒かった。
- Improvement:
  - Brain gate や shared threshold を変えず、
    addon-live 経路で strategy-local な context 要求を preserve する。
- Verification:
  - `pytest -q tests/addons/test_addon_live_broker_strategy_tag.py`
  - `pytest -q tests/addons/test_session_open_worker.py`
  - 次の session-open window で
    `sqlite3 logs/orders.db`
    から
    `session_open_breakout`
    の
    `entry_thesis.technical_context`
    と
    `live_setup_context`
    を再確認
- Verdict:
  - pending
- Next Action:
  - 次の
    `session_open_breakout`
    fill で
    `rsi/atr/gap`
    bucket が埋まるかを確認する。
  - それでも
    `unknown`
    が残るなら、
    session_open 側で
    `range_score` や
    `projection score`
    の top-level 露出を追加する。
- Status:
  - in_progress

## 2026-03-12 23:15 JST / `scalp_extrema_reversal_live` long `volatility_compression` loser lane を worker-local に追加 tightening

- Change:
  - `workers/scalp_extrema_reversal/worker.py`
    に、
    `long_setup_pressure_active`
    中だけ効く
    `long_positive_gap_probe_block`
    を追加した。
  - 条件は
    `volatility_compression`
    /
    `RANGE`
    /
    `not long_supportive`
    /
    `ma_gap_pips>=0.45`
    /
    `dist_low<=0.70`
    /
    `long_bounce<=0.20`
    /
    `tick_strength<=0.20`
    /
    `range_score<=0.46`
    /
    `rsi>=38`
    に限定した。
  - `ops/env/quant-scalp-extrema-reversal.env`
    の
    `SCALP_EXTREMA_REVERSAL_LONG_SETUP_PRESSURE_MIN_TRADES`
    を
    `6 -> 4`
    へ下げ、
    current loser lane に対する反応を早めた。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    に
    shallow positive-gap long の block / deeper long keep の回帰を追加した。
- Why:
  - 2026-03-12 23:00 JST 前後の local-v2 実測では、
    USD/JPY は
    `158.978`
    近辺、
    直近6hの実注文 spread は
    `0.8 pips`
    で安定、
    `data_lag_ms 506`
    /
    `decision_latency_ms 17.4`
    /
    `order_success_rate 0.97`
    /
    `reject_rate 0.03`
    と execution 側の崩れは主因ではなかった。
  - 一方で
    `scalp_extrema_reversal_live`
    の
    `long + volatility_compression`
    は直近約12hで
    `17 trades / net -34.121 JPY / avg -0.688 pips / sl_rate 0.529 / fast_sl_rate 0.412`
    だった。
  - 特に直近 loser は
    `dist_low 0.53-0.83`
    /
    `long_bounce 0.1-0.2`
    /
    `tick_strength 0.1-0.2`
    /
    `range_score 0.41-0.46`
    の shallow reclaim に偏り、
    13:03 UTC の最新 loser では
    `ma_gap_pips=0.635`
    の positive-gap lane が既存 `long_setup_pressure_block`
    を抜けていた。
- Hypothesis:
  - 現在の負け筋は
    「price が mean より上へ少し drift しているのに、
    reclaim の bounce / tick reversal が浅い long」
    で、
    これを recent setup pressure 中だけ落とせば
    scalp_fast の連続小損を減らせる。
  - `min_trades=4`
    へ下げることで、
    loser burst の検知が
    `6 trades`
    待ちより早くなり、
    同種の連続損失を縮められる。
- Expected Good:
  - `scalp_extrema_reversal_live`
    long `volatility_compression`
    の STOP_LOSS burst を減らす。
  - shared gate をいじらず、
    current loser lane だけを strategy-local に削れる。
- Expected Bad:
  - positive-gap reclaim long のうち、
    反発初動が浅い winner も少数取りこぼす可能性がある。
  - `min_trades=4`
    化で setup pressure が早く効きすぎると、
    loser burst 後の初回回復 long も 1-2 本捨てる可能性がある。
- Period:
  - `logs/orders.db`: 直近24h / 直近6h
  - `logs/trades.db`: 直近12h / 直近48h
  - `logs/health_snapshot.json`: 2026-03-12 23:14-23:15 JST 付近
- Fact:
  - `logs/trades.db`
    直近12hの
    `scalp_extrema_reversal_live`
    `long + volatility_compression`
    集計:
    `17 trades / net -34.121 JPY / sl_rate 0.529 / fast_sl_rate 0.412`。
  - 同 lane の
    `loss`
    平均は
    `dist_low=0.387 / bounce=0.208 / tick=0.175 / ma_gap=0.171 / range_score=0.446 / rsi=39.132`、
    `win`
    平均は
    `dist_low=0.585 / bounce=0.320 / tick=0.280 / ma_gap=-0.045 / range_score=0.466 / rsi=42.200`
    だった。
  - 新しい targeted test は
    `pytest -q tests/workers/test_scalp_extrema_reversal_worker.py`
    で
    `30 passed`
    を確認した。
- Failure Cause:
  - 既存 guard は
    `ma_gap<=0`
    を前提にした weak reclaim block へ寄っており、
    positive-gap の shallow reclaim loser が current lane として残っていた。
  - また
    long-side setup pressure は
    `min_trades=6`
    で、
    loser burst の早い段階では反応が遅かった。
- Improvement:
  - recent outcome が悪化している間だけ、
    positive-gap shallow reclaim long を worker local で reject する。
  - setup pressure の起動を
    `4 trades`
    へ前倒しし、
    long loser burst への追従を速める。
- Verification:
  - `pytest -q tests/workers/test_scalp_extrema_reversal_worker.py`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-order-manager,quant-position-manager`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-scalp-extrema-reversal,quant-scalp-extrema-reversal-exit`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-order-manager,quant-position-manager,quant-scalp-extrema-reversal,quant-scalp-extrema-reversal-exit`
  - `logs/local_v2_stack/quant-scalp-extrema-reversal.log`
    と
    `logs/orders.db`
    で
    `long_positive_gap_probe_block`
    相当 lane の entry 減少、
    `scalp_extrema_reversal_live`
    long `volatility_compression`
    の
    `sl_rate / fast_sl_rate / net_jpy`
    を次の 30-90 分で再確認する。
- Verdict:
  - pending
- Next Action:
  - まず 30-90 分で
    `scalp_extrema_reversal_live`
    long `volatility_compression`
    の新規約定数と
    `STOP_LOSS_ORDER`
    比率を見る。
  - cadence が落ちすぎる場合は、
    `long_positive_gap_probe_block`
    の
    `range_score_max`
    または
    `rsi_min`
    を戻す方向で再調整する。
- Status:
  - in_progress

## 2026-03-12 23:20 JST / local-v2: Brain safe canary に `session_open_breakout` を追加

- Change:
  - `ops/env/profiles/brain-ollama-safe.env`
    の
    `BRAIN_STRATEGY_ALLOWLIST`
    を
    `MicroLevelReactor, MomentumBurst-open_long, MomentumBurst-open_short, MicroTrendRetest-short`
    から
    `MicroLevelReactor, MomentumBurst-open_long, MomentumBurst-open_short, MicroTrendRetest-short, session_open_breakout`
    へ更新した。
  - `BRAIN_POCKET_ALLOWLIST=micro`
    と
    `timeout=4s / fail-open`
    は維持し、
    `scalp`
    pocket へは広げていない。
- Why:
  - 2026-03-12 23:11 JST 時点の
    `python3 scripts/prepare_local_brain_canary.py --dry-run`
    では、
    `USD/JPY bid=159.022 / ask=159.030 / spread=0.8p / atr_proxy=3.404p / recent_range_6m=7.4p`
    で
    `market_ready=true`
    だった。
  - 同時点で safe canary は
    `ollama_ready=true`
    /
    `profile_safe=true`
    を維持していたが、
    `logs/brain_state.db`
    直近2hには新しい Brain decision が無く、
    既存 allowlist の lane がその時間帯に発火していなかった。
  - 一方で
    `logs/orders.db`
    直近12hでは
    `session_open_breakout`
    が
    `8 rows / 2 filled`
    で live 発火しており、
    `entry_thesis`
    に
    `setup_fingerprint=session_open_breakout|{side}|transition|unknown|...`
    と
    `flow_regime=transition`
    が記録されていた。
  - `logs/trades.db`
    直近7dでは
    `session_open_breakout`
    が
    `8 trades / -2.8 JPY / avg -0.35 JPY`
    で、
    `MicroTrendRetest-long=-452.4 JPY`
    /
    `MicroTrendRetest-short=-531.9 JPY`
    より明らかにマイルドだった。
  - 直近2件は
    `STOP_LOSS_ORDER`
    で、
    long `-24.7 JPY`
    /
    short `-16.9 JPY`
    の transition probe が出ていた。
- Hypothesis:
  - いま実際に動いている
    `session_open_breakout`
    を safe canary に入れると、
    live Brain decision を増やしつつ、
    session-open の weak transition probe を
    `ALLOW/REDUCE`
    で軽く整流できる。
  - `workers/common/brain.py`
    は
    `entry_thesis.confidence`
    が無くても
    `entry_probability`
    から confidence を補完できるので、
    文脈不足で無意味な canary にはなりにくい。
- Expected Good:
  - `session_open_breakout`
    の recent loser probe に対して、
    local Brain が
    `REDUCE`
    か
    `BLOCK`
    を返す余地を持てる。
  - allowlist に入っているのに動かない状態から一歩進み、
    prompt/runtime tuning 用の current live data を取れる。
- Expected Bad:
  - `session_open_breakout`
    は件数がまだ少ないため、
    live LLM が shallow `REDUCE`
    を出し過ぎると cadence が鈍る可能性がある。
  - `benchmark_fresh=false`
    /
    `selection_fresh=false`
    の stale さは残っており、
    canary 拡張だけで readiness blocker 自体は消えない。
- Period:
  - `scripts/prepare_local_brain_canary.py --dry-run`: 現時点
  - `logs/orders.db`: 直近12h
  - `logs/trades.db`: 直近7d
  - `logs/brain_state.db`: 直近2h / 直近14d
- Fact:
  - `logs/orders.db`
    直近12hの strategy 別 rows では
    `session_open_breakout=8 rows / 2 filled / avg entry_probability=0.573 / setup_fingerprint付き=8 / flow_regime付き=8`
    だった。
  - `logs/trades.db`
    直近7dの
    `session_open_breakout`
    は
    `8 trades / realized_pl=-2.8 / avg=-0.35 / tp=0 / sl=2`
    で、
    recent 2 trade は
    `2026-03-12 22:04 JST short -16.9`
    と
    `2026-03-12 22:55 JST long -24.7`
    の
    `STOP_LOSS_ORDER`
    だった。
  - `logs/brain_state.db`
    直近14dでは
    `session_open_breakout`
    の decision はまだ 0 件で、
    allowlist 未投入が live Brain data 不足の直接要因だった。
  - `logs/local_v2_stack/quant-market-data-feed.log`
    は
    `tick_fetcher reconnect`
    をまだ出しているが、
    OANDA pricing stream は
    `HTTP 200`
    を返している。
- Failure Cause:
  - 直前の canary 拡張後も、
    allowlist に入れた lane がその時間帯に動かず、
    Brain を live で使う対象が十分広がっていなかった。
- Improvement:
  - 実際に現在動いている
    `session_open_breakout`
    を micro-only canary に追加し、
    scalp へは広げずに
    Brain decision の母数を増やす。
- Verification:
  - `python3 scripts/prepare_local_brain_canary.py --dry-run`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager`
  - `ps eww -p $(cat logs/local_v2_stack/pids/quant-order-manager.pid)`
    で
    `BRAIN_STRATEGY_ALLOWLIST`
    を確認
  - `sqlite3 logs/brain_state.db`
    で
    `session_open_breakout`
    の new decision を監視
- Verdict:
  - pending
- Next Action:
  - 次の session-open window で
    `logs/brain_state.db`
    に
    `session_open_breakout`
    の decision が入るか確認する。
  - `REDUCE/BLOCK`
    が過剰で cadence を落とすなら、
    allowlist を戻す前に prompt/runtime の shallow reduce 側を調整する。
  - `benchmark_fresh`
    /
    `selection_fresh`
    stale は別タスクで更新する。
- Status:
  - in_progress

## 2026-03-12 22:35 JST / local-v2: Brain safe canary を `MomentumBurst` / `MicroTrendRetest-short` へ限定拡張

- Change:
  - `ops/env/profiles/brain-ollama-safe.env`
    の
    `BRAIN_STRATEGY_ALLOWLIST`
    を
    `MicroLevelReactor`
    から
    `MicroLevelReactor, MomentumBurst-open_long, MomentumBurst-open_short, MicroTrendRetest-short`
    へ更新した。
  - `BRAIN_POCKET_ALLOWLIST=micro`
    は維持し、
    `scalp`
    pocket へは広げない方針を
    config / spec に明記した。
- Why:
  - 2026-03-12 22:24 JST 時点の local-v2 実測では、
    `logs/orders.db`
    直近24hが
    `filled=259 / rejected=8 / reject_rate=3.0%`
    で、
    `avg spread=0.8p`,
    `avg ATR=2.096p`,
    `avg preflight_to_fill=551.4ms`
    と、
    execution 側は極端な異常ではなかった。
  - `logs/brain_state.db`
    直近24hは
    `14 decisions / llm_ok=8 / llm_fail=6 / live_llm_ok=2 / avg live latency=3166.3ms`
    で、
    ローカル LLM 自体は使えている一方、
    safe canary の allowlist が
    `MicroLevelReactor`
    だけに留まっていた。
  - 同じ
    `logs/brain_state.db`
    の直近7dには、
    `MomentumBurst-open_short: 6 decisions / llm_ok=5`,
    `MomentumBurst-open_long: 3 / 3`,
    `MicroTrendRetest-short: 2 / 2`
    の Brain decision が既に残っていた。
  - `config/brain_prompt_profile_profit_micro.json`
    も
    `MomentumBurst-open_short`
    と
    `MicroTrendRetest-short`
    の loser lane を前提にした extra rule を持っており、
    prompt / runtime 側の準備はできていた。
- Hypothesis:
  - live 実績がある micro 戦略だけへ Brain を広げれば、
    `MicroLevelReactor`
    以外でも
    loser cluster の size cut / allow 判定を local LLM に使える。
  - ただし
    3 秒級の LLM latency を持つため、
    `scalp`
    まで広げると cadence を壊しやすく、
    safe canary は micro-only のままが妥当。
- Expected Good:
  - `MomentumBurst`
    の marginal setup と
    `MicroTrendRetest-short`
    の loser lane を
    order_manager preflight で局所的に絞れる。
  - `MicroLevelReactor`
    以外の live Brain data が増え、
    次の prompt/runtime tuning の根拠が取れる。
- Expected Bad:
  - `MomentumBurst-open_long`
    の current winner lane まで Brain が shallow `REDUCE`
    を出すと cadence が鈍る可能性。
  - `MicroTrendRetest-long`
    は live Brain 根拠が薄いため今回は未投入にしており、
    short だけ先行すると片側だけ tuning が進む。
- Period:
  - `logs/orders.db`: 直近24h
  - `logs/brain_state.db`: 直近24h / 直近7d
- Fact:
  - `logs/brain_state.db`
    直近7dの strategy 別 Brain decision:
    `MicroLevelReactor-bounce-lower=91`,
    `MomentumBurst-open_short=6`,
    `MomentumBurst-open_long=3`,
    `MicroTrendRetest-short=2`。
  - `logs/local_v2_stack/quant-market-data-feed.log`
    では
    pricing stream の `HTTP 200`
    は取れている一方、
    `tick_fetcher reconnect`
    警告は継続している。
  - したがって今回の変更は
    market-data-feed
    や
    scalp
    へ広げる理由にはせず、
    Brain の proven micro lane 拡張に限定した。
- Failure Cause:
  - local LLM は実際に動いていたが、
    safe canary allowlist が狭すぎて
    `MicroLevelReactor`
    以外に live preflight の知見を溜められていなかった。
- Improvement:
  - Brain safe canary を
    proven micro tag
    へだけ広げ、
    `fail-open / timeout=4s / micro-only`
    は維持した。
- Verification:
  - `python3 scripts/prepare_local_brain_canary.py --dry-run`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager`
  - `ps eww -p $(cat logs/local_v2_stack/pids/quant-order-manager.pid)`
    で
    `BRAIN_STRATEGY_ALLOWLIST`
    を確認
- Verdict:
  - pending
- Next Action:
  - next 30-90 分で
    `logs/brain_state.db`
    に
    `MomentumBurst-open_long/open_short`
    と
    `MicroTrendRetest-short`
    の新しい decision が入るか確認する。
  - `MomentumBurst-open_long`
    の winner lane が過剰に
    `REDUCE`
    されるなら、
    allowlist は維持したまま prompt/runtime rule を調整する。
  - `MicroTrendRetest-long`
    は live Brain 根拠が取れてから次段で追加判断する。
- Status:
  - in_progress

## 2026-03-12 22:06 JST / local-v2: `MomentumBurst` の high-RSI bull-run follow-through を strategy-local に少し前倒し

- Change:
  - `strategies/micro/momentum_burst.py`
    に
    long-side の
    `bull_run`
    context helper を追加し、
    `rsi>68`
    でも
    strong higher-TF uptrend + low range/chop + strong gap/DI/ROC/slope
    のときだけ
    transition long を通せるようにした。
  - `tests/strategies/test_momentum_burst.py`
    に
    high-RSI follow-through keep / choppy keep-block
    の回帰を追加した。
- Why:
  - 2026-03-12 22:06 JST 時点の local-v2 実測では
    `USD/JPY 158.848`
    /
    `M1 ATR 2.32 pips`
    /
    `ADX 25.7`
    /
    `vol_5m 1.96`
    /
    `open_positions 0`
    で、
    execution/流動性の異常は見えていなかった。
  - それでも
    `MomentumBurst`
    は
    24h
    `2 trades / +185.3 JPY`
    のまま、
    2026-03-12 当日は
    `preflight_start=0`
    だった。
  - 既に
    `stale participation artifact` 修正、
    `90s / reaccel 20s`
    cooldown、
    transition long の
    `mid-RSI`
    緩和は入っているため、
    次の scarcity は
    `rsi:overbought`
    winner lane の
    follow-through 判定側と見た。
- Hypothesis:
  - current winner の
    `MomentumBurst-open_long|...|gap:up_strong|...|tr:up_strong|rsi:overbought`
    は、
    broad loosening ではなく
    high-RSI continuation の quality 判定を
    strong trend 文脈に限定して少し前倒しすれば、
    expectancy を保ったまま cadence を増やせる。
- Expected Good:
  - `MomentumBurst-open_long`
    の
    overbought continuation winner lane で
    1 本早い entry を拾える。
  - range/chop / weaker higher-TF support / weak DI-ROC-slope
    では従来どおり block される。
- Expected Bad:
  - strong uptrend の終盤で
    earlier continuation long が増え、
    shallow exhaustion を掴む可能性がある。
- Period:
  - 直近24h の `logs/orders.db` / `logs/trades.db`
  - 2026-03-12 22:06 JST 時点の `logs/factor_cache.json` / `logs/market_context_latest.json`
- Fact:
  - `logs/orders.db`
    の
    `MomentumBurst`
    preflight は
    7d
    `62 attempts / avg_prob 0.842`
    だが、
    2026-03-12 は `0 attempts`。
  - `logs/trades.db`
    の直近 winner は
    2026-03-11 21:44-21:51 JST の
    `MomentumBurst-open_long`
    2本で、
    どちらも
    `entry_probability=0.955`
    /
    `flow_regime=transition`
    /
    `gap:up_strong`
    /
    `rsi:overbought`
    /
    `tr:up_strong`
    だった。
  - `pytest -q tests/strategies/test_momentum_burst.py`
    は
    `32 passed`
    だった。
- Failure Cause:
  - `mid-RSI`
    緩和後も、
    high-RSI continuation は
    `_indicator_quality_ok()`
    の
    strict overextension 条件
    (`strong_di_gap / strong_roc_push / slope>=0.0010`)
    に縛られ、
    strong trend 文脈でも no-entry が残っていた。
- Improvement:
  - `MomentumBurstMicro._long_bull_run_context_ok()`
    を追加し、
    `rsi<=72`
    /
    `range_score<=0.26`
    /
    `micro_chop_score<=0.54`
    /
    `gap_pips>=0.34`
    /
    `DI gap>=9`
    /
    `roc5>=0.024`
    /
    `ema_slope_10>=0.0008`
    /
    `trend_snapshot.direction=long`
    かつ strong higher-TF gap/ADX
    のときだけ、
    long bull-run を通すようにした。
  - short 条件・shared gate・forecast gate・cooldown は変更しない。
- Verification:
  - `pytest -q tests/strategies/test_momentum_burst.py`
  - restart 後
    30-60 分で
    `MomentumBurst-open_long`
    の
    `preflight_start / filled / entry_probability / confidence / close_reason`
    を確認する。
- Verdict:
  - pending
- Next Action:
  - 反映後も
    `MomentumBurst`
    が still `0 attempts`
    なら、
    次は shared gate ではなく
    `forecast gate` の strategy-local 閾値か
    explicit trend snapshot 要件を点検する。
- Status:
  - in_progress

## 2026-03-12 21:34 JST / local-v2: `scalp_extrema_reversal_live` は short ではなく long `volatility_compression` loser lane を先に切る

- Change:
  - `workers/scalp_extrema_reversal/worker.py`
    に
    long-side の
    `setup_pressure`
    guard を追加し、
    current loser の
    `volatility_compression`
    だけを
    recent outcome + weak reclaim 条件で reject するようにした。
  - `ops/env/quant-scalp-extrema-reversal.env`
    に
    long-side setup-pressure の dedicated 閾値を追加した。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    に
    weak long block / stronger long keep
    の回帰を追加した。
- Why:
  - 2026-03-12 21:23 JST 時点の local-v2 は
    24h
    `256 trades / net -170.7 JPY / win_rate 29.3%`
    で、
    current loser の主因は
    `PrecisionLowVol`
    と
    `scalp_extrema_reversal_live`
    だった。
  - `scalp_extrema_reversal_live`
    では
    recent tuning が short 側中心だった一方で、
    24h 実測は
    `long + volatility_compression`
    が
    `30 trades / -40.0 JPY`
    と、むしろ long 側へ損失が寄っていた。
  - 特に
    `long|range_compression|...|volatility_compression`
    は
    `15 trades / -14.8 JPY / sl_rate 0.733 / fast_sl_rate 0.667`
    で、
    `2026-03-12 20:22-20:24 JST`
    相当の recent 3 losers は
    `-3.128 / -2.992 / -2.754 JPY`
    かつ
    `ma_gap<=0`,
    `tick_strength<=0.3`,
    `range_score 0.48-0.51`,
    `ADX 9-13`
    の weak reclaim だった。
- Hypothesis:
  - long loser は
    strategy 名義ではなく
    `long + volatility_compression + weak reclaim`
    に偏っている。
  - shared gate をまた触らず、
    worker local で
    `recent negative outcome`
    と
    `dist_low / bounce / tick_strength / ma_gap / range_score / ADX`
    が弱い lane だけを落とせば、
    current loser を削りつつ stronger long は残せる。
- Expected Good:
  - `scalp_extrema_reversal_live`
    long
    `volatility_compression`
    の recent fast-SL cluster が減る。
  - short 側や stronger long 側の participation は維持する。
- Expected Bad:
  - weak reclaim から始まる一部の small winner も削る可能性がある。
- Period:
  - 直近24h（2026-03-11 21:23 JST - 2026-03-12 21:23 JST）。
- Fact:
  - `scalp_extrema_reversal_live`
    24h は
    `80 trades / -71.7 JPY / win_rate 25.0%`
    だった。
  - side×range_reason では
    `long|volatility_compression = 30 trades / -40.0 JPY / sl_rate 0.50 / fast_sl_rate 0.467`,
    `short|volatility_compression = 40 trades / -22.9 JPY / sl_rate 0.75 / fast_sl_rate 0.60`
    だった。
  - `long|range_compression|...|volatility_compression`
    loser は
    `avg dist_low 0.542 / long_bounce 0.258 / tick_strength 0.204 / ma_gap -0.094 / range_score 0.55 / RSI 39.97 / ADX 17.23`
    で、
    same bucket の positive sample は
    `ma_gap 0.31-0.70`
    の stronger reclaim が残っていた。
- Failure Cause:
  - recent 改善が short shallow probe に寄っており、
    current loser に移った
    long weak reclaim lane を取りこぼしていた。
- Improvement:
  - long-side の
    `setup_pressure`
    を
    `volatility_compression`
    専用で追加し、
    `trades>=6 / net<0 / sl_rate>=0.45 / fast_sl_rate>=0.40`
    の recent loser 圧力がある時だけ、
    `dist_low<=0.90 / bounce<=0.35 / tick_strength<=0.30 / ma_gap<=0 / range_score 0.45-0.55 / ADX<=23`
    の weak reclaim long を reject する。
- Verification:
  - `pytest -q tests/workers/test_scalp_extrema_reversal_worker.py -k 'recent_setup_pressure or stronger_long_even_under_setup_pressure or blocks_long_under_recent_setup_pressure'`
    が通ること。
  - restart 後の
    `orders.db / trades.db`
    で
    `scalp_extrema_reversal_live long volatility_compression`
    の
    fills / STOP_LOSS / realized_jpy
    を 30-60 分で再確認する。
- Verdict:
  - pending
- Next Action:
  - 次は shared gate を触らず、
    この long lane の結果だけを見る。
  - 改善しない場合も
    `PrecisionLowVol`
    と同時に触らず、
    `range_fade` と `range_compression`
    を分けて切る。
- Status:
  - in_progress

## 2026-03-12 20:58 JST / local-v2: `strategy_feedback_worker` が dedicated entry worker を発見できず `RangeFader` を落としていた

- Change:
  - `analysis/strategy_feedback_worker.py`
    に
    service 名 /
    worker module 名から
    canonical strategy tag を復元する fallback を追加した。
  - `tests/analysis/test_strategy_feedback_worker.py`
    に
    explicit tag env を持たない
    `quant-scalp-rangefader.service`
    /
    `quant-scalp-precision-lowvol.service`
    の discovery 回帰を追加した。
- Why:
  - 2026-03-12 20:24 JST 時点の local-v2 は
    24h
    `256 trades / net -170.7 JPY / PF 0.78 / win_rate 29.3%`
    だった。
  - その時点の
    `logs/strategy_feedback.json`
    は
    `RangeFader / PrecisionLowVol / DroughtRevert / VwapRevertS`
    を含まず、
    current loser / current live strategy への feedback coverage が欠けていた。
- Hypothesis:
  - `strategy_feedback_worker`
    は
    `*_MODE / *_TAGS / *_STRATEGY_TAG`
    を持つ env だけを tag source として扱っており、
    dedicated worker の一部は
    entry service が running でも
    `entry_active`
    を立てられていない。
  - service/module 名 fallback を入れれば、
    running な dedicated worker の feedback omission を解消できる。
- Expected Good:
  - `RangeFader`
    のような running entry worker が
    `strategy_feedback.json`
    へ戻る。
  - worker local / shared trim の current artifact coverage が改善する。
- Expected Bad:
  - fallback mapping が広すぎると、
    unrelated service 名を誤って canonical tag へ寄せる可能性がある。
- Period:
  - 2026-03-12 20:24-20:57 JST。
- Fact:
  - raw `trades.db` 14日窓では
    `RangeFader=295`,
    `PrecisionLowVol=50`,
    `DroughtRevert=30`,
    `VwapRevertS=14`,
    `WickReversalBlend=13`
    が存在した。
  - 修正前の payload では
    `RangeFader`
    が absent だったが、
    修正後の
    `python3 -m analysis.strategy_feedback_worker --nowrite`
    / `_build_payload()`
    では
    `RangeFader`
    が復帰した。
  - `main`
    へ
    `f83fe33f`
    を push して
    `quant-market-data-feed / quant-strategy-control / quant-order-manager / quant-position-manager / quant-strategy-feedback`
    を restart 後、
    actual
    `logs/strategy_feedback.json`
    は
    `5 strategies`
    となり
    `RangeFader`
    が復帰した。
  - 一方
    `PrecisionLowVol / DroughtRevert / VwapRevertS`
    は
    `scripts/local_v2_stack.sh status --profile trade_min`
    で
    `stopped`
    のため、
    修正後も payload には出ない。
- Failure Cause:
  - `strategy_feedback_worker`
    の discovery が
    env tag 依存で、
    `workers.scalp_rangefader.worker`
    や
    `workers.scalp_precision_lowvol.worker`
    のような dedicated worker module を
    canonical strategy tag へ変換できていなかった。
- Improvement:
  - service/module 名から
    `RangeFader / PrecisionLowVol / DroughtRevert / VwapRevertS / WickReversalBlend`
    を補完する fallback を追加した。
- Verification:
  - `pytest -q tests/analysis/test_strategy_feedback_worker.py -k 'discovers_local_v2_services or dedicated_worker_without_explicit_tag_env'`
    が通ること。
  - `_build_payload()`
    の live result で
    `RangeFader`
    が現れること。
- Verdict:
  - good
- Next Action:
  - `quant-scalp-precision-lowvol`,
    `quant-scalp-drought-revert`,
    `quant-scalp-vwap-revert`
    が stop のままなので、
    それらを別件で
    「なぜ止まっているか」
    まで切る。
  - 今回の修正は
    `RangeFader`
    の feedback coverage 復旧として維持する。
- Status:
  - done

## 2026-03-12 21:05 JST / local-v2: `PrecisionLowVol / DroughtRevert / VwapRevertS` は worker restart 後も `strategy_feedback` から欠落し、`systemd` 非依存の local pid discovery が必要だった

- Change:
  - `analysis/strategy_feedback_worker.py`
    に
    local pid-only service 名から
    synthetic unit body を生成する fallback を追加した。
  - `tests/analysis/test_strategy_feedback_worker.py`
    に
    `systemd/*.service`
    が無くても
    `quant-scalp-precision-lowvol`
    /
    `quant-scalp-drought-revert`
    /
    `quant-scalp-vwap-revert`
    を拾う回帰を追加した。
- Why:
  - `quant-scalp-precision-lowvol`,
    `quant-scalp-drought-revert`,
    `quant-scalp-vwap-revert`
    は stale pid で停止していたため restart したが、
    restart 後に
    `python3 -m analysis.strategy_feedback_worker`
    を回しても
    `logs/strategy_feedback.json`
    は
    `RangeFader`
    しか拾わなかった。
- Hypothesis:
  - local-v2 の一部 worker は
    `scripts/local_v2_stack.sh`
    管理の service であり、
    `systemd/*.service`
    や host systemd unit を持たない。
  - `strategy_feedback_worker`
    が
    local pid の running service 名だけを見ても
    unit body を解決できないため、
    `entry_active`
    を立てられていない。
- Expected Good:
  - local stack 上で実際に running な
    `PrecisionLowVol / DroughtRevert / VwapRevertS`
    が
    `strategy_feedback.json`
    へ復帰する。
- Expected Bad:
  - synthetic module 推定が広すぎると、
    strategy と無関係な local service を誤認する可能性がある。
- Period:
  - 2026-03-12 21:05-21:10 JST。
- Fact:
  - `scripts/local_v2_stack.sh status --profile trade_min`
    では
    上記 3 strategy の worker は
    restart 後に
    `running`
    へ戻った。
  - それでも修正前の
    `strategy_feedback_worker`
    は
    `systemd/*.service`
    が無い
    `quant-scalp-precision-lowvol`
    /
    `quant-scalp-drought-revert`
    /
    `quant-scalp-vwap-revert`
    を
    discovery できず、
    actual artifact は
    `RangeFader`
    のみだった。
  - `main`
    へ
    `e388be92`
    を push して
    `quant-market-data-feed / quant-strategy-control / quant-order-manager / quant-position-manager / quant-strategy-feedback`
    を restart 後、
    常駐
    `quant-strategy-feedback`
    が書く
    `logs/strategy_feedback.json`
    は
    `8 strategies`
    となり、
    subset は
    `DroughtRevert / PrecisionLowVol / RangeFader / VwapRevertS`
    へ復帰した。
- Failure Cause:
  - `_discover_from_systemd`
    は
    local pid から running service 名を得ても、
    systemd unit body を解決できない service を捨てていた。
  - さらに
    `PrecisionLowVol / DroughtRevert`
    は env の
    `*_PERF_GUARD_MODE=reduce`
    を strategy tag と誤認し、
    service-name fallback 自体も潰れていた。
- Improvement:
  - local pid-only service は
    `ops/env/{service}.env`
    と
    service 名から推定した worker module で
    synthetic unit body を構成し、
    既存 parser へ通すようにした。
- Verification:
  - `pytest -q tests/analysis/test_strategy_feedback_worker.py -k 'discovers_local_v2_services or dedicated_worker_without_explicit_tag_env or local_pid_only_service_without_systemd_unit'`
    が通ること。
  - `python3 -m analysis.strategy_feedback_worker`
    と
    restart 後の常駐 worker が書く
    `logs/strategy_feedback.json`
    で
    `PrecisionLowVol / DroughtRevert / VwapRevertS`
    が復帰すること。
- Verdict:
  - good
- Next Action:
  - `WickReversalBlend`
    系が current coverage に戻っていない理由は別件として切る。
- Status:
  - done

## 2026-03-12 20:38 JST / local-v2: participation allocator override が env key typo と未クォート値で無効化されていた

- Change:
  - `ops/env/local-v2-stack.env`
    の
    `LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOC_CMD`
    を
    `LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOCATOR_CMD`
    へ修正し、
    値を
    `"/Users/tossaki/App/QuantRabbit/.venv/bin/python .../scripts/participation_allocator.py ..."`
    の1文字列へ変更した。
  - `docs/OPS_LOCAL_RUNBOOK.md`
    に
    participation allocator job の override key と
    `CMD`
    クォート要件を追記した。
- Why:
  - 2026-03-12 20:24 JST 時点の local-v2 では
    24h
    `256 trades / net -170.7 JPY / PF 0.78 / win_rate 29.3%`
    と悪化していた。
  - 同時に
    `quant-position-manager.log`
    と
    `scripts/local_v2_stack.sh status`
    で
    `local-v2-stack.env: line 54: .../scripts/participation_allocator.py: Permission denied`
    が反復し、
    `config/participation_alloc.json`
    は intended override ではなく
    既定の
    `max_units_cut=0.22 / max_units_boost=0.24 / max_probability_boost=0.10`
    のままだった。
- Hypothesis:
  - env key typo と shell-unsafe な未クォート値のせいで、
    「chronic loser setup を deeper に trim する」override が
    runtime へ渡っていない。
  - key と quoting を直せば、
    local feedback cycle が intended policy
    `0.35 / 0.30 / 0.15`
    を適用できる。
- Expected Good:
  - env 読み込み時の
    `Permission denied`
    を止める。
  - `participation_allocator`
    が intended override で再計算され、
    loser setup の trim を深くできる。
- Expected Bad:
  - deeper trim が強すぎると、
    市況回復直後の scalp 参加率を落とす。
- Period:
  - 2026-03-12 19:30-20:35 JST。
- Fact:
  - `ops/env/local-v2-stack.env:54`
    は
    `LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOC_CMD=...python .../participation_allocator.py ...`
    となっており、
    shell source 時に command 実行へ解釈されていた。
  - `scripts/run_local_feedback_cycle.py`
    は
    `job_name=participation_allocator`
    に対して
    `LOCAL_FEEDBACK_CYCLE_PARTICIPATION_ALLOCATOR_*`
    を読む実装だった。
  - `config/participation_alloc.json`
    の policy は
    `max_units_cut=0.22`
    で、
    2026-03-12 の intended setting
    `0.35`
    が runtime に入っていなかった。
- Failure Cause:
  - env override key typo と未クォート command 値により、
    participation allocator の override が無効化され、
    しかも env 読み込みのたびに誤実行エラーを出していた。
- Improvement:
  - override key を正しい
    `...PARTICIPATION_ALLOCATOR_CMD`
    に合わせ、
    command 値を shell-safe にクォートした。
- Verification:
  - `source ops/env/local-v2-stack.env`
    相当の読み込みで
    `Permission denied`
    が出ないこと。
  - `scripts/run_local_feedback_cycle.py`
    実行後に
    `config/participation_alloc.json`
    の policy が
    `max_units_cut=0.35 / max_units_boost=0.30 / max_probability_boost=0.15`
    を示すこと。
- Verdict:
  - pending
- Next Action:
  - local feedback cycle を再実行し、
    `participation_alloc.json`
    の policy と
    `PrecisionLowVol / scalp_extrema_reversal_live`
    の trim 深度が変わることを確認する。
- Status:
  - in_progress

## 2026-03-12 16:05 JST / local-v2: `MicroLevelReactor-bounce-lower` の negative-forecast long を strategy-scoped leading profile で reject

- Change:
  - `ops/env/quant-micro-levelreactor.env`
    と
    `ops/env/local-v2-stack.env`
    に
    `MICROLEVELREACTOR_ENTRY_LEADING_PROFILE_*`
    を追加し、
    `MicroLevelReactor`
    long の leading profile を
    `forecast 60% / tech 15% / range 20% / micro 5%`
    で評価し、
    `adjusted entry_probability < 0.44`
    の lane を reject するようにした。
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    に
    current loser
    `bounce-lower`
    は reject、
    current winner
    `breakout-long`
    は keep
    の回帰を追加した。
- Why:
  - 2026-03-12 15:41 JST 時点の local-v2 は
    `146 trades / PF 0.44 / net -23.255 JPY`
    で、
    post-deploy の current loser は
    `MicroLevelReactor-bounce-lower`
    だった。
  - 直近 12h の同一 fingerprint
    `MicroLevelReactor-bounce-lower|long|range_fade|...|gap:down_lean|...|tr:dn_strong`
    は
    `6 trades / net -28.026 JPY / 4x STOP_LOSS`
    だった。
  - recent loser
    `459537 / 459541 / 459563 / 459571`
    は
    `TP_touch<=600s = 0/4`
    で、
    全件
    `forecast.reason=style_mismatch_range`
    /
    `forecast.expected_pips=-0.4551`
    /
    `forecast.p_up=0.331551`
    /
    `trend_state=strong_down`
    のまま通っていた。
- Hypothesis:
  - この lane は、
    forecast block を shared layer が
    `reduce`
    に留めた結果、
    negative forecast long が still filled されている。
  - `MicroLevelReactor`
    専用 leading profile で
    forecast の contra を強めに重み付けすれば、
    loser `bounce-lower`
    は切り、
    positive forecast の
    `breakout-long`
    は残せる。
- Expected Good:
  - `MicroLevelReactor-bounce-lower`
    の current loser long を reject し、
    `STOP_LOSS_ORDER`
    を減らす。
  - `breakout-long`
    の winner lane は維持する。
- Expected Bad:
  - `MicroLevelReactor`
    long の participation が減る。
  - threshold が強すぎると、
    反発勝ちの薄利 long も落とす。
- Period:
  - 直近 12h-24h（2026-03-12 15:41 JST 時点確認）。
- Fact:
  - `459537 / 459541 / 459563 / 459571`
    は
    `entry_probability_after_forecast_fusion ≈ 0.4736`
    から、
    提案値の leading profile simulation で
    `0.41897`
    へ低下し、
    reject になることを確認した。
  - 一方
    `458937 / 458929`
    の
    `MicroLevelReactor-breakout-long`
    は
    `entry_probability_after_forecast_fusion ≈ 0.6649`
    のまま
    pass することを確認した。
- Failure Cause:
  - `MicroLevelReactor-bounce-lower`
    の negative forecast long を
    shared forecast fusion が縮小だけで残し、
    same-lane burst が filled されていた。
- Improvement:
  - `MicroLevelReactor`
    専用
    `ENTRY_LEADING_PROFILE`
    を有効化し、
    negative forecast + weak range の long だけを
    strategy-scoped に reject する。
- Verification:
  - `PYTHONPATH=. pytest -q tests/execution/test_strategy_entry_forecast_fusion.py -k 'mlr_bounce_lower_negative_forecast or mlr_breakout_long_positive_forecast or entry_leading_profile_'`
    が通ること。
  - deploy 後 30-60 分で
    `MicroLevelReactor-bounce-lower`
    の
    `forecast_gate_block`
    と
    `entry_leading_profile_reject`
    が並び、
    filled が止まること。
- Verdict:
  - pending
- Next Action:
  - post-deploy で
    `MicroLevelReactor-bounce-lower`
    の
    `fills / STOP_LOSS_ORDER / realized_jpy`
    を確認し、
    まだ残るなら
    `bounce-lower`
    自体に worker local guard を追加する。
- Status:
  - in_progress

## 2026-03-12 16:15 JST / local-v2: `PrecisionLowVol` の `gap:up_flat` shallow short を worker local で遮断

- Change:
  - `workers/scalp_wick_reversal_blend/worker.py`
    に
    `short_up_flat`
    判定と
    `PrecisionLowVol`
    専用
    `up_flat_shallow_short_lane`
    guard を追加し、
    `range_compression + volatility_compression + gap:up_flat`
    の short で
    `projection<=0.28`
    かつ
    `setup_quality<0.50`
    の shallow lane を reject するようにした。
  - `workers/scalp_wick_reversal_blend/config.py`
    に対応閾値を追加した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に current loser lane block / stronger reclaim keep の回帰を追加した。
- Why:
  - 2026-03-12 16:07 JST 時点の local-v2 では
    `PrecisionLowVol`
    が
    24h
    `28 trades / net -131.147 JPY / win 32.1% / PF 0.50`
    の最大 loser だった。
  - recent loser
    `459601`
    は
    `-22.899 JPY / 10s`
    で、
    `PrecisionLowVol|short|range_compression|...|gap:up_flat|volatility_compression`
    の current lane だった。
  - 直近同型 loser
    `459483 / 459473 / 459429 / 459371`
    も同じ fingerprint 群で、
    winner
    `459411`
    との差は
    `setup_quality`
    が
    `0.301-0.455`
    と低いことだった
    （winner は `0.609`）。
- Hypothesis:
  - 現在の `PrecisionLowVol` short loser は
    「強い overbought short」ではなく、
    `gap:up_flat`
    の shallow reclaim を short している。
  - `setup_quality`
    を閾値にすれば、
    same fingerprint でも
    stronger reclaim short は残しつつ、
    loser probe だけを切れる。
- Expected Good:
  - `PrecisionLowVol`
    の current loser lane を減らし、
    `STOP_LOSS_ORDER`
    を止める。
  - `setup_quality`
    が高い reclaim short は残す。
- Expected Bad:
  - `PrecisionLowVol`
    short の participation が減る。
  - 閾値が厳しすぎると、
    薄利 winner も一部落ちる。
- Period:
  - 直近 12h-24h（2026-03-12 16:07 JST 時点確認）。
- Fact:
  - recent
    `gap:up_flat`
    loser
    `459601 / 459483 / 459473 / 459429 / 459371`
    は
    `setup_quality=0.301-0.455`
    で、
    `projection=0.015-0.275`
    だった。
  - same fingerprint の winner
    `459411`
    は
    `setup_quality=0.609`
    /
    `projection=0.275`
    で、
    `flow_guard`
    が明確に強かった。
- Failure Cause:
  - 既存 guard は
    `RSI>=59`
    の weak overbought short を中心に切っていたため、
    `RSI 51-56`
    の
    `gap:up_flat`
    shallow short を残していた。
- Improvement:
  - `short_up_flat`
    を別 cluster として切り出し、
    `projection<=0.28`
    /
    `setup_quality<0.50`
    の lane を worker local に reject する。
- Verification:
  - `PYTHONPATH=. pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py -k 'up_flat_shallow_short or marginal_short or weak_short'`
    が通ること。
  - deploy 後 30-60 分で
    `PrecisionLowVol|short|range_compression|...|gap:up_flat|volatility_compression`
    の
    `fills / STOP_LOSS_ORDER / realized_jpy`
    を確認する。
- Verdict:
  - pending
- Next Action:
  - post-deploy で
    `PrecisionLowVol`
    の同 fingerprint が still loser なら、
    `touch_ratio`
    まで使った reclaim strength guard を追加する。
- Status:
  - in_progress

## 2026-03-12 15:35 JST / local-v2: `scalp_extrema_reversal_live` short の marginal drift probe を worker local で遮断

- Change:
  - `workers/scalp_extrema_reversal/worker.py`
    に
    `short_drift_probe_block`
    を追加し、
    `short + volatility_compression + non-supportive`
    で
    `dist_high<=0.9`
    /
    `short_bounce<=0.15`
    /
    `tick_strength<=0.15`
    /
    `range_score<=0.48`
    /
    `0<=ma_gap_pips<=0.35`
    /
    `rsi<=60`
    の marginal short probe を reject するようにした。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    に
    loser lane block / bearish-gap short keep
    の回帰を追加した。
- Why:
  - 2026-03-12 15:18 JST 時点の local-v2 で
    `scalp_extrema_reversal_live`
    は
    24h
    `65 trades / net -54.296 JPY / win 23.1%`
    の main loser のままだった。
  - 直近 short loser
    `459489`
    /
    `459495`
    は
    `4.5s`
    /
    `22.5s`
    で
    `STOP_LOSS_ORDER`
    になり、
    tick validate でも
    `tp_touch<=600s なし`
    だった。
- Hypothesis:
  - 現在の short loser は
    「上方向へ強く伸び切った sell fade」ではなく、
    `ma_gap`
    がまだ中途半端に正で、
    `bounce/tick`
    も弱い shallow reversal probe を short している。
  - この marginal lane は、
    bearish gap の short や、
    強く stretch した reversion short とは分けて落とせる。
- Expected Good:
  - `scalp_extrema_reversal_live`
    short の current loser lane を削り、
    `STOP_LOSS_ORDER`
    を減らす。
  - bearish gap の short や、
    stronger stretch short は残す。
- Expected Bad:
  - short entry 数が減る。
  - 閾値がきつすぎると、
    薄いが勝てる short を落とす可能性がある。
- Period:
  - recent 24h / recent 7d
  - tick validation は 2026-03-12 UTC の recent short loser 2件
- Fact:
  - `logs/trades.db`
    の
    `scalp_extrema_reversal_live`
    `short + volatility_compression + non-supportive + dist_high<=0.9 + short_bounce<=0.15 + tick_strength<=0.15 + range_score<=0.48`
    を
    `ma_gap`
    で見ると、
    `0<=ma_gap<0.35`
    は
    `5 trades / 5 STOP_LOSS / net -5.590 JPY`
    だった。
  - 同じ shallow short でも、
    `459393`
    は
    `ma_gap=-0.320`
    の bearish gap で
    `+0.534 JPY`
    だった。
  - `459489`
    は
    `range_score 0.460 / ma_gap 0.270 / dist_high 0.886 / bounce 0.100 / tick 0.100 / tp_touch<=600s なし`。
  - `459495`
    は
    `range_score 0.443 / ma_gap 0.090 / dist_high 0.124 / bounce 0.100 / tick 0.100 / tp_touch<=600s なし`。
- Failure Cause:
  - 既存 short guard は
    `countertrend gap>=0.45`
    か
    `setup_pressure active`
    のときだけ強く効いていたため、
    その手前の
    `0-0.35 pip`
    程度の bullish drift short が current loser lane として残っていた。
- Improvement:
  - `short_drift_probe_block`
    で、
    current loser cluster に一致する
    marginal short probe だけを worker local に落とす。
- Verification:
  - `PYTHONPATH=. pytest -q tests/workers/test_scalp_extrema_reversal_worker.py -k 'short_drift_probe or bullish_gap'`
  - `python3 -m py_compile workers/scalp_extrema_reversal/worker.py tests/workers/test_scalp_extrema_reversal_worker.py`
  - deploy 後 30-60 分で
    `scalp_extrema_reversal_live`
    short の
    `STOP_LOSS_ORDER`
    /
    `fast<=5s`
    /
    `realized_jpy`
    を再確認する。
- Verdict:
  - pending
- Next Action:
  - post-deploy で
    `short_drift_probe_block`
    の skip ログと、
    recent short loser の消失を確認する。
  - まだ悪ければ
    `PrecisionLowVol`
    short を current 窓だけで再評価する。
- Status:
  - open

## 2026-03-12 15:20 JST / local-v2: `scalp_ping_5s_d_live` TP-enabled long の instant-SL lane を worker local で遮断

- Change:
  - `workers/scalp_ping_5s/worker.py`
    に
    `D_NEGATIVE_WINDOW_LONG_OPPOSITE`
    guard を追加し、
    `long`
    かつ
    `m1_opposite`
    /
    `horizon_neutral`
    /
    `direction_bias_side=long`
    の D momentum long で、
    `TP_ENABLED=1`
    の current variant に限り
    `signal_window_adaptive_live_score_pips<=-1.0`
    かつ
    `lookahead_edge_pips<=0.60`
    の lane を reject するようにした。
  - `workers/scalp_ping_5s/config.py`
    に D 専用閾値
    `SCALP_PING_5S_D_NEGATIVE_WINDOW_LONG_OPPOSITE_*`
    を追加した。
  - `tests/workers/test_scalp_ping_5s_worker.py`
    に
    block / pass
    の focused test を追加した。
- Why:
  - 2026-03-12 15:09 JST 時点の local-v2 は
    `USD/JPY 159.051 / spread 0.8 pips / open_trades 0`
    で execution 停止ではなく、
    24h は
    `141 trades / PF 0.48 / net -11.281 JPY`
    だった。
  - ユーザ指摘の
    `1秒前後のSTOP_LOSS`
    を ticket 単位で見ると、
    `scalp_ping_5s_d_live`
    の current TP-enabled long variant
    が
    `459319`
    /
    `459441`
    ともに
    `0.2-1.1s`
    で
    `STOP_LOSS_ORDER`
    になっていた。
- Hypothesis:
  - D long の
    `m1_opposite + horizon_neutral`
    でも、
    `direction_bias_side=long`
    に引っ張られて
    `signal_window_adaptive_live_score_pips`
    が深く負のまま入る lane は、
    current TP-enabled scalp variant では follow-through が足りず、
    broker TP/SL を付けても即時損切りになりやすい。
- Expected Good:
  - `scalp_ping_5s_d_live`
    current long variant の
    `STOP_LOSS_ORDER`
    と
    `fast<=2s`
    loss を減らす。
  - D の long participation 全体は止めず、
    stronger live-score / edge の long は残す。
- Expected Bad:
  - D long の entry 数がさらに減る。
  - 今後この lane が改善した場合、
    初動の細い winner long も落ちる可能性がある。
- Period:
  - recent 7d
    / ticket validation は 2026-03-12 UTC の current loser trades
- Fact:
  - `logs/trades.db`
    の
    `scalp_ping_5s_d_live`
    で
    `m1_opposite + horizon_neutral + direction_bias_side=long + tp_pips>0.5`
    は
    `2 trades / 2 STOP_LOSS / 2 fast<=2s / avg live_score -1.313 / avg edge 0.277 / avg sl 0.965p / avg tp 1.35p / net -3.870 JPY`
    だった。
  - `ticket 459319`
    は
    `hold 0.2s / pl -1.0p / tp_touch<=600s なし / MAE 9.8p / MFE 0.0p`。
  - `ticket 459441`
    は
    `hold 1.1s / pl -1.0p / tp_touch<=600s なし / MAE 3.6p / MFE 1.1p`。
- Failure Cause:
  - SL が単に近すぎたのではなく、
    `negative live-score`
    のまま
    `direction_bias long`
    に押されて入った current long lane 自体の質が悪かった。
- Improvement:
  - `TP_ENABLED=1`
    の D long current variant に限定し、
    `negative-window opposite long`
    を worker local で entry 前に落とす。
- Verification:
  - `PYTHONPATH=. pytest -q tests/workers/test_scalp_ping_5s_worker.py -k 'd_negative_window_short_align_block_reason or d_negative_window_long_opposite_block_reason'`
  - `python3 -m py_compile workers/scalp_ping_5s/config.py workers/scalp_ping_5s/worker.py tests/workers/test_scalp_ping_5s_worker.py`
  - deploy 後 30-60 分で
    `scalp_ping_5s_d_live`
    の
    `STOP_LOSS_ORDER`
    /
    `fast<=2s`
    /
    `realized_jpy`
    を再確認する。
- Verdict:
  - pending
- Next Action:
  - post-deploy で
    `scalp_ping_5s_d_live`
    の
    `d_negative_window_long_opposite_block`
    log 出現と、
    long の instant-SL 消失を確認する。
  - なお悪ければ次は
    `scalp_extrema_reversal_live`
    short の shallow probe を current window だけで再評価する。
- Status:
  - open

## 2026-03-12 14:40 JST / local-v2: `scalp_ping_5s_d_live` short の negative-window momentum lane を worker local で遮断

- Change:
  - `workers/scalp_ping_5s/worker.py`
    に
    `D_NEGATIVE_WINDOW_SHORT_ALIGN`
    guard を追加し、
    `short`
    かつ
    `m1_align_boost`
    で、
    `horizon_neutral|align_weak|counter_scaled`
    のまま
    `signal_window_adaptive_live_score_pips`
    が深く負、
    かつ
    `lookahead_edge_pips`
    が薄い lane を reject するようにした。
  - `workers/scalp_ping_5s/config.py`
    に
    D 専用閾値
    `SCALP_PING_5S_D_NEGATIVE_WINDOW_SHORT_ALIGN_*`
    を追加した。
  - `tests/workers/test_scalp_ping_5s_worker.py`
    に
    loser lane block / supported lane pass
    の回帰を追加した。
- Why:
  - ユーザ指摘どおり、
    `logs/trades.db`
    では
    `STOP_LOSS_ORDER`
    が
    `0.5-5 秒`
    で刺さる lane が残っており、
    その一部は
    `scalp_ping_5s_d_live`
    の short に集中していた。
  - 2026-03-12 14:19 JST 時点の local-v2 実測は
    `USD/JPY 159.052 / spread 0.8 pips / open_trades 0`
    で execution 停止ではなく、
    24h は
    `scalp_ping_5s_d_live 9 trades / net -17.529 JPY / win 0%`
    の current loser だった。
- Hypothesis:
  - D short の
    `m1_align_boost`
    だから通している momentum lane でも、
    `signal_window_adaptive_live_score_pips`
    が深く負で
    `lookahead_edge`
    が薄いものは実際には follow-through が足りず、
    即 SL の確率が高い。
- Expected Good:
  - `scalp_ping_5s_d_live`
    short の
    `STOP_LOSS_ORDER`
    と
    sub-5s loss
    を減らす。
  - D の short participation は全面停止せず、
    stronger edge の short だけ残す。
- Expected Bad:
  - D short の entry 数が落ちる。
  - 閾値がきつすぎると、
    `horizon_neutral`
    の薄い winner short も削る可能性がある。
- Period:
  - current 24h / recent 7d (`logs/trades.db`, `logs/orders.db`, 2026-03-12 14:19-14:34 JST 集計)
- Fact:
  - current 24h fast-SL 集計では
    `scalp_ping_5s_d_live short`
    が
    `6 trades / fast<=5s 5 / fast<=2s 2 / avg_hold 7.922s / net -11.349 JPY`
    だった。
  - recent 7d cluster では
    `short + horizon_neutral + m1_align_boost`
    が
    `13 trades / net -52.875 JPY / avg lookahead_edge 0.270 / avg live_score -1.529`
    と継続悪化だった。
  - 直近 losers の代表は
    `2026-03-11 23:00 UTC`
    close の short で
    `hold 4.148s / tp_pips 1.4 / lookahead_edge 0.342 / live_score -1.312 / horizon_neutral / m1_align_boost`
    だった。
- Failure Cause:
  - `m1_align_boost`
    だけでは short momentum の continuation を保証できず、
    adaptive signal-window が強く negative の時点で lane quality が崩れていた。
- Improvement:
  - D short だけに限定して、
    `live_score <= -0.85`
    かつ
    `lookahead_edge <= 0.40`
    の negative-window momentum short を worker local で reject する。
- Verification:
  - `PYTHONPATH=. pytest -q tests/workers/test_scalp_ping_5s_worker.py -k 'd_negative_window_short_align_block_reason'`
    で
    `2 passed`
  - `python3 -m py_compile workers/scalp_ping_5s/config.py workers/scalp_ping_5s/worker.py tests/workers/test_scalp_ping_5s_worker.py`
    は成功。
  - `tests/workers/test_scalp_ping_5s_worker.py`
    全体は今回差分と無関係な既存失敗があるため、
    targeted test のみを採用。
- Verdict:
  - pending
- Next Action:
  - deploy 後 30-60 分で
    `scalp_ping_5s_d_live short`
    の
    `fills / STOP_LOSS_ORDER / hold_sec<=5 / realized_jpy`
    を再確認する。
  - まだ fast-SL が残るなら、
    次は
    `scalp_extrema_reversal_live`
    の short fast-SL cluster を追加で削る。
- Status:
  - in_progress

## 2026-03-12 14:15 JST / local-v2: `scalp_wick_reversal_blend` 系の perf guard 無効化フラグが逆読まれて participation を落としていた

- Change:
  - `workers/scalp_wick_reversal_blend/worker.py`
    の
    `_perf_guard_bypass_enabled()`
    を修正し、
    `SCALP_PRECISION_*_PERF_GUARD_ENABLED=0`
    を本当に
    `perf guard off`
    として扱うようにした。
  - `tests/workers/test_scalp_wick_reversal_blend_dispatch.py`
    に
    `guard disabled -> bypass on`
    と
    `guard enabled -> bypass off`
    の回帰を追加した。
- Why:
  - 2026-03-12 14:07 JST 時点の local-v2 実測では
    `USD/JPY 159.070 / spread 0.8 pips / open_trades 0`
    で、
    last 60 分の entry は
    `13:12 JST`
    の
    `DroughtRevert`
    と
    `PrecisionLowVol`
    各1本だけだった。
  - live log では
    `DroughtRevert`
    と
    `PrecisionLowVol`
    に
    `perf guard blocked`
    が連発していた一方、
    dedicated env は
    `SCALP_PRECISION_DROUGHT_REVERT_PERF_GUARD_ENABLED=0`
    と
    `SCALP_PRECISION_LOWVOL_PERF_GUARD_ENABLED=0`
    になっていた。
- Hypothesis:
  - worker の
    `_perf_guard_bypass_enabled()`
    が
    `PERF_GUARD_ENABLED`
    をそのまま
    `bypass`
    として読んでおり、
    `0`
    でも perf guard が active のままになっていた。
  - その結果、
    本来 worker-local quality guard で選別したい
    `DroughtRevert / PrecisionLowVol / WickReversalBlend`
    系が strategy-wide に止まり、
    participation を不必要に落としていた。
- Expected Good:
  - dedicated env で
    `PERF_GUARD_ENABLED=0`
    にしている mode は、
    worker-local guard だけで entry 候補を評価できるようになる。
  - `DroughtRevert`
    と
    `PrecisionLowVol`
    の candidate 空白が縮む。
- Expected Bad:
  - loser lane まで戻し過ぎると、
    entry 数だけ増えて損失も戻る。
- Period:
  - 直近60分 live orders/logs + current dedicated env
- Fact:
  - last 60 分 orders は
    `DroughtRevert filled 1`,
    `PrecisionLowVol filled 1`,
    `scalp_ping_5s_c_live entry_probability_reject 1`
    で、
    その後の candidate は極端に薄かった。
  - live log は
    `2026-03-12 12:xx-14:xx JST`
    に
    `perf guard blocked tag=DroughtRevert reason=pf=0.89 win=0.54 n=13`
    と
    `perf guard blocked tag=PrecisionLowVol reason=pf=0.87 win=0.50 n=24`
    を出していた。
  - dedicated env は
    `...PERF_GUARD_ENABLED=0`
    を明示していた。
- Failure Cause:
  - mode-specific env の
    `PERF_GUARD_ENABLED`
    を bypass flag として逆向きに読んでいた実装ミス。
- Improvement:
  - `_perf_guard_bypass_enabled()`
    を
    `not _env_bool(..., True)`
    に修正し、
    env の意味と worker の挙動を一致させた。
- Verification:
  - `pytest -q tests/workers/test_scalp_wick_reversal_blend_dispatch.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - restart 後に
    `quant-scalp-drought-revert`
    と
    `quant-scalp-precision-lowvol`
    の
    `perf guard blocked`
    ログ消失と
    `preflight_start`
    の戻りを確認する。
- Verdict:
  - pending
- Next Action:
  - restart 後 30-60 分で
    `DroughtRevert / PrecisionLowVol`
    の `preflight / filled / realized_jpy`
    を見て、
    participation だけ戻って quality が崩れるなら
    perf guard ではなく setup-local block を追加する。
- Status:
  - in_progress

## 2026-03-12 13:45 JST / local-v2: `MomentumBurst` cadence を stale participation artifact と cooldown で詰めていた箇所を修正

- Change:
  - `scripts/entry_path_aggregator.py`
    の lookback 判定を
    `ts >= datetime('now', ?)`
    から
    `julianday(ts) >= julianday('now', ?)`
    へ変更した。
  - `ops/env/quant-micro-momentumburst.env`
    の
    `MICRO_MULTI_STRATEGY_COOLDOWN_SEC`
    を
    `120 -> 90`,
    `MOMENTUMBURST_REACCEL_COOLDOWN_SEC`
    を
    `35 -> 20`
    へ短縮した。
- Why:
  - 2026-03-12 13:33 JST 時点の local-v2 実測では
    `USD/JPY 159.060 / spread 0.8 pips / open_trades 0`
    で execution は正常、
    24h は
    `141 trades / win 29.1% / PF 0.50 / net -19.453 JPY`
    だった。
  - `MomentumBurst`
    は 24h
    `2 trades / +185.320 JPY / win 100%`
    で winner なのに、
    13時台の live では fills が増えていなかった。
  - その確認中、
    `config/participation_alloc.json`
    の
    `lookback_hours=6`
    にもかかわらず
    `MomentumBurst-open_long attempts=2`
    が残っており、
    `2026-03-11 12:44-12:46 UTC`
    の stale order が current window に混入していることを確認した。
- Hypothesis:
  - `orders.ts` は
    `2026-03-11T12:46:30.837953+00:00`
    形式で保存されており、
    文字列比較の
    `ts >= datetime('now', ?)`
    では current 6h 窓を正しく切れない。
  - stale entry-path artifact が残ると、
    current participation の押し引きが live 市況とズレる。
  - あわせて
    `MomentumBurst`
    は 7d で
    `60 trades`
    の entry 間隔のうち
    `28本`
    が
    `120s未満`,
    `24本`
    が
    `90s未満`
    だったため、
    `120s`
    cooldown は valid cluster の一部を削っている可能性がある。
- Expected Good:
  - current 6h participation artifact から stale lane が消え、
    live cadence 制御が current window に揃う。
  - `MomentumBurst`
    の clustered winner/reaccel を少し多く拾える。
- Expected Bad:
  - short-interval re-entry が増え、
    burst 局面での連敗幅が大きくなる可能性がある。
- Period:
  - 直近6h artifact / 直近24h live results / 直近7d `MomentumBurst` cadence
- Fact:
  - `python3 scripts/pdca_profitability_report.py --top-n 10`
    で
    `2026-03-12 13:33 JST`
    snapshot は
    `open_trades=0`,
    `reject_rate=0`,
    `order_success_rate=1.0`
    だった。
  - `logs/orders.db`
    の latest `MomentumBurst-open_long`
    order は
    `2026-03-11T12:46:30.837953+00:00`
    で、
    current 6h 窓には本来入らない。
  - それでも修正前 artifact は
    `MomentumBurst-open_long attempts=2`
    を出しており、
    lookback 判定が壊れていた。
  - 7d `MomentumBurst`
    の entry interval は
    `median 133.8s`,
    `<120s 28/59`,
    `<90s 24/59`
    だった。
- Failure Cause:
  - stale order を current entry-path summary に残す query と、
    `MomentumBurst`
    の base cooldown がやや長いことの組み合わせで、
    cadence の改善判断と actual cadence の両方が鈍っていた。
- Improvement:
  - `entry_path_aggregator`
    を time-aware query に修正し、
    `MomentumBurst`
    の dedicated cooldown を
    `90s / reaccel 20s`
    に下げた。
- Verification:
  - `pytest -q tests/scripts/test_entry_path_aggregator.py tests/workers/test_micro_multistrat_trend_flip.py`
  - `python3 scripts/entry_path_aggregator.py --lookback-hours 6 --limit 6000 --top-k 8`
  - `python3 scripts/participation_allocator.py --lookback-hours 6 --min-attempts 12 --setup-min-attempts 2 --max-units-cut 0.22 --max-units-boost 0.24 --max-probability-boost 0.10`
  - restart 後に
    `MomentumBurst` の
    `fills / avg units / realized_jpy`
    を 30-60 分で確認する。
- Verdict:
  - pending
- Next Action:
  - current 30-60 分で
    `MomentumBurst-open_long`
    の fill が増えない場合は、
    cadence ではなく entry condition 側の long follow-through / forecast gate を切る。
- Status:
  - in_progress

## 2026-03-12 13:00 JST / local-v2: `MomentumBurst` current winner lane の size を小幅に戻す

- Change:
  - `ops/env/local-v2-stack.env`
    と
    `ops/env/quant-micro-momentumburst.env`
    の
    `MICRO_MULTI_STRATEGY_UNITS_MULT`
    を
    `MomentumBurst:1.05 -> 1.20`
    へ引き上げた。
- Why:
  - 2026-03-12 12:53 JST 時点の local-v2 実測では、
    市況は通常帯
    (`USD/JPY 159.007 / spread 0.8 pips / open_trades 0 / reject_rate 0 / order_success_rate 1.0`)
    で、
    `資本を使えていない`
    状態だった。
  - 24h の `MomentumBurst`
    は
    `2 trades / +185.320 JPY / win 100%`
    で、
    strategy 別では最もまともな winner だった。
- Hypothesis:
  - current winner setup が
    `MomentumBurst-open_long|long|transition|...|gap:up_strong`
    に寄っており、
    7d loser setup は
    `dynamic_alloc` の setup override で既に
    `0.14`
    まで薄くなっている。
  - そのため strategy-wide multiplier を
    `1.20`
    へ小幅に戻しても、
    loser setup を大きく増やさず
    strong open_long の期待値を少し押せる。
- Expected Good:
  - `MomentumBurst` の current strong open_long が出たときの
    realized_jpy を増やせる。
- Expected Bad:
  - non-winning setup まで拾う局面では
    損失幅もやや増える。
- Period:
  - 直近24h-7d の local-v2 trades / current live metrics
- Fact:
  - 24h `MomentumBurst`: `2 trades / +185.320 JPY / avg abs units 4674.5`
  - 直近 winner は
    `2026-03-11 21:46-21:51 JST`
    の
    `MomentumBurst-open_long|long|transition|...|gap:up_strong`
    2本で、
    `entry_probability=0.955`,
    `tech_score=0.058-0.064`,
    `+107.208 JPY`,
    `+78.112 JPY`
    だった。
  - `config/dynamic_alloc.json`
    では current loser setup が
    `lot_multiplier=0.14`
    まで落ちている一方、
    strategy 本体は
    `sum_realized_jpy=181.23`
    を維持していた。
- Failure Cause:
  - winner はあるのに、
    shared micro sizing が
    `MomentumBurst:1.05`
    のままで
    current live winner lane を押し切れていなかった。
- Improvement:
  - shared / dedicated 両方の
    `MomentumBurst` units multiplier を
    `1.20`
    へ上げて、
    current live winner lane の size を少し戻す。
- Verification:
  - deploy 後に
    `quant-micro-momentumburst`
    の再起動を確認し、
    次の `30-60m`
    で
    `MomentumBurst filled / avg units / realized_jpy`
    を確認する。
- Verdict:
  - pending
- Next Action:
  - `MomentumBurst`
    の次回 fill が無ければ
    size 調整より
    participation/cadence 側を見直す。
  - fill は出るが expectancy が悪化するなら
    `1.20 -> 1.05`
    に戻す。
- Status:
  - in_progress

## 2026-03-12 12:45 JST / local-v2: `RangeFader` long `range_fade|p0` の weak probe を worker local で遮断

- Change:
  - `workers/scalp_rangefader/worker.py`
    に
    `_rangefader_long_weak_probe_guard()`
    を追加し、
    `RangeFader` long の
    `buy-fade` / `neutral-fade`
    のうち current loser lane だけを
    `projection + tech forecast + entry_probability`
    で reject するようにした。
  - `tests/workers/test_scalp_rangefader_worker.py`
    に
    `neutral-fade` の block/allow と
    `buy-fade` の block 回帰を追加した。
- Why:
  - 2026-03-12 12:37 JST の local-v2 実測では
    市況は通常帯
    (`USD/JPY 159.062 / spread 0.8 pips / open_trades 0 / OANDA正常`)
    で、
    `RangeFader`
    は 24h 集計で
    `27 trades / net -7.764 JPY / win 11.1% / PF 0.59`
    とまだ赤字だった。
  - loser の中心は
    `RangeFader|long|neutral-fade|range_fade|p0`
    `14 trades / -20.8 JPY`
    と
    `RangeFader|long|buy-fade|range_fade|p0`
    `9 trades / -4.1 JPY`
    だった。
- Hypothesis:
  - `neutral-fade`
    は
    `forecast.expected_side_pips<0`
    かつ
    `directional_edge<0`
    で
    `tech_score / projection / entry_probability`
    も薄い lane だけを落とせば、
    positive reclaim sample は残したまま
    main loser cluster を削れる。
  - `buy-fade`
    は 24h で winner が無く、
    `projection<=0.20`
    かつ
    `tech_score<=0.20`
    かつ
    `entry_probability<=0.38`
    の shallow probe を止めれば、
    期待値の悪い long を先に減らせる。
- Expected Good:
  - `RangeFader` long
    `volatility_compression`
    の
    `MARKET_ORDER_TRADE_CLOSE / STOP_LOSS_ORDER`
    のマイナス回数が減る。
- Expected Bad:
  - weak probe から始まる small winner の一部も削る可能性がある。
- Period:
  - 直近24h（2026-03-11 12:37 JST - 2026-03-12 12:37 JST）
- Fact:
  - `RangeFader|long|neutral-fade|range_fade|p0`
    は
    `14 trades / -20.8 JPY`
    で、
    平均
    `projection.score=0.242`,
    `forecast_side_pips=-0.470`,
    `forecast_edge=-0.064`,
    `tech_score=0.132`,
    `entry_probability=0.408`,
    `gap_ratio=0.353`
    だった。
  - 同 lane の唯一の positive sample は
    `forecast_side_pips=0.225`,
    `forecast_edge=0.026`,
    `tech_score=0.271`,
    `gap_ratio=0.565`
    で、
    blanket stop は不要だった。
  - `RangeFader|long|buy-fade|range_fade|p0`
    は
    `9 trades / -4.1 JPY / 0 winners`
    で、
    全 sample が
    `projection.score<=0.20`,
    `tech_score<=0.20`,
    `entry_probability<=0.379`
    に収まっていた。
  - system-wide の最新3クローズは
    2026-03-12 12:29-12:40 JST に
    `+0.534 / +17.440 / +14.256 JPY`
    で連続プラスだったため、
    市況異常ではなく loser lane の修正を続けられる状態だった。
- Failure Cause:
  - `RangeFader` long は
    strategy signal だけでは
    `forecast/tech` の弱さを最終拒否に使っておらず、
    `range_fade|p0` の shallow probe を残していた。
- Improvement:
  - worker local で
    `range_reason=volatility_compression`
    かつ
    `flow_regime=range_fade`
    かつ
    `continuation_pressure=0`
    の long を対象に、
    `buy-fade` と `neutral-fade` で別閾値の
    weak-probe guard を入れる。
  - positive reclaim context
    (`forecast>0`, `gap_ratio高め`, `tech_score高め`)
    の `neutral-fade` は残す。
- Verification:
  - `PYTHONPATH=. pytest -q tests/workers/test_scalp_rangefader_worker.py`
    で
    `10 passed`
    を確認する。
  - live では次の `30-60m` で
    `RangeFader long` の
    `fills / STOP_LOSS_ORDER / realized_jpy`
    を確認する。
- Verdict:
  - pending
- Next Action:
  - deploy 後に
    `RangeFader|long|neutral-fade|range_fade|p0`
    と
    `...buy-fade...`
    の出現数と収支を確認し、
    まだ負けるなら次は
    `order-manager` の
    `file is not a database`
    警告を優先して切る。
- Status:
  - in_progress

## 2026-03-12 10:20 JST / local-v2: `DroughtRevert` long の loser lane を recent outcome で動的に絞る

- Change:
  - `workers/scalp_wick_reversal_blend/config.py` に
    `DROUGHT_SETUP_PRESSURE_*`
    を追加し、`DroughtRevert` long の recent-outcome guard を
    dedicated env から調整できるようにした。
  - `workers/scalp_wick_reversal_blend/worker.py` に
    recent `DroughtRevert` long close を集計する
    `_drought_revert_setup_pressure()` を追加し、
    active 時だけ weak reclaim long を reject するようにした。
  - `ops/env/quant-scalp-drought-revert.env`
    に current live 用の `...DROUGHT_SETUP_PRESSURE_*`
    運用値を追加した。
- Why:
  - 直近24hの local-v2 実測では
    `DroughtRevert` 全体が `19 trades / -24.094 JPY`、
    特に `long × volatility_compression`
    が `13 trades / -33.770 JPY / PF 0.605`
    で赤字寄与になっていた。
  - 同 lane のうち
    `projection.score<=0.08` かつ `setup_quality<0.40`
    は `10 trades / -45.164 JPY` で、
    小さな勝ちを混ぜても expectancy を押し下げていた。
- Hypothesis:
  - recent 8-trade 程度で
    `sl_rate / fast_sl_rate / net_jpy`
    が悪化している間だけ、
    `projection / continuation_pressure / reversion_support / setup_quality`
    が弱い reclaim long を止めれば、
    loser lane のマイナス回数を先に減らせる。
- Expected Good:
  - `DroughtRevert` long の `STOP_LOSS_ORDER` 回数と
    negative expectancy が改善する。
- Expected Bad:
  - active pressure 中は small winner の一部も削る可能性がある。
- Period:
  - 直近24h（2026-03-11 10:20 JST - 2026-03-12 10:20 JST）
- Fact:
  - `DroughtRevert long / volatility_compression`
    直近24h: `13 trades / -33.770 JPY`
  - `projection.score<=0.08 && setup_quality<0.40`
    直近24h: `10 trades / -45.164 JPY`
- Failure Cause:
  - long 側 winner はあるが、
    negative / flat projection の shallow reclaim を
    recent outcome 悪化中でも拾っていた。
- Improvement:
  - strategy-local の `setup_pressure`
    を recent outcome から計算し、
    active 時だけ weak reclaim long を block する。
  - strong reclaim probe
    (`touch_ratio / rev_strength / setup_quality / reversion_support / projection`)
    は維持する。
- Verification:
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    で weak/strong 両 lane を固定する。
  - live では次の `30-60m`
    に `DroughtRevert long` の
    `filled / STOP_LOSS_ORDER / net_jpy`
    を確認する。
- Verdict:
  - pending
- Next Action:
  - `DroughtRevert long`
    の `STOP_LOSS_ORDER`
    と `expectancy_jpy`
    が改善しなければ、
    次は `RangeFader long`
    の setup-scoped loser lane を切る。
- Status:
  - in_progress

## 2026-03-12 10:35 JST / local-v2: `scalp_wick_reversal_blend` 系が市況ラベルを binary に潰していたのを止める

- Change:
  - `workers/scalp_wick_reversal_blend/worker.py`
    の `flow_regime` 上書きをやめ、
    worker-local の label は
    `flow_headwind_regime`
    として別名保持するようにした。
- Why:
  - この worker 系は `continuation_pressure`
    から `flow_regime=continuation_headwind|range_fade`
    を書き込んでいたため、
    `strategy_entry`
    が持つ richer な
    `range_compression / transition / trend_*`
    の setup context を潰していた。
- Hypothesis:
  - binary 化を止めれば、
    shared feedback / dynamic alloc / participation alloc
    が current setup を
    より market-structure に沿って学習できる。
- Expected Good:
  - `DroughtRevert` / `PrecisionLowVol` / `WickReversalBlend`
    の loser cluster が
    `range_fade` 一色に潰れず、
    `range_compression` や `trend_short`
    を別 setup として扱える。
- Expected Bad:
  - 既存 artifact は coarse label 前提のものがあり、
    短期的に setup key の分布が変わる。
- Period:
  - 2026-03-12 10:35 JST 時点の current code / local-v2 logs
- Fact:
  - 直近24hの `DroughtRevert`
    には `range_compression`
    と `range_fade`
    が混在していたが、
    worker-local `flow_guard`
    経由の thesis では binary label に寄っていた。
- Failure Cause:
  - 市況の複雑さを読む基盤は別にあったが、
    worker 側が coarse label を先に固定していた。
- Improvement:
  - binary label は
    `flow_headwind_regime`
    として残し、
    `flow_regime`
    自体は richer live setup context に委ねる。
- Verification:
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    で
    `flow_headwind_regime`
    と `flow_regime`
    の分離を固定する。
- Verdict:
  - pending
- Next Action:
  - 次回 entry から
    `entry_thesis.flow_regime`
    と `setup_fingerprint`
    が richer label で保存されるかを確認する。
- Status:
  - in_progress

## 2026-03-12 09:45 JST / local-v2: `WickReversalBlend` を市況 + recent outcome で動的化し、同値 protection refresh を止める

- Change:
  - `workers/scalp_wick_reversal_blend/policy.py` に
    `WickReversalBlend` long の weak countertrend lane block を追加した。
  - `workers/scalp_wick_reversal_blend/worker.py` に
    recent long loser を読む `setup_pressure` を追加し、
    active 時だけ shallow long を追加で reject するようにした。
  - `execution/order_manager.py` に
    submit-time protection の cache seed を追加し、
    fill 後の identical `on_fill_protection`
    refresh を no-op にした。
- Why:
  - user-facing では 09:30 台に `注文取消` が並んだが、
    実態は `WickReversalBlend` long が約定してすぐ SL を踏み、
    その前後で child order の replace / sibling cancel が見えていた。
  - 取消の説明だけでは不十分で、
    実際に money を削っている current lane と
    無駄な protection refresh の両方を潰す必要があった。
- Hypothesis:
  - `volatility_compression` の weak countertrend long を entry 前に落とせば、
    09:30 型の immediate SL を減らせる。
  - さらに recent loser streak が active な間だけ
    `bbw / range_score / wick_quality / projection_score`
    を使った `setup_pressure`
    を掛ければ、fixed stop ではなく
    current market regime に応じて long participation を絞れる。
  - submit-time `SL/TP` と actual fill 基準の target が同じケースでは
    protection refresh を送らなければ、
    OANDA UI 上の cancel noise と不要な child order churn を減らせる。
- Expected Good:
  - `WickReversalBlend` の `09:30` 型 long loser が減る。
  - normal fill 直後に identical protection refresh が走らず、
    user-facing cancel notice が減る。
- Expected Bad:
  - `WickReversalBlend` long の aggressive reclaim が一部減る可能性がある。
  - broker 側で protection 未作成なのに cache seed だけ入るケースがあれば、
    same-value refresh を skip してしまうリスクがある。
- Period:
  - incident:
    - JST `2026-03-12 09:29-09:30`
  - RCA sample:
    - last 14 days `trades.db`
- Fact:
  - `quant-order-manager.log` では
    `2026-03-12 09:30:16-09:30:17 JST`
    に
    `WickReversalBlend buy 2012 units`
    が `OPEN_REQ sl/tp=159.173/159.220`
    で送られ、
    fill 後の `on_fill_protection`
    も同じ `159.173/159.220`
    を target にしていた。
  - `trades.db` では
    `ticket=459305`
    が `09:30:17 JST fill 159.192`
    から
    `09:30:22 JST STOP_LOSS_ORDER 159.173 / -38.228 JPY`
    で閉じていた。
  - same detailed lane
    (`technical_context.side=long`,
    `range_reason=volatility_compression`,
    `rsi<50`,
    `projection.score<=0.15`,
    `macd_hist_pips<-0.05`)
    は直近14日で
    `3 trades / 0 wins / -44.845 JPY`
    だった。
- Failure Cause:
  - `WickReversalBlend` が
    neutral-to-weak RSI かつ bearish micro momentum の
    countertrend long を strong wick/tick だけで通していた。
  - `order_manager` は
    submit-time protection を cache していなかったため、
    realign 不要でも same-value `TradeCRCDO`
    を送り直しうる状態だった。
- Improvement:
  - `wick_blend_entry_quality()` に
    `range_reason / macd_hist_pips / di_gap`
    を渡し、
    weak countertrend long lane を
    `weak_countertrend_lane`
    として reject する。
  - `WickReversalBlend` long の recent outcome を
    `setup_pressure`
    として集計し、
    current loser streak が active な間だけ
    shallow long を worker local で絞る。
  - fill 後は `_remember_protections()`
    で submit-time `SL/TP`
    を seed し、
    actual fill への change が無いケースは
    `on_fill_protection`
    を skip する。
- Verification:
  - `pytest -q tests/workers/test_scalp_wick_reversal_blend_policy.py`
  - `pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - `pytest -q tests/execution/test_order_manager_dynamic_protection.py`
  - `pytest -q tests/execution/test_order_manager_preflight.py`
- Verdict:
  - pending
- Next Action:
  - restart 後の next `30-60分` で
    `WickReversalBlend` long の
    `filled / STOP_LOSS_ORDER / net_jpy`
    と、
    fill 直後の `on_fill_protection`
    cancel noise が実際に減るかを確認する。
- Status:
  - in_progress

## 2026-03-12 04:57 JST / local-v2: `scalp_extrema_reversal_live` の late probability reject を撤去

- Change:
  - `ops/env/quant-order-manager.env` に
    `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=0.35`
    と
    `ORDER_MANAGER_PRESERVE_INTENT_MIN/MAX_SCALE_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=1.00`
    を追加した。
- Why:
  - live では worker が signal を出して order path まで届いているのに、
    `quant-order-manager` の final probability-scale で
    `entry_probability_below_min_units`
    へ落ち続け、entry cadence が潰れていた。
- Hypothesis:
  - `analysis_feedback` / `participation_alloc` / `auto_canary` / `dynamic_alloc`
    の dynamic trim は残したまま、
    order-manager の二重縮小だけ外せば
    `scalp_extrema_reversal_live`
    の small intent entry が live に戻る。
- Expected Good:
  - `scalp_extrema_reversal_live` の
    `entry_probability_reject`
    が減り、`submit_attempt` / `filled` が増える。
  - weak setup は `entry_probability<0.35`
    で引き続き弾かれる。
- Expected Bad:
  - current loser setup でも
    low-mid probability の entry が通りやすくなるため、
    loser lane の再開が早すぎる可能性がある。
- Period:
  - RCA window:
    - JST `2026-03-12 04:31-04:43`
  - 参照集計:
    - past 1 day `orders.db`
- Fact:
  - current live の `quant-order-manager.log` では
    `2026-03-12 04:31:56-04:43:27 JST`
    に `scalp_extrema_reversal_live`
    の `OPEN_SKIP note=entry_probability:entry_probability_below_min_units`
    が連続していた。
  - `orders.db` の latest reject row では
    `entry_probability 0.4888`,
    `entry_units_intent 55`,
    `confidence 64`
    で、`analysis_feedback -> leading_profile -> participation_alloc -> auto_canary`
    を通過した後、
    `order_manager_probability_gate`
    だけが block していた。
  - past 1 day の `entry_probability_reject`
    は `scalp_extrema_reversal_live 99`
    で、live starvation の主要因の 1 本だった。
- Failure Cause:
  - worker/local/shared で既に dynamic に薄くした後、
    order-manager が same probability を使って
    preserve-intent scale をもう一度掛けており、
    final units が `min_units` を割っていた。
- Improvement:
  - `scalp_extrema_reversal_live`
    だけは order-manager の preserve-intent scale を `1.00`
    へ固定し、
    common layer は truly weak (`<0.35`) reject だけに戻した。
- Verification:
  - restart 後に
    `logs/local_v2_stack/quant-order-manager.log`
    で `scalp_extrema_reversal_live`
    の new signal が
    `entry_probability_below_min_units`
    へ落ちず、
    `submit_attempt` か `filled`
    へ進むことを確認する。
- Verdict:
  - pending
- Next Action:
  - next live signal で
    `scalp_extrema_reversal_live`
    の `entry_probability_reject`
    が消えるかを確認し、
    まだ詰まるなら `analysis_feedback` と
    `auto_canary` の stacked trim 上限を見直す。
- Status:
  - in_progress

## 2026-03-12 07:40 JST / local-v2: `scalp_extrema_reversal_live` の mid-RSI loser probe を worker-local に切る

- Change:
  - `workers/scalp_extrema_reversal/worker.py` に
    non-supportive `mid-RSI` probe block を追加し、
    `range_compression / volatility_compression`
    の current loser short/long を worker local で落とすようにした。
  - `ops/env/quant-scalp-extrema-reversal.env` に
    `LONG_MID_RSI_PROBE_*` / `SHORT_MID_RSI_PROBE_*`
    を追加し、current live threshold を明示した。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    に short loser / short winner / long loser の回帰を追加した。
- Why:
  - entry は戻ったが、
    current loser の `ExtremaReversal`
    は `entry_probability` より
    `mid-RSI / shallow bounce / non-supportive`
    な reversal probe が主因だった。
- Hypothesis:
  - `RSI 55-56` の short と
    `RSI 40-42` の shallow long を
    current range-compression loser として切れば、
    winner short (`RSI 69`) を残したまま
    30 分窓の downside を減らせる。
- Expected Good:
  - `scalp_extrema_reversal_live`
    の immediate `STOP_LOSS_ORDER`
    が減る。
  - high-RSI short winner lane は残る。
- Expected Bad:
  - range compression の
    early reversal long/short を削りすぎると cadence が落ちる。
- Period:
  - RCA window:
    - JST `2026-03-12 06:56-07:28`
- Fact:
  - loser short `459162` は
    `RSI 56.3 / dist_high 0.85 / tick_strength 0.1`
    で `sl_hit_s=0`, `MFE=-1.6p`。
  - loser long `459152` は
    `RSI 41.5 / dist_low 0.331 / ADX 26.4`
    で `sl_hit_s=0`, `MFE=-1.6p`。
  - winner short `459180` は
    `RSI 69.0 / dist_high 0.55`
    で `+1.0p` を確保した。
- Failure Cause:
  - current loser は
    shallow probe が range compression の真ん中で反転に失敗し、
    TP 側へほぼ触れずに逆行していた。
- Improvement:
  - `short_mid_rsi_probe_block` と
    `long_mid_rsi_probe_block`
    を追加し、non-supportive かつ
    `mid-RSI + shallow bounce`
    の probe だけを reject する。
- Verification:
  - restart 後に
    `scalp_extrema_reversal_live`
    の recent fills で
    `RSI 55-56` short と `RSI 40-42` long が減り、
    `STOP_LOSS_ORDER` が下がることを確認する。
- Verdict:
  - pending
- Next Action:
  - live で 3-5 trade 見て、
    まだ loser が続くなら
    `DroughtRevert` の current long `gap:up_flat`
    lane を次に切る。
- Status:
  - in_progress

## 2026-03-12 08:00 JST / local-v2: shared participation は zero-profit setup を boost しない

- Change:
  - `scripts/participation_allocator.py` の
    `boost_participation` 条件を tightened し、
    `realized_jpy == 0` の strategy / setup は
    fill-rate が高くても boost しないようにした。
  - `tests/scripts/test_participation_allocator.py`
    に strategy-level と setup-level の
    zero-profit no-boost 回帰を追加した。
- Why:
  - current live では
    `PrecisionLowVol|short|range_fade|...|gap:down_flat`
    が realized `0.0` のまま
    `boost_participation` され、
    `dynamic_alloc` の trim と衝突していた。
- Hypothesis:
  - 「勝っていない lane は押し上げない」を shared participation に入れると、
    loser / flat lane の無駄な cadence 増を止められる。
- Expected Good:
  - zero-profit setup の
    `probability_boost / units boost`
    が消える。
  - `PrecisionLowVol` の current loser/flat lane を
    shared artifact が誤って押し上げない。
- Expected Bad:
  - close lag で realized がまだ 0 の fresh winner を
    早く押せなくなる可能性がある。
- Fact:
  - current `participation_alloc` では
    `PrecisionLowVol|short|range_fade|...|gap:down_flat`
    が `realized_jpy=0.0` でも
    `boost_participation / probability_boost>0`
    になっていた。
- Improvement:
  - shared participation の boost は
    `positive realized_jpy` を必須にした。
- Verification:
  - regenerated `config/participation_alloc.json`
    で zero-profit lane の boost が消えていることを確認する。
- Verdict:
  - pending
- Next Action:
  - regenerate artifact 後、
    `PrecisionLowVol` の current setup override を見て
    boost conflict が消えたことを確認する。
- Status:
  - in_progress

## 2026-03-12 09:20 JST / local-v2: `PrecisionLowVol` weak overbought short を setup-pressure 前段で遮断

- Change:
  - `workers/scalp_wick_reversal_blend/worker.py` の
    `_signal_precision_lowvol()` に、
    `range_reason=volatility_compression` の short で
    `flow_guard` 付き `rsi>=60`, `projection.score<=0`,
    `setup_quality<0.46` を満たす weak overbought lane を
    reject する guard を追加した。
  - `workers/scalp_wick_reversal_blend/config.py` と
    `ops/env/quant-scalp-precision-lowvol.env` に
    `PREC_LOWVOL_WEAK_SHORT_*` を追加し、
    current threshold を dedicated env で固定した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に block / allow の回帰を追加した。
- Why:
  - current loser は burst 成立後だけではなく、
    setup-pressure が立つ前に
    weak short を同時に複数持って削る entry も残っていた。
- Hypothesis:
  - `flow_guard` が付き、
    `rsi>=60`, `projection.score<=0`, `setup_quality<0.46`
    の weak overbought short を前段で落とせば、
    `PrecisionLowVol` の短期 loser lane を削りつつ
    positive projection short は残せる。
- Expected Good:
  - `PrecisionLowVol` の current short loser が減る。
  - setup-pressure が発火する前の simultaneous weak short を減らせる。
  - 直近30分の scalp expectancy が改善する。
- Expected Bad:
  - overbought short の cadence が少し落ちる。
  - weak 判定が厳しすぎると positive reclaim を取り逃がす可能性がある。
- Fact:
  - 2026-03-12 09:10 JST 時点の市況:
    - USD/JPY `159.212`
    - spread `0.8 pips`
    - M1 ATR14 `2.9 pips`
    - OANDA pricing/candle latency `246-332 ms`
  - 2026-03-12 09:10 JST 時点の直近30分:
    - 全体 `200 trades / +13.0 JPY / PF 1.029`
    - `PrecisionLowVol 24 trades / -81.71 JPY`
  - 直近7日 `PrecisionLowVol short volatility_compression` のうち
    `flow_guard` 付き `rsi>=60`, `projection.score<=0`,
    `setup_quality<0.46` は
    `7 trades / 0 wins / -91.923 JPY` だった。
  - 直近 loser 例:
    - `2026-03-12 09:01 JST`
      `rsi=61.146 / projection=-0.065 / setup_quality=0.435`
      `-> -13.244 JPY`
    - `2026-03-12 09:01 JST`
      `rsi=61.146 / projection=-0.065 / setup_quality=0.416`
      `-> -13.640 JPY`
- Observed/Fact:
  - current burst guard は recent loser burst には効くが、
    burst が成立する前の weak short 同時持ちまでは止められなかった。
  - loser lane は shared gate ではなく、
    `PrecisionLowVol` worker local の current factor だけで説明できる。
- Verdict:
  - pending
- Next Action:
  - restart 後の next `30-60m` で
    `PrecisionLowVol` の
    `filled / net_jpy / STOP_LOSS_ORDER / entry_thesis.flow_guard.setup_quality`
    を再確認する。
  - まだ current short loser が残るなら、
    `projection_score_max` か `setup_quality_max`
    のどちらを tighter にするかを切り分ける。
- Status:
  - done

## 2026-03-12 03:34 JST / local-v2: `PrecisionLowVol` short repeated-loss burst を setup-pressure guard で抑制

- Change:
  - `workers/scalp_wick_reversal_blend/worker.py` に
    `_precision_lowvol_setup_pressure()` を追加し、
    recent `PrecisionLowVol` short `volatility_compression` close から
    `sl_rate / fast_sl_rate / net_jpy / stop_loss_streak /
    fast_stop_loss_streak / last_close_age_sec` を集計するようにした。
  - `_signal_precision_lowvol()` は
    `2x STOP_LOSS_ORDER` と `1x fast SL (<=35s)` が
    `180s` 以内に並ぶ burst 中は weak short re-entry を reject し、
    `touch_ratio / rev_strength / setup_quality / reversion_support /
    projection.score` が強い reclaim short だけを残す。
  - `workers/scalp_wick_reversal_blend/config.py` と
    `ops/env/quant-scalp-precision-lowvol.env` に
    `PREC_LOWVOL_SETUP_PRESSURE_*` を追加し、
    current threshold を dedicated env で明示した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py` と
    `tests/workers/test_scalp_wick_reversal_blend_dispatch.py`
    に回帰を追加した。
- Why:
  - user 指摘どおり `PrecisionLowVol` は
    RR だけではなく
    same-direction の short 再突入 burst が
    数秒-数十秒の broker SL を連打して資産を削っていた。
- Hypothesis:
  - `PrecisionLowVol` short `volatility_compression` の
    immediate loser burst を worker local にだけ絞れば、
    strong reclaim short を残したまま
    「数秒で SL を何回も繰り返す」 lane を止められる。
- Expected Good:
  - `PrecisionLowVol` の short repeated-loss cluster が減る。
  - same lane の `STOP_LOSS_ORDER` と fast-SL count が落ちる。
  - strong reclaim short は維持される。
- Expected Bad:
  - burst 直後の本来 winner だった short まで
    一時的に落とす可能性がある。
  - setup-pressure 条件が緩すぎると guard が効かず、
    厳しすぎると cadence を落とす。
- Period:
  - RCA window:
    - JST `2026-03-11 16:04-22:11`
    - tick validation:
      - `2026-03-11 09:00-2026-03-12 03:30 JST`
- Fact:
  - recent 6h aggregate:
    - `PrecisionLowVol 14 trades / -49.987 JPY`
    - `STOP_LOSS_ORDER 7 trades / -87.233 JPY / avg_hold_sec 26.42`
    - `TAKE_PROFIT_ORDER 2 trades / +37.58 JPY / avg_hold_sec 242.98`
  - all recent `PrecisionLowVol` fills は short で、
    `2026-03-11 16:14 JST` と `17:18 JST` に
    short stop-loss burst が集中していた。
    - 例:
      - `458138` `-1.7 pips` after `34s`
      - `458146` `-1.5 pips` after `1s`
      - `458169` `-1.6 pips` after `3s`
      - `458245` `-2.0 pips` after `9s`
      - `458259` `-1.6 pips` after `22s`
  - tick validation:
    - `TP_touch<=120s 3/14`
    - `TP_touch<=300s 5/14`
    - `TP_touch<=600s 6/14`
    - つまり current loser は
      「全部 tight-SL」ではなく、
      RR 問題と repeated re-entry の両方を含んでいた。
- Failure Cause:
  - `PrecisionLowVol` short `volatility_compression` が
    weak re-entry のまま同方向に連打され、
    short cooldown だけでは burst を止め切れていなかった。
  - hostile projection guard と wider SL の後でも、
    same-direction burst への current local brake が不足していた。
- Improvement:
  - strategy-wide stop ではなく、
    recent repeated-loss burst にだけ効く
    setup-pressure guard を worker local へ追加した。
  - strong reclaim short の allow 条件は残し、
    `setup_pressure` を thesis に保存して
    post-trade 監査を可能にした。
- Verification:
  - `pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py tests/workers/test_scalp_wick_reversal_blend_dispatch.py`
  - `python3 -m py_compile workers/scalp_wick_reversal_blend/config.py workers/scalp_wick_reversal_blend/worker.py`
  - restart 後に `PrecisionLowVol` の
    `filled / STOP_LOSS_ORDER / fast_sl_count / realized_jpy /
    setup_pressure.active / last_close_age_sec`
    を再集計する。
- Verdict:
  - pending
- Next Action:
  - restart 後の next `30-60m` で
    `PrecisionLowVol` short `volatility_compression` の
    burst が減るかを見る。
  - まだ repeated-loss が続く場合は、
    `allow_touch_ratio / allow_reversion_support / active_max_age_sec`
    のどれが甘いかを詰める。
- Status:
  - done

## 2026-03-12 00:05 JST / local-v2: `strategy_feedback_coverage_gap` を canonical remap + 120s loop で解消

- Change:
  - `analysis/strategy_feedback_worker.py` で
    `participation_alloc` の boosted low-sample lane を
    discovered strategy の canonical key へ remap するようにした。
    例: `MomentumBurst-open_long -> MomentumBurst`。
  - `scripts/publish_health_snapshot.py` も
    boosted low-sample lane の表示を canonical key に揃えた。
  - `ops/env/local-v2-stack.env` の
    `STRATEGY_FEEDBACK_LOOP_SEC` を `600 -> 120` へ短縮した。
  - `tests/analysis/test_strategy_feedback_worker.py`,
    `tests/scripts/test_publish_health_snapshot.py`
    に回帰を追加した。
- Why:
  - stack 自体は正常でも、
    `strategy_feedback.json` が restart 直後の薄い payload のまま残ると
    `health_snapshot.json` に `strategy_feedback_coverage_gap` が出ていた。
- Hypothesis:
  - boosted probe の key を canonical strategy に揃え、
    feedback loop を 2 分化すれば、
    coverage gap は restart 後も短時間で self-heal する。
- Expected Good:
  - `MomentumBurst` など active strategy の coverage gap が消える。
  - restart 後の health 赤化時間が短くなる。
- Expected Bad:
  - feedback artifact の更新頻度が上がるため、
    log 書き込み回数は増える。
- Period:
  - JST `2026-03-11 23:54-2026-03-12 00:05`
- Fact:
  - `strategy_feedback.json` の on-disk payload は
    `2 strategies` まで落ちる瞬間があり、
    health では `MicroLevelReactor`, `MomentumBurst`,
    `scalp_extrema_reversal_live` が missing と見えていた。
  - 同時に worker の one-shot build 自体は `9 strategies` を返せていたため、
    runtime 停止ではなく payload freshness / key alignment の問題だった。
- Failure Cause:
  - boosted low-sample lane の key が directional のまま残り、
    health / feedback の canonical strategy key と揃っていなかった。
  - 加えて feedback loop が `600s` で、
    restart 直後の薄い payload が長く残りやすかった。
- Improvement:
  - boosted probe key を canonical remap し、
    loop を `120s` に短縮した。
- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/analysis/test_strategy_feedback_worker.py tests/scripts/test_publish_health_snapshot.py`
  - restart 後に `strategy_feedback.json` と `health_snapshot.json` を再生成して
    `coverage_gap` が消えることを確認する。
- Verdict:
  - pending
- Next Action:
  - restart 後の `strategy_feedback.json` strategies count と
    `eligible_missing_strategies` を再確認する。
- Status:
  - done

## 2026-03-11 23:35 JST / local-v2: shared participation を faster profit cadence 向けに 2 分化

- Change:
  - `scripts/participation_allocator.py` の
    fast winner / loss-drag 分岐を current short window 前提で強め、
    `2 fills` の profitable lane を早めに `boost_participation` へ上げ、
    `realized_jpy <= -8` 級の current loser を早めに `trim_units` へ寄せた。
  - `scripts/run_local_feedback_cycle.py` の
    `entry_path_aggregator` / `participation_allocator` 既定周期を
    `300s -> 120s` に短縮し、
    `--setup-min-attempts 2` を現行運用値にした。
  - `execution/strategy_entry.py` は
    participation cap を `mult_max=1.24`,
    `prob_boost_max=0.12` まで受けられる current runtime に更新した。
  - `tests/scripts/test_participation_allocator.py`,
    `tests/scripts/test_run_local_feedback_cycle.py`
    を更新し、
    `tests/execution/test_strategy_entry_adaptive_layers.py`
    で runtime cap の既存受け口を再検証した。
- Why:
  - local-v2 実測では service / execution は正常で、
    直近 `30m=-5.895 JPY`, `60m=-16.174 JPY`, `120m=+160.610 JPY`
    だった。
  - つまり「遅い」の主因は latency ではなく、
    `MomentumBurst-open_long` winner を shared が押す前に、
    `DroughtRevert` / `PrecisionLowVol` / `scalp_ping_5s_d_live`
    の current loser が先に資金を削ることだった。
- Hypothesis:
  - artifact 再計算を 5 分待たず 2 分で回し、
    さらに `2-attempt` setup まで current loser trim を前倒しすれば、
    winner lane の participation 回復と loser lane の削減が
    30 分窓の中で間に合う。
- Expected Good:
  - `MomentumBurst-open_long` の current winner lane が
    `1.20x` 近辺の shared push を早く受けられる。
  - `DroughtRevert|...|gap:down_flat`,
    `PrecisionLowVol current loser short`,
    `ping_d current loser` の current drag を
    artifact 側で先に薄くできる。
- Expected Bad:
  - short window の bias が強くなり、
    zero-realized / noise lane を誤 boost するリスクがある。
  - `2-attempt` trim は sample noise を拾いやすいため、
    clean setup を一時的に削る副作用があり得る。
- Period:
  - JST `2026-03-11 21:35-23:21`
  - RCA window:
    - `30m`, `60m`, `120m`
- Fact:
  - market/account:
    - `USD/JPY 158.5145-158.5155`
    - `NAV 35578.7594 JPY`
    - open trade `0`
  - execution quality:
    - `decision_latency_ms ≈ 16.7-17.6`
    - `data_lag_ms ≈ 534-549`
    - `preflight 13 -> submit 8 -> filled 7` in 120m
  - current winner:
    - `MomentumBurst-open_long` の 120m 粗利は `+185.32 JPY`
  - current losers:
    - `DroughtRevert -10.279 JPY`
    - `PrecisionLowVol -8.152 JPY`
    - `scalp_ping_5s_d_live -4.53 JPY`
- Failure Cause:
  - shared participation が
    「winner/loser を認識できない」のではなく、
    「認識と反映が遅い」ことが main drag だった。
  - 特に 5 分周期では、
    short-window loser lane が artifact に反映される前に
    次の数発を食っていた。
- Improvement:
  - winner boost の閾値を
    current `2/2 fills + positive realized` lane で即座に有効化し、
    loser trim は `2-attempt` setup まで前倒しした。
  - auto cycle 自体を 2 分化し、
    runtime TTL `30s` と合わせて live 反映を早めた。
- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/scripts/test_participation_allocator.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/scripts/test_run_local_feedback_cycle.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/execution/test_strategy_entry_adaptive_layers.py`
  - restart 後に `config/participation_alloc.json` と `orders.db / trades.db` で
    `MomentumBurst-open_long` の boost と
    current loser setup override の反映を確認する。
- Verdict:
  - pending
- Next Action:
  - 次の `30-60m` で
    `MomentumBurst-open_long`,
    `DroughtRevert gap:down_flat`,
    `PrecisionLowVol current short`,
    `scalp_ping_5s_d_live`
    の `fills / realized_jpy / setup_fingerprint` を再集計する。
  - short-window noise が見えたら
    `2-attempt trim` は loser 条件を維持したまま、
    boost 側だけを stricter に戻す。
- Status:
  - done

## 2026-03-11 21:15 JST / local-v2: `scalp_ping_5s_d_live` を countertrend lane reject + broker TP 前提へ修正

- Change:
  - `workers/scalp_ping_5s/worker.py` に
    D variant 専用の `countertrend_horizon_m1_block` を追加し、
    `horizon_composite_side != neutral` で horizon に逆らい、
    さらに `m1_trend_gate == m1_opposite` の entry を
    strategy-local に reject するようにした。
  - `ops/env/scalp_ping_5s_d.env` と `ops/env/local-v2-stack.env` で
    `SCALP_PING_5S_D_TP_ENABLED=1` を明示し、
    `ping_d` が broker TP 付きで small-win を取りに行く形へ戻した。
  - `tests/workers/test_scalp_ping_5s_extrema_routes.py` に
    D variant countertrend guard の unit test を追加した。
- Why:
  - 2026-03-11 20:45 JST 前後の restart 後に、
    資金を減らしていた immediate loser は `scalp_ping_5s_d_live` だった。
  - user goal は「1円を積み上げる」型であり、
    no-TP の countertrend momentum を維持する理由がない。
- Hypothesis:
  - D の current loser は
    「non-neutral horizon に逆らい、M1 でも逆行している momentum entry」
    に集中しているため、
    その lane だけを worker 内で切れば cadence を大きく落とさずに
    小さな損切りの連打を止められる。
  - broker TP を有効化すれば、
    exit worker 任せの no-TP hold より small-win capture が安定する。
- Expected Good:
  - `scalp_ping_5s_d_live` の `-2 JPY` 級 stop/close churn が減る。
  - scalp_fast の small-win cadence が
    「逆行を薄く打つ」より
    「neutral / aligned setup を短く利確する」側へ寄る。
- Expected Bad:
  - trend window で D の fill 数が一時的に減る。
  - TP が短すぎると一部の大きい winner を早取りし過ぎる。
- Period:
  - 2026-03-08 21:15 JST 〜 2026-03-11 21:15 JST
  - post-restart immediate window: 2026-03-11 20:44 JST 以降
- Fact:
  - local snapshot: `USD/JPY 158.5345`, filled spread は `0.8 pips`,
    filled thesis ATR は概ね `1.77-2.05 pips`,
    `nav=35429.58`, `free_margin_ratio≈0.987`。
  - `scalp_ping_5s_d_live` の post-restart 実績は
    `5 trades / -8.48 JPY / avg -2.22 pips / avg hold 13.2s / avg TP 0.0 pips`。
  - 同 strategy の直近3日で
    `horizon_composite_side != neutral` かつ `m1_trend_gate=m1_opposite`
    は `10 trades / -22.0 JPY / positive trades 0` だった。
- Failure Cause:
  - D worker が
    「horizon に逆らい、M1 trend でも逆行」の momentum lane を
    縮小だけで通しており、
    no-TP のまま small stop を繰り返していた。
- Improvement:
  - shared gate ではなく D worker 内で countertrend lane を reject し、
    D runtime 自体も broker TP を持つ前提へ戻した。
- Verification:
  - `pytest -q tests/workers/test_scalp_ping_5s_extrema_routes.py`
  - restart 後に `orders.db` / `trades.db` で
    `scalp_ping_5s_d_live` の `countertrend_horizon_m1_block` skip と
    `tp_pips > 0` の fill を確認する。
- Verdict:
  - pending
- Next Action:
  - 次の 30-60 分で `scalp_ping_5s_d_live` の
    `filled / realized_jpy / avg_hold_sec / tp_pips / horizon_composite_side`
    を再集計する。
  - なお negative が残る場合は
    `horizon_neutral + m1_opposite` lane も setup-local に切る。
- Status:
  - done

## 2026-03-11 18:32 JST / local-v2: `PrecisionLowVol` short hostile lane を strong reversal probe 条件つきへ調整

- Change:
  - `workers/scalp_wick_reversal_blend/worker.py` の
    `PrecisionLowVol` short hostile projection guard を follow-up 調整し、
    weak hostile lane だけを reject し、
    `rev_strength` と `touch_ratio` が十分強い strong reversal probe は
    confidence / size を落として残すようにした。
  - `tests/workers/test_scalp_wick_reversal_blend_dispatch.py` に
    weak hostile short が落ち、
    strong reversal hostile short は縮小で残ることを追加した。
- Why:
  - `PrecisionLowVol` の hostile short を strategy-local に潰す方針は妥当だが、
    participation を落とし過ぎず、
    本当に reversal が強い probe は残したい。
- Hypothesis:
  - negative projection の short でも
    `rev_strength >= max(min_strength+0.46, 0.82)` かつ `touch_ratio >= 0.55`
    の probe だけを残せば、
    weak loser lane を抑えつつ short participation は維持できる。
- Expected Good:
  - `PrecisionLowVol` short の weak hostile churn を減らしつつ、
    strong reversal short はまだ取れる。
  - 「止める」ではなく、same lane の quality 差で trade を選別できる。
- Expected Bad:
  - strong probe 条件が甘いと hostile lane の一部が still 通る。
  - 条件が厳しすぎると hostile projection の candidate がほぼ全落ちして、
    short cadence が細る。
- Period:
  - UTC `2026-03-11 09:26` - `2026-03-11 09:44`
  - JST `2026-03-11 18:26` - `2026-03-11 18:44`
- Fact:
  - market check:
    - recent tick `158.424-158.474`, spread `0.8 pips`
    - `data_lag_ms=736.2`, `decision_latency_ms=17.0`
  - post-restart `orders.db`:
    - `PrecisionLowVol` short fill が
      `projection.score=0.0`, `setup_quality=0.239` と
      `projection.score=0.075`, `setup_quality=0.34`
      の lane で残っていた。
    - つまり current runtime は already
      supportive / neutral projection short を still 通している。
  - new test:
    - weak hostile short (`projection.score=-0.125`, low quality, weaker reversal) は reject
    - strong hostile short (`projection.score=-0.125`, low quality, stronger reversal) は
      `size_mult<=1.02` で allow
- Failure Cause:
  - 直前の guard は hostile projection lane を block できたが、
    user goal である「止めずに改善する」には
    hostile setup の中の strong probe を残す余地が必要だった。
- Improvement:
  - `PrecisionLowVol` short は
    hostile projection lane を blanket reject せず、
    weak lane のみ reject、strong reversal probe は de-rate して通す。
- Verification:
  - `./.venv/bin/pytest -q tests/workers/test_scalp_wick_reversal_blend_dispatch.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - 反映後 2-6h で
    `PrecisionLowVol` short の `filled`, `STOP_LOSS_ORDER`, `net_jpy`, `avg pl_pips`
    と `projection.score<=-0.10` lane の surviving fills を再監査する。
- Verdict:
  - pending
- Next Action:
  - `projection.score<=-0.10` で surviving fill が still 悪いなら、
    `rev_strength` と `touch_ratio` の gate をさらに引き上げる。
  - short cadence が細り過ぎるなら、
    positive / neutral projection lane 側の confidence/size recovery を検討する。
- Status:
  - in_progress

## 2026-03-11 18:24 JST / local-v2: `PrecisionLowVol` short の hostile projection lane を strategy-local に遮断

- Change:
  - `workers/scalp_wick_reversal_blend/worker.py` の `_signal_precision_lowvol()` に、
    short side で `projection.score` が明確に逆風、`vwap_gap/ATR` が過伸長、
    `flow_guard.setup_quality` が低い lane を block する guard を追加した。
  - `tests/workers/test_scalp_wick_reversal_blend_dispatch.py` に、
    supportive short が残りつつ hostile short が落ちることを追加した。
- Why:
  - `RangeFader` の long-side shallow probe を締めた後、
    live loser は `PrecisionLowVol` short へ寄っており、
    fixed-SL attach 後も low-quality short fade が SL を連打していた。
- Hypothesis:
  - `projection.score <= -0.10`, `vwap_gap/ATR >= 2.5`,
    `setup_quality < 0.40`, `rsi < max(short_min+10, 60)` の short を
    entry 前に strategy-local で落とせば、
    continuation 寄りの shallow short fade を減らして
    per-trade loss の連発を止められる。
- Expected Good:
  - `PrecisionLowVol` の hostile short lane が減り、
    `STOP_LOSS_ORDER` 連打の churn が薄くなる。
  - shared gate を増やさず、`PrecisionLowVol` 自身の quality 判定だけで
    live loser lane を抑えられる。
- Expected Bad:
  - short cadence が落ち過ぎると、
    positive projection の clean revert short まで巻き添えで減る可能性がある。
  - projection score 閾値が浅いと、
    hostile lane の一部が still 通る可能性がある。
- Period:
  - UTC `2026-03-11 05:00` - `2026-03-11 09:24`
  - JST `2026-03-11 14:00` - `2026-03-11 18:24`
- Fact:
  - market check:
    - `USD/JPY 158.3965` 近辺、tick spread `0.8 pips`
    - recent tick range `158.38-158.45`（約 `7.0 pips`）
    - `data_lag_ms=1314.8`, `decision_latency_ms=18.6`
    - `openTrades=[]`
  - `logs/trades.db` current live window:
    - `27 trades / -63.9 JPY / avg -0.67 pips`
    - `PrecisionLowVol 8 trades / -92.2 JPY / avg -1.387 pips / win_rate 0.125`
  - current loser lane:
    - `2026-03-11 07:15-08:19 UTC` の `PrecisionLowVol` short で
      `setup_quality=0.262-0.378`, `flow_regime=range_fade`,
      `projection.score=-0.125/-0.14`, `vwap_gap=7.0-29.4 pips`,
      `rsi=59.18-61.42`, `adx=13.0-16.8`
    - 同 cluster は `STOP_LOSS_ORDER` で `-1.5` から `-2.0 pips` を連打した。
- Failure Cause:
  - fixed-SL は attach 済みで tail loss は止まったが、
    `_signal_precision_lowvol()` は
    negative projection かつ stretched short fade を still 通していた。
  - その結果、broker SL は正しく付いていても
    shallow short fade の期待値自体が負のまま連発していた。
- Improvement:
  - `PrecisionLowVol` short にだけ hostile projection guard を追加し、
    positive/supportive projection の short は残したまま
    stretched loser lane だけを落とす。
- Verification:
  - `./.venv/bin/pytest -q tests/workers/test_scalp_wick_reversal_blend_dispatch.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - 反映後 2-6h で
    `PrecisionLowVol` の `filled`, `STOP_LOSS_ORDER`, `net_jpy`, `avg pl_pips`
    を直前窓と比較する。
  - `orders.db` で `projection.score <= -0.10` の short fill が
    0 近傍まで減るかを確認する。
- Verdict:
  - pending
- Next Action:
  - `PrecisionLowVol` short の cadence が落ち過ぎず、
    `projection.score > 0` の clean short が残るかを次窓で監査する。
  - hostile short が still 残る場合は
    `flow_guard.setup_quality` と `rsi` 側の quality floor をさらに引き上げる。
- Status:
  - in_progress

## 2026-03-11 13:59 JST / local-v2: `RangeFader` の shallow range-fade long probe を strategy-local に遮断

- Change:
  - `strategies/scalping/range_fader.py` に
    `range_fade + continuation_pressure=0` の shallow long probe を落とす
    `shallow_probe_guard` を追加した。
  - guard は `buy-fade` / `neutral-fade` long に対して、
    `range_score`, `setup_quality`, `momentum_pips / ATR`, `RSI distance`
    を同時に見て block する。
  - `tests/strategies/test_scalp_thresholds.py` に
    shallow `buy-fade` / `neutral-fade` が block され、
    `buy-supportive` は通ることを追加した。
- Why:
  - fixed-SL 欠落の是正だけでは
    `RangeFader` の current loser lane が残るため、
    strategy-local の次優先改善が必要だった。
- Hypothesis:
  - `RangeFader|long|neutral-fade|range_fade|p0` と
    `RangeFader|long|buy-fade|range_fade|p0` の shallow probe を
    entry 前に worker 内で遮断すれば、
    spread 負け主体の churn を減らせる。
- Expected Good:
  - `RangeFader` の long-side loser cluster が減り、
    `max_hold_loss` まで持たされる shallow fade が減る。
  - shared `entry_probability_reject` に頼らず、
    same loser setup の preflight を早い段で減らせる。
- Expected Bad:
  - `buy-supportive` に近い marginal long まで落とすと、
    long-side winner cadence も削る可能性がある。
  - loser setup の定義が浅いと、
    sell-side や supportive lane の観測を十分に増やせない。
- Period:
  - UTC `2026-03-10 04:59` - `2026-03-11 04:59`
  - JST `2026-03-10 13:59` - `2026-03-11 13:59`
- Fact:
  - market check:
    - `USD/JPY 158.144/158.152`, spread `0.8 pips`
    - OANDA `pricing` latency `240ms`
    - `M5 ATR14=3.607 pips`, recent 1h range `11.1 pips`
    - `data_lag_ms=330`, `decision_latency_ms=18`
  - `logs/trades.db` 24h:
    - `346 trades / net_jpy=-440.9 / PF=0.452 / win_rate=0.289`
    - `RangeFader 211 trades / net_jpy=-156.0 / exp_jpy=-0.7`
  - current loser cluster:
    - `RangeFader|long|neutral-fade|range_fade|p0`: `18 trades / -21.6 JPY / win_rate=0.111`
    - `RangeFader|long|buy-fade|range_fade|p0`: `16 trades / -8.2 JPY / win_rate=0.000`
    - fresh quality 付き sample でも
      `buy-fade=-2.2 JPY (6 trades)`, `neutral-fade=-8.3 JPY (21 trades)` が継続。
  - fresh loser sample は
    `hold_sec≈180-817`, `close_reason=MARKET_ORDER_TRADE_CLOSE` で、
    `max_hold_loss` まで保有して spread 負けへ寄るケースがあった。
- Failure Cause:
  - current `flow_headwind` guard は trend continuation 側には効いていたが、
    `range_fade p0` の shallow long probe は
    `continuation_pressure=0` のため通過していた。
  - その結果、shared trim 後も shallow long fade が残り、
    低い期待値の churn を作っていた。
- Improvement:
  - trend headwind ではなく
    `range_score + setup_quality + momentum/ATR + RSI distance`
    で shallow range probe 自体を strategy-local に block する。
  - `buy-supportive` は guard 対象から外し、
    supportive winner lane は維持する。
- Verification:
  - `pytest -q tests/strategies/test_scalp_thresholds.py`
  - 反映後 2-6h で
    `RangeFader|long|neutral-fade|range_fade|p0` と
    `RangeFader|long|buy-fade|range_fade|p0` の trade count / net_jpy / win_rate を直前窓と比較する。
  - `orders.db` で `RangeFader-buy-fade` / `RangeFader-neutral-fade` の
    `preflight_start` と `filled` の比率変化を確認する。
- Verdict:
  - pending
- Next Action:
  - local-v2 反映後に
    long loser cluster の減少と `buy-supportive` の cadence 低下有無を同時に監査する。
  - `RangeFader-sell-fade|range_fade|p1` は別 cluster として継続監視し、
    必要なら short 側も setup-local に追加 tightening する。
- Status:
  - in_progress

## 2026-03-11 14:15 JST / local-v2: fixed-SL dedicated worker の broker SL 欠落を order-manager override で補修

- Change:
  - `ops/env/quant-order-manager.env` に
    `DroughtRevert / PrecisionLowVol / VwapRevertS / WickReversalBlend / TickImbalance /
    LevelReject / FalseBreakFade / SqueezePulseBreak / session_open_breakout /
    scalp_macd_rsi_div_live / scalp_macd_rsi_div_b_live`
    向けの `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_*` を追加した。
  - `tests/execution/test_order_manager_sl_overrides.py` に
    underscore を含む strategy tag でも generic override が効くことを固定した。
- Why:
  - current local-v2 は `quant-v2-runtime.env` の `ORDER_FIXED_SL_MODE=0` を baseline にしている一方、
    dedicated worker 側は `ORDER_FIXED_SL_MODE=1` を前提にしていた。
    しかし order submit を担当する `quant-order-manager` にはその worker env が入らないため、
    live fill に broker `stopLossOnFill` が付かず、想定SLを大きく超える負けが出ていた。
- Hypothesis:
  - fixed-SL 前提の戦略だけ order-manager 側で explicit override すれば、
    新規 fill に broker SL が付き、exit worker の遅延や range fade hold で tail loss が膨らむ経路を止められる。
- Expected Good:
  - `VwapRevertS / WickReversalBlend` などの `avg_loss_vs_sl` が改善する。
  - OANDA `openTrades` の新規建玉に `stopLossOrder` が付与される。
- Expected Bad:
  - タイトな SL を持つ戦略では `STOP_LOSS_ON_FILL_LOSS` や early stop の増加で cadence が落ちる可能性がある。
  - strategy-local exit より broker SL が先に約定することで、従来より profit factor が荒れる可能性がある。
- Period:
  - UTC `2026-03-10 04:51` - `2026-03-11 04:51`
  - JST `2026-03-10 13:51` - `2026-03-11 13:51`
- Fact:
  - market check:
    - OANDA `pricing/summary/openTrades` は全て `200 OK`、latency は約 `238-243ms`。
    - `USD/JPY 158.138/158.146`, spread `0.8 pips`, tick 由来 `M1 ATR14=1.293 pips`, recent range `2.8 pips`。
  - `logs/trades.db` 24h:
    - `346 trades / net_jpy=-440.9 / net_pips=-396.7 / win_rate=0.289 / PF=0.452 / expectancy=-1.3 JPY`
    - 比較窓 `prev24h` は `398 trades / net_jpy=-1642.1 / PF=0.558`。
  - loss vs intended SL:
    - `WickReversalBlend avg_loss_vs_sl=3.84x, max=5.46x`
    - `VwapRevertS avg_loss_vs_sl=3.66x, max=7.61x`
  - 実例:
    - `VwapRevertS` は `sl_pips=1.8` に対して `-13.7p / -12.4p`
    - `WickReversalBlend` は `sl_pips=1.83` に対して `-10.0p / -8.8p`
  - `ops/env/quant-scalp-*.env` 側では該当 worker が `ORDER_FIXED_SL_MODE=1` を持つ一方、
    `ops/env/quant-order-manager.env` には同 tag の `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_*` が存在しなかった。
- Failure Cause:
  - worker 側 dedicated env の fixed-SL 意図が、
    発注主体である `quant-order-manager` に伝播していなかった。
  - そのため global baseline `ORDER_FIXED_SL_MODE=0` が優先され、
    live order payload から `stopLossOnFill` が抜けていた。
- Improvement:
  - fixed-SL dedicated strategy の attach 可否を order-manager 側で明示し、
    live order submit の source of truth を 1 箇所に揃える。
- Verification:
  - 新規 fill の `orders.request_json.order.stopLossOnFill` を確認する。
  - `python3 scripts/oanda_open_trades.py` で新規建玉に `stopLossOrder` が出ることを確認する。
  - 24h 後に `WickReversalBlend / VwapRevertS` の `avg_loss_vs_sl` と `STOP_LOSS_ON_FILL_LOSS` 件数を比較する。
- Verdict:
  - pending
- Next Action:
  - local-v2 restart 後に order-manager log と openTrades を確認し、
    `STOP_LOSS_ON_FILL_LOSS` が過剰なら戦略別 `ORDER_ENTRY_MAX_SL_PIPS_*` の見直しへ進む。
- Status:
  - in_progress

## 2026-03-11 11:00 JST / local-v2: `RangeFader` を setup-local entry/exit へ寄せ、thin fade を worker 内で処理

- Change:
  - `strategies/scalping/range_fader.py` が `setup_quality / setup_size_mult` を live factor から算出し、
    thin fade を strategy-local に block するようにした。
  - `workers/scalp_rangefader/worker.py` が `setup_size_mult` を sizing に反映し、
    `setup_*` を `entry_thesis` へ保存するようにした。
  - `workers/scalp_rangefader/exit_worker.py` が `setup_quality / continuation_pressure / flow_regime / setup_fingerprint`
    から `soft_adverse / take_profit / trail / hold` を trade-local に再計算するようにした。
- Why:
  - post-restart loser は `RangeFader-sell-fade` の `range_fade p0/p1` cluster に偏っており、
    thin fade が同じ static exit で処理されていた。
- Hypothesis:
  - thin fade を worker 内で block し、通した trade も low-quality hostile setup では
    `soft_adverse` と早めの `take_profit` に寄せれば、
    `RangeFader` の marginal loser を shared layer 依存なしに薄くできる。
- Expected Good:
  - `RangeFader` の loser cluster が `setup_fingerprint` 単位で薄くなる。
  - 保有時間と per-trade risk が hostile setup で短くなる。
- Expected Bad:
  - `setup_quality` 判定が浅いと winner cadence まで削る。
  - `setup_size_mult` が強すぎると winner も細り、件数だけ減る可能性がある。
- Period:
  - 2026-03-11 10:42-11:00 JST の live loser cluster を根拠に実装。
- Fact:
  - fresh `RangeFader` loser sample は `entry_probability 0.29-0.30`,
    `flow_regime=range_fade`, `continuation_pressure=0/1`,
    `setup_fingerprint=RangeFader|short|sell-fade|range_fade|p0/p1` に集中していた。
  - focused test では hostile low-quality setup が `soft_adverse` と早めの `take_profit` を使うことを確認対象とした。
- Failure Cause:
  - `setup_fingerprint` は live にあっても、
    entry size と exit threshold が still static で、
    setup quality が保有ロジックまでつながっていなかった。
- Improvement:
  - signal 時点の `setup_quality` を entry/exit 両方へ引き回し、
    `RangeFader` の worker 内で dynamic sizing / dynamic exit を完結させる。
- Verification:
  - post-restart の `RangeFader` fresh trade を再集計し、
    `range_fade p0/p1` loser の件数・平均保有時間・平均 pips を直前 window と比較する。
- Verdict:
  - pending
- Next Action:
  - local-v2 反映後に `logs/trades.db` と strategy log で `RangeFader` fresh trade を監査し、
    cadence を削り過ぎていないかも同時に確認する。
- Status:
  - in_progress

## 2026-03-11 04:35 JST / local-v2: RangeFader は通常 spread 下でも `entry_probability_below_min_units` 優勢、current RCA を task 化

- Change:
  - runtime logic は追加変更せず、
    current local-v2 の RangeFader reject profile を実測で再確認した。
  - `docs/TASKS.md`, `docs/OPS_CURRENT.md`, `docs/WORKER_REFACTOR_LOG.md`,
    `docs/ARCHITECTURE.md` へ
    artifact 運用と next verification を同期する前提を整理した。
- Why:
  - 2026-03-11 02:29/02:48 JST の shared participation 改善後も、
    「今の通常相場で RangeFader reject がどこまで減ったか」が
    current diary と task board に残っていなかった。
- Hypothesis:
  - 現在の main blocker は market abnormal や margin ではなく、
    `risk_mult_perf` と `order_probability_scale` を経た
    `entry_probability_below_min_units` 優勢である。
  - よって今やるべきなのは追加の共通緩和ではなく、
    shared trim 後の fill/reject 比率を next live window で再検証すること。
- Expected Good:
  - current state を `OPS_CURRENT` / `TASKS` / change diary で一貫して追える。
  - `RangeFader` の blockage を margin/API 障害と誤認せず、
    participation/perf trim の follow-up に絞れる。
- Expected Bad:
  - current snapshot だけで conclusion を固定し過ぎると、
    次の market regime 変化を取り逃がす可能性がある。
  - shared trim を維持したまま live 監視だけに留めるため、
    短期では reject 件数自体は大きく変わらない可能性がある。
- Period:
  - UTC `2026-03-10 19:25-19:35`
  - JST `2026-03-11 04:25-04:35`
- Fact:
  - 市況は通常帯:
    - `USD/JPY bid=158.044 / ask=158.052 / spread=0.8p`
    - `M1 close=158.001 / ATR14=2.17p / regime=Mixed`
    - `M5 ATR14=7.23p`
    - `H1 ATR14=21.05p / regime=Mixed`
  - account / OANDA:
    - `margin_used=316.096`
    - `free_margin_ratio=0.9911`
    - `USD/JPY short_units=50`
  - `RangeFader` 直近10分:
    - `entry_probability_reject=128`
    - `filled=3`
    - `preflight_start=3`
    - `probability_scaled=3`
    - `submit_attempt=3`
  - metrics:
    - `risk_mult_perf=0.55`
    - `order_probability_scale=0.3781`
    - `data_lag_ms=780.5`
    - `decision_latency_ms=14.3`
- Failure Cause:
  - current blockage は `margin` 系 reject ではなく、
    `RangeFader-sell-fade` が performance / probability trim の後段で
    `entry_probability_below_min_units` へ落ちていること。
  - `shared trim` の改善は入っているが、
    current live window では still `reject >> fill` の比率だった。
- Improvement:
  - 追加の global gate / time block / common relax は入れない。
  - generated artifact の Git 方針を固定し、
    current snapshot を `OPS_CURRENT` と `TASKS` に同期する。
- Verification:
  - `sqlite3 logs/orders.db "select status, count(*) ... client_order_id like '%rangefad%'"` で直近10分を集計
  - `sqlite3 logs/metrics.db "select ts, metric, value ..."` で `risk_mult_perf` / `order_probability_scale` / latency を確認
  - `jq` / `sed` で `tick_cache.json`, `factor_cache.json`,
    `oanda_account_snapshot_live.json`, `oanda_open_positions_live_USD_JPY.json` を確認
- Verdict:
  - `pending`
- Next Action:
  - next live window で
    `RangeFader-sell-fade` の `entry_probability_reject / filled / realized_jpy`
    を再集計し、shared trim の追加見直しが必要か判定する。
  - その際も共通 gate ではなく、
    strategy-local quality か shared participation artifact のどちらで詰めるかを切り分ける。
- Status:
  - in_progress

## 2026-03-11 02:48 JST / local-v2: shared loser-side trim を二段化し、`RangeFader-neutral-fade` は units trim のみに留める

- Change:
  - `scripts/participation_allocator.py` の loser trim を二段化し、
    `trim_units` は維持したまま、
    negative `probability_offset` は stronger loser 条件を満たす lane に限定した。
  - `tests/scripts/test_participation_allocator.py` に
    mild loser と high-reject small-loss loser の回帰を追加した。
- Why:
  - `RangeFader-neutral-fade` は still loser だが、
    `buy/sell-fade` と同じ強さの `probability_offset` を入れるには loss 根拠が弱かった。
- Hypothesis:
  - loser lane を一律に probability trim せず、
    軽症 lane は `trim_units` のみに留めれば、
    `RangeFader-neutral-fade` の取り過ぎだけを抑えつつ fill を潰しにくくできる。
- Expected Good:
  - `RangeFader-buy/sell-fade` の強い trim は維持しつつ、
    `neutral-fade` の mild loser を shared 側で締め過ぎない。
  - `order_manager` の reject gate を変えずに、
    loser lane の重症度で trim 強度を分けられる。
- Expected Bad:
  - stronger loser 条件が緩すぎると、
    本来 probability trim すべき lane を units trim のみに留める可能性がある。
  - `neutral-fade` の live 悪化が続く場合は、
    この変更で reject が減る代わりに負け trade を増やす可能性がある。
- Period:
  - UTC `2026-03-10 17:35-17:48`
  - JST `2026-03-11 02:35-02:48`
- Fact:
  - 市況は変更継続可の通常帯:
    - `USD/JPY bid=157.599 / ask=157.607 / spread=0.8p`
    - recent range `5m=5.0p / 15m=23.1p / 60m=23.8p`
    - `ATR14(M1)=3.175p`, `ATR14(M5)=7.289p`, `ATR14(H1)=20.632p`
    - `data_lag_ms=548.170`, `decision_latency_ms=17.100`
  - current `entry_path_summary_latest.json` では
    `RangeFader-buy-fade attempts=668 fills=52 share_gap=0.1852 hard_block_rate=0.7266`,
    `sell-fade 578/29 share_gap=0.2009 hard_block_rate=0.7136`,
    `neutral-fade 537/35 share_gap=0.1662 hard_block_rate=0.6100`。
  - current `participation_alloc.json` の再生成後は
    `buy-fade probability_offset=-0.0443`,
    `sell-fade=-0.0463`,
    `neutral-fade=0.0` となり、
    `trim_units` 自体は 3 lane で維持された。
  - targeted test は
    `tests/scripts/test_participation_allocator.py`,
    `tests/execution/test_strategy_entry_adaptive_layers.py`,
    `tests/scripts/test_run_local_feedback_cycle.py`,
    `tests/scripts/test_publish_health_snapshot.py`
    で `25 passed`。
- Failure Cause:
  - 直前の shared loser trim は
    mild loser でも severe loser と近い `probability_offset` を返し得た。
  - `RangeFader-neutral-fade` は loss が軽い一方で
    share/reject 指標だけで `buy/sell-fade` と近い trim 強度になっていた。
- Improvement:
  - negative `probability_offset` を
    `loss_pressure` または stronger loser 条件へ分離した。
  - これにより `neutral-fade` は
    `trim_units + probability_offset=0` の shared trim に留まる。
- Verification:
  - `pytest -q tests/scripts/test_participation_allocator.py tests/execution/test_strategy_entry_adaptive_layers.py tests/scripts/test_run_local_feedback_cycle.py tests/scripts/test_publish_health_snapshot.py`
  - `python3 scripts/participation_allocator.py --entry-path-summary logs/entry_path_summary_latest.json --trades-db logs/trades.db --output config/participation_alloc.json --lookback-hours 24 --min-attempts 20`
- Verdict:
  - `pending`
- Next Action:
  - next live window で
    `RangeFader-neutral-fade` の `filled` と `entry_probability_reject` の比率が改善するかを監視する。
  - `neutral-fade` まで負けが拡大するなら、
    shared trim ではなく strategy-local quality 側へ戻って詰める。
- Status:
  - in_progress

## 2026-03-11 02:40 JST / ここまでの改善サマリ: change diary 固定、shared feedback coverage 復旧、review draft 自動化、RangeFader loser lane 前捌き

- Change:
  - `docs/TRADE_FINDINGS.md` を
    `good/bad/pending + Next Action` が追える change diary として固定した。
  - boosted low-sample lane を
    shared `strategy_feedback` / `health_snapshot` coverage へ接続し、
    `strategy_feedback_worker` の zero-win / zero-loss crash を除去した。
  - `scripts/trade_findings_diary_draft.py` を追加し、
    `run_local_feedback_cycle` 後段で review-only draft
    (`logs/trade_findings_draft_latest.{json,md}`) を自動生成するようにした。
  - shared participation に loser-side `probability_offset` を追加し、
    `RangeFader` の late reject を pre-order trim 側へ寄せる改善を入れた。
- Why:
  - 改善自体は進んでいたが、個別エントリだけだと
    「ここまで何を直して、どこが終わっていて、何がまだ pending か」が
    一目で追いにくかった。
- Hypothesis:
  - 改善群を 1 本の要約として束ねておけば、
    次の RCA / 調整 / 引き継ぎが「前提整理」から始まらず、
    すぐ次の改善へ入れる。
- Expected Good:
  - 直近の改善スタックを 1 画面で把握できる。
  - `done` と `pending` が分かれ、次の監視ポイントが明確になる。
- Expected Bad:
  - この要約が更新されなくなると、逆に古いサマリが誤解を招く。
  - 詳細を省きすぎると、個別エントリを読まずに判断してしまう可能性がある。
- Period:
  - UTC `2026-03-10 16:40-17:30`
  - JST `2026-03-11 01:40-02:30`
- Fact:
  - 市況は変更継続可の通常帯:
    - `USD/JPY bid=157.638 / ask=157.646 / spread=0.8p`
    - recent range `5m=8.1p`
    - `data_lag_ms=854.1`, `decision_latency_ms=25.4`, `mechanism_integrity.ok=true`
  - ここまでの改善は大きく 4 本:
    1. change diary 化
    2. boosted low-sample lane の shared feedback / health coverage 復旧
    3. review-only draft 自動生成
    4. `RangeFader` loser lane の shared pre-order trim 強化
  - 状態の整理:
    - `change diary` 固定: `done`
    - `shared feedback coverage + worker crash fix`: `good`
    - `review draft 自動化`: `done`（運用ノイズ評価は継続）
    - `RangeFader` loser-side `probability_offset`: `pending`
- Failure Cause:
  - 調整は「記録運用」「shared feedback」「review automation」「shared participation」の
    別々のエントリに分かれており、全体進捗を俯瞰するまとめがなかった。
- Improvement:
  - 直近の改善群を 1 本の要約へまとめ、
    「何を直したか / 何が効いたか / 何がまだ監視中か」を先頭で読めるようにした。
- Verification:
  - 本ファイル内の 2026-03-11 01:55 JST / 02:13 JST / 02:18 JST / 02:29 JST の各エントリ
  - `docs/WORKER_REFACTOR_LOG.md` の対応追記
- Verdict:
  - `mixed`
- Next Action:
  - `RangeFader-buy/sell-fade` の `entry_probability_reject` / `perf_block` が
    前窓より低下するかを継続監視する。
  - `trade_findings_draft` は draft ノイズが増えないかを数窓運用で確認する。
  - 次の改善を入れたら、このサマリも更新して「いま何が最新か」を保つ。
- Status:
  - done

## 2026-03-11 02:29 JST / local-v2: shared participation に loser-side `probability_offset` を追加し、`RangeFader` の late reject を pre-order trim へ寄せる

- Change:
  - `scripts/participation_allocator.py` が
    `trim_units` lane で `share_gap`, bounded `hard_block_rate`,
    negative `realized_jpy` を合成した負の `probability_offset` を出せるようにした。
  - `execution/strategy_entry.py` は
    `probability_offset<0` を pre-order の `entry_probability` へ反映し、
    `entry_thesis["participation_alloc"]` に
    `entry_probability_before/after` と `reason=overused_trim` を残すようにした。
  - `execution/order_manager.py` の reject gate 自体は変更していない。
- Why:
  - current の `RangeFader` は loser lane の試行過多が続いており、
    shared trim が units だけだと late `entry_probability_below_min_units`
    直前まで signal を流し込んでいた。
- Hypothesis:
  - loser lane に bounded な negative `probability_offset` を追加すれば、
    order-manager の late reject へ届く前に shared artifact 側で前捌きできる。
- Expected Good:
  - overused loser lane の `entry_probability_reject` / `perf_block` が減りやすくなる。
  - `RangeFader-neutral-fade` のような hold lane を巻き込まずに
    `buy-fade` / `sell-fade` だけを shared trim できる。
  - `order_manager` の責務は guard/reject のまま維持できる。
- Expected Bad:
  - loser 判定が短期ノイズに引っ張られると、
    recover 手前の lane を shared 側で少し締め過ぎる可能性がある。
  - `probability_offset` と units trim が同時に効くため、
    live の見かけ上の entry 数は一時的にさらに減る可能性がある。
- Period:
  - UTC `2026-03-10 17:20-17:30`
  - JST `2026-03-11 02:20-02:30`
- Fact:
  - 市況は変更継続可の通常帯:
    - `USD/JPY bid=157.439 / ask=157.447 / spread=0.8p`
    - recent range `5m=6.8p / 15m=9.8p / 60m=13.6p`
    - `ATR14(M1)=2.922p`, `ATR14(M5)=6.934p`, `ATR14(H1)=20.632p`
    - `data_lag_ms=548.528`, `decision_latency_ms=16.972`
  - 直近24h の `RangeFader` order status は
    `buy-fade entry_probability_reject=1159 / perf_block=616 / filled=54`,
    `sell-fade entry_probability_reject=913 / perf_block=549 / filled=44`,
    `neutral-fade entry_probability_reject=344 / perf_block=501 / filled=46`。
  - 直近24h の closed trade は `RangeFader 144 trades / -128.086 JPY / -33.0p / win_rate 0.632`。
  - 変更後に再生成した `config/participation_alloc.json` では
    `RangeFader-buy-fade action=trim_units lot_multiplier=0.8366 probability_offset=-0.0443`,
    `RangeFader-sell-fade action=trim_units lot_multiplier=0.8307 probability_offset=-0.0463`,
    `RangeFader-neutral-fade action=hold probability_offset=0.0`、
    `allocation_policy.negative_probability_offsets_enabled=true` を確認した。
- Failure Cause:
  - `RangeFader` の loser lane は
    leading profile, dynamic alloc, blackboard coordination, participation trim の後でも
    order-manager の probability gate に多く到達していた。
  - shared participation は winner boost だけで、
    loser の probability 側を前倒し調整できていなかった。
- Improvement:
  - overused loser lane に `probability_offset<0` を追加し、
    shared artifact で units と probability の両方を軽く trim できるようにした。
  - 적용点は `execution/strategy_entry.py` に限定し、
    `order_manager` の reject semantics は維持した。
- Verification:
  - `pytest -q tests/execution/test_strategy_entry_adaptive_layers.py tests/scripts/test_participation_allocator.py`
  - `pytest -q tests/scripts/test_run_local_feedback_cycle.py tests/scripts/test_publish_health_snapshot.py`
  - `python3 scripts/participation_allocator.py --entry-path-summary logs/entry_path_summary_latest.json --trades-db logs/trades.db --output config/participation_alloc.json --lookback-hours 24 --min-attempts 20`
- Verdict:
  - `pending`
- Next Action:
  - restart 後の current window で
    `RangeFader-buy/sell-fade` の `entry_probability_reject` と `perf_block` の比率が
    前窓より下がるかを監視する。
  - `neutral-fade` が hold を維持しつつ fill を落としていないかも併せて確認する。
- Status:
  - in_progress

## 2026-03-11 01:55 JST / docs運用: `TRADE_FINDINGS` を「良かった/悪かった/保留」が追える change diary として固定

- Change:
  - `docs/TRADE_FINDINGS.md` の冒頭ルールに
    `Change / Why / Hypothesis / Expected Good / Expected Bad / Verdict / Next Action`
    を追加し、短い記入テンプレートを定義した。
  - `AGENTS.md`, `docs/AGENT_COLLAB_HUB.md`, `docs/INDEX.md`,
    `docs/WORKER_REFACTOR_LOG.md` も同じ前提へ同期した。
- Why:
  - 記録場所自体は既にあったが、変更ごとの「良かった/悪かった/次に何をする」が
    一目で追いにくく、改善判断が時系列比較しづらかった。
- Hypothesis:
  - 変更ごとに期待値と副作用、実測結果、次アクションを固定で残せば、
    同じ失敗の再発と「効かなかった調整の再実施」を減らせる。
- Expected Good:
  - 変更の当たり外れを `good/bad/mixed/pending` で横比較しやすくなる。
  - 次の改善が「前回の続き」から始まり、勘ではなく履歴で回せる。
- Expected Bad:
  - 記録項目が増えることで、追記が雑になるか、更新コストを嫌って未記入が増える可能性がある。
- Period:
  - UTC `2026-03-10 16:40-16:55`
  - JST `2026-03-11 01:40-01:55`
- Fact:
  - 市況は変更継続可の通常帯:
    - `USD/JPY bid=157.536 / ask=157.544 / spread=0.8p`
    - recent range `5m=5.5p / 15m=15.8p / 60m=15.8p`
    - `data_lag_ms=261.6`, `decision_latency_ms=11.7`, `mechanism_integrity.ok=true`
  - 既存ルールでは `docs/TRADE_FINDINGS.md` を単一台帳と定義していたが、
    冒頭 Rules には `Verdict` と `Next Action` の明示必須化がなかった。
- Failure Cause:
  - 「記録先がない」のではなく、「改善日記としての読み方が固定されていない」ことが課題だった。
- Improvement:
  - 単一台帳方針は維持したまま、change diary の最低項目とテンプレートを明文化した。
- Verification:
  - `git diff -- AGENTS.md docs/AGENT_COLLAB_HUB.md docs/INDEX.md docs/TRADE_FINDINGS.md docs/WORKER_REFACTOR_LOG.md`
  - 追記後の `docs/TRADE_FINDINGS.md` 冒頭で必須項目とテンプレートを確認。
- Verdict:
  - `pending`
- Next Action:
  - 次の strategy/runtime 変更からこの形式で 3-5 件連続記録し、
    「読みやすさが上がったか」「未記入が増えていないか」を再評価する。
- Status:
  - done

## 2026-03-11 02:18 JST / local-v2: boosted low-sample lane の shared feedback coverage を復旧し、`strategy_feedback_worker` crash を除去

- Change:
  - `analysis/strategy_feedback_worker.py` が
    active + `boost_participation` lane に `feedback_probe` metadata を出せるようにした。
  - `scripts/publish_health_snapshot.py` は
    fresh `participation_alloc` の boosted low-sample lane を
    active 時のみ `strategy_feedback` coverage 対象へ含めるようにした。
  - `scripts/participation_allocator.py` の `hard_block_rate` を
    `hard_blocks / (attempts + hard_blocks)` へ修正した。
  - 同 worker の zero-win / zero-loss lane での `ZeroDivisionError` を除去した。
- Why:
  - profitable probe lane を `boost_participation` しても、
    shared feedback/health から見えず、worker 自体も一部 lane で落ち得た。
- Hypothesis:
  - boosted lane を active feedback coverage へ入れ、hard-block 指標を bounded に直せば、
    shared participation と shared feedback の blind spot を減らせる。
- Expected Good:
  - active な boosted lane 欠落を `health_snapshot` が即検知できる。
  - `strategy_feedback_worker` が zero-win / zero-loss lane で止まらない。
  - `hard_block_rate` が 1.0 超で暴れず、quality score が安定する。
- Expected Bad:
  - low-sample lane を metadata-only で feedback bus に載せるため、
    `strategy_feedback.json` の strategy 件数は増える。
  - `boost_participation` lane の active 判定が過剰なら health が敏感になり過ぎる可能性がある。
- Period:
  - UTC `2026-03-10 16:45-17:18`
  - JST `2026-03-11 01:45-02:18`
- Fact:
  - 市況は変更継続可の通常帯:
    - `USD/JPY bid=157.638 / ask=157.646 / spread=0.8p`
    - `ATR14(M1)=2.331p`, `ATR14(M5)=6.547p`, `ATR14(H1)=20.873p`
    - `data_lag_ms=547.499`, `decision_latency_ms=16.894`
  - 直近24h の entry path summary は
    `preflight_start=2112`, `filled=404`, `entry_probability_reject=2242`, `perf_block=1672`。
  - 変更前は `config/participation_alloc.json` に
    `PrecisionLowVol`, `session_open_breakout`, `scalp_ping_5s_c_live`
    の `boost_participation` が存在した一方、
    `strategy_feedback.json` は 5 strategy しか持たなかった。
  - さらに `python3 -m analysis.strategy_feedback_worker` は
    zero-win lane の `avg_win` 計算で `ZeroDivisionError` を起こした。
  - 変更後は targeted test
    `tests/scripts/test_participation_allocator.py`,
    `tests/analysis/test_strategy_feedback_worker.py`,
    `tests/scripts/test_publish_health_snapshot.py`,
    `tests/analysis/test_strategy_feedback.py`,
    `tests/scripts/test_run_local_feedback_cycle.py`
    で `26 passed`。
  - 再生成後の `logs/strategy_feedback.json` は `11 strategies` を持ち、
    `scalp_ping_5s_c_live` に `strategy_params.feedback_probe` が出力された。
  - `logs/health_snapshot.json` の
    `mechanism_integrity.ok=true`,
    `strategy_feedback.boosted_low_sample_strategies=['MicroTrendRetest-long','PrecisionLowVol','scalp_ping_5s_c_live','scalp_ping_5s_d_live','session_open_breakout']`,
    `eligible_missing_strategies=[]` を確認した。
- Failure Cause:
  - shared participation と shared feedback の接続が
    `min_trades` 閾値で切れており、
    boosted lane が active でも coverage から落ちていた。
  - `hard_block_rate` は attempts を超える reject count で 1.0 超へ壊れ得た。
  - `strategy_feedback_worker` は zero-win / zero-loss lane を前提にしていなかった。
- Improvement:
  - boosted low-sample lane を metadata-only probe として feedback bus へ露出した。
  - health は active boosted lane のみ追加 coverage 対象にした。
  - `hard_block_rate` を bounded 化し、
    quality score が極端に歪まないようにした。
  - stats helper に zero guard を追加した。
- Verification:
  - `pytest -q tests/analysis/test_strategy_feedback_worker.py tests/scripts/test_publish_health_snapshot.py tests/scripts/test_participation_allocator.py tests/analysis/test_strategy_feedback.py tests/scripts/test_run_local_feedback_cycle.py`
  - `python3 -m analysis.strategy_feedback_worker`
  - `python3 scripts/publish_health_snapshot.py`
- Verdict:
  - `good`
- Next Action:
  - `PrecisionLowVol` / `session_open_breakout` が active に戻った窓で
    `feedback_probe` が自動出力されることを next monitoring で確認する。
  - `RangeFader` の reject 過多は別件なので、
    strategy-local quality と shared trim のどちらで処理するかを分離して次回詰める。
- Status:
  - done

## 2026-03-11 02:13 JST / local-v2: `trade_findings_draft` を feedback cycle 後段へ追加し、change diary の下書きを自動生成

- Change:
  - `scripts/trade_findings_diary_draft.py` を追加し、
    `health_snapshot / pdca_profitability / strategy_feedback /
    trade_counterfactual / replay_quality_gate`
    から `logs/trade_findings_draft_latest.json` /
    `logs/trade_findings_draft_history.jsonl` /
    `logs/trade_findings_draft_latest.md` を生成するようにした。
  - `scripts/run_local_feedback_cycle.py` に `trade_findings_draft` job を追加し、
    既存の interval/lock/output 契約へ載せた。
  - whiteboard 通知は opt-in とし、同一 fingerprint で重複投稿しないようにした。
- Why:
  - analysis artifact は自動生成されていたが、
    `docs/TRADE_FINDINGS.md` の change diary へ昇格する前に散りやすく、
    「何が悪くて次に何を変えるか」の起点が毎回手作業だった。
- Hypothesis:
  - feedback cycle の最後で review draft を自動生成すれば、
    次の改善 entry は facts をゼロから集め直さずに始められ、
    `good/bad/pending` 判定までの時間を短縮できる。
- Expected Good:
  - `TRADE_FINDINGS` へ昇格すべき候補が whiteboard と draft artifact に残り、
    JSON 群の見落としを減らせる。
  - live 発注 worker や health snapshot を汚さず、analysis 後段だけで日記化できる。
- Expected Bad:
  - stale artifact や weak signal まで draft 化すると、review queue がノイズ化する。
  - fingerprint 設計が粗いと、同内容の再通知や history spam が起きる。
- Period:
  - UTC `2026-03-10 17:13-17:24`
  - JST `2026-03-11 02:13-02:24`
- Fact:
  - 市況は変更継続可の通常帯:
    - `USD/JPY bid=157.486 / ask=157.494 / spread=0.8p`
    - recent range `5m=6.0p / 15m=11.0p / 60m=25.3p`
    - `data_lag_ms=1155.1`, `decision_latency_ms=14.1`, `mechanism_integrity.ok=true`
  - 現状の local artifact には
    `health_snapshot`, `pdca_profitability`, `strategy_feedback`,
    `trade_counterfactual`, `replay_quality_gate` が既にあり、
    `run_local_feedback_cycle.py` も job 拡張を受け入れる構造だった。
- Failure Cause:
  - 事実の収集は自動化されていた一方、
    change diary に必要な「人が読むための束ね直し」が自動化されていなかった。
- Improvement:
  - local feedback cycle 後段で review-only draft を自動生成し、
    `docs/TRADE_FINDINGS.md` 本体へは自動追記せず、
    同一 fingerprint の history/whiteboard 重複を抑止した。
- Verification:
  - `python3 -m py_compile scripts/trade_findings_diary_draft.py scripts/run_local_feedback_cycle.py tests/scripts/test_trade_findings_diary_draft.py tests/scripts/test_run_local_feedback_cycle.py`
  - `pytest -q tests/scripts/test_trade_findings_diary_draft.py tests/scripts/test_run_local_feedback_cycle.py`
- Verdict:
  - `pending`
- Next Action:
  - 実運用で `python3 scripts/trade_findings_diary_draft.py --no-whiteboard-enabled` の出力を確認し、
    draft がノイズ過多なら headline/fingerprint 条件をさらに絞る。
- Status:
  - done

## 2026-03-10 23:28 JST / local-v2: shared participation alloc を「低試行 winner の cadence 回復」まで広げ、counterfactual overlay で全戦略の TP/SL を動的化

- Period:
  - UTC `2026-03-10 14:15-14:28`
  - JST `2026-03-10 23:15-23:28`
- Fact:
  - 市況は変更継続可の通常帯:
    - `USD/JPY bid=157.730 / ask=157.738 / spread=0.8p`
    - `ATR14(M1)=2.834p`, `ATR14(M5)=6.863p`, `ATR14(H1)=20.959p`
    - `data_lag_ms=549.341`, `decision_latency_ms=17.026`
  - 変更前の `config/participation_alloc.json` は
    `PrecisionLowVol 10 attempts / 10 fills / +33.67 JPY` と
    `session_open_breakout 4 attempts / 4 fills / +0.804 JPY` を
    どちらも `action=hold` に据え置いていた。
  - 同時に `RangeFader-buy/sell-fade` は
    `666/575 attempts`, `filled_rate=7.81%/4.52%`, `realized_jpy=-71.60/-8.84`,
    `action=trim_units`, `cadence_floor=0.9` で過剰参加側の trim が残っていた。
  - 変更後に `python3 scripts/participation_allocator.py ...` を再実行すると、
    `PrecisionLowVol -> boost_participation / lot_multiplier=1.0296 / probability_boost=0.0089 / cadence_floor=1.0766`
    `session_open_breakout -> boost_participation / lot_multiplier=1.0241 / probability_boost=0.0069 / cadence_floor=1.0651`
    へ更新された。
- Failure Cause:
  - shared `participation_alloc` は `min_attempts` 未満の profitable winner を
    保守的に `hold` へ残しやすく、`micro_runtime` 側も mild `dynamic_alloc` trim があると
    cadence boost を実効頻度へ変換しきれなかった。
  - `analysis/strategy_feedback.py` は counterfactual overlay を
    主に `units/probability` へ使っていたため、
    live 相場で「細かく取る / 大きく狙う」を全戦略へ共通反映する幅が不足していた。
- Improvement:
  - `scripts/participation_allocator.py`
    - low-sample winner の必要試行数を `min_attempts*0.2` 基準へ下げ、
      `filled_rate` と `fill_share-attempt_share` で `session_open_breakout` 級の
      4-trade winner も `boost_participation` に昇格できるようにした。
  - `workers/micro_runtime/worker.py`
    - `boost_participation` の `cadence_floor>1.0` を cooldown 短縮へ反映。
    - `dynamic_alloc` が mild trim のときは cadence boost が一部相殺できるようにし、
      loser lane の `trim + dyn trim` は従来どおり強く減速させる。
  - `workers/scalp_rangefader/worker.py`
    - `boost_participation` でも `cadence_floor` を解釈し、
      trim は延長、boost は短縮の両方向 cooldown を許可した。
  - `analysis/strategy_feedback.py`
    - `trade_counterfactual_latest.json` の `reentry_overrides / side_actions` から
      `sl_distance_multiplier / tp_distance_multiplier` も導出し、
      `strategy_entry` 経由で all-strategy の TP/SL オーバーレイへ反映するようにした。
  - `analysis/strategy_feedback_worker.py`
    - `min_trades` を超えた active strategy は、
      tuning knob が閾値未満でも metadata-only payload を残し、
      `strategy_feedback_coverage_gap` で health が赤化しないようにした。
- Verification:
  - `pytest -q tests/scripts/test_participation_allocator.py tests/analysis/test_strategy_feedback.py tests/analysis/test_strategy_feedback_worker.py tests/workers/test_scalp_rangefader_worker.py tests/workers/test_micro_multistrat_trend_flip.py` → `46 passed`
  - `python3 scripts/participation_allocator.py --entry-path-summary logs/entry_path_summary_latest.json --trades-db logs/trades.db --output config/participation_alloc.json --lookback-hours 24 --min-attempts 20`
  - `config/participation_alloc.json` で `PrecisionLowVol` / `session_open_breakout` の
    `boost_participation` を確認。
- Status:
  - done

## 2026-03-10 12:30 JST / local-v2: `health_snapshot` に `mechanism_integrity` を追加し、forecast / strategy_feedback / dynamic_alloc / pattern_book / blackboard の欠落を即検知

- Period:
  - UTC `2026-03-10 03:21-03:30`
  - JST `2026-03-10 12:21-12:30`
- Fact:
  - 市況は変更継続可の通常帯:
    - `USD/JPY bid=157.860 / ask=157.868 / spread=0.8p`
    - `M1 ATR14=1.872p / RSI=59.82 / ADX=28.46`
    - `M5 ATR14=5.800p / RSI=64.88 / ADX=20.76`
    - `openTradeCount=0`, `marginRate=0.04`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-forecast,quant-strategy-feedback`
    では core 4 本と `quant-forecast`, `quant-strategy-feedback` がすべて `running`。
  - 直前の `logs/health_snapshot.json` は鮮度メトリクスを持っていたが、
    `strategy_feedback` の coverage gap や `entry_intent_board` 欠落のような
    「仕組みが一部抜けている」状態は即座に見えなかった。
- Failure Cause:
  - live 導線では core service が `running` でも、
    `strategy_feedback.json` の戦略カバレッジ欠落、
    `dynamic_alloc/pattern_book/forecast runtime` の stale、
    `entry_intent_board` 消失は別レイヤで起こり得る。
  - 既存 `collect_local_health.sh` は snapshot 鮮度までしか表示せず、
    欠落の検知がログ JSON 深部依存だった。
- Improvement:
  - `scripts/publish_health_snapshot.py`
    - `health_snapshot.json.mechanism_integrity` を追加。
    - `strategy_feedback.json`, `dynamic_alloc.json`, `pattern_book.json`,
      `forecast_improvement_latest.json` の freshness を監査。
    - `analysis.strategy_feedback_worker` の active strategy discovery と
      recent trades remap を再利用し、
      `min_trades` を超えた active entry strategy が
      `strategy_feedback.json` に不在なら `strategy_feedback_coverage_gap` を立てる。
    - `quant-forecast` は port probe だけでなく
      `http://127.0.0.1:8302/health` の `ok=true` でも healthy 扱いにする。
    - `entry_intent_board` は table 不在だけでなく、
      `orders.db` 不在/読取不能でも `entry_intent_board_missing` として拾う。
  - `scripts/collect_local_health.sh`
    - 実行後に `mechanism_integrity=yes|no missing=...` を即表示するよう更新。
- Verification:
  - `pytest -q tests/scripts/test_publish_health_snapshot.py tests/analysis/test_strategy_feedback.py tests/analysis/test_strategy_feedback_worker.py` → `12 passed`
  - `python3 -m py_compile scripts/publish_health_snapshot.py tests/scripts/test_publish_health_snapshot.py`
  - `bash scripts/collect_local_health.sh`
    - `updated=yes`
    - `mechanism_integrity=yes missing=-`
  - `logs/health_snapshot.json`
    - `generated_at=2026-03-10T03:30:13.665454+00:00`
    - `data_lag_ms=686.2`, `decision_latency_ms=17.4`
    - `mechanism_integrity.ok=true`
    - `missing_mechanisms=[]`
    - `strategy_feedback.eligible_missing_strategies=[]`
    - `forecast_service.health={"ok": true, "service": "quant-forecast"}`
    - `blackboard.recent_rows_24h=1`
- Status:
  - done

## 2026-03-10 11:00 JST / local-v2: `trade_cover` profile を追加し、既存 dedicated worker を regime-cover 常駐へ昇格

- Period:
  - UTC `2026-03-10 01:55-02:00`
  - JST `2026-03-10 10:55-11:00`
- Fact:
  - `logs/tick_cache.json`
    - `USD/JPY bid=157.622 / ask=157.630 / spread=0.8p`
    - 直近レンジ `5分 2.0p / 15分 3.9p / 60分 11.0p`
  - `logs/factor_cache.json`
    - `M1 RSI=42.3 / ADX=16.5`
    - `M5 RSI=43.8 / ADX=20.7`
    - `H1 RSI=39.0 / -DI優位`
    - `H4 RSI=52.8 / +DI優位`
    で transition/chop。
  - `logs/orders.db` / `logs/trades.db` 直近60分:
    - `filled=5`
    - `probability_scaled=10`
    - `entry_probability_reject=3`
    - winner は `scalp_extrema_reversal_live 3 trades / +0.56 JPY`
    - `scalp_ping_5s_d_live` は negative-edge block 優勢
- Failure Cause:
  - current `trade_min` は winner 集中には向くが、
    transition/chop で有効な reversion / failed-break / vwap 系 lane が薄く、
    valid setup 不在時に participation が細る。
- Improvement:
  - `scripts/local_v2_stack.sh` に `PROFILE_trade_cover` を追加。
  - `trade_cover` は `trade_min` の構成へ次の pair を重ねた:
    - `quant-scalp-rangefader(+exit)`
    - `quant-scalp-extrema-reversal(+exit)`
    - `quant-scalp-failed-break-reverse(+exit)`
    - `quant-scalp-false-break-fade(+exit)`
    - `quant-scalp-macd-rsi-div-b(+exit)`
    - `quant-scalp-pullback-continuation(+exit)`
    - `quant-scalp-tick-imbalance(+exit)`
    - `quant-scalp-squeeze-pulse-break(+exit)`
    - `quant-micro-rangebreak(+exit)`
    - `quant-micro-trendmomentum(+exit)`
    - `quant-micro-vwapbound(+exit)`
    - `quant-micro-vwaprevert(+exit)`
    - `quant-micro-momentumpulse(+exit)`
    - `quant-micro-momentumstack(+exit)`
  - `brain.py` / `order_manager.py` / `strategy_entry.py` / `pattern_gate.py` /
    `ops/env/local-v2-stack.env` は触らない。
- Verification:
  - `bash -n scripts/local_v2_stack.sh scripts/install_local_v2_launchd.sh`
  - `scripts/local_v2_stack.sh restart --profile trade_cover --env ops/env/local-v2-stack.env`
  - `scripts/local_v2_stack.sh status --profile trade_cover --env ops/env/local-v2-stack.env`
  - `scripts/install_local_v2_launchd.sh --profile trade_cover --env ops/env/local-v2-stack.env`
- Status:
  - done

## 2026-03-10 01:58 UTC / 2026-03-10 10:58 JST - local-v2: `strategy_feedback` が directional split-tag を base 戦略へ接続できておらず、`MicroTrendRetest` 系へ分析補正が届いていなかった

Period:
- 調査/実装: UTC `01:41-01:58` / JST `10:41-10:58`
- 対象（実測）:
  - `logs/strategy_feedback.json`
  - `logs/trades.db`
  - `analysis/strategy_feedback.py`
  - `analysis/strategy_feedback_worker.py`
  - `tests/analysis/test_strategy_feedback.py`
  - `tests/analysis/test_strategy_feedback_worker.py`

Fact:
- 市況は変更継続可の通常帯:
  - `USD/JPY bid=157.680 / ask=157.688 / spread=0.8p`
  - `health_snapshot generated_at=2026-03-10T01:38:04Z`, `data_lag_ms=733`, `decision_latency_ms=10.1`
  - `openTradeCount=0`, `marginRate=0.04`
- `logs/strategy_feedback.json` は `MomentumBurst` / `MicroLevelReactor` を出している一方、
  `MicroTrendRetest` / `MicroTrendRetest-long` / `MicroTrendRetest-short` は欠落していた。
- `logs/trades.db` の直近3日では `MicroTrendRetest-long=17 trades / -290.23 JPY / -45.7p`,
  `MicroTrendRetest-short=2 trades / -540.6 JPY / -10.6p` で、long 側は
  `STRATEGY_FEEDBACK_MIN_TRADES=12` を超えていた。
- 原因は canonical key の食い違い:
  - worker discovery は dedicated service から base key `MicroTrendRetest` を検知
  - trade 集計は live tag `MicroTrendRetest-long` / `-short` を別 key で保持
  - runtime `current_advice()` も base fallback を持たず、split-tag から
    base feedback を引けなかった

Failure Cause:
- `strategy_feedback` が alias/hash suffix だけでなく directional split-tag も
  canonical base 戦略へ束ねる前提になっておらず、
  `execution/strategy_entry.py` の analysis feedback 段で
  `MicroTrendRetest-long/-short` が実質 fail-open になっていた。

Improvement:
- `analysis/strategy_feedback_worker.py`
  - trade stats を discovered canonical strategy keys に再解決し、
    `MicroTrendRetest-long/-short` のような directional tag を
    active base 戦略 `MicroTrendRetest` へ集約するよう更新。
- `analysis/strategy_feedback.py`
  - `current_advice()` に base-tag fallback を追加し、
    runtime の split-tag から base `strategy_feedback` を参照可能にした。
- `tests/analysis/test_strategy_feedback.py`
  - base `MicroTrendRetest` feedback が `MicroTrendRetest-long` へ適用される回帰を追加。
- `tests/analysis/test_strategy_feedback_worker.py`
  - directional trade tag が discovered base strategy へ再集約されて payload に出る回帰を追加。

Verification:
- `pytest -q tests/analysis/test_strategy_feedback.py tests/analysis/test_strategy_feedback_worker.py` → `7 passed`
- `python3 -m py_compile analysis/strategy_feedback.py analysis/strategy_feedback_worker.py tests/analysis/test_strategy_feedback.py tests/analysis/test_strategy_feedback_worker.py`
- 反映後は `logs/strategy_feedback.json` に `MicroTrendRetest` が出ること、
  `strategy_feedback.current_advice("MicroTrendRetest-long")` が `None` でなくなることを確認する。

Status:
- in_progress

## 2026-03-10 01:32 UTC / 2026-03-10 10:40 JST - local-v2: `MomentumBurst` short の tight oversold sell-chase を non-reaccel だけ止める

Period:
- 調査/実装: UTC `01:32-01:40` / JST `10:32-10:40`
- 対象（実測）:
  - `logs/tick_cache.json`, `logs/factor_cache.json`
  - `logs/trades.db`, `logs/orders.db`
  - `logs/local_v2_stack/quant-market-data-feed.log`
  - `strategies/micro/momentum_burst.py`
  - `tests/strategies/test_momentum_burst.py`

Fact:
- 市況は作業継続可の通常帯:
  - `USD/JPY bid=157.624 / ask=157.632 / spread=0.8p`
  - `ATR14(M1)=3.183p`, `ATR14(M5)=8.185p`, `M1 RSI=42.97`, `M1 ADX=22.50`
  - `quant-market-data-feed.log` は `2026-03-10 08:19 JST` に stream reconnect を 1 回出したが、
    `08:19:47 JST` と `08:25:31 JST` に `pricing/stream HTTP 200 OK` で復帰した。
- `trades.db` 24h 集計は `403 trades / net_jpy=-1643.03 / win_rate=46.7% / PF=0.558`。
- `MomentumBurst` は `24 trades / net_jpy=-617.21 / win_rate=50.0% / PF=0.675`。
- 同 strategy の short は `11 trades / net_jpy=-598.47 / win_rate=36.4%` で、ほぼ毀損の本体。
- short loser cluster は
  `tr:dn_strong|rsi:os|vol:tight|atr:low|d:short`
  に偏り、`rsi=23.06-33.67`, `ema20 gap=-7.97~-19.35p`, `ma gap=-1.53~-7.84p`
  のまま売り追いした例が上位損失を占めた。
- 一方で winner short は `spin/doji/pullback` を伴うものが多く、
  `reaccel` か、oversold でも entry bar が marubozu ではないケースに寄っていた。

Failure Cause:
- `MomentumBurst` の non-reaccel short は、tight/low-ATR の oversold 帯で
  bearish marubozu をそのまま追うケースと、
  直前2本が高値引け bullish reclaim の rebound squeeze を再ショートするケースを
  十分に弾けていなかった。
- shared gate や order-manager の問題ではなく、strategy-local short quality の境界が粗かった。

Improvement:
- `strategies/micro/momentum_burst.py`
  - tight short context の中で、`non-reaccel` かつ `oversold/stretch` の short に
    追加 exhaustion guard を導入。
  - entry bar が `大陰線 + close near low + 上ヒゲほぼ無し` の breakdown chase を reject。
  - 直前2本が `bullish + close near high` の rebound squeeze も reject。
  - `reaccel` lane とそれ以外の long/clean short 条件は維持。
- `tests/strategies/test_momentum_burst.py`
  - oversold breakdown chase reject
  - oversold rebound squeeze reject
  を回帰テストとして追加。

Verification:
- `pytest -q tests/strategies/test_momentum_burst.py` → `24 passed`
- `python3 -m py_compile strategies/micro/momentum_burst.py tests/strategies/test_momentum_burst.py`
- 反映後 2h/24h で次を確認:
  - `MomentumBurst short` の `STOP_LOSS_ORDER` 比率と `avg adverse pips` が悪化しないこと
  - `MomentumBurst` 全体の filled cadence が `reaccel` lane を中心に維持されること
  - `RangeFader` / `MicroLevelReactor` の cadence を食わないこと

Status:
- in_progress

## 2026-03-09 13:17 UTC / 2026-03-09 22:17 JST - local-v2: `MomentumBurst` の downside は cadence 不足ではなく SL 幅過多だったため、entry SL と loss-cut drift を同時に締める

Period:
- 調査/実装: UTC `13:09-13:17` / JST `22:09-22:17`
- 対象（実測）:
  - `logs/tick_cache.json`, `logs/factor_cache.json`, `logs/health_snapshot.json`
  - `logs/orders.db`, `logs/trades.db`, `logs/oanda_open_positions_live_USD_JPY.json`
  - `strategies/micro/momentum_burst.py`, `config/strategy_exit_protections.yaml`

Fact:
- 市況は live 変更を入れてよい通常帯:
  - `USD/JPY mid=158.452`, `spread=0.8p`
  - `ATR14(M1)=2.573p`, `ATR14(M5)=6.146p`
  - `data_lag_ms=457.2`, `decision_latency_ms=18.5`
  - `USD/JPY open positions=0`
- ローカルV2 24h は `356 trades / win_rate=54.21% / PF=0.665 / net_jpy=-1111.8`。
- 主因は `MomentumBurst` の downside:
  - `22 trades / -486.4 JPY / win_rate=54.5%`
  - `STOP_LOSS_ORDER=9 / net=-1689.8 JPY / avg=-4.044p`
  - `MARKET_ORDER_TRADE_CLOSE=13 / net=+1203.4 JPY / avg=+2.238p`
- `orders.db` の filled payload では、stop-out cluster の planned SL は `3.2-4.6p` で、
  realized loss とほぼ一致していた。execution latency より broker-side SL 幅が主因。

Failure Cause:
- `MomentumBurst` は entry 数や broad reject より、1トレードあたりの許容損失が重すぎた。
- さらに `MomentumBurst.exit_profile` は現行運用の想定値 `loss_cut_max_hold_sec=900`,
  `loss_cut_cooldown_sec=4` に対して実設定が `1800/6` のまま残っており、
  tail-loss clamp が stale だった。

Improvement:
- `strategies/micro/momentum_burst.py`
  - entry SL を `atr_pips * 1.25` から `atr_pips * 1.15` へ引き締め、
    `SL floor=2.4p` は維持。
  - TP 算式や shared gate には未変更。
- `config/strategy_exit_protections.yaml`
  - `MomentumBurst.exit_profile.loss_cut_max_hold_sec=900`
  - `MomentumBurst.exit_profile.loss_cut_cooldown_sec=4`
- shared `order_manager` / Brain / shared micro gate は変更しない。

Verification:
- `pytest -q tests/strategies/test_momentum_burst.py`
- `python3 -m py_compile strategies/micro/momentum_burst.py`
- 反映後 2h/24h で次を確認:
  - `MomentumBurst` の `STOP_LOSS_ORDER avg_pl_pips` が current 窓より改善すること
  - `MomentumBurst` の `net_jpy` が current 窓より悪化しないこと
  - `RangeFader` / `MicroLevelReactor` の filled cadence が食われないこと

Status:
- in_progress

## 2026-03-09 13:22 UTC / 2026-03-09 22:22 JST - local-v2: `MomentumBurst` 損失拡大抑制のため loss_cut hard_mult を引き締め

Period:
- 調査/実装: UTC `13:10-13:22` / JST `22:10-22:22`
- 対象（実測）:
  - `logs/orders.db`, `logs/trades.db`
  - `config/strategy_exit_protections.yaml`

Fact:
- `MomentumBurst` が 24h で `-486.38 JPY / -7.3p` と赤方に偏っている状態が継続。
- エントリー数は維持されているにもかかわらず、`STOP_LOSS_ORDER` 系の逆行幅が長いケースが目立つ。
- `MomentumBurst` の失敗クラスタは、entry後の逆行が 4p 超で止まるパターンが多く、`loss_cut` の硬制限が効いている構成。

Failure Cause:
- エントリー判定自体の抑制ではなく、`MomentumBurst` の per-trade downside が広く、逆行が一定値を超えた時点での即時収束が不足していた。

Improvement:
- `config/strategy_exit_protections.yaml`
  - `MomentumBurst`:
    - `loss_cut_hard_sl_mult` を `1.50 -> 1.20` に更新。
  - `loss_cut_hard_pips` は 0 のまま据え置き、SL 派生の hard cut を活かしつつ上限を引き下げ。
- shared gate / Brain / order-manager / strategy_local entry 条件には未変更。

Verification:
- 本体再起動後、`MomentumBurst` の次 6h/24h の `STOP_LOSS_ORDER` 比率が低下し、
  `avg_pips` と `max adverse drawdown` が改善することを監視。

Status:
- in_progress

## 2026-03-09 13:00 UTC / 2026-03-09 22:00 JST - local-v2: Brain は shallow `REDUCE` を strong setup で `ALLOW` へ戻し、entry 頻度を落とさず quality 制御を優先

Period:
- 調査/実装: UTC `12:44-13:00` / JST `21:44-22:00`
- 対象（実測）:
  - `logs/brain_state.db`
  - `logs/brain_canary_readiness_latest.json`
  - `workers/common/brain.py`
  - `config/brain_runtime_param_profile_profit_micro.json`

Fact:
- 市況確認（latest readiness）:
  - `USD/JPY bid=158.372 / ask=158.380 / spread=0.8p`
  - `atr_proxy_pips=3.0`, `recent_range_pips_6m=4.2p`
  - `market_ready=true`, `ollama_ready=true`, `quality_gate_ok=true`
- 直近24hの `brain_state.db`:
  - `micro llm ALLOW=7`, `micro llm REDUCE=7`
  - `micro llm_fail ALLOW=6`, `micro llm_fail REDUCE=71`
  - `MomentumBurst-open_long/open_short` は strong setup でも `scale=0.8` 前後の shallow `REDUCE` が残っていた
- common Brain prompt / runtime guard は、これまで `BLOCK -> REDUCE` の補正は持っていたが、
  healthy spread / strong setup での `REDUCE -> ALLOW` 補正は持っていなかった

Failure Cause:
- local LLM は使えていても、strong setup まで shared Brain が shallow `REDUCE` に寄せると、
  strategy-local cadence は維持されても participation と size が不必要に痩せる。
- 特に strong setup で spread / latency / reject 劣化が無い窓では、
  common Brain の役割は entry を減らすことではなく、hard risk だけを止めることだった。

Improvement:
- `workers/common/brain.py`
  - runtime profile に `reduce_to_allow_scale` を追加。
  - strong setup (`entry_probability>=0.80`, `confidence>=75`, spread/ATR 正常) では、
    `REDUCE` が shallow (`scale >= reduce_to_allow_scale`) かつ hard risk reason でない限り
    `ALLOW` へ戻す `activity_preserve_allow` を追加。
  - `spread / latency / reject / execution / slippage` を含む reason は hard risk として uplift 対象外に固定。
  - `llm_fail` 経路でも runtime guard に live context を渡し、同じ preserve 判定を使えるようにした。
- `config/brain_runtime_param_profile_profit_micro.json`
  - `reduce_to_allow_scale=0.78`
- `config/brain_runtime_param_profile.json`
  - `reduce_to_allow_scale=0.80`

Verification:
- `python3 -m py_compile workers/common/brain.py tests/workers/test_brain_history_prompt_autotune.py`
- `pytest -q tests/workers/test_brain_history_prompt_autotune.py tests/workers/test_brain_ollama_backend.py`
  - `21 passed`

Status:
- done

## 2026-03-09 12:39 UTC / 2026-03-09 21:39 JST - local-v2: Brain は strong setup を `BLOCK` しすぎないよう `REDUCE` 優先へ補正

Period:
- 調査/実装: UTC `12:31-12:39` / JST `21:31-21:39`
- 対象（実測）:
  - `logs/brain_canary_readiness_latest.json`
  - `logs/brain_state.db`
  - `logs/orders.db`
  - `workers/common/brain.py`

Fact:
- 市況確認（UTC `12:39` / JST `21:39`）:
  - USD/JPY `bid=158.416 / ask=158.424 / spread=0.8p`
  - `atr_proxy_pips=2.834`, `recent_range_pips_6m=3.0p`
  - `market_ready=true`, `ollama_ready=true`, `quality_gate_ok=true`
- 直近6hの `brain_state.db`:
  - `micro REDUCE no_llm_reduce=71`
  - `micro ALLOW no_llm=8`
  - `micro BLOCK` は直近上位に出ていない一方、common Brain prompt は
    `Prefer blocking on uncertainty or missing context.` を持っていた
- 直近6hの `orders.db`:
  - `filled=293`, `rejected=2`
  - `entry_probability_reject=51`, `strategy_cooldown=45`, `brain_shadow=21`

Failure Cause:
- local LLM の common prompt が uncertainty 時に `BLOCK` へ寄りやすく、
  runtime guard も「過去に block し過ぎた」時しか `BLOCK -> REDUCE` へ矯正しなかった。
- このままだと strong setup まで common Brain が止める方向へ寄り、entry 頻度を削る余地が残っていた。

Improvement:
- `workers/common/brain.py`
  - prompt を `BLOCK` 優先から `REDUCE` 優先へ変更
  - `entry_probability>=0.80` かつ `confidence>=75`、spread/ATR が通常帯の strong setup は
    runtime guard で `BLOCK -> REDUCE` へ矯正
  - spread shock / 高 spread-to-ATR 比では preserve を無効化
- `tests/workers/test_brain_history_prompt_autotune.py`
  - strong setup preserve と spread 劣化時の `BLOCK` 維持を固定

Verification:
- `python3 -m py_compile workers/common/brain.py tests/workers/test_brain_history_prompt_autotune.py tests/workers/test_brain_ollama_backend.py`
- `pytest -q tests/workers/test_brain_history_prompt_autotune.py tests/workers/test_brain_ollama_backend.py`

Status:
- done

## 2026-03-09 12:40 UTC / 2026-03-09 21:40 JST - local-v2: `RangeFader` の reject は解消したため、次の entry 増は `MomentumBurst` reaccel の再突入間隔だけを短縮

Period:
- 調査/実装: UTC `12:31-12:40` / JST `21:31-21:40`
- 対象（実測）:
  - `logs/health_snapshot.json`
  - `logs/orders.db`, `logs/trades.db`
  - `logs/local_v2_stack/quant-micro-momentumburst.log`
  - `logs/local_v2_stack/quant-micro-levelreactor.log`
  - `ops/env/quant-micro-momentumburst.env`

Fact:
- `health_snapshot.json`（UTC `12:35:32` / JST `21:35:32`）:
  - `data_lag_ms=137.4`, `decision_latency_ms=17.4`, `trades_count_24h=340`, `trades_last_entry=12:31:29 UTC`
- 24h strategy 損益:
  - `MicroLevelReactor: 224 trades / +73.52 JPY / +188.5p`
  - `RangeFader: 64 trades / +68.91 JPY / +75.1p`
  - `MomentumBurst: 22 trades / -486.38 JPY / -7.3p`
- 直近2h:
  - `RangeFader: 7 trades / +15.02 JPY / win_rate=85.7% / entry_probability_reject=0`
  - `MicroLevelReactor: 14 trades / -25.70 JPY`
  - `MomentumBurst: 1 trade / +244.40 JPY / +4.6p`
- `MicroLevelReactor` は同窓で `mlr_range_gate_block` が `6` 本あったが、
  latest block は `chop=0.533` / `0.55` でも、成績が負けていたため今この窓で gate を広げる根拠にはならなかった。

Failure Cause:
- `RangeFader` の current 窓ではすでに cadence 改善が効いており、次の scarcity は `MomentumBurst` の reaccel 再突入間隔。
- `MicroLevelReactor` は 24h winner でも直近2hは負けており、ここを broad に緩めると精度毀損リスクが先に立つ。

Improvement:
- `ops/env/quant-micro-momentumburst.env`
  - `MOMENTUMBURST_REACCEL_COOLDOWN_SEC=45 -> 35`
- `MomentumBurst` の non-reaccel 条件 / shared micro gate / order-manager / Brain は変更しない

Verification:
- `pytest -q tests/strategies/test_momentum_burst.py tests/workers/test_micro_multistrat_trend_flip.py`
- `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-micro-momentumburst`
- `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-momentumburst`
- `ps eww -p <quant-micro-momentumburst pid> | rg 'MOMENTUMBURST_REACCEL_COOLDOWN_SEC=35'`

Status:
- done

## 2026-03-09 12:15 UTC / 2026-03-09 21:15 JST - local-v2: precision は許容帯、winner cadence を増やすため `RangeFader` cooldown を短縮し、`MomentumBurst` reaccel を監査可能化

Period:
- 調査/実装: UTC `12:03-12:15` / JST `21:03-21:15`
- 対象（実測）:
  - OANDA `pricing`, `summary`, `openTrades`
  - `logs/replay/USD_JPY/USD_JPY_ticks_20260309.jsonl`
  - `logs/replay/USD_JPY/USD_JPY_M5_20260309.jsonl`
  - `logs/health_snapshot.json`
  - `logs/orders.db`, `logs/trades.db`
  - `python3 scripts/analyze_entry_precision.py --db logs/orders.db --limit 200`

Fact:
- 市況確認（UTC `12:09-12:10` / JST `21:09-21:10`）:
  - OANDA live `bid=158.445 / ask=158.458 / spread=1.3p / tradeable=true / pricing=310.2ms(200)`
  - local tick spread は直近 `0.8p`
  - `M5 bars=60`, `range=49.4p`, `ATR14=6.114p`
  - `health_snapshot.json`: `data_lag_ms=1037.2`, `decision_latency_ms=14.0`, `trades_count_24h=329`
- 24h 損益:
  - `329 trades / win_rate=55.9% / PF=0.685 / expectancy=-3.04 JPY / net_jpy=-999.49`
  - strategy別:
    - `MicroLevelReactor: 218 trades / +80.65 JPY / avg +0.37 JPY`
    - `RangeFader: 58 trades / +54.30 JPY / avg +0.94 JPY`
    - `MomentumBurst: 22 trades / -486.38 JPY / avg -22.11 JPY`
- execution quality（`analyze_entry_precision.py`）:
  - `cost_vs_mid_pips mean=0.390`
  - `slip_vs_side_pips mean=-0.009`
  - `spread_pips mean=0.800`
  - `latency_submit p50=196.7ms`

Failure Cause:
- 直近の主因は execution cost ではなく、winner 側の cadence 不足と loser 側の寄与が重い strategy mix。
- `RangeFader` は勝っているのに回転がまだ足りず、`MomentumBurst` は reaccel fill を監査しにくく、改善後の頻度/精度を `entry_thesis` から直接追えなかった。

Improvement:
- `ops/env/quant-scalp-rangefader.env`
  - `RANGEFADER_COOLDOWN_SEC=24.0 -> 20.0`
- `workers/micro_runtime/worker.py`
  - `MomentumBurst` の reaccel signal だけ `entry_thesis["reaccel"]=true` を記録し、
    `orders.db` / `trades.db` から cadence と quality を直接追跡可能にした
- shared order-manager / 共通 gate / 共通 sizing には追加の一律判定を入れない

Verification:
- `pytest -q tests/strategies/test_momentum_burst.py tests/workers/test_micro_multistrat_trend_flip.py`
- `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-scalp-rangefader,quant-micro-momentumburst,quant-micro-levelreactor,quant-micro-trendretest,quant-micro-vwapbound`
- `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-scalp-rangefader,quant-micro-momentumburst,quant-micro-levelreactor,quant-micro-trendretest,quant-micro-vwapbound`
- `sqlite3 logs/orders.db "SELECT ts,status,json_extract(request_json,'$.entry_thesis.strategy_tag'),json_extract(request_json,'$.entry_thesis.reaccel') FROM orders WHERE ts >= datetime('now','-2 hours') AND json_extract(request_json,'$.entry_thesis.strategy_tag') LIKE 'MomentumBurst%' ORDER BY ts DESC LIMIT 20;"`

Status:
- done

## 2026-03-09 12:17 UTC / 2026-03-09 21:17 JST - local-v2: `RangeFader` の cadence を tag+side 単位へ寄せ、reject に cooldown を消費しないよう修正

Period:
- 調査/実装: UTC `11:55-12:17` / JST `20:55-21:17`
- 対象（実測）:
  - OANDA API `pricing`, `summary`, `openTrades`, `candles(M5)`
  - `logs/health_snapshot.json`
  - `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`
  - `logs/local_v2_stack/quant-order-manager.log`
  - `workers/scalp_rangefader/worker.py`
  - `ops/env/quant-scalp-rangefader.env`

Fact:
- 市況確認（OANDA live, UTC `12:09-12:10` / JST `21:09-21:10`）:
  - USD/JPY `bid=158.449 / ask=158.457 / spread=0.8p`
  - `ATR20(M5)=6.53p`, `last 12xM5 range=13.8p`
  - API 応答品質: `pricing=310-373ms(200)`, `summary=289ms(200)`, `openTrades=289ms(200)`, `candles=292-298ms(200)`
  - `openTradeCount=0`, `marginUsed=0.0`, `marginAvailable=36840.1962`
- execution quality（`python3 scripts/analyze_entry_precision.py --db logs/orders.db --limit 120`）:
  - `slip_vs_side mean=-0.027p`, `cost_vs_mid mean=0.373p`, `spread mean=0.8p`
  - `latency_submit p50=203.9ms`, `latency_preflight p95=2518.0ms`
- 24h strategy 実績:
  - `MicroLevelReactor: 218 trades / +190.8p / win_rate=51.4%`
  - `RangeFader: 58 trades / +68.0p / win_rate=93.1%`
  - `MomentumBurst: 22 trades / -7.3p / win_rate=54.5%`
- `RangeFader` の取りこぼし:
  - `orders.db` 24h では `entry_probability_reject=34` が残り、全件 `entry_probability_below_min_units`
  - 集中帯は UTC `01-05` で、例: `RangeFader-buy-fade prob=0.400 units_intent=263`, `RangeFader-sell-fade prob=0.346 units_intent=788`
  - 同時に `workers/scalp_rangefader/worker.py` は `market_order()` の戻りが `None` でも global `last_entry_ts` を更新しており、
    reject/no-fill でも `30s` cooldown を消費して次の setup を落としていた
  - cooldown も strategy 全体で 1 本だったため、`buy-fade` の直後に `sell-fade` / `neutral-fade` が出ても同じ枠で抑止される構造だった
- `quant-order-manager` は UTC `12:12` / JST `21:12` の再起動後、
  実効 env で `ORDER_MIN_UNITS_STRATEGY_RANGEFADER*=60` を読み込んでいることを `ps eww -p 23904` で確認した

Failure Cause:
- `RangeFader` の no-entry は shared gate ではなく、winner flow の local cadence が粗かったことが主因。
- 特に `entry_probability_below_min_units` の no-fill にまで global cooldown が掛かり、
  同一レンジ帯の連続 signal や反対側 fade の再入場機会まで 30 秒単位で失っていた。

Improvement:
- `workers/scalp_rangefader/worker.py`
  - cooldown を global 1本から `signal_tag + side` 単位へ変更
  - `market_order()` が truthy な結果を返したときだけ cooldown を更新し、reject/no-fill では消費しない
- `ops/env/quant-scalp-rangefader.env`
  - `RANGEFADER_COOLDOWN_SEC=30.0 -> 24.0`
- `tests/workers/test_scalp_rangefader_worker.py`
  - cooldown が tag+side ごとに独立すること
  - cooldown 無効時は block しないこと

Verification:
- `pytest -q tests/workers/test_scalp_rangefader_worker.py`
- `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-scalp-rangefader`
- `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-scalp-rangefader`
- `ps eww -p <quant-order-manager pid> | rg 'ORDER_MIN_UNITS_STRATEGY_RANGEFADER|ORDER_MIN_UNITS_STRATEGY_SCALP_RANGEFAD'`
- `sqlite3 logs/orders.db "SELECT substr(ts,1,13) AS hour_utc,status,COUNT(*) FROM orders WHERE ts >= datetime('now','-2 hours') AND json_extract(request_json,'$.entry_thesis.strategy_tag') LIKE 'RangeFader%' GROUP BY 1,2 ORDER BY 1,2"`
- `sqlite3 logs/trades.db "SELECT COUNT(*), ROUND(SUM(pl_pips),1), ROUND(AVG(pl_pips),2) FROM trades WHERE close_time >= datetime('now','-2 hours') AND strategy_tag LIKE 'RangeFader%'"`

Status:
- done

## 2026-03-09 09:30 UTC / 2026-03-09 18:30 JST - local-v2: 微益側の回転を維持するため、micro runtime に chop 文脈を追加して `MomentumBurst` / `MicroLevelReactor` を再配分

Period:
- 調査/実装: UTC `09:05-09:30` / JST `18:05-18:30`
- 対象（実測）:
  - `logs/health_snapshot.json`
  - `logs/trades.db`, `logs/orders.db`
  - `logs/local_v2_stack/quant-micro-levelreactor.log`
  - `logs/local_v2_stack/quant-market-data-feed.log`
  - `python3 scripts/pdca_profitability_report.py --out-json /tmp/qr_pdca_20260309.json --top-n 10`
  - OANDA `summary` / `open_trades`

Fact:
- 市況確認（UTC `09:30` / JST `18:30`）:
  - USD/JPY `mid=158.435`, `spread=0.8p`, `open_trades=0`
  - `balance/nav=37152.90`
  - `health_snapshot.json` では `data_lag_ms` は概ね sub-2s、`decision_latency_ms` は low teens。
- 24h 損益:
  - `313 trades / win_rate=56.9% / PF(pips)=1.72 / net_pips=211.2 / net_jpy=-686.78`
  - strategy別:
    - `MomentumBurst: -730.78 JPY / 21 trades`
    - `MicroLevelReactor: +99.22 JPY / 210 trades`
    - `RangeFader: +54.30 JPY / 58 trades`
- 時間帯分解:
  - JST `11:00-15:00`
    - `MicroLevelReactor: 120 trades / +254.2 JPY / win_rate=63.3%`
    - `RangeFader: 49 trades / +77.7 JPY / win_rate=100%`
  - JST `17:00-18:30`
    - `MomentumBurst: 10 trades / -283.8 JPY`
    - `RangeFader: 4 trades / -43.0 JPY`
- log:
  - `quant-micro-levelreactor.log` は `18:23-18:27 JST` に
    `mlr_range_gate_block active=False` を連続出力。
  - `quant-market-data-feed.log` は `17:45-17:48 JST` に pricing reconnect failure があったが、
    その後 `200 OK` で復旧。

Failure Cause:
- 今日は「細かく稼ぐ刀」は JST `11-15` の `MicroLevelReactor` / `RangeFader` だったが、
  後半は `MicroLevelReactor` が strict range gate で止まり、`MomentumBurst` が chop/range にそのまま出て毀損した。
- shared sizing でも `MomentumBurst` がまだ重く、逆に微益側は late session の回転再開時に取り分が薄かった。

Improvement:
- `workers/micro_runtime/worker.py`
  - recent M1 から `micro_chop` を算出し、`entry_thesis` へ監査情報を残す。
  - `MicroLevelReactor` は chop score が十分なときだけ strict range gate を override 可能にした。
  - `MICRO_MULTI_CHOP_STRATEGY_UNITS_MULT` を units 計算へ追加し、
    chop 時は `MomentumBurst` を縮小、`MicroLevelReactor` を増量する。
- `strategies/micro/momentum_burst.py`
  - `range_active/range_score/micro_chop_*` を使って strategy-local に confidence 減衰 / skip を追加。
  - `reaccel` signal は例外として通し、「大きく取る」導線は残す。
- env:
  - `ops/env/local-v2-stack.env`
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MomentumBurst:1.05,MicroLevelReactor:1.35`
    - `MICRO_MULTI_CHOP_STRATEGY_UNITS_MULT=MomentumBurst:0.74,MicroLevelReactor:1.22`
    - `MICRO_MULTI_MLR_CHOP_*` を追加
  - `ops/env/quant-micro-momentumburst.env`
    - `MOMENTUMBURST_*` の context tilt 閾値を追加
  - `ops/env/quant-scalp-rangefader.env`
    - `RANGEFADER_BASE_UNITS=14000`

Verification:
- `pytest -q tests/workers/test_micro_multistrat_trend_flip.py tests/strategies/test_momentum_burst.py`
- `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-momentumburst,quant-micro-levelreactor,quant-scalp-rangefader`
- `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-momentumburst,quant-micro-levelreactor,quant-scalp-rangefader`
- `sqlite3 logs/orders.db "SELECT ts,status,side,units,json_extract(request_json,'$.entry_thesis.strategy_tag'),json_extract(request_json,'$.entry_thesis.micro_chop.score') FROM orders WHERE ts >= datetime('now','-2 hours') ORDER BY ts DESC LIMIT 20"`
- `sqlite3 logs/trades.db "SELECT strategy_tag, COUNT(*), ROUND(SUM(realized_pl),2) FROM trades WHERE close_time >= datetime('now','-2 hours') GROUP BY strategy_tag ORDER BY SUM(realized_pl) DESC"`

Status:
- done

## 2026-03-09 07:32 UTC / 2026-03-09 16:32 JST - local-v2: `MomentumBurst` の staircase 条件が主因で entry を落としていたため、strategy-local に 1 本ノイズ許容へ緩和

Period:
- 調査/実装: UTC `07:24-07:32` / JST `16:24-16:32`
- 対象（実測）:
  - OANDA API `pricing`, `summary`, `openTrades`, `candles(M1/M5)`
  - `logs/health_snapshot.json`
  - `logs/orders.db`, `logs/trades.db`
  - `strategies/micro/momentum_burst.py`

Fact:
- 市況確認（OANDA live, UTC `07:24` / JST `16:24`）:
  - USD/JPY `bid=158.586 / ask=158.594 / spread=0.8p`
  - `ATR14(M5)=8.786p`
  - `range_last_12xM5=23.1p`
  - API 応答品質: `pricing=219ms(200)`, `summary=228ms(200)`, `openTrades=239ms(200)`, `candles=223ms(200)`
  - `openTradeCount=0`, `marginUsed=0.0`, `marginAvailable=37589.9852`
- ローカルV2稼働:
  - `logs/health_snapshot.json` at UTC `07:22:33`
    - `trades_count_24h=283`
    - `orders_status_1h`: `filled=7`, `entry_probability_reject=5`, `probability_scaled=6`
- 直近6h実績:
  - `trades.db`
    - `MicroLevelReactor: 153 trades / +194.92 JPY / avg_pips +1.085`
    - `RangeFader: 49 trades / +77.73 JPY / avg_pips +1.567`
    - `MomentumBurst: 10 trades / -260.00 JPY / avg_pips -0.720`
- `MomentumBurst` strategy-local 診断:
  - 直近180本の OANDA M1 を使った near-miss 集計では
    `price_action_direction` のみ不成立が
    `short=33`, `long=21` と最多。
  - 次点は `gap_or_reaccel short=4`, `long_rsi=3`, `long_bull_run=1`, `short_adx=1`。
  - 直近360本 M1 を side別 `90s` cooldown で診断すると、
    strict staircase の candidate `24` 本 (`short=17`, `long=7`) が、
    `3遷移中2票` へ緩和すると `45` 本 (`short=28`, `long=17`) になる。
  - これは candidate bar 数であり、fill 数そのものではない。

Failure Cause:
- `MomentumBurst` は 2026-03-09 の `reaccel` 緩和後も、
  `_price_action_direction()` が recent 4 candles の high/low を
  `3/3` で順行させる strict staircase を要求していた。
- そのため gap / ADX / EMA offset / RSI は十分でも、
  inside bar や wick ノイズ 1 本で continuation が落ち、
  strategy-local の no-entry が残っていた。

Improvement:
- `strategies/micro/momentum_burst.py`
  - `_price_action_direction()` を
    recent 4 candles の `3/3` 順行必須から、
    `3遷移中2票以上` の majority 判定へ変更。
  - `reaccel`, `RSI`, `EMA offset`, shared micro gate, cooldown は変更しない。
- `tests/strategies/test_momentum_burst.py`
  - 1 本だけ逆行するノイズ bar を含む short continuation で `OPEN_SHORT` を返す回帰テストを追加。
  - 2 本以上崩れるケースは reject 維持の回帰テストを追加。

Verification:
- `pytest -q tests/strategies/test_momentum_burst.py`
- `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-momentumburst`
- `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-momentumburst`
- `sqlite3 logs/orders.db "SELECT ts,status,side,units,json_extract(request_json,'$.entry_thesis.strategy_tag') FROM orders WHERE ts >= datetime('now','-2 hours') AND json_extract(request_json,'$.entry_thesis.strategy_tag') LIKE 'MomentumBurst%' ORDER BY ts DESC LIMIT 20"`

Status:
- done

## 2026-03-09 06:10 UTC / 2026-03-09 15:10 JST - local-v2: micro winner の size を戻し、`MomentumBurst` は頻度増に合わせて per-trade risk を再配分

Period:
- 調査/実装: UTC `06:03-06:10` / JST `15:03-15:10`
- 対象（実測）:
  - `logs/health_snapshot.json`
  - `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`
  - `logs/local_v2_stack/quant-micro-momentumburst.log`
  - `logs/local_v2_stack/quant-micro-levelreactor.log`
  - `config/dynamic_alloc.json`

Fact:
- 市況/health:
  - `logs/health_snapshot.json` at UTC `06:03:12` / JST `15:03:12`
    - `data_lag_ms=681.29`
    - `decision_latency_ms=13.41`
    - `orders_last_ts=2026-03-09T05:52:38.745612+00:00`
    - `trades_last_entry=2026-03-09T05:52:38.745612+00:00`
- 直近2h損益:
  - `trades.db`
    - `MicroLevelReactor`: `153 trades / +194.92 JPY / avg_pips +1.085`
    - `MomentumBurst`: `3 trades / -597.77 JPY / avg_pips -3.567`
    - `RangeFader`: `49 trades / +77.73 JPY / avg_pips +1.567`
- サイズ配分:
  - `logs/local_v2_stack/quant-micro-levelreactor.log`
    - UTC `05:06-05:10` / JST `14:06-14:10` で `sent units=1120 -> 483`, `dyn=0.68`, `s_mult=0.80`
  - `logs/local_v2_stack/quant-micro-momentumburst.log`
    - UTC `05:29:01` / JST `14:29:01` の short は `sent units=-10045`, `dyn=0.68`, `s_mult=1.60`
  - `config/dynamic_alloc.json`
    - `MicroLevelReactor.score=0.222`
    - `MomentumBurst.score=0.55`
  - つまり shared micro override が `MomentumBurst` を winner size のまま厚くし、
    直近 winner の `MicroLevelReactor` は `s_mult=0.80` で抑え込んでいた。

Failure Cause:
- `MomentumBurst` は just-fixed の re-acceleration signal で今後エントリー回数が増える見込みだが、
  shared micro override では still `s_mult=1.60` と大きく、直近負け局面に対して per-trade risk が重い。
- 一方で `MicroLevelReactor` は直近2hの実績が最良なのに、
  `s_mult=0.80` と `dyn=0.68` の積で lot が細り、利益拡大の余地を取りこぼしている。

Improvement:
- `ops/env/local-v2-stack.env`
  - `MICRO_MULTI_STRATEGY_UNITS_MULT`
    - `MomentumBurst:1.60 -> 1.35`
    - `MicroLevelReactor:0.80 -> 1.10`
- `ops/env/quant-micro-momentumburst.env`
  - `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=180 -> 120`
  - これで `MomentumBurst` は re-acceleration 後の再投入を増やしつつ、
    1回あたりの risk は少し落として過剰集中を避ける。
  - `MicroLevelReactor` は current winner として size を戻し、entry 数が同じでも稼ぎを増やす。

Verification:
- `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-momentumburst,quant-micro-levelreactor`
- `ps eww -p <pid>` で `MICRO_MULTI_STRATEGY_UNITS_MULT` / `MICRO_MULTI_STRATEGY_COOLDOWN_SEC` の実値確認
- `sqlite3 logs/trades.db "SELECT strategy_tag, COUNT(*), ROUND(SUM(realized_pl),2) FROM trades WHERE close_time >= datetime('now','-2 hours') GROUP BY strategy_tag ORDER BY SUM(realized_pl) DESC"`
- `logs/local_v2_stack/quant-micro-momentumburst.log` / `logs/local_v2_stack/quant-micro-levelreactor.log` の `sent units` を増分監査

Status:
- done

## 2026-03-09 06:08 UTC / 2026-03-09 15:08 JST - local-v2: `RangeFader` の winner flow を増やすため、entry reject 閾値を局所緩和し sizing を少し戻す

Period:
- 調査/実装: UTC `06:00-06:08` / JST `15:00-15:08`
- 対象（実測）:
  - `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`
  - `logs/orderbook_snapshot.json`, `logs/oanda_account_snapshot_live.json`, `logs/oanda_open_positions_live_USD_JPY.json`
  - `ops/env/quant-scalp-rangefader.env`

Fact:
- 市況確認（ローカル/OANDA live）:
  - `logs/orderbook_snapshot.json` at UTC `06:03:37` / JST `15:03:37`
    - USD/JPY `bid=158.390 / ask=158.398 / spread=0.8p`
    - stream latency `216.3ms`
  - `logs/oanda_account_snapshot_live.json`
    - `balance=37254.1602`, `margin_used=0.0`, `free_margin_ratio=1.0`
  - `logs/metrics.db` last 2h
    - `data_lag_ms avg=568.704 max=2512.702`
    - `decision_latency_ms avg=17.101 max=130.631`
    - `reject_rate=0.0`
- `RangeFader` 24h 実績:
  - `trades.db`
    - `49 trades / +77.73 JPY / avg_pips +1.567 / expectancy +1.586 JPY per trade`
    - `close_reason=MARKET_ORDER_TRADE_CLOSE` のみで利益化
- `RangeFader` の entry 詰まり:
  - `orders.db`
    - filled: `buy=14`, `sell=18`, `neutral=17`
    - `entry_probability_reject`: `buy=6`, `sell=22`, `neutral=6`
  - reject の `entry_probability` は主に `0.341-0.350` 帯へ集中
    - 例: `RangeFader-sell-fade` at UTC `02:00-02:04` / JST `11:00-11:04`
      `entry_probability=0.341-0.350`
  - filled は主に `0.372-0.415` 帯
    - 例: `RangeFader-buy-fade` / `sell-fade` / `neutral-fade`
      `entry_probability=0.373-0.415`
  - つまり `entry leading profile reject` の境界が winner flow の喉元に残っている。

Failure Cause:
- `ops/env/quant-scalp-rangefader.env` の
  `RANGEFADER_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.34` が、
  直近の profitable な `RangeFader-sell-fade` / `buy-fade` 候補の一部を落としていた。
- 同時に `RANGEFADER_BASE_UNITS=11000` では live filled units が `80-138` 帯まで薄く、
  winner 戦略の取り分が小さかった。

Improvement:
- `ops/env/quant-scalp-rangefader.env`
  - `RANGEFADER_ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.34 -> 0.30`
  - `RANGEFADER_BASE_UNITS: 11000 -> 12500`
- 意図:
  - 共通 gate や loser 戦略は緩めず、`RangeFader` の profitable zone だけ通過を回復する。
  - 低確率帯を無制限に通すのではなく、`0.34-0.35` 近辺の winner flow を局所的に拾う。
  - 同時に per-trade の取り分を少し増やすが、過大サイズへは戻さない。

Verification:
- `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-scalp-rangefader`
- `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-scalp-rangefader`
- `sqlite3 logs/orders.db "SELECT json_extract(request_json,'$.entry_thesis.strategy_tag'), status, COUNT(*) FROM orders WHERE ts >= datetime('now','-2 hours') AND json_extract(request_json,'$.entry_thesis.strategy_tag') LIKE 'RangeFader%' GROUP BY 1,2 ORDER BY 1,2"`
- `sqlite3 logs/trades.db "SELECT COUNT(*), ROUND(SUM(realized_pl),2), ROUND(AVG(pl_pips),3) FROM trades WHERE close_time >= datetime('now','-2 hours') AND strategy_tag='RangeFader'"`
  で `filled` 増分が出つつ、avg_pips が崩れていないことを確認する

Status:
- done

## 2026-03-09 05:54 UTC / 2026-03-09 14:54 JST - local-v2: `MomentumBurst` が反発後の再下落を MAクロス待ちで取り逃す経路を是正

Period:
- 調査/実装: UTC `05:45-05:54` / JST `14:45-14:54`
- 対象（実測）:
  - `logs/trades.db`, `logs/orders.db`, `logs/metrics.db`
  - `logs/local_v2_stack/quant-micro-momentumburst.log`
  - `logs/replay/USD_JPY/USD_JPY_M1_20260309.jsonl`
  - `logs/factor_cache.json`, `logs/health_snapshot.json`

Fact:
- 市況確認（ローカル/OANDA live）:
  - `logs/health_snapshot.json` at UTC `05:48:53` / JST `14:48:53`
    - `data_lag_ms=517.0`, `decision_latency_ms=14.69`
    - `orders_last_ts=2026-03-09T05:30:14.480224+00:00`
    - `trades_last_entry=2026-03-09T05:29:01.031691+00:00`
  - `logs/orders.db`
    - `2026-03-09 14:29 JST` の `MomentumBurst` short fill 以降、micro の新規 reject は無し
- 取り逃し局面:
  - `logs/replay/USD_JPY/USD_JPY_M1_20260309.jsonl`
    - UTC `05:46:59` / JST `14:46:59` close `158.425`
    - UTC `05:47:59` / JST `14:47:59` close `158.388`
    - UTC `05:48:59` / JST `14:48:59` close `158.340`
  - 同時点の再計算 factor:
    - UTC `05:46:59`
      - `ema20=158.473`, `rsi=39.29`, `adx=32.40`
      - `minus_di=27.18 > plus_di=15.82`
      - `roc5=-0.0398`
      - 直近3本安値ブレイク成立
    - ただし `ma10=158.4623 > ma20=158.4385` で、既存 `gap_pips<=-0.20` 条件だけ未成立
- つまり bearish re-acceleration 自体は出ていたが、`MomentumBurst` は MA10/MA20 の再デッドクロス待ちでショートを再投入できなかった。

Failure Cause:
- `strategies/micro/momentum_burst.py` の short 条件は
  - `gap_pips<=-0.20`
  - `price_action_direction(short)` の4本連続 lower-high/lower-low
  を同時要求しており、反発直後の再下落では再加速を認識する前に遅れる。
- 14:46-14:48 JST のケースでは、価格は `ema20` を明確に割り込み、
  `DI / ROC / ADX` は bearish だった一方で、MAクロスだけが遅れていた。

Improvement:
- `strategies/micro/momentum_burst.py`
  - recent 3-bar の高値/安値ブレイクと `ema20` 乖離、`DI` 優位、`roc5`, `ema_slope_10`
    を使う `reaccel_break` を追加。
  - `MomentumBurst` は既存の `MA gap + staircase` 条件に加えて、
    反発後の再加速条件でも `OPEN_SHORT` / `OPEN_LONG` を返せるようにした。
- `tests/strategies/test_momentum_burst.py`
  - 2026-03-09 14:46 JST 相当の factor/candle で `OPEN_SHORT` になる回帰テストを追加。
  - 14:45 JST 相当の「まだ break 不成立」ケースでは発火しないことも固定。

Verification:
- `pytest -q tests/strategies/test_momentum_burst.py`
- `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env --services quant-micro-momentumburst`
- `sqlite3 logs/orders.db "SELECT ts,status,side,units,strategy_tag FROM orders WHERE strategy_tag LIKE 'MomentumBurst%' AND ts >= datetime('now','-2 hours') ORDER BY ts DESC LIMIT 20"`
- `logs/local_v2_stack/quant-micro-momentumburst.log` で 14:46-14:48 JST 型の再加速局面の `sent units` を監査

Status:
- done

## 2026-03-09 05:34 UTC / 2026-03-09 14:34 JST - local-v2: `MomentumBurst` の `rsi_take` が薄利で負け決済化する経路を是正

Period:
- 調査/実装: UTC `05:28-05:34` / JST `14:28-14:34`
- 対象（実測）:
  - `logs/trades.db`, `logs/orders.db`, `logs/metrics.db`
  - `logs/local_v2_stack/quant-micro-momentumburst.log`
  - `logs/local_v2_stack/quant-micro-momentumburst-exit.log`
  - `logs/replay/USD_JPY/USD_JPY_ticks_20260309.jsonl`
  - `logs/replay/USD_JPY/USD_JPY_M1_20260309.jsonl`
  - `logs/orderbook_snapshot.json`, `logs/oanda_account_snapshot_live.json`, `logs/oanda_open_positions_live_USD_JPY.json`

Fact:
- 市況確認（ローカル/OANDA live）:
  - `logs/orderbook_snapshot.json` at UTC `05:34:25` / JST `14:34:25`
    - USD/JPY `bid=158.392 / ask=158.400 / spread=0.8p`
    - stream latency `113.3ms`
  - `logs/metrics.db`
    - `data_lag_ms=434-1429`
    - `decision_latency_ms=14-24`
    - `reject_rate=0.0` for `MomentumBurst-open_short`
  - `logs/oanda_account_snapshot_live.json`
    - `balance=37258.9602`, `margin_used=0.0`
  - `logs/oanda_open_positions_live_USD_JPY.json`
    - `long_units=0`, `short_units=0`
- 対象約定:
  - ユーザー提示の `454072` は `trades.transaction_id=454072` に一致し、
    実 trade は `ticket_id=454064`, `strategy_tag=MomentumBurst`, `client_order_id=qr-1773034140605-micro-momentumc2a9d46cd`
  - open: UTC `05:29:01.031691` / JST `14:29:01.031691`
    - short `-5422`, entry `158.394`
  - close: UTC `05:30:00.287523` / JST `14:30:00.287523`
    - `MARKET_ORDER_TRADE_CLOSE`, exit `158.409`, `pl_pips=-1.5`, `realized_pl=-81.33`
  - `logs/local_v2_stack/quant-micro-momentumburst-exit.log`
    - UTC `05:30:00.373` / JST `14:30:00.373` で `reason=rsi_take pnl=0.10p`
- tick 照合:
  - `python3 ~/.codex/skills/qr-tick-entry-validate/scripts/tick_entry_validate.py --trades-db logs/trades.db --ticks logs/replay/USD_JPY/USD_JPY_ticks_20260309.jsonl --instrument USD_JPY --ticket 454064`
  - 結果:
    - `mae_120s=3.6p`
    - `mfe_120s=3.0p`
    - `mfe_300s=5.3p`
    - `tp_touch<=300s=0/1`
  - つまり full TP `8.37p` までは未達だが、決済後 2-5 分では short 利益側へ再度伸びていた。
- 傾向:
  - `trades.db` 14d の `MomentumBurst` は `MARKET_ORDER_TRADE_CLOSE=97`, `avg_pips=-2.044`, `sum_pl=-2802.22`
  - 同期間の `TAKE_PROFIT_ORDER=76`, `avg_pips=+4.917`, `sum_pl=+8347.79`

Failure Cause:
- `workers/micro_runtime/exit_worker.py` の `rsi_take` は「`pnl > 0` かつ RSI 閾値到達」だけで発火しており、
  `MomentumBurst` 固有の最低利益バッファを持っていなかった。
- 今回は `mark_pnl_pips` ベースで `+0.10p` の時点で `rsi_take` が出たが、
  fast move + spread で実約定は `-1.5p` へ反転した。
- `orders.db` ではその後の再試行が `close_reject_profit_buffer`（`min_profit_pips=1.2`, `est_pips=1.0`）で拒否されており、
  初回の薄利 `rsi_take` だけが通ってしまう構造が露出した。

Improvement:
- `workers/micro_runtime/exit_worker.py`
  - `exit_profile` から `rsi_take_min_pips` / `rsi_take_tp_ratio` を読めるようにし、
    `rsi_take` はその閾値以上の利益がある場合だけ許可する。
- `config/strategy_exit_protections.yaml`
  - `MomentumBurst.exit_profile.rsi_take_min_pips=1.6` を追加。
  - これにより `MomentumBurst` の `rsi_take` は `+0.1p` のような thin edge では出ず、
    spread/slippage を吸収できる帯まで待つ。

Verification:
- `pytest -q tests/workers/test_micro_runtime_exit_worker.py`
- `python3 ~/.codex/skills/qr-tick-entry-validate/scripts/tick_entry_validate.py --trades-db logs/trades.db --ticks logs/replay/USD_JPY/USD_JPY_ticks_20260309.jsonl --instrument USD_JPY --ticket 454064`
- `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env --services quant-micro-momentumburst-exit`
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env --services quant-micro-momentumburst-exit`
- `sqlite3 logs/orders.db "SELECT status, COUNT(*) FROM orders WHERE client_order_id LIKE 'qr-%micro-momentum%' AND ts >= datetime('now','-24 hours') GROUP BY status"`
  で `close_reject_profit_buffer` の churn と `rsi_take` 発火帯を監査する

Status:
- done

## 2026-03-07 13:03 UTC / 2026-03-07 22:03 JST - local-v2: counterfactual/forecast/tag解決の閉ループ欠線を docs へ反映

Period:
- 確認/記録: UTC `12:42-13:03` / JST `21:42-22:03`
- 対象（実測）: `logs/pdca_profitability_latest.json`, `logs/brain_canary_readiness_latest.json`, `logs/trade_counterfactual_latest.json`, `logs/forecast_improvement_latest.json`
- 対象（コード読解）: `analysis/strategy_feedback.py`, `execution/strategy_entry.py`, `analysis/forecast_improvement_worker.py`, `workers/common/forecast_gate.py`, `analysis/replay_quality_gate_worker.py`, `execution/reentry_gate.py`, `analysis/strategy_feedback_worker.py`, `scripts/dynamic_alloc_worker.py`, `utils/strategy_tags.py`

Fact:
- 市況確認（ローカル実測、週末 stale）:
  - `logs/pdca_profitability_latest.json`
    - pricing time=`2026-03-06T21:59:05.068919348Z`
    - USD/JPY `bid=157.790 / ask=157.853 / mid=157.8215 / spread=6.3p`
    - OANDA 応答品質: `pricing=202.6ms(200)`, `summary=213.1ms(200)`, `openTrades=198.9ms`
  - `logs/brain_canary_readiness_latest.json`
    - `tick_age_sec=50560.1`
    - `atr_proxy_pips` / `recent_range_pips_6m` は `None`
- `logs/trade_counterfactual_latest.json`（generated_at=`2026-03-07T12:54:45.522578+00:00`）には
  `strategy_like=scalp_ping_5s_b_live%`、`policy_hints.reentry_overrides.mode=tighten`、
  `same_dir_reentry_pips_mult=1.35`、`side_actions.short=block` が出力済み。
- `logs/forecast_improvement_latest.json` は inspection 時点で
  generated_at=`2026-03-04T13:50:54.857403+00:00`, `verdict=mixed`, `returncode=0` だが、
  `runtime_overrides` は未出力で、今回の実装反映後 audit はまだ未実行。
- コード接続確認では:
  - `analysis/strategy_feedback.py` が `trade_counterfactual_latest.json` を追加で読み、
    `policy_hints.reentry_overrides` と `side_actions` を
    `entry_units_multiplier` / `entry_probability_multiplier` /
    `entry_probability_delta` へ bounded に合成する。
  - `execution/strategy_entry.py` は `current_advice()` に `side` を渡し、
    long/short 別の soft feedback を live entry 側で受け取る。
  - `analysis/forecast_improvement_worker.py` は
    `logs/forecast_improvement_latest.json.runtime_overrides` へ
    `enabled/reason/max_age_sec/env_overrides/...` を保存できる。
  - `workers/common/forecast_gate.py` は runtime override 適用前に base env へ戻し、
    `enabled=true` かつ fresh な payload だけを採用し、
    `missing/stale/degraded` では override を使わない。
  - `utils/strategy_tags.py` の `resolve_strategy_tag` /
    `normalize_strategy_lookup_key` / `strategy_like_matches` を
    `strategy_feedback_worker`, `dynamic_alloc_worker`,
    `replay_quality_gate_worker`, `reentry_gate`, `strategy_feedback` が共有し、
    alias や一時 suffix 付き tag を同じ canonical key へ寄せる。

Failure Cause:
- `trade_counterfactual` の知見は主に
  `replay_quality_gate -> worker_reentry.yaml` にしか流れず、
  live entry を触る `strategy_feedback` 側には直接入っていなかった。
- `forecast_improvement` は before/after 監査を残していたが、
  `forecast_gate` が即時参照できる runtime handoff が欠けていた。
- strategy tag の正規化が複数箇所で重複しており、
  alias/hash suffix/LIKE 判定のずれで feedback・replay・reentry の集計単位が割れやすかった。

Improvement:
- `strategy_feedback` は counterfactual を live へ hard block ではなく
  soft overlay として反映する。
  `reentry_overrides` と `side_actions` は bounded な
  units/probability 調整へ変換し、`strategy_params.counterfactual_feedback`
  と `_meta.counterfactual` に由来を残す。
- `forecast_improvement_worker` は、現在の評価で使った env を
  `runtime_overrides.env_overrides` として出力し、
  `forecast_gate` は `FORECAST_GATE_RUNTIME_OVERRIDE_PATH` 経由で
  それを runtime 読み込みする。
  degrade / stale / empty payload 時は base env に自動復帰する。
- strategy tag canonicalization は `utils/strategy_tags.py` に統合され、
  feedback / replay / reentry / dynamic_alloc が同じ tag 解決規約を共有する。

Verification:
- `git diff -- analysis/strategy_feedback.py analysis/forecast_improvement_worker.py workers/common/forecast_gate.py analysis/replay_quality_gate_worker.py execution/reentry_gate.py analysis/strategy_feedback_worker.py scripts/dynamic_alloc_worker.py execution/strategy_entry.py utils/strategy_tags.py`
- `rg -n "current_advice\\(|runtime_overrides|resolve_strategy_tag|strategy_like_matches" analysis/strategy_feedback.py execution/strategy_entry.py analysis/forecast_improvement_worker.py workers/common/forecast_gate.py analysis/replay_quality_gate_worker.py execution/reentry_gate.py analysis/strategy_feedback_worker.py scripts/dynamic_alloc_worker.py utils/strategy_tags.py`
- `python3` で `logs/trade_counterfactual_latest.json` と `logs/forecast_improvement_latest.json` のキーを spot-check
- 週末 stale のため、このタスクでは live 発注導線の再検証は未実施

Status:
- done

## 2026-03-07 12:22 UTC / 2026-03-07 21:22 JST - local-v2: `quant-strategy-feedback` は live 接続済み、残課題は loop 回帰保証だった

Period:
- 確認/改善: UTC `12:19-12:22` / JST `21:19-21:22`
- 対象（実測）: `logs/local_v2_stack/quant-strategy-feedback.log`, `logs/strategy_feedback.json`, `scripts/local_v2_stack.sh status`, `ps`
- 対象（テスト）: `tests/analysis/test_strategy_feedback_worker.py`

Fact:
- `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-strategy-feedback`
  で `quant-strategy-feedback` は `running`、PID は `6427`。
- `logs/strategy_feedback.json` と `logs/local_v2_stack/quant-strategy-feedback.log` の mtime は
  UTC `12:19` / JST `21:19` で一致し、現行プロセスは生存継続している。
- 一方で test coverage は `payload build` のみで、`main()` の loop 経路
  (`loop start -> _run_once -> time.sleep`) を通す検証がなかった。
- コード接続確認では:
  - `analysis/strategy_feedback.py` / `execution/strategy_entry.py` が
    `strategy_feedback.json` を live entry へ反映
  - `config/dynamic_alloc.json` は worker と `strategy_entry` で使用
  - `config/pattern_book*.json` は `order_manager` preflight で使用
  - `trade_counterfactual` は主に `replay_quality_gate_worker -> worker_reentry.yaml`
    の改善提案系へ接続

Failure Cause:
- `strategy_feedback` 自体の live 接続は成立していたが、
  常駐 loop 経路を固定するテストがなく、`sleep` まわりの回帰を事前に検知できなかった。
- ログには旧クラッシュ履歴が残るため、「今も死んでいる」のか「過去ログが残っているだけか」の判別が遅れやすかった。

Improvement:
- `tests/analysis/test_strategy_feedback_worker.py`
  に `test_main_loop_runs_once_and_sleeps` を追加し、
  常駐モードで 1 iteration 実行後に `time.sleep()` へ到達することを固定化。
- `docs/ARCHITECTURE.md`
  で local watchdog 導線の責務を修正し、
  `quant-strategy-feedback` が常駐更新を担当、
  `run_local_feedback_cycle.py` は既定で `strategy_feedback` を回さない構成を明記。

Verification:
- `pytest -q tests/analysis/test_strategy_feedback_worker.py`
- `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-strategy-feedback`
- `ps -Ao pid,ppid,lstart,command | rg "analysis.strategy_feedback_worker"`

Status:
- done

## 2026-03-07 11:49 UTC / 2026-03-07 20:49 JST - local Brain/Ollama を compact-context + shadow canary として local-v2 へ反映

Period:
- 実装/検証: UTC `11:45-11:49` / JST `20:45-20:49`
- 対象: `workers/common/brain.py`, `execution/order_manager.py`, `tests/workers/test_brain_history_prompt_autotune.py`
- 反映対象: `quant-order-manager`, `quant-strategy-control`

Fact:
- 直前 readiness:
  - `logs/brain_canary_readiness_latest.json`
    - `profile_safe=true`
    - `quality_gate_ok=true`
    - `ollama_ready=true`
    - `market_ready=false`
- offline benchmark:
  - `logs/brain_local_llm_benchmark_latest.json`
    - `qwen2.5:7b parse_pass_rate=1.0`
    - `latency_p95_ms=3332.901`
- 市況（週末 stale）:
  - USD/JPY `bid=157.790 / ask=157.853 / spread=6.3p`
  - `tick_age_sec=49694.4`

Failure Cause:
- Brain は `entry_thesis/meta` の大きい JSON をそのまま prompt に入れており、ローカルLLMの parse/latency 安定性を落としていた。
- 週明け canary も `shadow` 観測なしでいきなり `block/scale` へ進む形だと、月曜オープン直後の切り戻しが重い。

Improvement:
- `workers/common/brain.py`
  - prompt/context を compact scalar 中心へ変更
  - `context_json` / `response_json` を valid JSON で保存
  - `factors.M1`, `forecast_fusion`, `dynamic_alloc` の要点だけを保持
- `execution/order_manager.py`
  - `ORDER_MANAGER_BRAIN_GATE_MODE=shadow` 時は Brain 判定を `brain_shadow` と metric に記録し、
    実際の `block/scale` は適用しない
- 反映:
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env,ops/env/profiles/brain-ollama-safe.env --services quant-order-manager,quant-strategy-control`
  - 週末クローズ帯のため、live fill 検証ではなく「safe shadow profile を本線へ読ませる」反映に限定

Verification:
- `python3 -m py_compile workers/common/brain.py execution/order_manager.py scripts/prepare_local_brain_canary.py`
- `pytest -q tests/workers/test_brain_history_prompt_autotune.py tests/workers/test_brain_ollama_backend.py tests/scripts/test_apply_brain_model_selection.py tests/scripts/test_prepare_local_brain_canary.py`
- `python3 scripts/prepare_local_brain_canary.py --warmup`
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager`

Status:
- done

## 2026-03-07 02:03 UTC / 2026-03-07 11:03 JST - local-v2: local Brain/Ollama の月曜 canary 準備を固定化

Period:
- 確認: UTC `02:00-02:03` / JST `11:00-11:03`
- 対象（実測）: `logs/trades.db`, `logs/orders.db`, `logs/brain_state.db`, `logs/brain_local_llm_benchmark_latest.json`, `logs/brain_model_selection_latest.json`, `logs/health_snapshot.json`
- 対象（OANDA API）: `pricing`, `candles(M1, count=180)`
- 対象（ローカルLLM）: `ollama list`, `http://127.0.0.1:11434/api/tags`

Fact:
- 市況（OANDA live, UTC `02:03` / JST `11:03`）:
  - USD/JPY `bid=157.790 / ask=157.853 / spread=6.3p`
  - `tick_age_sec=48825.1`（週末 stale）
  - `pricing=0.218s`, `candles=0.191s`
- ローカルLLM:
  - Ollama 稼働、installed models は `qwen2.5:7b`, `gpt-oss:20b`, `llama3.1:8b`, `gemma3:4b`
  - live local-v2 は `BRAIN_ENABLED=0` / `ORDER_MANAGER_BRAIN_GATE_ENABLED=0` のまま
  - benchmark / selection artifact は存在し、latest selection は `preflight=qwen2.5:7b`, `autotune=gpt-oss:20b`
- 直近7dの strategy 実測:
  - winner: `MomentumBurst +1856.9 JPY`, `MicroLevelReactor +74.2 JPY`
  - loser: `M1Scalper-M1 -6172.5 JPY`, `scalp_ping_5s_flow_live -7131.1 JPY`

Failure Cause:
- safe canary profile の実体が欠けており、週明けに Brain を戻す導線が「aggressive profile をそのまま使う」か「手作業で env を組む」かの二択になっていた。
- readiness 判定も未整備で、benchmark/selection の鮮度、safe profile の形、Ollama 到達性、market 条件を1回で判定できなかった。

Improvement:
- `ops/env/profiles/brain-ollama-safe.env`
  - `micro-only`
  - `MomentumBurst, MicroLevelReactor, MicroRangeBreak, MicroTrendRetest` に限定
  - `brain_gate_mode=shadow`
  - `ORDER_MANAGER_SERVICE_WORKERS=1`
  - `fail-open`
  - `sample_rate=0.35`
  - `ttl=15s`
  - `timeout=4s`
  - auto-tune off
- `scripts/prepare_local_brain_canary.py`
  - benchmark → safe profile selection sync（`--timeout-cap-sec 4`）
  - safe profile shape の検査
  - selected preflight model の `parse_pass_rate >= 0.90` / `latency_p95_ms <= 4000` を必須化
  - Ollama server / model presence
  - OANDA market sanity（spread / tick age）
  - `logs/brain_canary_readiness_latest.json` を出力
  - 実行結果（2026-03-07 11:42 JST）: `profile_safe=true`, `selection_sync_ok=true`, `ollama_ready=true`, blocker は `market_ready` のみ
- `scripts/apply_brain_model_selection.py`
  - benchmark の全variantが `min_parse_pass_rate` 未達でも、poor-quality model を preflight 採用しないよう修正
  - safe canary は fallback `qwen2.5:7b` を維持

Verification:
- 週末市況のため live restart は未実施
- 月曜の enable 条件:
  - `python3 scripts/prepare_local_brain_canary.py`
  - `ready.enable_recommended=true`
  - spread / tick age が通常化していること
  - その上で `quant-order-manager,quant-strategy-control` だけ safe profile で restart

Status:
- done

## 2026-03-06 15:56 UTC / 2026-03-07 00:56 JST - local-v2: Git再開判断（OANDA account endpoint 復旧確認）

Period:
- 確認: UTC `15:54-15:56` / JST `00:54-00:56`
- 対象（実測）: `logs/health_snapshot.json`, `logs/local_v2_stack/quant-market-data-feed.log`, `logs/local_v2_stack/quant-order-manager.log`, `logs/local_v2_stack/quant-position-manager.log`
- 対象（OANDA API）: `pricing`, `summary`, `openTrades`, `candles(M5, count=30)`

Fact:
- 市況（OANDA live, UTC `15:56` / JST `00:56`）:
  - USD/JPY `mid=157.582 / bid=157.578 / ask=157.586 / spread=0.8p`
  - `ATR14(M5)=10.364p` / `range_last_12xM5=22.7p`
- ローカル導線:
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で主要サービスと `trade_min` 戦略群は running
  - `scripts/collect_local_health.sh` は `snapshot_age_sec=0` で更新継続
- OANDA API 応答品質（UTC `15:56` / JST `00:56`, 各2回直接確認）:
  - `pricing`: `200/200`, latency `210ms/221ms`
  - `summary`: `200/200`, latency `199ms/267ms`
  - `openTrades`: `200/200`, latency `210ms/314ms`
  - `candles`: `200/200`, latency `211ms/222ms`

Failure Cause:
- 直前の保留判断では `summary/openTrades` の live `503` が継続していたが、今回の再確認で account 系 endpoint の劣化は解消した。

Improvement:
- Git 処理の保留を解除し、差分検証後に commit/push を再開する。
- 運用ログ `logs/ops_v2_audit_20260307_0056_git_resume.json` に復旧確認を退避。

Verification:
- `summary/openTrades` が連続 `200`
- `health_snapshot` 更新継続
- `local_v2_stack` の主要サービスが running

Status:
- done

## 2026-03-06 15:50 UTC / 2026-03-07 00:50 JST - local-v2: flow負エッジ reject + severe loser縮退強化 + micro snapshot fallback

Period:
- 集計: `logs/pdca_profitability_latest.json` generated_at=`2026-03-07 00:50 JST`
- 市況確認: UTC `15:46-15:50` / JST `00:46-00:50`
- 対象（実測）: `logs/trades.db`, `logs/orders.db`, `logs/metrics.db`, `logs/health_snapshot.json`, `logs/tick_cache.json`, `logs/factor_cache.json`, OANDA pricing/summary/openTrades

Fact:
- 市況（OANDA live + local cache）:
  - USD/JPY `mid=157.632-157.680`, `spread=0.8p`
  - `ATR14(M1)=4.19p`, 直近5分レンジ `10.5p`
  - pricing は `200`, 一方 `summary/openTrades` は `503` が断続
- 直近24h（bot only）:
  - 全体 `net=-8571.0 JPY`, `PF=0.65`, `win_rate=45.8%`
  - loser 上位: `scalp_ping_5s_flow_live=-5901.8 JPY`, `M1Scalper-M1=-4479.4 JPY`
  - winner 上位: `MomentumBurst=+1526.1 JPY`, `MicroLevelReactor=+259.5 JPY`
- 実行品質:
  - `spread_mean=0.805p`, `cost_vs_mid_mean=0.407p`, `latency_submit_p50=190ms`
  - 執行コストよりも strategy expectancy 側が主因
- 安定性:
  - `MicroLevelReactor` worker は `get_account_snapshot()` の `503` で再起動を繰り返していた
  - `orders.db` では `margin_snapshot_failed=16`, `api_error=4`, `quote_unavailable=14`

Failure Cause:
- `scalp_ping_5s_flow_live` は既存の `signal_window_adaptive_live_score_pips` / `lookahead_edge_pips` が負でも通る経路があり、低エッジ約定を量産していた。
- `M1Scalper-M1` は service env の既定値が local-v2 recovery override より緩く、再起動経路で loser setup / size が戻りうる状態だった。
- profitable な `MicroLevelReactor` が OANDA `summary 503` を worker 側で吸収できず、winner 側の機会を落としていた。

Improvement:
- `workers/scalp_ping_5s/config.py` / `workers/scalp_ping_5s/worker.py`
  - `SIGNAL_WINDOW_ADAPTIVE_LIVE_SCORE_MIN_PIPS`
  - `LOOKAHEAD_EDGE_HARD_REJECT_PIPS`
  を追加し、負エッジを strategy local で hard reject。
- `ops/env/scalp_ping_5s_flow.env`
  - flow clone で `LOOKAHEAD_GATE_ENABLED=1`
  - `SIGNAL_WINDOW_ADAPTIVE_LIVE_SCORE_MIN_PIPS=0.0`
  - `LOOKAHEAD_EDGE_HARD_REJECT_PIPS=0.0`
  を追加。
- `scripts/dynamic_alloc_worker.py`
  - `sum_realized_jpy`, `market_close_loss_share`, `realized_jpy_per_1k_units`, `jpy_downside_share`
    が同時に悪い severe loser を `lot_multiplier=0.12-0.18` まで圧縮する段を追加。
- `ops/env/quant-m1scalper.env` / `ops/env/local-v2-stack.env`
  - `M1SCALP_ALLOW_REVERSION=0`
  - `M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest-long,nwave-long,vshape-rebound-long`
  - `M1SCALP_BASE_UNITS=1200`
  - `M1SCALP_MARGIN_USAGE_HARD=0.88`
  - `M1SCALP_DYN_ALLOC_MULT_MIN=0.12`
  へ同期。
- `workers/micro_runtime/worker.py`
  - account snapshot を stale fallback 付きで解決し、`503` 時は cached snapshot で loop 継続。
- `scripts/local_v2_stack.sh`
  - `scalp_ping_5s_flow` の thin wrapper worker が `up/restart` の親シェル終了に巻き込まれて stale pid 化する事象を確認。
  - detached session launcher へ切り替え、local-v2 runtime と `status` の整合を戻した。

Verification:
1. `logs/local_v2_stack/quant-micro-levelreactor.log` で `account_snapshot_unavailable` による loop skip は許容しても、process crash が再発しないこと。
2. `config/dynamic_alloc.json` で `M1Scalper-M1` の `lot_multiplier` が `0.12` 近辺まで縮退すること。
3. flow を再開する場合、`orders.db` で `adaptive_live_score_block` / `lookahead_edge_hard_block` が観測され、負エッジ約定が減ること。
4. 反映後 1-3h で `MomentumBurst` / `MicroLevelReactor` の filled が維持されつつ、`M1Scalper-M1` の `net_jpy` 悪化速度が鈍ること。

Status:
- in_progress

## 2026-03-06 15:48 UTC / 2026-03-07 00:48 JST - local-v2: Git保留判断（OANDA summary/openTrades 503 継続）

Period:
- 確認: UTC `15:45-15:48` / JST `00:45-00:48`
- 対象（実測）: `logs/orders.db`, `logs/health_snapshot.json`, `logs/local_v2_stack/quant-market-data-feed.log`, `logs/local_v2_stack/quant-order-manager.log`, `logs/local_v2_stack/quant-position-manager.log`
- 対象（OANDA API）: `pricing`, `summary`, `openTrades`, `candles(M5, count=30)`

Fact:
- 市況（OANDA live, UTC `15:48` / JST `00:48`）:
  - USD/JPY `mid=157.705 / bid=157.701 / ask=157.709 / spread=0.8p`
  - `ATR14(M5)=10.8p` / `range_last_12xM5=28.8p`
- ローカル導線:
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で主要サービスは running
  - `logs/health_snapshot.json` は `generated_at=2026-03-06T15:46:38Z`, `data_lag_ms=745.9`, `decision_latency_ms=16.5`
  - `logs/orders.db` の直近1hは `filled=1370`, `margin_snapshot_failed=13`, `api_error=4`
- OANDA API 応答品質（UTC `15:48` / JST `00:48`, 各2回直接確認）:
  - `pricing`: `200/200`, latency `278ms/218ms`
  - `candles`: `200/200`, latency `229ms/311ms`
  - `summary`: `503/503`, latency `208ms/301ms`
  - `openTrades`: `503/503`, latency `221ms/295ms`
- ログ実測:
  - `quant-market-data-feed` は `pricing/stream` `200 OK` を継続
  - `quant-position-manager` / `quant-order-manager` は `summary` と `openTrades` の `503` を継続記録

Failure Cause:
- 価格配信と candle 取得は正常だが、OANDA の account 系 endpoint (`summary`, `openTrades`) が live で継続 `503`。
- AGENTS の着手前チェック要件では API 応答品質悪化時は作業保留とするため、Git 操作を先行させる判断を不採用とした。

Improvement:
- 本タスクでは実装変更・commit/push を行わず保留。
- account 系 endpoint が `200` に戻るまで、Git 操作は見送り。
- 運用ログ `logs/ops_v2_audit_20260307_0048_git_hold.json` に同内容を退避。

Verification:
- 再開条件:
  - `summary` と `openTrades` が連続 `200` を返すこと
  - `quant-position-manager.log` / `quant-order-manager.log` で `503` が収束していること
  - `scripts/collect_local_health.sh` の snapshot 更新が継続していること

Status:
- open

## 2026-03-06 14:22 UTC / 2026-03-06 23:22 JST - local-v2: M1Scalper setup絞り込み + Flow低品質entry圧縮 + OANDA 503耐性

Period:
- 集計: `logs/pdca_profitability_report_latest.json` generated_at=`2026-03-06T23:21:50 JST`
- 市況確認: UTC `14:20-14:22` / JST `23:20-23:22`
- 対象（実測）: `logs/trades.db`, `logs/orders.db`, `logs/metrics.db`, `logs/local_v2_stack/*.log`, OANDA pricing/openTrades

Fact:
- 市況（OANDA + local candle, UTC `14:21-14:22` / JST `23:21-23:22`）:
  - USD/JPY `mid=157.976 / spread=0.8p`
  - `ATR14(M1)=4.56p` / `ATR60(M1)=6.79p` / `range_120m=70.6p`
  - pricing は継続 `200 OK`、一方で `openTrades/summary` は `503` が断続
- 直近24h（bot only, `pdca_profitability_report_latest.md`）:
  - `trades=2482 / win_rate=49.9% / PF(pips)=0.69 / net_jpy=-9909.0`
- M1Scalper-M1 の source tag 別（同24h, `trades.db entry_thesis.source_signal_tag`）:
  - `trend-long: 473 trades / -1759.4 JPY / -354.2 pips`
  - `sell-rally: 861 trades / -1557.8 JPY / -442.3 pips`
  - `buy-dip: 328 trades / -1557.7 JPY / -417.6 pips`
  - 一方で `nwave-long: 30 trades / +98.4 JPY / +43.3 pips`
  - `breakout-retest-long: 2 trades / +81.2 JPY / +6.5 pips`
- Flow (`scalp_ping_5s_flow_live`) の直近負けトレード `entry_thesis` では
  - `signal_window_adaptive_live_score_pips=-0.58 〜 -0.89`
  - それでも `filled=366` / `net_jpy=-6998.5` が出ており、低品質entryが通過していた
- ローカル稼働:
  - `health_snapshot` は `data_lag_ms≈84 / decision_latency_ms≈21`
  - `quant-m1scalper` は OANDA `/summary` `503` で worker crash を起こし、stale pid 相当の再起動が発生

Failure Cause:
- `M1Scalper-M1` は当日ソース別で `trend-long` と `sell-rally` が損失寄与の大半を占め、setup の絞り込み不足が継続。
- `scalp_ping_5s_flow_live` は leading profile 無効のまま low-edge signal を大量通過させていた。
- OANDA `/summary` の瞬断時に M1 worker が例外落ちし、稼働継続性を損ねていた。

Improvement:
- `ops/env/local-v2-stack.env`
  - Flow:
    - `SCALP_PING_5S_FLOW_ENTRY_LEADING_PROFILE_ENABLED=1`
    - `...REJECT_BELOW=0.58` / `...SHORT=0.66`
    - `BASE_ENTRY_UNITS=80`
    - `MAX_ACTIVE_TRADES=1`
  - M1:
    - `M1SCALP_ALLOW_REVERSION=0`
    - `M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest-long,nwave-long`
    - `M1SCALP_BASE_UNITS=1200`
    - `M1SCALP_MARGIN_USAGE_HARD=0.88`
- `workers/scalp_m1scalper/worker.py`
  - account snapshot をキャッシュ付きで扱い、`/summary` `503` では loop skip / cached snapshot fallback に変更

Verification:
- 反映後:
  - `quant-m1scalper` が `/summary 503` で即死せず、`account snapshot ... cached snapshot` ログで継続すること
  - `orders.db` で `client_order_id like '%scalp-m1scalperm1%'` の `source_signal_tag` が `nwave-long / breakout-retest-long` 中心になること
  - `scalp_ping_5s_flow_live` の filled 件数は減っても、`avg_pips / PF` が改善方向へ動くこと
  - 次の24hで `PF(pips)>0.85` を暫定回復目標、14dで `PF>1.0` を再判定

Status:
- in_progress

## 2026-03-06 05:56 UTC / 2026-03-06 14:56 JST - M1Scalper-M1: 損失縮小のため exit チューニング（profit_buffer拒否多発の抑制）

Period:
- 集計: 直近24h（`logs/pdca_profitability_latest.json` generated_at=`2026-03-06T14:51 JST`）
- 対象（実測）: `logs/trades.db`, `logs/orders.db`

Fact:
- 直近24h（trades, strategy=`m1scalper-m1`）:
  - `PF(pips)=0.5648`（≈0.55）/ `win_rate=0.6205` / `trades=1913`
  - `gross_loss_pips=3413.2 / losses=696` → `avg_loss≈4.90p`
  - `gross_win_pips=1927.7 / wins=1187` → `avg_win≈1.62p`
  - **勝率は高いが avg_loss が avg_win を大幅に上回り、期待値が負け**。
- 直近24h（orders, client_order_id like `%m1scalperm1%`）:
  - `close_reject_profit_buffer=509`（多発）
- Ops note:
  - entry は一時停止（2026-03-06 14:42 JST頃）として扱う（再開は別判断）。
  - ただし `orders.db` では `m1scalperm1` の `filled` が `2026-03-06 14:53 JST` にも観測されており、停止が適用されていない可能性がある（要確認）。

Failure Cause:
- `lock_floor` / 早期の利確系が「極小利益帯」で頻発し、`close_reject_profit_buffer` が多発して EXIT が安定しない。
- 負け側は avg_loss が大きく、損失上限（max_adverse）と負け側の cut 条件が弱い。

Improvement:
- `ops/env/quant-m1scalper-exit.env`（exitのみ）:
  - `M1SCALP_EXIT_LOCK_TRIGGER_MIN_PIPS=1.8`（from `1.00`）
  - `M1SCALP_EXIT_COMPOSITE_MIN_SCORE=2.0`
  - `M1SCALP_EXIT_MAX_ADVERSE_PIPS=4.0`
  - `M1SCALP_EXIT_PROFIT_TAKE_PIPS=3.0`
  - `M1SCALP_EXIT_LOCK_BUFFER_PIPS=0.3`
- entry は停止維持（再開は次サイクル判断）。

Verification:
- 反映後24hで以下を確認:
  - `m1scalper-m1` の `avg_loss` が低下し、`PF(pips)` が改善していること（目標: `PF>0.85` へ回復）。
  - `close_reject_profit_buffer` が減少していること（目標: `<= 1/3`）。
  - `quant-m1scalper-exit` のログで例外/連続close失敗が増えていないこと。

Status:
- in_progress

## 2026-03-06 13:25 JST / local-v2: 資産減少/稼げてない RCA（scalp_ping_5s_flow_live + M1Scalper-M1 寄与大、margin closeout 高止まり）

Period:
- 集計: 2026-03-05 13:19〜2026-03-06 13:19 JST（UTC 2026-03-05 04:19〜2026-03-06 04:19）
- 比較: 直近7d（UTC 2026-02-27 04:19〜2026-03-06 04:19）
- 対象（実測）: `logs/metrics.db`, `logs/orders.db`, `logs/trades.db`
- 対象（OANDA API）: account summary / transactions（TRANSFER_FUNDS, DAILY_FINANCING）/ openTrades / pricing

Fact:
- 市況（OANDA pricing, UTC 04:10 / JST 13:10）:
  - USD/JPY `bid=157.510 / ask=157.518 / spread=0.8p`
  - pricing latency `~200ms`
  - replay(M5) `ATR14_pips~3.14` / `range_60m_pips~6.3`
- 口座状態（OANDA summary, UTC 04:14 / JST 13:14）:
  - `balance=38652.5 / NAV=38321.7 / unrealized=-330.8`
  - `marginCloseoutPercent=0.9428` / `marginAvailable=2191.9`（**closeout近傍**）
  - openTrades: `n=13`、`USD_JPY net_units=+4175 / gross_units=7303`（unrealized寄与: micro=-262 / scalp=-60 / scalp_fast=-8）
- 直近24h（account metrics, practice=false）:
  - `ΔNAV=-12845.9 / Δbalance=-10454.5`（含み変動が約 `-2391`）
- 直近24h（bot only, trades.db realized_pl）:
  - `n=2459 / win_rate=0.536 / PF=0.508 / net=-10773 / net_pips=-1743.6`
  - win率>50%でも **avg_loss が avg_win を大幅に上回る**（期待値負け）
- 直近7d（trades.db realized_pl）:
  - bot: `net=-11096 / PF=0.506`、manual: `net=-7851 / PF=0.057`（manual寄与が大きい）
  - OANDA DAILY_FINANCING: `-66.9(24h) / -653.2(7d)`（主因ではないが確実に減る）
- 実行コスト（orders.db 直近2000 filled, analyze_entry_precision）:
  - `spread_pips mean=0.805 (p50=0.8)` / `cost_vs_mid_pips mean=0.401` / `slip_p95=0.3p`
  - `latency_submit p50~193ms` / `latency_preflight p50~228ms`（**執行自体は通常**）
  - strategy別（要点）:
    - `scalp_ping_5s_flow_live`: `tp_mean~1.44p / sl_mean~1.32p`（**spread(0.8p)に対して取り幅が小さすぎる**）
    - `M1Scalper-M1`: `tp_mean~7.35p / sl_mean~6.07p` だが、実現では `avg_win~+5.32 / avg_loss~-17.71`（損失が勝ちの約3.3倍）
- 拒否/ブロック（orders.db, client_order_id最終ステータス, 直近24h）:
  - OANDA `rejected=3/2934`（低い）
  - 一方で `margin_usage_projected_cap=119` / `strategy_control_entry_disabled=146` / `margin_usage_exceeds_cap=26` が観測（主にscalp_fast）

Failure Cause:
- 主因（戦略期待値）:
  - `scalp_ping_5s_flow_live` が **PF=0.30台**で大幅な赤字寄与（取り幅が spread/コストに対して小さく、期待値が構造的に負け）。
  - `M1Scalper-M1` は勝率は高いが、**avg_loss が avg_win の約3倍**で PF<1 に沈む（負けトレードの損失拡大）。
- 副因（リスク/稼働制約）:
  - margin closeout が高止まり（`marginCloseoutPercent~0.94`）し、risk cap 系ブロックが出て「悪化局面での建玉・調整」が難しくなる。

Improvement:
- P0（当日・安全）: margin closeout 回避を最優先。新規entryを抑え、exitは継続。
  - `SCALP_PING_5S_FLOW_*` の entry を一時縮小/停止（`STRATEGY_CONTROL_ENTRY_SCALP_PING_5S_FLOW=0` または units/active_trades を強制縮小）し、口座余力を回復させる。
  - `quant-m1scalper` は `M1SCALP_BASE_UNITS` を縮小し、`M1SCALP_MARGIN_USAGE_HARD` を下げて過剰レバを抑える。
- P1（当日〜1日）: scalp_ping_5s_flow_live の「期待値負け」を構造的に解消。
  - forecast gate: `FORECAST_GATE_EXPECTED_PIPS_MIN_*` と `TARGET_REACH_MIN_*` を **cost_vs_mid(≈0.4p)+spread余裕**以上へ引き上げ（low edgeを排除）。
  - local gating: `SCALP_PING_5S_FLOW_ENTRY_LEADING_PROFILE_REJECT_BELOW` を引き上げ、低品質エントリを減らす。
  - perf/profit guard（strategy限定）を有効化し、PF悪化時は scale-to-low へ自動退避。
- P1（1〜3日）: M1Scalper-M1 の「損失拡大」を抑制（exit_worker/損切り設計を重点監査）。
  - 負け側の cut を早める（損失上限・早期撤退条件の追加/強化）。
  - replay で「どの局面が大負け源泉か」を再現し、entryフィルタ or exit改善へ落とす。

Verification:
- 24hで `PF>1.0` / `expectancy>0` を最低目標（まず n>=300 で暫定判定 → 14d で確定）。
- `scalp_ping_5s_flow_live` の `tp_mean/spread` 比が改善し、net_pips がマイナス継続しないこと。
- `marginCloseoutPercent` と `account.margin_usage_ratio` が 0.90 未満へ戻ること（急変時は即時縮小）。

Status:
- open

## 2026-03-05 17:30 JST / local-v2: Pattern Gateが実質無効化されていた問題を修正（preserve_strategy_intent下でも評価）+ 運用キー整理 + trades.entry_thesis backfill 追加

Period:
- 調査: 2026-03-05 16:55〜17:30 JST（UTC 07:55〜08:30）
- 市況確認: 2026-03-05 17:00 JST（UTC 08:00）
- 対象（実測）: `logs/trades.db`, `logs/orders.db`, `logs/patterns.db`, `logs/tick_cache.json`, `logs/oanda/candles_*_latest.json`
- 対象（実装）: `execution/order_manager.py`, `ops/env/local-v2-stack.env`, `scripts/backfill_entry_thesis_from_orders.py`

Fact:
- 市況（local tick/candles, UTC 08:00 / JST 17:00）:
  - USD/JPY `bid=157.343 / ask=157.351 / spread=0.8p`（tick直近200: min/median/p90/max=0.8p）
  - `ATR14_pips(M1)=2.04` / `range_60m_pips(M1)=11.0`
  - `ATR14_pips(H1)=12.69` / `ATR14_pips(H4)=31.82`
- 直近7d（観測メモ, manual含む）:
  - `net=-7291` / `PF=0.21`
  - 最大損失: manual pocket `2026-03-02` の `MARKET_ORDER_MARGIN_CLOSEOUT (-7696)`
  - OANDA openTrades: `-6998 units` の巨大shortが残存し、`margin_usage_ratio~0.88 / health_buffer~0.12`
- scalp_fast（観測メモ, 直近24h）:
  - `scalp_ping_5s_b_live` の long 側 `504 trades net=-183`、short 側 `201 trades net=+1.8`
  - close_reason は `SL 539件 (avg -1.42pips)` が支配
  - ただし `ops/env/local-v2-stack.env` で `SCALP_PING_5S_B_SIDE_FILTER=sell` に切替後は long close が 0（設定ドリフト起因の偏損が濃厚）
- Pattern Gate が “死んでいた” 根拠（コード+DB）:
  - `execution/order_manager.py` の Pattern Gate は `not preserve_strategy_intent` 条件で常にスキップされ得る。
  - `ORDER_MANAGER_PRESERVE_STRATEGY_INTENT=1`（既定）かつ `ORDER_MANAGER_PATTERN_GATE_ENABLED` 未設定（既定false）だと、orders.db に `pattern_block/pattern_scale_*` が出ない。
  - 一方 `logs/patterns.db` には `st:scalp_ping_5s_b_live...` 等のスコアが大量に存在し、avoid/weak の識別が可能。

Failure Cause:
- Pattern Gate の運用キーが二重（`ORDER_PATTERN_GATE_ENABLED` と `ORDER_MANAGER_PATTERN_GATE_ENABLED`）かつ、
  order_manager側の実行条件が `preserve_strategy_intent` に依存していたため、**opt-in戦略でも gate が実行されない**状態になっていた。

Improvement:
- Pattern Gate を preserve_strategy_intent 下でも評価する:
  - `execution/order_manager.py`: `Pattern gate` 条件から `and not preserve_strategy_intent` を除去（pattern_gate自体はopt-inなので仕様整合）。
- ローカルV2導線で Pattern Gate を有効化:
  - `ops/env/local-v2-stack.env`: `ORDER_MANAGER_PATTERN_GATE_ENABLED=1`
- （任意/母集団浄化）過去 trades の entry_thesis 契約欠損を backfill:
  - `scripts/backfill_entry_thesis_from_orders.py` を追加し、`orders.db submit_attempt.request_json` から `entry_thesis` を復元して `trades.db` を更新（バックアップ/ロック付）。

Verification:
- Pattern Gate の作動確認:
  - `orders.db` に `status='pattern_block'` / `status='pattern_scale_below_min'` が出ること（該当戦略の opt-in 前提）。
  - `request_json.entry_thesis.pattern_gate` が payload を持つこと（allowed/scale/reason/pattern_id）。
- backfill（dry-run→適用）:
  - `python scripts/backfill_entry_thesis_from_orders.py --dry-run`
  - `python scripts/backfill_entry_thesis_from_orders.py --until-utc 2026-02-27T23:59:59+00:00`
- 収益の再評価:
  - scalp_fast の `SL率/avg_pips` が改善方向か、`PF/expectancy` が悪化しないこと（n>=300 の短期判定→14dで再評価）。

Status:
- in_progress

## 2026-03-05 15:50 JST / local-v2: PF悪化RCA（scalp_ping_5s_b寄与大）+ Brain autopdca既定OFF(=opt-in)化

Period:
- 集計: 2026-03-04 15:46:50〜2026-03-05 15:46:50 JST（UTC 2026-03-04 06:46:50〜2026-03-05 06:46:50）
- 市況確認: 2026-03-05 15:40 JST（UTC 06:40）
- 対象: `logs/trades.db`, `logs/factor_cache.json`, `logs/health_snapshot.json`, `scripts/local_v2_autorecover_once.sh`, `scripts/local_v2_stack.sh`

Fact:
- 市況（OANDA実測 + local factor, UTC 06:40 / JST 15:40）:
  - USD/JPY `bid=157.106 / ask=157.114 / spread=0.8p`
  - pricing latency `avg=255ms`（samples `[247,266,251]`）
  - `ATR14_pips(M1)=2.64`, `ATR14_pips(M5)=5.87`, `ATR14_pips(H1)=18.02`
- 直近24h（manual除外, `trades.db`, realized_pl）:
  - `n=706`, `win_rate=0.218`, `PF=0.423`, `expectancy=-0.73 JPY/trade`
  - `net=-518.5 JPY`, `net_pips=-567.1`
- 負け寄与上位（`pocket|strategy_tag`, count>=5, net昇順）:
  - `micro|MicroPullbackEMA n=25 net=-133.9 PF=0.111`
  - `scalp_fast|scalp_ping_5s_b_live n=606 net=-133.2 PF=0.436`
- 直近14d（count>=10, net上位）:
  - `scalp|M1Scalper-M1 n=16 net=+413.3 PF=4.092`
  - `micro|MicroRangeBreak n=27 net=+6.4 PF=1.035`

Failure Cause:
- `scalp_ping_5s_b` が **高頻度かつ期待値マイナス**のため、取引回数の大半を占有しPF/期待値を押し下げた。
- `MicroPullbackEMA` が micro pocket 内で大きなマイナス寄与（PF=0.111）。
- （運用リスク）`local_v2_autorecover_once.sh` の Brain autopdca が既定ONだと、意図せず cycle/restart が走り得る（opt-in運用と不整合）。

Improvement:
- `trade_min` の構成を **core + MicroRangeBreak(+exit) + M1Scalper(+exit)** に寄せ、`scalp_ping_5s_b(+exit)` を除外（stack側で反映）。
- Brain autopdca を **既定OFF（opt-inのみ）**へ変更:
  - `QR_LOCAL_V2_BRAIN_AUTOPDCA_ENABLED` 既定 `0`
  - `QR_LOCAL_V2_BRAIN_AUTOPDCA_ALLOW_RESTART` 既定 `0`（未指定時は `--dry-run` で実行し restart しない）

Verification:
- profile反映後:
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で想定workerのみが `running` であること（`scalp_ping_5s_b` が起動していない）。
- 次の再評価（まず24h、次に14d）:
  - 同一集計で `PF>1.0` / `expectancy>0` を目標（`n>=300` で判定、未達なら更にRCA）。
- Brain autopdca:
  - デフォルトで `local_v2_autorecover.log` に autopdca cycle が出ないこと。
  - opt-in時（`QR_LOCAL_V2_BRAIN_AUTOPDCA_ENABLED=1`）でも、`QR_LOCAL_V2_BRAIN_AUTOPDCA_ALLOW_RESTART=1` を明示しない限り restart されないこと（cycle output の `dry_run=true`）。

Status:
- in_progress

## 2026-03-05 15:40 JST / local-v2: MicroPullbackEMAの勝率改善（ATRスケール+M5/H1確認）+ strategy_control hard stop解除

Period:
- 対応: 2026-03-05 15:34〜15:40 JST（UTC 06:34〜06:40）
- 市況確認: 2026-03-05 15:34 JST（UTC 06:34）
- 対象: `strategies/micro/pullback_ema.py`, `workers/micro_runtime/worker.py`, `ops/env/local-v2-stack.env`

Fact:
- 市況（local tick/factor + OANDA, UTC 06:34 / JST 15:34）:
  - USD/JPY spread(avg/p95)=0.8p（tick直近500）/ mid_last=157.134
  - `ATR14(M1)=2.17p`, `ATR14(M5)=5.71p`, `ATR14(H1)=18.02p`
  - M1 `regime=Range`, `ADX=19.97` / H1 `regime=Trend`, `ADX=35.61`
- 直近7d（manual除外, `trades.db`, micro pocket）:
  - `MicroPullbackEMA n=25`, **all long**, `win_rate=0.28`, `PF=0.062`, `net_pips=-45.6`
- 運用設定（対応前）:
  - `STRATEGY_CONTROL_ENTRY_* =0` により entry を hard stop（停止済み戦略の注文試行が `strategy_control_entry_disabled` になりノイズ化）。

Failure Cause:
- MicroPullbackEMA が M1のみで方向決定し、M5/H1の逆行トレンドで long 側が連続損失。
- pullback/ma-gap が固定幅で、低ATR/弱gapで深いpullbackを許容しやすく、レンジ寄りで stop hit が増加。

Improvement:
- MicroPullbackEMA（strategy）:
  - gap/pullback を ATR スケール化（弱トレンド/低vol の誤爆を抑止）
  - `plus_di/minus_di` で方向整合を追加
  - `abs(pullback) <= abs(gap) + buffer(ATR連動)` で深い pullback を抑制
  - range_bias 時は ADX 閾値を上げる
- micro_runtime worker:
  - MicroPullbackEMA に M5/H1 の MA-gap+ADX 確認ゲートを追加（counter-trend を遮断）
- local-v2 env:
  - `STRATEGY_CONTROL_ENTRY_*` を 1 に戻し、停止ではなくフィルタでリスク制御へ
  - `LOCAL_V2_EXTRA_ENV_FILES=` を維持（Brain gate無効のまま）

Verification:
- deploy後（〜2h）:
  - `orders.db` 新規の `strategy_control_entry_disabled` が増えない（矛盾/ノイズ解消）
  - MicroPullbackEMA の `win_rate/PF` が改善（最低でも `PF>1.0` / `expectancy>0` を目標、`n>=30` で再評価）
  - M5/H1 逆行時に `pullback_mtf_block` ログが出ていること
  - Brain gate が無効のまま（order_managerで `BRAIN_ENABLED=0` 維持）

Status:
- in_progress

## 2026-03-05 15:00 JST / local-v2: trade_all常駐の解消（watchdog監視対象の縮退）+ dyn alloc sampling bias修正

Period:
- 対応: 2026-03-05 14:53〜15:00 JST（UTC 05:53〜06:00）
- 市況確認: 2026-03-05 14:43 JST（UTC 05:43）
- 対象: `scripts/local_v2_stack.sh`, `scripts/install_local_v2_launchd.sh`, `~/Library/LaunchAgents/com.quantrabbit.local-v2-autorecover.plist`, `scripts/dynamic_alloc_worker.py`, `config/dynamic_alloc.json`

Fact:
- 市況（OANDA実測, UTC 05:43 / JST 14:43）:
  - USD/JPY `bid=157.039 / ask=157.047 / spread=0.8p`
  - `ATR14(M5)=6.712p(Wilder)`, `range_60m=21.4p`
  - pricing latency `350ms`, candles(M5) latency `318ms`
- local-v2稼働状況（対応直前）:
  - `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で多数workerが `running` のまま残存していた（trade_min前提の運用と不整合）。
  - `orders.db` 直近では `strategy_control_entry_disabled` が連続し、entry停止済み戦略が注文試行を継続していた（ノイズ/負荷）。
- sizing:
  - `scripts/dynamic_alloc_worker.py` が `--limit 300` 固定だと、高頻度戦略（例: `scalp_ping_5s_b_live`）の直近取引だけで埋まり、
    低頻度/別ポケット戦略（例: `MicroRangeBreak`）が `config/dynamic_alloc.json` に出ないことがあった。

Failure Cause:
- trade_min想定でも trade_all worker が常駐し、entry停止しても「注文試行/ログ/CPU負荷」が積み上がる。
- dyn alloc のサンプルが「直近N件」偏重で、戦略間の比較・ロット配分が歪む（高頻度が全枠を占有）。

Improvement:
- watchdog/launchd の監視対象を明示縮退:
  - `scripts/install_local_v2_launchd.sh` を `--services "quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-rangebreak,quant-micro-rangebreak-exit"` で再インストール（`StartInterval=20s`）。
- trade_all 常駐の解消:
  - `scripts/local_v2_stack.sh down --profile trade_all --env ops/env/local-v2-stack.env`
  - `scripts/local_v2_stack.sh up --services "quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-micro-rangebreak,quant-micro-rangebreak-exit" --env ops/env/local-v2-stack.env`
- dyn alloc sampling bias修正:
  - `scripts/dynamic_alloc_worker.py` の `--limit` を既定 `5000` に拡張し、`--limit 0` で full lookback を許可。
  - `--limit 0 --min-trades 24` で再生成し、`MicroRangeBreak` が alloc に含まれることを確認（例: `lot_multiplier=0.681`, `jpy_pf=1.033`）。

Verification:
- `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で `running` が縮退サービスのみになる。
- `config/dynamic_alloc.json` に `MicroRangeBreak` が出力され、`lot_multiplier` が floor (`0.45`) に貼り付かない。
- `orders.db` で停止済み戦略由来の `strategy_control_entry_disabled` が新規に増えない（stop後は `orders_last_ts` が更新されない/必要戦略のみ更新される）。

Status:
- in_progress（縮退後の損益・約定・負荷の前後比較が必要）

## 2026-03-05 14:05 JST / Brain(ollama)タイムアウト起因のpreflight遅延を除去 + strategy_control env override修正 + 赤字戦略のentry停止

Period:
- 調査/反映: 2026-03-05 13:38〜14:05 JST（UTC 04:38〜05:05）
- 対象: `ops/env/local-v2-stack.env`, `logs/orders.db`, `logs/trades.db`, `logs/brain_state.db`, `logs/strategy_control.db`, `logs/local_v2_stack/quant-order-manager.log`

Fact:
- 市況（OANDA実測, UTC 04:38 / JST 13:38）:
  - USD/JPY `bid=157.112 / ask=157.120 / spread=0.8p`
  - `ATR14(M5)=6.383p(EMA)`, `range_60m=25.9p`
  - pricing latency `avg=285ms (max=337ms)`, openTrades latency `216ms`
- 直近24hの収益（`trades.db`, manual除外）:
  - `n=797`, `win_rate=0.2095`, `PF=0.421`, `net_jpy=-545.863`, `net_pips=-649.2`
  - 赤字寄与上位: `scalp_ping_5s_b_live(-181.408)`, `MicroPullbackEMA(-112.892)`, `MicroTrendRetest-long(-91.344)`, `scalp_ping_5s_d_live(-84.1)`, `scalp_ping_5s_flow_live(-79.128)`
- 実行品質（`scripts/analyze_entry_precision.py`, filled entry 直近1113）:
  - `spread_pips p95=0.8`, `latency_submit p95=303ms`
  - **`latency_preflight p95=7509ms`**（スキャルプに致命的）
- Brain決定（`brain_state.db`, 直近24h）:
  - `source=llm_fail` が多発し、`avg_latency ≒ 6.0s`（例: `scalp_ping_5s_b_live n=104 avg=5994ms`）
  - order_manager 側で `slow_request elapsed=8〜14s` が連発し、strategy 側で `order_manager Read timed out (8s)` が発生していた。

Failure Cause:
- `ops/env/local-v2-stack.env` の `LOCAL_V2_EXTRA_ENV_FILES=ops/env/profiles/brain-ollama.env` により Brainゲートが常時有効化され、
  ollama呼び出しが `BRAIN_TIMEOUT_SEC=6` で連続失敗 → preflight遅延 + market_order応答タイムアウト → stale entry/損失増大。

Improvement:
- BrainゲートをローカルV2のデフォルト導線から外す:
  - `ops/env/local-v2-stack.env` の `LOCAL_V2_EXTRA_ENV_FILES` を空に変更。
  - `quant-order-manager` を `--env ops/env/local-v2-stack.env` で再起動し、ログで `BRAIN_ENABLED=0 / ORDER_MANAGER_BRAIN_GATE_ENABLED=0` を確認。
  - `brain_decisions` は直近30秒で `0件`（新規 `llm_fail` 停止）。
  - `order/market_order` は reject でも `~70ms` で応答（preflightがタイムアウトしないことを確認）。
- 入口制御の再発防止:
  - `workers/common/strategy_control.py` の env override が `value` を参照しており無効だったため修正（`_env_bool(key)`）。
  - `STRATEGY_CONTROL_ENTRY_*` で赤字戦略（`scalp_ping_5s_{b,c,d,flow}`, `micropullbackema`, `microtrendretest`）の entry を停止し、
    `strategy_control.db` で `entry_enabled=0` を確認（exitは維持）。

Verification:
- 短期（〜1h）:
  - `brain_decisions` の新規 `llm_fail=0` を維持。
  - strategy 側の `order_manager Read timed out` が再発しない。
  - `orders.db` の `brain_scale_below_min` が新規に増えない（Brain無効化の確認）。
- 中期（1〜3h）:
  - 稼働中戦略の `PF>1.0` / `expectancy>0` が確認できたものから順次有効範囲を広げる（停止戦略は原因分析→改善→段階復帰）。

Status:
- in_progress（稼働後の損益前後比較が必要）

## 2026-03-05 11:16 JST / scalp_ping_5s_b 約定停止の即効チューニング（side_filter解除 + lookahead遮断緩和）

Period:
- 調査時刻: 2026-03-05 11:06〜11:16 JST
- 対象: `ops/env/local-v2-stack.env`, `logs/orders.db`, `logs/trades.db`, `logs/local_v2_stack/quant-scalp-ping-5s-b.log`

Fact:
- 着手前手順:
  - `sed -n '/^## 運用手順/,/^## /p' docs/AGENT_COLLAB_HUB.md` を実行し、ローカルV2運用手順を確認。
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で主要サービスが `running` を確認。
- 市況/執行プロキシ（JST 11時台）:
  - 価格帯 proxy（`orders.filled.executed_price` 直近20件）: `156.846 - 157.078`
  - spread/range proxy（`scalp_ping_5s_b` lookaheadログ）: `cost_pips ≒ 1.096-1.132`, `range ≒ 0.1-0.8p`
  - OANDA API応答品質: `quant-scalp-ping-5s-b.log` で `/v3/accounts/.../pricing` の `HTTP 200` 継続を確認。
- 約定停止の事実（JST明記）:
  - `orders` 最終約定: `2026-03-05 09:10:20 JST`（`2026-03-05T00:10:20.711262+00:00`）
  - `trades` 最終クローズ: `2026-03-04 22:54:44 JST`（`2026-03-04T13:54:44.442316+00:00`、前日）
- skip内訳（`quant-scalp-ping-5s-b.log`）:
  - `entry-skip summary side=long total=14 no_signal:side_filter_block=14`
  - `entry-skip summary side=short total=23 lookahead_block=23`
  - side_filter解除後の実測: `side_filter=(unset)` は反映されたが、`lookahead edge_negative_block` が主遮断で `orders` 更新なし。

Failure Cause:
- 主因1: `SCALP_PING_5S_B_SIDE_FILTER=short` 固定で long 側が実質停止し、シグナルが `side_filter_block` に集中。
- 主因2: short 側は `lookahead_block` 比率が高止まりし、entry 変換率が低下したまま約定停止へ遷移。

Improvement:
- `ops/env/local-v2-stack.env` を local-v2 即効チューニングとして更新:
  - `SCALP_PING_5S_B_SIDE_FILTER=none`
  - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1`
  - `SCALP_PING_5S_B_DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.35`
  - `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=0`
  - `SCALP_PING_5S_B_LOOKAHEAD_SLIP_SPREAD_MULT=0.18`
  - `SCALP_PING_5S_B_LOOKAHEAD_SLIP_RANGE_MULT=0.08`
  - `SCALP_PING_5S_B_LOOKAHEAD_LATENCY_PENALTY_PIPS=0.01`
- 追加施策: `LOOKAHEAD_GATE_ENABLED=0` を適用し、`edge_negative_block` 主遮断を一時的に外して即時の約定再開を優先。
- 狙い:
  - 約定再開（long側の `side_filter_block` 解消）
  - short側 lookahead の過剰遮断を緩和しつつ、`DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.35` で過度な逆張りを抑制。

Verification:
- 反映後に `orders.filled` が再開し、最終約定時刻が更新されること。
- `entry-skip summary` で `side_filter_block`（long）と `lookahead_block`（short）の比率が低下すること。
- `HTTP 200` 継続と `rejected=0`（orders）を維持すること。

Status:
- in_progress

## 2026-03-05 06:35 JST / Hourly RCA HOLD（trade_min停止 + trades stale + OANDA DNS失敗）

Period:
- 調査時刻: 2026-03-05 06:29〜06:35 JST
- 対象: `docs/AGENT_COLLAB_HUB.md`, `docs/OPS_LOCAL_RUNBOOK.md`, `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `scripts/local_v2_stack.sh`, `scripts/check_oanda_summary.py`

Fact:
- 着手前手順:
  - `sed -n '/^## 運用手順/,/^## /p' docs/AGENT_COLLAB_HUB.md` を実行し運用手順を確認。
  - `sed -n '/^## 運用原則/,/^## /p' docs/OPS_LOCAL_RUNBOOK.md` を実行（該当見出し抽出は空出力）。
- DB鮮度（最終再確認, JST）:
  - `orders.db=05:22:04 (age 72.4分)`, `trades.db=02:26:29 (age 248.6分)`, `metrics.db=06:35:11 (age 0.3分)`
  - 30分超 stale 条件該当のため `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` を実行し、120秒待機後に再確認。
- スタック状態:
  - `up` は `quant-order-manager port=8300 remains occupied` で失敗（占有PID: `38068,38234,38235,38236,38238,38239,38240`）。
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` は対象サービスが全て `stopped`。
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は `NameResolutionError(api-fxtrade.oanda.com)` で失敗。
  - secret欠損エラーではないため `refresh_env_from_gcp.py` 条件には非該当。
- 市況/執行プロキシ（直近2h, `logs/*.db` 実測）:
  - 価格帯（filled）: `157.002 - 157.094`、mid帯（entry_thesis）: `156.998 - 157.089`
  - spread: `avg/min/max = 0.8 / 0.8 / 0.8 pips`
  - ATR/range proxy: `atr_m1_pips avg=1.58`, `signal_range_pips avg=0.59`
  - 約定/拒否: `filled=10`, `submit_attempt=10`, `close_ok=2`, `reject=0`
  - 応答品質 proxy: `data_lag_ms avg=838.452 (max=2457.185)`, `decision_latency_ms avg=32.899 (max=98.952)`
- 直近2h PnL分解:
  - `trades.db` 実績 `0件`（strategy別/時間帯別/拒否理由別とも集計不可）。
  - 実行コスト proxy: `slippage vs ideal_entry = +0.15 pips`, `spread avg=0.8 pips`

Failure Cause:
- 利益阻害 Top3（数値根拠）:
  1. `trades.db` が `248.6分` stale で、2h PnLの一次データが欠損（`trades=0`）。
  2. `check_oanda_summary.py` が `NameResolutionError` で `1/1失敗`、API品質の正常判定が不能。
  3. `quant-order-manager` の `8300` 競合により `trade_min` 復旧失敗、`status` は主要サービス `stopped`。

Improvement:
- 本時間帯の Hourly RCA は `HOLD`。
- 次の1アクション:
  - `8300` 占有プロセス解消で `local_v2_stack up --profile trade_min` を成功させ、
    `orders/trades <=10分` と `check_oanda_summary.py` 成功を同時達成した次ランで
    2h PnL本分解（strategy別/時間帯別/拒否理由別/実行コスト別）を再開する。

Verification:
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で主要サービスが `running`。
- `logs/orders.db` と `logs/trades.db` の最終更新が `now-10m` 以内。
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功終了。

Status:
- in_progress（HOLD）

## 2026-03-05 05:36 JST / Hourly RCA HOLD（trades stale継続 + Summary API DNS失敗）

Period:
- 調査時刻: 2026-03-05 05:29〜05:36 JST
- 対象: `docs/AGENT_COLLAB_HUB.md`, `docs/OPS_LOCAL_RUNBOOK.md`, `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `scripts/local_v2_stack.sh`, `scripts/check_oanda_summary.py`

Fact:
- 着手前手順:
  - `docs/AGENT_COLLAB_HUB.md` の「運用手順」を確認（runbook準拠）。
  - `docs/OPS_LOCAL_RUNBOOK.md` の「運用原則」は該当見出しの抽出実行のみ（出力なし）。
- DB鮮度（最終再確認, JST）:
  - `orders.db=05:22:04 (age 13.35分)`, `trades.db=02:26:29 (age 188.93分)`, `metrics.db=05:35:18 (age 0.13分)`
  - 初回鮮度チェックで `trades.db` が30分超 stale のため、
    `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` を実行。
- スタック起動結果:
  - `quant-order-manager port=8300 remains occupied (start aborted for safety)` で `up` は失敗（120秒待機後も改善なし）。
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` は対象サービス `stopped`。
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は
    `NameResolutionError(api-fxtrade.oanda.com)` で失敗（secret/env欠損ではない）。
- 市況/執行プロキシ（直近2h, `logs/*.db` 実測）:
  - 価格帯（`orders.filled.executed_price`）: `157.002 - 157.131`（range `12.9 pips`）
  - spread（`preflight_start.entry_thesis.spread_pips`）: `avg=0.8 pips`（min/max `0.8/0.8`）
  - ATR proxy（`preflight_start.entry_thesis`）:
    - `mtf_regime_atr_m1 avg=1.639 pips`（`1.502-1.873`）
    - `mtf_regime_atr_m5 avg=4.823 pips`（`4.526-5.175`）
  - range proxy（`signal_range_pips`）: `avg=0.760 pips`（`0.3-1.3`）
  - 約定/拒否（orders 2h）: `total=58`, `filled=15`, `close_ok=4`, `reject_like=0`, `order_success_rate avg=1.0`, `reject_rate avg=0.0`
  - API応答品質 proxy（metrics 2h）:
    - `data_lag_ms avg=858.79, p95=1776.41, max=2794.15`
    - `decision_latency_ms avg=34.14, p95=62.34, max=98.95`
- 直近2h PnL分解:
  - `trades.db` ベース実績: `trades=0`, `realized_pnl=0`（strategy/hour/reject/execution cost いずれも実績ゼロ）
  - `orders.db` 代理集計（filled↔close_ok CID対応）:
    - strategy別: `scalp_ping_5s_b_live trades=4, net_pips=+0.20, net_unit_pips=+27.0`
    - hour別: `03:00JST +2.0p`, `04:00JST -0.8p`, `05:00JST -1.0p`
    - reject理由別: `0件`
    - execution cost proxy: `closed=4`, `avg_hold_sec=200.83`, `loss_count=3`, `avg_loss=-0.60p`, `avg_win=+2.00p`

Failure Cause:
- 利益阻害 Top3（数値根拠）:
  1. `trades.db` stale (`188.93分`) により、2h realized PnL分解が `0件` となり本来のRCA軸が欠損。
  2. `check_oanda_summary.py` が `NameResolutionError` で `1/1失敗`、API正常性を確認不能。
  3. `quant-order-manager` の `8300` 競合で `local_v2_stack up` が失敗し、復旧オペレーションが閉塞。

Improvement:
- 次の1アクション（復旧ゲート）:
  - `8300` 占有元を解消して `local_v2_stack up --profile trade_min` を成功させ、
    `trades.db age <= 10分` と `check_oanda_summary.py` 成功を同時達成した次ランで
    2h PnL本分解（strategy/hour/reject/execution cost）を再開する。

Verification:
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で主要サービスが `running`。
- `logs/trades.db` 最終更新が `now-10m` 以内。
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功終了。

Status:
- in_progress（HOLD）

## 2026-03-05 04:34 JST / Hourly RCA HOLD（OANDA Summary API DNS失敗継続）

Period:
- 調査時刻: 2026-03-05 04:29〜04:34 JST
- 対象: `docs/OPS_LOCAL_RUNBOOK.md`, `docs/AGENT_COLLAB_HUB.md`, `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `scripts/local_v2_stack.sh`, `scripts/check_oanda_summary.py`

Fact:
- 着手前手順:
  - `docs/OPS_LOCAL_RUNBOOK.md` の運用原則、および `docs/AGENT_COLLAB_HUB.md` の運用手順を確認。
- DB鮮度（初回確認, JST）:
  - `orders.db=03:50:07`, `trades.db=02:26:29`, `metrics.db=04:29:45`
  - `trades.db` が30分超 stale のため、指示どおり `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` を実行。
- スタック起動結果:
  - `quant-order-manager port=8300 remains occupied (start aborted for safety)` で `up` は完了せず（feed/control のみ起動ログあり）。
  - 120秒待機後の再確認でも `trades.db=02:26:29 JST` のまま更新なし。
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` を2回実行し、2/2で `NameResolutionError(api-fxtrade.oanda.com)`。
  - secret欠損ではなく DNS 解決失敗。
- 市況/執行プロキシ（`logs/*.db` 直近2h, window UTC `17:34:42-19:34:42`）:
  - 価格帯（`filled.executed_price` + `close_ok.exit_context.mid`）: `156.988 - 157.156`
  - spread: `avg 0.8 pips`
  - ATR proxy: `mtf_regime_atr_m1 avg 1.899 pips`, `mtf_regime_atr_m5 avg 5.408 pips`
  - range proxy: `signal_range_pips avg 0.94 pips`
  - 約定/拒否: `orders=37`, `filled=10`, `reject_like=0`, `order_success_rate_avg=1.0`, `reject_rate_avg=0.0`
  - API応答品質 proxy（strategy_control）: `data_lag_ms avg=894.061 / max=20850.001`, `decision_latency_ms avg=34.566 / max=195.601`
  - `trades_2h=0`, `net_pnl=0`（`trades.db`最終更新が約340分 stale）

Failure Cause:
- ローカル市況/執行プロキシ自体は極端悪化ではないが、`check_oanda_summary.py` が継続してDNS失敗し、
  API応答品質を「正常」と判定できない。
- さらに `trades.db` が長時間 stale のままで、PnL分解の入力品質を満たさない。

Improvement:
- 本時間帯の Hourly RCA は `HOLD` とし、通常の2h PnL分解（strategy別/時間帯別/拒否理由別/実行コスト別）は実施しない。
- 次の1アクション（再開ゲート）:
  - `check_oanda_summary.py` 成功（DNS復旧）かつ `orders.db/trades.db` 更新が `<=10分` に戻るまで復旧対応を優先し、達成後ランでRCA本体を再開する。

Verification:
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功終了すること。
- `logs/orders.db` と `logs/trades.db` の最終更新が `now-10m` 以内になること。
- 条件成立後にのみ、直近2h PnL分解（strategy/hour/reject/execution cost）を再開すること。

Status:
- in_progress（HOLD）

## 2026-03-05 02:35 JST / Hourly RCA HOLD（OANDA DNS劣化でAPI品質異常）

Period:
- 調査時刻: 2026-03-05 02:29〜02:35 JST
- 対象: `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `logs/tick_cache.json`, `logs/factor_cache.json`, `scripts/check_oanda_summary.py`

Fact:
- DB鮮度（`age_min`）:
  - `orders.db=2.77分`, `trades.db=3.55分`, `metrics.db=0.01分`（30分以内のため `local_v2_stack up` は未実行）
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は
    `api-fxtrade.oanda.com` の `NameResolutionError` で失敗（secret欠損ではない）
- 市況（ローカル実測）:
  - `tick_cache` 最新: `bid/ask/mid=156.976/156.984/156.980`、`spread=0.8 pips`
  - 直近15分バンド: `156.926-157.038`（`range=11.2 pips`）
  - 直近60分バンド: `156.884-157.038`（`range=15.4 pips`）
  - `factor_cache` ATR: `M1=2.633 pips`, `M5=6.753 pips`
- 約定・拒否実績（直近2h）:
  - `orders` ステータス: `filled=4`, `submit_attempt=4`, `preflight_start=4`, `close_ok=2`, `rejected=0`
  - `trades` close件数: `0`、`realized_pl=0`
- 実行/応答品質（直近2h）:
  - `data_lag_ms`: `avg=1462.76`, `max=42302.97`（`n=717`）
  - `decision_latency_ms`: `avg=38.33`, `max=378.20`（`n=717`）

Failure Cause:
- ローカル価格レンジ/スプレッド自体は異常なしだが、OANDA account summary API のDNS解決失敗が継続し、
  API品質を「通常」と判定できない。
- API品質異常時に2h RCA（strategy別/時間帯別/拒否理由別/execution cost別）を進めると、
  市況前提の欠損を含む誤判定リスクが高い。

Improvement:
- 本時間帯の Hourly RCA は `HOLD` とし、PnL分解は実施しない。
- 次の単一アクション（数値ゲート付き）:
  - `check_oanda_summary.py` が成功するまでネットワーク/DNSを復旧し、
    復旧後ランで `API成功` かつ `orders/trades 更新 <=10分` を満たした時点でRCA本体を再開する。

Verification:
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功終了すること。
- `logs/orders.db` / `logs/trades.db` の最終更新が `now-10m` 以内であること。
- 上記成立後にのみ、直近2hのPnL分解（strategy/hour/reject/execution cost）を再実施する。

Status:
- in_progress（HOLD）

## 2026-03-05 01:36 JST / Hourly RCA HOLD（API DNS失敗継続 + 約定DB更新停止）

Period:
- 調査時刻: 2026-03-05 01:29〜01:36 JST
- 対象: `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `logs/orderbook_snapshot.json`, `logs/factor_cache.json`, `scripts/check_oanda_summary.py`

Fact:
- DB最終更新（JST）:
  - `orders.db`: `2026-03-04 23:03:15`（約152.7分 stale）
  - `trades.db`: `2026-03-05 00:04:08`（約91.9分 stale）
  - `metrics.db`: `2026-03-05 01:35:56`（fresh）
- 指示どおり `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` を実行したが、
  `quant-order-manager port=8300 remains occupied (start aborted for safety)` で再起動不可。
- OANDA API確認:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は
    `api-fxtrade.oanda.com` の `NameResolutionError` で失敗（DNS解決不可）。
- 市況確認（ローカル実測）:
  - `orderbook_snapshot`: `bid=156.960 / ask=156.968 / spread=0.8 pips`
  - `factor_cache(M1)`: `close=156.998`, `ATR=2.908 pips`, `range15=7.7 pips`, `range60=33.7 pips`
  - 時刻: `2026-03-04T16:34:59Z`（約定DBより新しいが、注文/約定記録は停止）
- 約定・拒否実績（直近2h, `datetime(ts)` 基準）:
  - `orders=0`, `rejected=0`, `filled=0`
  - `trades=0`, `realized_pl=0`
- 応答品質（直近2h, `metrics.db`）:
  - `data_lag_ms`: `last=579.137`, `p95=2502.624`（`n=557`）
  - `decision_latency_ms`: `last=44.872`, `p95=63.332`（`n=557`）

Failure Cause:
- OANDA account summary API が DNS 解決できず、live API品質確認が不能。
- `orders.db` / `trades.db` が90分超 stale で、RCA本体（直近2h PnL分解）の入力が欠損。
- `quant-order-manager` ポート競合で `trade_min` 再起動が完了しないため、復旧確認に進めない。

Improvement:
- 本時間帯の Hourly RCA は `HOLD` とし、通常の「strategy別/時間帯別/拒否理由別/実行コスト別」PnL分解は実施しない。
- 次の1アクション（数値ゲート）:
  1. `8300` 競合を解消し `local_v2_stack up` を成功させ、`orders.db` と `trades.db` の更新を `<=10分` に戻す。
  2. `check_oanda_summary.py` のDNS解決復旧を確認し、成功レスポンスを取得する。
  3. 上記2条件達成後、直近2hのPnL分解を再開する。

Verification:
- `orders.db/trades.db` の最終更新が `now-10m` 以内。
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` 成功。
- 条件達成後にのみ Hourly RCA本体（PnL分解）を再開。

Status:
- in_progress（HOLD）

## 2026-03-05 00:50 JST / ローカル収益RCA第4段 + 停止耐性（watchdog/launchd）固定

Period:
- 分析窓: 直近24h（`logs/trades.db`, `logs/orders.db`, `logs/metrics.db`）
- 実装/検証: 2026-03-05 00:20〜00:50 JST（ローカルV2導線）

Fact:
- 市況（ローカル実測）:
  - `USD/JPY bid=157.266 ask=157.274 spread=0.8 pips`
  - `ATR14(M1)=5.057 pips`, `range60(M1)=17.4 pips`
  - pricing応答 `avg=255ms, p95=276ms`
- 収益分解（`scalp_ping_5s_b_live`）:
  - `n=608`, `net=-175.605`, `win_rate=19.6%`, `PF=0.416`
  - `STOP_LOSS_ORDER=470 / net=-280.685`
  - side別: `buy n=412 net=-177.957` / `sell n=196 net=+2.352`
  - 保有秒バケット: `<5s net=-78.525`, `5-15s net=-64.523`, `15-30s net=-50.495`（短期偏損）
- 反実仮想:
  - `sell_only` では `net=+2.352`、`sell_no_momentum_hz` では `net=+2.812`

Failure Cause:
- 損失主因は buy側（特に `momentum*` 系）の逆選別。
- 30秒未満の超短期エントリーでSL偏重が発生し、期待値を継続的に毀損。
- 停止耐性は「起動コマンド依存」の運用余地が残り、ネット断/スリープ復帰時の再開確実性が不足。

Improvement:
- 収益改善（`ops/env/scalp_ping_5s_b.env`）:
  - `SIGNAL_MODE_BLOCKLIST=momentum_sidefilter,momentum_hz,momentum_hz_slflip_smflip_hz`
  - `ENTRY_COOLDOWN_SEC=4.5`, `MAX_ORDERS_PER_MINUTE=2`
  - `CONF_FLOOR=82`, `LOOKAHEAD_EDGE_MIN_PIPS=0.30`, `LOOKAHEAD_SAFETY_MARGIN_PIPS=0.18`
  - `MAX_SPREAD_PIPS=0.80`
  - `MIN_TICKS=5`, `MIN_SIGNAL_TICKS=4`, `SHORT_MIN_TICKS=5`, `SHORT_MIN_SIGNAL_TICKS=4`
  - `LOOKAHEAD_ALLOW_THIN_EDGE=0`
  - `REVERT_MIN_TICKS=3`, `REVERT_CONFIRM_TICKS=2`, `REVERT_MIN_TICK_RATE=0.60`
  - force-exit損失側を早期化（`MAX_FLOATING_LOSS_PIPS=0.65`, `MIN_HOLD_SEC=1` など）
- 停止耐性（手動起動不要化）:
  - `scripts/local_v2_watchdog.sh` を新設（`start/run/once/stop/status`）
  - `scripts/local_v2_stack.sh` に `watchdog/watchdog-stop/watchdog-status` を追加
  - `scripts/local_v2_autorecover_once.sh` に state管理を追加（polling gap検知 + network down/up検知）
  - network復帰時に `quant-market-data-feed` を自動再起動（既定ON, cooldown付き）
  - `scripts/install_local_v2_launchd.sh` を watchdog導線へ更新（既定10秒間隔）

Verification:
- テスト:
  - `pytest -q tests/workers/test_scalp_ping_5s_worker.py -k "signal_mode_blocked or resolve_final_signal_for_side_filter"` → `6 passed`
- スクリプト構文:
  - `bash -n scripts/local_v2_stack.sh scripts/local_v2_watchdog.sh scripts/local_v2_autorecover_once.sh scripts/install_local_v2_launchd.sh`
- watchdog実動:
  - `local_v2_stack.sh watchdog --daemon` → `watchdog-status` で `running`、`watchdog-stop` で `stopped`
- launchd反映:
  - `install_local_v2_launchd.sh --interval-sec 10 ...` 実行後、`status_local_v2_launchd.sh` で `run interval = 10 seconds` と env 注入（`QR_LOCAL_V2_NET_RECOVERY_RESTART_MARKET_DATA=1`）を確認
- 稼働確認:
  - `local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で8サービス `running`

Status:
- in_progress（直近 30〜90 分で `STOP_LOSS_ORDER 比率` と `scalp_ping_5s_b_live net` を追跡）

## 2026-03-05 00:34 JST / Hourly RCA HOLD（API名前解決失敗 + 約定DB更新停止）

Period:
- 調査時刻: 2026-03-05 00:29〜00:34 JST
- 対象: `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `logs/tick_cache.json`, `logs/factor_cache.json`, `scripts/check_oanda_summary.py`

Fact:
- DB更新時刻:
  - `orders.db`: `2026-03-04 23:03:15 JST`（調査時点で約91分経過）
  - `trades.db`: `2026-03-05 00:04:08 JST`（調査時点で約30分経過）
  - `metrics.db`: `2026-03-05 00:32:09 JST`（最新）
- `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` は
  `quant-order-manager port=8300 remains occupied (start aborted for safety)` で完了不可。
- API疎通:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` は
    `api-fxtrade.oanda.com` の `NameResolutionError` で失敗（DNS解決不可）。
- 市況（ローカル実測）:
  - `USD/JPY bid=157.234 ask=157.242 mid=157.238`, `spread=0.8 pips`
  - `ATR(M1)=3.664 pips`, `ATR(M5)=7.979 pips`
  - `range(15m)=13.0 pips`, `range(60m)=19.0 pips`
- 約定・拒否実績:
  - `orders` 直近2h: `115` 件、`rejected=2`
  - reject上位（直近24h）: `error_code=(none) 28件`
- 応答品質（`metrics.db` 直近2h）:
  - `data_lag_ms last=1206.68 / p95=1366022.22`
  - `decision_latency_ms last=12.24 / p95=96.70`

Failure Cause:
- OANDA APIのDNS解決失敗により、live APIベースの確認が不能。
- `orders.db` と `trades.db` の更新が停止/遅延しており、直近2時間PnL分解の前提データ品質を満たせない。
- `local_v2_stack up` もポート競合で完了せず、即時復旧を確認できない。

Improvement:
- 本時間帯のRCAは `HOLD` とし、PnL分解（strategy別/時間帯別/拒否理由別/実行コスト別）は保留。
- 次アクション（優先順）:
  1. `8300` 占有PIDの解消後に `local_v2_stack up` を再実行し、`orders/trades` 更新再開を確認。
  2. `check_oanda_summary.py` の再試行で API 名前解決復旧を確認。
  3. 復旧後に直近2h PnL分解を再実施して通常RCAへ戻す。

Verification:
- `orders.db/trades.db` の最終更新が `now-10m` 以内に戻ること。
- `check_oanda_summary.py` が成功し、`pricing/account summary` を取得できること。
- 上記2条件を満たした時点で、2時間PnL分解を再開すること。

Status:
- in_progress（HOLD）

## 2026-03-05 03:34 JST / Hourly RCA HOLD（API DNS失敗継続 + order/trade更新停止）

Period:
- 調査時刻: 2026-03-05 03:27〜03:34 JST
- 対象: `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `scripts/local_v2_stack.sh`, `scripts/check_oanda_summary.py`

Fact:
- DB更新時刻:
  - `orders.db`: `2026-03-05 02:27:16 JST`（約67.5分 stale）
  - `trades.db`: `2026-03-05 02:26:29 JST`（約68.3分 stale）
  - `metrics.db`: `2026-03-05 03:34:44 JST`（更新継続）
- スタック再起動:
  - `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` は
    `quant-order-manager port=8300 remains occupied` で失敗（`120s` 待機後も改善なし）。
- API疎通:
  - `PYTHONPATH=. python3 scripts/check_oanda_summary.py` を2回実行し、両方 `NameResolutionError(api-fxtrade.oanda.com)`。
- 市況プロキシ（`orders.db` 直近2h、`preflight_start`）:
  - spread: `avg=0.801 pips`（min `0.8`, max `1.0`）
  - ATR proxy: `mtf_regime_atr_m1=3.429`, `mtf_regime_atr_m5=9.525`
  - range proxy: `signal_range_pips=0.51`
  - 価格帯 proxy（`filled/close_ok`）: `156.884 - 157.594`（range `0.710`）
- 約定/拒否実績（直近2h）:
  - `orders=2843`, `filled=671`, `reject_like=412`
  - `trades=704`, `net_pnl=-299.270`
  - reject上位: `entry_probability_reject=383`, `STOP_LOSS_ON_FILL_LOSS=27`, `api_error(502)=1`
- API応答品質 proxy（`metrics.db` 直近2h）:
  - `data_lag_ms avg=6,534,745.994`（max `3,385,784,138.449`）
  - `decision_latency_ms avg=39.121`（max `3025.991`）

Failure Cause:
- 利益阻害トップ3（数値根拠）:
  1. API品質異常: `check_oanda_summary` 2/2失敗（DNS解決不可）。
  2. 執行導線停止: `orders/trades` が `67-68分` stale、`local_v2_stack up` も `port 8300` 競合で復旧失敗。
  3. 成績劣化: 直近2h `net_pnl=-299.270`、かつ `reject_like=412/2843 (14.5%)`、主要拒否は `entry_probability_reject=383`。

Improvement:
- 本時間帯のRCAは `HOLD` とし、通常の2h分解（strategy別・時間帯別・拒否理由別・実行コスト別）は保留。
- 次の1アクション:
  - `8300` 占有プロセス（`PID: 38068, 38234, 38235, 38236, 38238, 38239, 38240`）の整理を最優先し、
    `local_v2_stack up` 成功と `orders/trades <=10m` 回復を確認してからRCA再開する。

Verification:
- `orders.db/trades.db` の最終更新が `now-10m` 以内に戻ること。
- `PYTHONPATH=. python3 scripts/check_oanda_summary.py` が成功すること。
- 上記2条件を満たした次ランで、直近2h PnLの4軸分解を再実行すること。

Status:
- in_progress（HOLD）

## Hourly RCA 改善案バックログ（automation: qr-hourly-rca）

- `[status=in_progress]` API到達性と約定DB鮮度の復旧確認（`check_oanda_summary` 成功 + `orders/trades` 更新 `<=10m`）
- `[status=in_progress]` 8300競合（`quant-order-manager`）の占有元特定と競合解消手順の固定化（PID群まで確認済み）
- `[status=open]` 復旧後の直近2h PnL分解（strategy別・時間帯別・拒否理由別・実行コスト別）を再実施
- `[status=done]` 2026-03-05 06:35 JST ランのHOLD判定を台帳へ追記（trade_min停止 + trades stale + DNS失敗）
- `[status=done]` 2026-03-05 05:36 JST ランのHOLD判定を台帳へ追記（trades stale + DNS失敗 + 8300競合）
- `[status=done]` 2026-03-05 03:34 JST ランのHOLD判定を台帳へ追記（DNS失敗 + 8300競合継続）
- `[status=done]` 2026-03-05 02:35 JST ランのHOLD判定を台帳へ追記（API DNS失敗継続）
- `[status=done]` 2026-03-05 01:36 JST ランのHOLD判定を台帳へ追記
- `[status=done]` 2026-03-05 04:34 JST ランのHOLD判定を台帳へ追記（Summary API DNS失敗 + trades stale）

## 2026-03-04 14:22 UTC / 2026-03-04 23:22 JST - `SCALP_PING_5S_B_SIDE_FILTER` の fail-closed 強化（空許可の誤適用防止）

Period:
- 調査時刻: 2026-03-04 14:18〜14:22 UTC（23:18〜23:22 JST）
- 対象: `workers/scalp_ping_5s_b/worker.py` の環境変数マッピング

Fact:
- 起動ログで過去に `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` が混入した際、
  `side_filter=(unset)` で起動していた履歴があった。
- 直近起動では `side_filter=sell` を確認したが、未設定時に空sideを許容し得る分岐が残っていた。

Failure Cause:
- `ALLOW_NO_SIDE_FILTER=1` のとき、`SIDE_FILTER` が未設定でも空値を許容する実装だったため、
  意図しない no-filter 起動が起きる余地があった。

Improvement:
- `workers/scalp_ping_5s_b/worker.py` を修正し、
  `ALLOW_NO_SIDE_FILTER=1` でも **`SIDE_FILTER` が明示設定された場合のみ** 空sideを許容するよう変更。
- `SIDE_FILTER` 未設定時は常に `sell` へ fail-closed するよう固定。
- `tests/workers/test_scalp_ping_5s_b_worker_env.py` に再発防止テストを追加。

Verification:
- `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py` が `17 passed`。
- `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` かつ `SCALP_PING_5S_B_SIDE_FILTER` 未設定でも
  `SCALP_PING_5S_SIDE_FILTER=sell` になることをテストで確認。

Status:
- done

## 2026-03-04 14:15 UTC / 2026-03-04 23:15 JST - ローカル運用タスクでの VM 導線実行ミスの是正

Period:
- 発生時刻: 2026-03-04 13:54〜14:14 UTC（22:54〜23:14 JST）
- 対象: 運用手順の適用判断（ローカル運用タスク）

Fact:
- ユーザー依頼はローカル運用改善だったが、VM/GCP 導線（`scripts/vm.sh` / `deploy_to_vm.sh`）を実行した。
- 実行結果は IAM 権限不足で失敗し、ローカル改善タスクの進行を阻害した。

Failure Cause:
- ローカル運用モードの例外条件（「VMは明示指示時のみ」）を、実行前判断で徹底できなかった。

Improvement:
- `docs/OPS_LOCAL_RUNBOOK.md` の運用原則に、
  「ローカル運用タスクでは VM/GCP コマンドを実行しない。例外は明示依頼のみ」を追記。
- 以後の本タスクはローカルDB + OANDA API のみに限定して実施する。

Verification:
- 以後同種タスクで `scripts/vm.sh` / `deploy_to_vm.sh` / `gcloud compute *` を使わずに完了できること。
- 変更はローカル導線（`local_v2_stack` / `logs/*.db` / OANDA API）でのみ再現確認すること。

Status:
- done

## 2026-03-04 14:08 UTC / 2026-03-04 23:08 JST - `scalp_ping_5s_b_live` 収益悪化RCA第3段（SL偏重の即時圧縮）

Period:
- 集計窓: 直近24h（`logs/orders.db`, `logs/trades.db`, `logs/metrics.db`）
- 市況確認: OANDA API（`USD_JPY` pricing/candles/openTrades）
- 監査時刻: 2026-03-04 13:54〜14:08 UTC（22:54〜23:08 JST）

Fact:
- 市況は稼働可能:
  - `bid/ask=157.302/157.310`、`spread=0.8 pips`
  - `ATR14(M1)=3.393 pips`、`range_60m=18.8 pips`
  - API応答: pricing `mean=262ms`、candles `408ms`、openTrades取得成功
- 戦略収益（`scalp_fast / scalp_ping_5s_b_live`）:
  - `n=608`, `win_rate=19.57%`, `PF=0.416`, `expectancy=-0.752 pips`, `net=-175.6 JPY`
  - `close_reason`: `STOP_LOSS_ORDER=470 (net=-280.7 JPY, avg=-1.421 pips)` /
    `MARKET_ORDER_TRADE_CLOSE=138 (net=+105.1 JPY, avg=+1.527 pips)`
  - side別: `buy n=412 net=-178.0 JPY`、`sell n=196 net=+2.4 JPY`
  - `entry_probability>=0.85` の buy が `n=310 net=-142.4 JPY` で、高確率帯でも逆選別が発生

Failure Cause:
- 低品質エントリーがSLへ偏る構造が主因（約77%がSL終了）。
- 特に buy 側で確率校正が崩れ、`high-probability` 帯でも損失寄与が継続。
- spread/latencyの実行品質は致命劣化ではなく（spread平均0.802p, submit p50≈202ms）、
  エントリー品質とside配分の問題が優勢。

Improvement:
- `ops/env/scalp_ping_5s_b.env` を第3段調整（停止ではなく品質圧縮）:
  - エントリー厳格化: `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.62`, `ENTRY_PROBABILITY_ALIGN_FLOOR=0.58`
  - カウンター抑制: `ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX=0.28`
  - side実績連動ロット圧縮: `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT=0.45`,
    `...MAX_MULT=0.78`, `SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.22`,
    `SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.28`, `SIDE_ADVERSE_STACK_DD_MIN_MULT=0.40`
  - lookaheadの最低エッジ強化: `LOOKAHEAD_EDGE_MIN_PIPS=0.16`, `LOOKAHEAD_SAFETY_MARGIN_PIPS=0.08`
  - コスト耐性: `MAX_SPREAD_PIPS=0.90`、SL reject低減で `SL_MIN_PIPS=1.00` / `SHORT_SL_MIN_PIPS=1.00`

Verification:
- 反映後 60分/240分で次を確認:
  - `STOP_LOSS_ORDER 比率 <= 70%`
  - `buy` の net寄与がマイナス拡大しない（直近窓で `net_buy >= -20 JPY` を目安）
  - `PF >= 0.85` への回復傾向（最低でも `expectancy_pips > -0.20`）
  - `rejected:STOP_LOSS_ON_FILL_LOSS` の件数低下

Status:
- in_progress

## 2026-03-04 13:31 UTC / 2026-03-04 22:31 JST - sidecar `POSITION_MANAGER_SERVICE_PORT` 未反映の修正（18301運用を有効化）

Period:
- 調査時刻: 2026-03-04 13:29〜13:31 UTC（22:29〜22:31 JST）
- 対象: `workers/position_manager/worker.py`, `ops/env/local-v2-sidecar-ports.env`, `scripts/local_v2_stack.sh`

Fact:
- `local_v2_stack.sh` 側は `POSITION_MANAGER_SERVICE_PORT` を参照していたが、
  `workers.position_manager.worker` の `uvicorn.run` は `port=8301` 固定だった。
- この不整合により、`--env ops/env/local-v2-sidecar-ports.env` 指定時でも
  sidecar 起動は `8301` bind を試行し、parity と競合して失敗していた。

Failure Cause:
- ポート設定の責務分離が不完全で、起動スクリプトの env 設計と worker 実装が一致していなかった。

Improvement:
- `workers/position_manager/worker.py` に `_service_port()` を追加し、
  `POSITION_MANAGER_SERVICE_PORT`（未設定時 `8301`）を読む実装へ変更。
- `tests/workers/test_position_manager_worker_env.py` を追加し、
  default (`8301`) と env override（例: `9315`）の両方を検証。

Verification:
- `pytest -q tests/workers/test_position_manager_worker_env.py` が `2 passed`。
- parity 稼働中に
  `scripts/local_v2_stack.sh up --services quant-position-manager --env ops/env/local-v2-sidecar-ports.env --force-conflict`
  を実行し、`http://127.0.0.1:18301/health` 応答を確認。
- 同コマンドで `down` まで完了し、conflict-safe mode で parity 側を巻き込まず停止できることを確認。

Status:
- done

## 2026-03-04 13:30 UTC / 2026-03-04 22:30 JST - `scalp_ping_5s_b_live` 第2段チューニング（lookahead過剰block緩和 + long過大ロット抑制）

Period:
- 調査時刻: 2026-03-04 13:20〜13:30 UTC（22:20〜22:30 JST）
- 対象: ローカル parity ログ `logs/local_vm_parity/quant-scalp-ping-5s-b.log` と `ops/env/scalp_ping_5s_b.env`

Fact:
- 直近 lookahead block は `edge_negative_block` のみ（`241/241 = 100%`）。
- side内訳は `short=207`, `long=34`（short 偏重）。
- blockサンプルは `pred ~0.10-0.31p` に対し `cost ~1.12-1.19p` で、`edge` が恒常的にマイナス。
- `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=1` のため、`edge <= 0` は即 block（thin-edge 設定では回避不可）。

Failure Cause:
- spread + slippage 見積りに対して短期予測値（pred）が不足し、lookahead が常時 `edge_negative_block` へ収束。
- 一方で long 側は過去の負け寄与が大きく、entry 復帰時のロット上振れを抑える安全弁が不足。

Improvement:
- `ops/env/scalp_ping_5s_b.env` を第2段で更新:
  - 方向安全弁（時限）: `SIDE_FILTER=sell`, `ALLOW_NO_SIDE_FILTER=0`（long側の即時遮断）
  - lookahead の予測項を引き上げ: `HORIZON_SEC=2.80`, `MOMENTUM_WEIGHT=1.15`, `FLOW_WEIGHT=0.50`, `TRIGGER_WEIGHT=0.45`, `BIAS_WEIGHT=0.42`, `COUNTER_PENALTY=0.30`
  - cost見積りを過剰保守から緩和: `SLIP_BASE_PIPS=0.04`, `SLIP_SPREAD_MULT=0.10`, `SLIP_RANGE_MULT=0.06`
  - long過大ロット抑制: `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.08`,
    `ENTRY_PROBABILITY_ALIGN_UNITS_MAX_MULT=0.94`,
    `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT=0.96`,
    `SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.34`
- 既存の `LOOKAHEAD_GATE_ENABLED=1` は維持し、無条件エントリー化は行わない。
- long再開は時限条件付き（long単独で `PF>1` かつ `avg_pips>=0` を一定件数で確認後）。

Verification:
- parity 再起動後、30分/120分で以下を確認する。
  - `lookahead block` 件数と `edge_negative_block` 比率が低下すること
  - `filled` / `preflight_start` 比率の回復
  - `PF`, `win_rate`, `STOP_LOSS_ORDER 比率` が第1段より悪化しないこと
  - long/short の平均 units が再び long 側へ偏りすぎないこと

Status:
- in_progress

## 2026-03-04 13:23 UTC / 2026-03-04 22:23 JST - `local_v2_stack` と parity supervisor 競合による ENTRY 減少の再発防止

Period:
- 調査時刻: 2026-03-04 12:40〜13:23 UTC（21:40〜22:23 JST）
- 対象: ローカルV2導線（`scripts/local_v2_stack.sh`）と parity 導線（`scripts/local_vm_parity_supervisor.py`）

Fact:
- `ps` の親子関係で、`workers.position_manager.worker` / `workers.order_manager.worker` は
  `local_vm_parity_supervisor.py`（`screen: qr-local-parity`）配下で稼働。
- `lsof` で `:8300` と `:8301` は parity 側の worker が LISTEN。
- 同時に `local_v2_stack.sh up` を実行すると、`quant-order-manager.log` /
  `quant-position-manager.log` で `Errno 48 (address already in use)` が発生し、
  worker が不安定化して ENTRY 停滞が発生。

Failure Cause:
- 同一 repo で「`local_v2_stack` と parity supervisor を同時運転」し、
  同じ worker と同じ固定ポート（8300/8301）を奪い合う運用競合が発生していた。

Improvement:
- `scripts/local_v2_stack.sh` に排他ガードを追加。
  - `up/down/restart` 実行時に次を検出したら既定拒否:
    - `screen` セッション `qr-local-parity`
    - `scripts/local_vm_parity_supervisor.py` プロセス（repo配下）
  - 案内: `scripts/local_vm_parity_stack.sh stop` を表示。
  - 例外: 意図的な実行時のみ `--force-conflict` でバイパス可能。
- `docs/OPS_LOCAL_RUNBOOK.md` に「`local_v2_stack` と parity は排他運用」を明記。
- `docs/WORKER_REFACTOR_LOG.md` に監査ログ追記。

Verification:
- `scripts/local_v2_stack.sh up --services quant-position-manager --env ops/env/local-v2-stack.env`
  が parity 稼働中に `EXIT:3` で拒否されること。
- `scripts/local_v2_stack.sh up --services quant-position-manager --env ops/env/local-v2-stack.env --force-conflict`
  でガードがバイパスされること（その後の成否は環境依存）。
- `scripts/local_v2_stack.sh status` / `logs` が parity 稼働中でも実行できること。

Status:
- done

## 2026-03-04 12:40 UTC / 2026-03-04 21:40 JST - `scalp_ping_5s_b_live` 緊急収益改善（品質閾値+サイズ圧縮）

Period:
- 集計窓: 直近24h（`logs/trades.db`, `logs/orders.db`, `logs/metrics.db`）
- 市況確認: OANDA API 直近取得（`USD/JPY`）

Fact:
- 市況は稼働可能レンジ:
  - `bid/ask=157.214/157.222`, `spread=0.8 pips`
  - `ATR14(M1)=3.3429 pips`, `range15m=19.1 pips`, `range60m=20.2 pips`
  - API応答 `avg=246ms`, error `0`
- 戦略収益（`pocket <> manual`）:
  - `n=537`, `win_rate=20.3%`, `PF=0.43`, `expectancy=-0.721 pips`, `net=-164.864 JPY`
  - `close_reason`: `STOP_LOSS_ORDER=414 (77.09%, net=-270.768 JPY)`, `MARKET_ORDER_TRADE_CLOSE=123 (net=+105.904 JPY)`
- side寄与:
  - `long: n=397, avg_units=63.4, net=-169.127 JPY`
  - `short: n=146, avg_units=1.9, net=+2.882 JPY`
- entry_probability帯:
  - `0.80-0.90` が最大赤字（`n=297, net=-118.285 JPY, exp=-0.969 pips`）
- 直近24hの注文:
  - `filled=593`, `entry_probability_reject=383`, `rejected=22`
  - reject理由は `STOP_LOSS_ON_FILL_LOSS` が全件

Failure Cause:
- `scalp_ping_5s_b_live` が高頻度エントリーのまま SL 偏重（勝率20%台）で、ロング側の実効サイズが過大。
- 高確率帯（0.80-0.90）で期待値が崩れており、既存の確率補正/帯別配分が過大評価を抑え切れていない。

Improvement:
- `ops/env/scalp_ping_5s_b.env` を即時更新（停止ではなく品質選別を強化）:
  - 取引密度: `MAX_ACTIVE_TRADES 2->1`, `MAX_ORDERS_PER_MINUTE 6->4`
  - ロット上限: `BASE_ENTRY_UNITS 70->45`, `MAX_UNITS 700->180`
  - spread閾値: `MAX_SPREAD_PIPS 2.00->1.00`
  - entry品質: `CONF_FLOOR 72->78`, `CONF_SCALE_MIN_MULT 0.92->0.80`
  - 方向過大化抑制: `DIRECTION_BIAS_ALIGN_UNITS_BOOST_MAX 0.08->0.03`, `SIDE_BIAS_BLOCK_THRESHOLD 0.08->0.12`
  - 確率補正強化:
    - `ENTRY_PROBABILITY_ALIGN_PENALTY_MAX 0.20->0.28`
    - `ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX 0.22->0.30`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN 0.74->0.82`
    - `ENTRY_PROBABILITY_ALIGN_FLOOR 0.35->0.50`
  - 確率帯配分を縮小側へ:
    - `ENTRY_PROBABILITY_BAND_ALLOC_HIGH_REDUCE_MAX 0.65->0.82`
    - `ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT 0.82->0.65`
    - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT 0.82->0.68`
  - adverse stackの縮小強化:
    - `SIDE_ADVERSE_STACK_UNITS_MIN_MULT 0.72->0.60`
    - `SIDE_ADVERSE_STACK_DD_MIN_MULT 0.78->0.65`
  - spread guard実値:
    - `spread_guard_max_pips=1.00`
    - `spread_guard_release_pips=0.85`
    - `spread_guard_hot_trigger_pips=1.10`
    - `spread_guard_hot_cooldown_sec=8`

Verification:
- 適用後 60分/180分で以下を確認する。
  - `PF >= 0.80`（まずは負け幅圧縮）
  - `win_rate >= 30%`
  - `STOP_LOSS_ORDER 比率 <= 65%`
  - `net_pips` の時間帯連続悪化（3連続マイナス）解消
  - `rejected` が `STOP_LOSS_ON_FILL_LOSS` 偏重のまま増加しないこと

Status:
- in_progress

## 2026-03-04 08:49 UTC / 2026-03-04 17:49 JST - `scalp_ping_5s_b` 第2段調整（long偏重SL抑制）

Period:
- 変更時刻: 2026-03-04 08:49 UTC / 17:49 JST
- 対象: `ops/env/scalp_ping_5s_b.env`, `ops/env/quant-order-manager.env`

Fact:
- `STOP_LOSS_ORDER` が long 側へ集中し、同方向の連続損切りが収益を圧迫。
- `direction_cap` 判定が優勢な局面で long の過密継続が残存。
- short 側の通過率は維持されており、sell-only 固定へ戻さず long 側の連打のみを抑制する方針が妥当。

Failure Cause:
- エントリー間隔・方向反転クールダウン・long 側モメンタム閾値が浅く、連続longの抑制が不足。

Improvement:
- `ops/env/scalp_ping_5s_b.env`:
  - `ENTRY_COOLDOWN_SEC: 2.8 -> 4.0`
  - `FAST_DIRECTION_FLIP_COOLDOWN_SEC: 0.6 -> 1.2`
  - `BASE_ENTRY_UNITS: 120 -> 90`
  - `LONG_MOMENTUM_TRIGGER_PIPS: 0.14 -> 0.20`
  - `DIRECTION_BIAS_BLOCK_SCORE: 0.52 -> 0.60`
  - `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT: 0.28 -> 0.20`
- `ops/env/quant-order-manager.env`:
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE): 6 -> 4`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B(_LIVE): 0.10 -> 0.15`

Verification:
- 15分窓で `STOP_LOSS_ORDER(long)` 件数を追跡。
- 15分窓で `submit/filled` の side 比（long:short）を追跡。
- 15分窓で `net_jpy` と `pf` を追跡。

Status:
- in_progress

## 2026-03-04 08:42 UTC / 2026-03-04 17:42 JST - `scalp_ping_5s_b` の過剰エントリー抑制（sell-only化なし）

Period:
- 変更時刻: 2026-03-04 08:42 UTC / 17:42 JST
- 対象: `ops/env/scalp_ping_5s_b.env`

Fact:
- 直前設定では `ENTRY_COOLDOWN_SEC=2.2`, `MAX_ORDERS_PER_MINUTE=6` で、過密エントリー継続時に SL 連打が再発しやすい構成。
- `SCALP_PING_5S_B_SIDE_FILTER=none` は維持されており、sell-only 固定ではない。

Failure Cause:
- エントリー間隔・分間発注上限・方向別圧縮下限がタイトで、短時間に同質シグナルが重なった際の連続損切りを抑え切れていない。

Improvement:
- `ops/env/scalp_ping_5s_b.env` を更新（重複キーなし、各キーは単一値で整理）。
  - `SCALP_PING_5S_B_ENTRY_COOLDOWN_SEC: 2.2 -> 2.8`
  - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE: 6 -> 4`
  - `SCALP_PING_5S_B_MAX_ACTIVE_TRADES: 2 -> 2`（据え置き明示）
  - `SCALP_PING_5S_B_MAX_ACTIVE_PER_DIRECTION: (未定義) -> 1`（明示追加）
  - `SCALP_PING_5S_B_DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT: 0.95 -> 1.00`
  - `SCALP_PING_5S_B_SIDE_BIAS_SCALE_FLOOR: 0.55 -> 0.45`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT: 0.60 -> 0.65`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT: 0.55 -> 0.60`

Verification:
- `rg` で対象8キーの最終値を確認し、同一キー重複がないことを確認する。
- `SIDE_FILTER=none` が維持され、sell-only 化していないことを確認する。

Status:
- in_progress

## 2026-03-04 04:30 UTC / 2026-03-04 13:30 JST - ローカルV2で `scalp_ping_5s_b_live` の敗因集中をデリスク（sell-only継続＋品質閾値引き上げ）

Period:
- 集計窓: 直近24h（`logs/trades.db`, `logs/orders.db`）
- 市況確認: `logs/tick_cache.json`（直近約38分）

Fact:
- 24h（`pocket <> manual`）: `n=80`, `net=-45.456 JPY`, `net_pips=-82.6`, `win_rate=16.25%`, `PF=0.3262`。
- 戦略寄与: `scalp_ping_5s_b_live` が同期間の実トレード全損益を占有（`n=80`, `net=-45.456 JPY`）。
- 注文内訳（24h）: `entry_probability_reject=376`, `filled=83`, `rejected=2`（`STOP_LOSS_ON_FILL_LOSS`）。
- 市況（直近tick）: `USD/JPY mid=157.333`, `spread_avg=0.8 pips`, `range(5m)=2.7 pips`, `range(15m)=9.2 pips`, `range(30m)=23.9 pips`。
- 実行品質（metrics直近1000点）: `decision_latency_ms p50=27.0 / p95=48.9`、`data_lag_ms p50=630.6 / p95=1566.2`（外れ値あり）。
- OANDA応答品質: workerログで pricing API `HTTP 200` 継続、`oanda.positions.error` は直近発生なし（最終 2026-02-21）。

Failure Cause:
- `scalp_ping_5s_b_live` の long 側でSLヒット連鎖が発生し、低エッジ・低品質シグナルの流入で期待値が崩れた。
- 併せてローカル起動導線でワーカープロセス重複/残骸が出る経路があり、実行品質の不安定化要因になっていた。

Improvement:
- `workers/scalp_ping_5s_b/worker.py`: `subprocess.run` 経路を `os.execvpe` に変更し、ラッパー子プロセス残骸を抑止。
- `scripts/local_vm_parity_stack.sh`: 広すぎる `pkill` を repo 配下限定へ縮小、`stop/status` の stale PID 判定を明確化。
- `ops/env/scalp_ping_5s_b.env`:
  - `MAX_ORDERS_PER_MINUTE 12 -> 8`
  - `BASE_ENTRY_UNITS 140 -> 90`
  - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY 0.60 -> 0.68`
  - `CONF_FLOOR 60 -> 72`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN 0.68 -> 0.74`
  - `ENTRY_PROBABILITY_BAND_ALLOC_HIGH_REDUCE_MAX 0.78 -> 0.65`
  - `ENTRY_NET_EDGE_MIN_PIPS 0.12 -> 0.20`
- `ops/env/local-v2-full.env`: parity系にも同方針（sell-only + 閾値強化 + サイズ縮小）を反映。
- `scripts/dynamic_alloc_worker.py` 再計算で `scalp_ping_5s_b_live lot_multiplier=0.45` を維持（攻めず縮小継続）。

Verification:
- 実プロセス環境で `SCALP_PING_5S_B_*` の新値反映を確認（`ENTRY_NET_EDGE_MIN_PIPS=0.20`, `CONF_FLOOR=72`, `SIDE_FILTER=sell` など）。
- `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` でV2群が `running`。
- 直近ログで `entry-skip summary` の long 側は `side_filter_block` が継続し、不要long流入を遮断していることを確認。
- 次判定条件（再評価）:
  - 24hで `PF >= 1.0` かつ `net_jpy > 0`
  - `entry_probability_reject` 比率の過剰偏り緩和（目安 `< 45%`）
  - `data_lag_ms p95 < 1200` を維持

Status:
- in_progress

## 2026-03-02 02:35 UTC / 2026-03-02 11:35 JST - `fx-trader*` 同時稼働の一時的収束

Period:
- 監査時点: 2026-03-02 02:00–02:35 UTC
- 対象コマンド: `scripts/ensure_single_trading_vm.sh`

Fact:
- `fx-trader*` 系の稼働インスタンスが `fx-trader-vm-es1a`, `fx-trader-c-repair`, `fx-trader-vm-es1c` で
  重複していた状態を確認。
- `--target=fx-trader-vm-es1a` で `ensure_single_trading_vm.sh` を実行し、非対象の稼働/起動中インスタンスを停止指示。
- 再確認時点（`--dry-run`）で `fx-trader-vm-es1a` のみが `RUNNING` と判定。
- `fx-trader-c-repair` は `TERMINATED`、`fx-trader-vm-es1c` は `STOPPING` -> `TERMINATED` へ移行中（運用監視対象）。

Failure Cause:
- 同時稼働を放置した状態が、compute 料金・I/O競合・復元監査ノイズを増加させ、月間コスト（約23,000円規模）悪化要因になり得る。
- BQ面はこの時点の主要原因ではなく、過去監査と合わせて「同時稼働 + 不要リソース残存」が主軸。

Improvement:
- `AGENTS.md` のセクション9を更新し、1台運用（RUNNING 1台）を運用条項化。
- 併せて `quant-bq-sync` の `--bq-interval 900`/`--disable-lot-insights` 方針と
  `ensure_single_trading_vm.sh --dry-run` を監査手順として固定化。

Verification:
- `gcloud compute instances list --filter='name~^fx-trader' --format='value(name,status,zone)'` で
  `RUNNING` が1件のみ。
- `./scripts/ensure_single_trading_vm.sh -p quantrabbit -m fx-trader-vm-es1a -P fx-trader --dry-run`
  が `OK` を返すこと。

Status:
- done

## 2026-03-04 (JST) / scalp_ping_5s_b を「売り限定解除 + 両方向適応」へ再調整（ローカルV2）

Source:
- `logs/local_v2_stack/quant-scalp-ping-5s-b.log`
- `logs/orders.db`
- `logs/trades.db`
- `curl http://127.0.0.1:8301/position/sync_trades`

Findings:
- `SCALP_PING_5S_B_SIDE_FILTER=none` + `ALLOW_NO_SIDE_FILTER=1` で固定sideは解除済み。
- 直近ログで `short` 候補はあるが `units_below_min` 比率が高く、実約定は long 偏重。
- 直近クローズでは `STOP_LOSS_ORDER` が連続し、反転局面で long 連打が残る。

Action:
- `ops/env/scalp_ping_5s_b.env` を更新（ローカル）
  - `ENTRY_COOLDOWN_SEC=2.2`（連打抑制）
  - `MAX_ORDERS_PER_MINUTE=6`（過剰回転抑制）
  - `CONF_SCALE_MIN_MULT=0.92`
  - `DIRECTION_BIAS_OPPOSITE_UNITS_MULT=0.68`
  - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.95`
  - `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.28`
  - `SIDE_BIAS_SCALE_FLOOR=0.40`
  - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MIN_MULT=0.60`
  - `FAST_DIRECTION_FLIP_DIRECTION_SCORE_MIN=0.58`
  - `FAST_DIRECTION_FLIP_HORIZON_SCORE_MIN=0.35`
  - `FAST_DIRECTION_FLIP_HORIZON_AGREE_MIN=2`
  - `SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.60`
  - `SIDE_ADVERSE_STACK_DD_MIN_MULT=0.65`
- `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env` で反映。

Verification:
- `orders.db` では `submit_attempt/filled` が継続して発生（エントリー導線は稼働）。
- ただし `short` 側の `units_below_min` は残存し、追加観測が必要。
- `position_manager` が断続的に再起動する時間帯があり、短時間で `open_positions` 呼び出し失敗が発生することを確認。

Next:
- `position_manager` の起動安定化を先に固定。
- その後 20〜30分窓で `short fill件数 / units_below_min比率 / PF` を再測定して再調整する。

## 2026-03-02 02:20 UTC / 2026-03-02 11:20 JST - `quant-bq-sync` / `quant-policy-cycle` で BQ 負荷を先回り抑制

Period:
- 対象:
  - `systemd/quant-bq-sync.service`
  - `systemd/quant-policy-cycle.timer`
  - `scripts/run_sync_pipeline.py`
  - `AGENTS.md` / `docs/GCP_PLATFORM.md`
- 根拠データ:
  - 過去監査での `quant-bq-sync` 実引数 (`--interval 60 --bq-interval 300 --limit 1200`) と
    `quant-policy-cycle.timer` (`15min`) の設定差分

Fact:
- 事後監査で、`quant-bq-sync` の BQ 同期が 5 分刻み（`--bq-interval 300`）で回る運用を確認し、`lot insights` 解析も毎サイクル有効な状態だった。
- `systemd/quant-bq-sync.service` を `--bq-interval 900 --disable-lot-insights` へ変更し、現在は
  - BQ エクスポートを 15 分間隔ベースまで緩和
  - ロットインサイトの毎回生成を停止
  - 既定で `--limit 1200` による送信上限を維持
  する運用へ更新した。
- `systemd/quant-policy-cycle.timer` を `15min` から `60min` に変更し、policy-cycle の重複実行圧を低下。
- `AGENTS.md` に 1 台運用・BQ 原価抑制の運用規則を明文化し、月間約 23,000 円要因が「同時 RUNNING VM」や未使用リソース固定化と合致する運用根拠へ寄せた。

Failure Cause:
- コスト上振れの主因は `BQ` 単独より、`run_sync` の高頻度実行と `fx-trader*` 複数台起動が重なった状態である可能性が高い。
- lot insights と policy-cycle の同時高頻度化により、DB I/O 競合・再処理量増大を誘発していた。

Improvement:
- `quant-bq-sync` の実行を `--bq-interval 900` / `--disable-lot-insights` へ変更。
- `scripts/run_sync_pipeline.py` を lot insights 無効時でも継続稼働できるガード実装へ修正。
- `quant-policy-cycle.timer` を 60 分周期化。
- 運用文書（`AGENTS.md`/`GCP_PLATFORM.md`）へ1台化・BQ抑制設定を反映。

Verification:
- 運用反映後に `systemd/quant-bq-sync.service` の `ExecStart`、`systemd/quant-policy-cycle.timer` の `OnUnitActiveSec` を確認。
- `gcloud compute instances list --filter='name~^fx-trader' --format='value(name,status)'` で RUNNING の `fx-trader*` が 1 台のみであることを確認。
- `systemctl cat quant-bq-sync.service` と `systemctl cat quant-policy-cycle.timer` を `ops` 監査ログへ保存。
- 24h 程度で `logs/{orders,metrics}.db` の拒否率と `B/Q` 再実行率（`run_sync_pipeline`/`googleapis`）を比較し、悪化がないことを確認。

Status:
- in_progress

## 2026-03-03 (JST) / 並行2レーン判定の運用化（min-trades導入）

Source:
- `scripts/compare_live_lanes.py`
- `scripts/watch_lane_winner.sh`
- `ops/env/local-llm-lane.env`

Fact:
1. 比較ロジックに `--min-trades` を追加し、サンプル不足時は `winner=insufficient_data` を返すよう変更。
2. ローカルレーンは `ops/env/local-llm-lane.env` をロードして local LLM（Ollama）前提で運用可能化。
3. `watch_lane_winner.sh` で比較結果を `logs/lane_winner_latest.json` と履歴へ出力可能。

Verification:
1. `python scripts/compare_live_lanes.py --hours 24 --min-trades 5`
2. `scripts/watch_lane_winner.sh`

Status:
- in_progress

## 2026-03-03 (JST) / ローカルLLM並行売買レーン導入（VM+Local 比較運用）

Source:
- OANDA API 実測（`scripts/check_oanda_summary.py` + 20回サンプリング）
- OANDA pricing/candles 実測（USD/JPY）
- VM health snapshot（`gs://fx-ui-realtime/realtime/health_fx-trader-vm-es1c.json`）
- VM serial（`gcloud compute instances get-serial-port-output fx-trader-vm-es1c`）
- local live log（`logs/codex_long_autotrade.log`）

Market Check (2026-03-03 22:00 JST 台):
1. USD/JPY: mid `157.756`, spread `0.8 pips`
2. M1 ATR14: `3.236 pips`
3. 直近レンジ推移: `last15m=50.2 pips`, `prev15m=54.5 pips`（比率 `0.921`）
4. OANDA API応答品質: 20/20 成功, `p95=315ms`, `max=345ms`
5. OANDA openTrades: manual short `-7000` のみ（bot open なし）

Fact:
1. VM側 health snapshot は `2026-03-02T09:10:55Z` で更新停止気味、bot lane の約定が薄い。
2. serial に `oom-kill`（`quant-scalp-false-break-fade` 系 python）を確認。
3. local lane は `codex_long_autotrade.log` で実売買中、`gpt-oss:20b` 判定は1回 `~27s` と高遅延。
4. 直近24h比較:
   - local lane: `1 trade / net -7.316 JPY / -5.4 pips`
   - vm lane: `0 trades`

Action:
1. BrainゲートへローカルLLM backend を追加（`BRAIN_BACKEND=ollama`）。
2. Brain失敗時ポリシーを `BRAIN_FAIL_POLICY=allow|reduce|block` で制御可能化。
3. `ORDER_MANAGER_BRAIN_GATE_APPLY_WITH_PRESERVE_INTENT` を追加し、必要時のみ preserve intent と Brain の併用を許可。
4. 2系統比較スクリプト `scripts/compare_live_lanes.py` を追加（local log + vm trades.db）。
5. まず `local lane` を実運用継続し、`vm lane` は OOM/SSH不安定解消後に再評価する方針へ。

Verification:
1. `python scripts/compare_live_lanes.py --hours 24` を定期実行し、`winner` を監視。
2. Brainをollamaで有効化する場合は `brain_latency_ms` と `order_brain_block` の比率を同時監査。
3. VM側は OOM 解消（unit整理/メモリ圧迫タスク抑制）後に `filled` 復帰を再確認。

Status:
- in_progress

## 2026-03-03 (JST) / no-entry継続への追加対応（ping5s閾値再緩和 + OOM要因記録）

Source:
- OANDA account summary (`2026-03-03T10:45:54Z`): `openTradeCount=1`, `lastTransactionID=413002`（manualのみ、bot新規約定なし）
- OANDA pricing/candles (`2026-03-03T10:23:56Z`): `USD/JPY bid=157.730 ask=157.738 spread=0.008`, `ATR14(M5)=0.0765`, `range_last_60m=0.313`, API latency `~220-250ms`
- VM serial (`gcloud compute instances get-serial-port-output`): `quant-scalp-false-break-fade.service` の `oom-kill` 発生（`2026-03-03 10:17:54Z` 付近）

Hypothesis:
1. ping5s B/C/D の `reject_under` と `min_units` が still strict で、`submit_attempt` に到達しないケースが残っている。
2. OOMイベントが発生した時間帯では、worker群の安定稼働が崩れて entry チャンスを取りこぼす。

Action:
- `ops/env/quant-order-manager.env`
  - `ORDER_SUBMIT_MAX_ATTEMPTS: 1 -> 2`
  - `ORDER_PROTECTION_FALLBACK_MAX_RETRIES: 0 -> 1`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B(_LIVE): 0.15 -> 0.10`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE): 0.15 -> 0.10`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_D(_LIVE): 0.12 -> 0.10`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE): 20 -> 10`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE): 20 -> 10`

Impact:
- manual建玉操作は未実施（`BLOCK_MANUAL_NETTING=0` 方針維持）。
- 変更は order_manager 側の preflight 緩和のみで、V2導線・entry_thesis契約は不変。

Verification (post-deploy):
1. OANDA `lastTransactionID` が `413002` から更新し、manual以外の新規open tradeが増えること。
2. `orders.db` で ping5s系の `submit_attempt_count` / `filled` が 0 から復帰すること。
3. OOM再発の有無（`quant-scalp-false-break-fade.service` の `oom-kill`）を serial/journal で監視すること。

Status:
- in_progress

## 2026-03-01 01:40 UTC / 2026-03-01 10:40 JST - `replay_quality_gate.py` のシナリオ同義語互換を `exit_workers_groups` と統一

Period:
- 対象:
  - `scripts/replay_quality_gate.py`
  - `tests/analysis/test_replay_quality_gate_script.py`
- 対象データ:
  - ユニットテスト観測（シナリオ名正規化）

Fact:
- `replay_quality_gate.py` 側で受理シナリオが旧仕様のままだと、`replay_exit_workers_groups.py` が新規対応した
  `trend_up` / `trend_down` / `gap` / `stale` などを指定できず、品質ゲート運用に分断が発生していた。
- `wide` / `uptrend` / `gapdown` / `stale_ticks` といった実運用で使われる別名が
  `replay_quality_gate.py` でも正規化されることで、シナリオ指定の往復が一貫化した。

Failure Cause:
- リプレイ品質評価（walk-forward）とシナリオ分類エンジンの許容名が不一致で、`--scenarios` の実運用運用系が断絶していた。

Improvement:
- `replay_quality_gate.py` のシナリオ正規化を拡張し、`SCENARIO_OPTIONS` を exit 側と同等に更新。
- 受理対象に `trend_up` / `trend_down` / `gap_*` / `stale` を追加。
- 同義語を一元変換する補助関数を追加。

Verification:
- `tests/analysis/test_replay_quality_gate_script.py::test_resolve_scenario_names_validates_supported_grouped_scenarios`
  で同義語を含む指定が期待 canonical 名へ変換されることを固定化。

Status:
- done

## 2026-03-01 01:25 UTC / 2026-03-01 10:25 JST - `replay_exit_workers_groups` のシナリオ網羅性を拡張（trend/gap/stale）

Period:
- 対象:
  - `scripts/replay_exit_workers_groups.py` の `_parse_scenarios`, `_build_tick_scenarios`
  - `tests/scripts/test_replay_exit_workers_groups.py`
- 対象データ:
  - スクリプト内ユニット合成データ（シナリオ分類の観測固定）

Fact:
- 従来の `wide_spread/tight_spread/high_vol/low_vol/trend/range` に加え、
  - `trend_up`/`trend_down`（方向別トレンド）、
  - `gap`/`gap_up`/`gap_down`（tick間急変）、
  - `stale`（tick間時間遅延）
  を付与できるように分類ロジックが拡張された。
- 同時に `uptrend`/`downtrend`/`stale_ticks`/`high_volatility` といった同義語がシナリオCLIへ反映される実装を追加。

Failure Cause:
- 先行のシナリオ選別が上下トレンド・価格ギャップ・欠落ティックに弱く、再現シナリオを運用的に増やしにくい状態だった。

Improvement:
- `_parse_scenarios` に同義語正規化を追加し、既存 `SCENARIO_OPTIONS` の拡張シナリオを受理。
- `_build_tick_scenarios` で tick ミリ単位のギャップ/遅延判定を導入し、方向別トレンドの分類を追加。
- `all + 指定シナリオ` でシナリオごとのフィルタ対象を維持したまま再実行可能にし、検証の粒度を上げる。

Verification:
- `tests/scripts/test_replay_exit_workers_groups.py` にシナリオ同義語展開と拡張フラグ分類の固定テストを追加。

Status:
- done

## 2026-03-01 01:05 UTC / 2026-03-01 10:05 JST - `replay_exit_workers_groups` のスキーマ拡張対応（`OPEN_*`, `signals`, `created_at`）

Period:
- 対象:
  - `scripts/replay_exit_workers_groups.py` の `_load_entries_from_replay`
- 根拠データ:
  - `tests/scripts/test_replay_exit_workers_groups.py` 追加シナリオ

Fact:
- `action` が `OPEN_LONG` / `OPEN_SHORT` 形式、または `signals` / `entries` 配下に格納される形式で、または
  `created_at` / `entry` / `entry_px` / `units_signed` / `target_pips` を用いた形式の
  リプレイ行が観測された。
- `trades` 固定キー前提のままだと、旧実装で取りこぼし増加と不要なスキップが発生しうる。

Failure Cause:
- リプレイ生成経路やバージョン差分により、`entry` フィールド名・方向文字列・時間キーが揺れるのに対し
  受け口側の許容幅が不足していた。

Improvement:
- `_load_entries_from_replay` で受け口を拡張。
  - `trades` 未定義時に `entries`/`signals`/`actions`/`data` を探索。
  - `OPEN_*` 系アクションを方向として解釈。
  - `created_at`, `entry`, `entry_px`, `units_signed`, `take_profit`, `stop_loss`, `tp_distance`, `sl_distance`, `target_pips` を受理。
- 既存 `tp_price`/`sl_price` 優先順序と `tp_pips`/`sl_pips` 補完の方針を維持したまま、別名キーへの後方互換を追加。
- `units_signed` が負数でも `abs` 変換後に受理する前提を明記。方向キーとの整合で、符号付き数量由来のエントリー拒否を回避。

Verification:
- 追加テストで `OPEN_LONG` / `signals` / `created_at` / `units_signed` ケースを通過したことを確認。

Status:
- done

## 2026-03-01 00:45 UTC / 2026-03-01 09:45 JST - `replay_exit_workers_groups` の入力パース脆弱性是正（壊れたレコードでリプレイ停止を防止）

Period:
- 対象:
  - `scripts/replay_exit_workers_groups.py` の `replay_workers_*` 出力パース
- 根拠データ:
  - スキーマ揺れ（`entry_time`/`open_time`/`time`/`ts`/`timestamp`、`direction`/`side`/`action`）を含むローカル再現ログ
  - `tests/scripts/test_replay_exit_workers_groups.py` に追加した再現データ

Fact:
- `replay_workers_*_*.json` が 1 件の不正行を含むと、旧ローダーは `ValueError` で全件停止しやすく、再現レポートが生成不能になる状態を確認。
- 同じ入力で、`entry` スキーマのキー差分を吸収しない trade が混在しても、妥当行は継続して読み込めるようになった。
- `_parse_dt` が文字列エポック値と `ms` 単位数値にも対応し、時刻欠損行をスキップしながら継続処理できることを確認。

Failure Cause:
- `entry` 取得時に `entry_time` と `direction` と `entry_price` を固定キー前提で扱っていたため、仕様差分や欠損行で即停止する設計だった。

Improvement:
- `_load_entries_from_replay` を以下の方針に修正:
  - JSON パース失敗は空配列で継続。
  - `trades` を dict/list どちらでも受ける。
  - `entry` フィールドキーを `entry_time`/`open_time`/`time`/`ts`/`timestamp` 等へ拡張。
  - 方向/価格/数量の異形キーへ柔軟対応し、必須判定を保ちながら不正行をスキップ継続。
  - `tp_price`/`sl_price` と `tp_pips`/`sl_pips` を併用可能にし、`entry` 受理を阻害しない。
  - 時刻を UTC 正規化し、文字列/数値エポック（ms含む）を許容。
- `tests/scripts/test_replay_exit_workers_groups.py` を新規追加し、壊れた入力での継続挙動を固定化。

Verification:
- `pytest -q tests/scripts/test_replay_exit_workers_groups.py`
- 同期間の replay パイプラインで `summary_all.json` の生成中断件数が減少することを確認（運用環境再実行時に比較）。

Status:
- done

## 2026-03-01 00:30 UTC / 2026-03-01 09:30 JST - 収益阻害Top3（prob/perf/intent reject）に対する縮小継続チューニング適用

Period:
- 監査対象:
  - VM実データ `logs/orders.db`, `logs/trades.db`, `logs/metrics.db`, `journalctl`
  - 期間: 24h / 72h / 168h（JST 7-8時はメンテ時間として評価除外）

Fact:
- 24h収益は `PF=0.727`, `net=-737 JPY`。
- 168hでは `net=-31,346.5 JPY`、赤字寄与上位は `MicroPullbackEMA`, `scalp_ping_5s_c_live`, `scalp_ping_5s_b_live`。
- 直近5000 ordersで reject/guard 偏在:
  - `entry_probability_reject=609 (12.18%)`
  - `perf_block=588 (11.76%)`
  - `entry_intent_guard_reject=200 (4.00%)`
- `latency_preflight` の長い尾（p95級）と `fetch_recent_trades timeout(8s)`、`order-manager child died` を同時観測。

Failure Cause:
1. B/C戦略の `entry_probability` / perf guard / preserve-intent の閾値が重なり、同一経路で reject が常態化。
2. `scalp_ping_5s_b_live` の preserve-intent scale 範囲に逆転値（`MIN_SCALE > MAX_SCALE`）が存在し、縮小ロジックの意図維持を阻害。
3. V2 runtime に `WORKER_ONLY_MODE=true` と `MAIN_TRADING_ENABLED=1` が同居し、導線方針に矛盾。
4. position/order 周辺 timeout が長く、preflight遅延時の fail-fast が弱い。

Improvement:
- `ops/env/quant-v2-runtime.env`
  - `MAIN_TRADING_ENABLED=0`、`ORDER_MANAGER_SERVICE_FALLBACK_LOCAL=0`、`POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=0`。
  - order/position timeout を fail-fast 側へ調整。
  - B/C の forecast/net-edge/perf 閾値を 1 段緩和し、`hard block` 常態化を抑制。
- `ops/env/quant-order-manager.env`
  - `ORDER_MANAGER_SERVICE_FALLBACK_LOCAL=0`。
  - B の `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE` を `0.30 -> 0.85` へ修正（逆転解消）。
  - B/C の `reject_under`, `min_units`, perf/forecast 閾値を再調整。
- `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
  - B/C の base units を軽く抑え、probability floor/perf guard 閾値を再調整。

Verification:
- デプロイ後24hで以下を同条件再監査:
  - `orders.db` 直近5000件で
    - `entry_probability_reject <= 5%`
    - `perf_block <= 5%`
    - `entry_intent_guard_reject <= 1%`
  - `metrics.db` で `latency_preflight p95 < 2000ms`（取得可能な同等指標で代替可）
  - `trades.db`（non-manual）で `PF > 1.00`, `net_jpy > 0`

Status:
## Entry Template
```

## 2026-02-28 23:55 UTC / 2026-02-29 08:55 JST - EXIT共通制御で strategy_tag 欠損建玉が制御外を通過しうる事象を確認（2/24 追跡）

Period:
- 監査対象:
  - VM実データ `logs/orders.db`, `logs/trades.db`（2026-02-24 UTC中心）
  - `execution/order_manager.py` の実装査読・監査スクリプト再計算

Fact:
- `orders.db` で `status="strategy_control_exit_disabled"` が 10,277 件確認（主として2/24 UTC 02:00-07:00）。
- 同期間で閉鎖されていない/反復失敗 trade の主要 13件（`384420`,`384425`,`384430`,`384435`,`384797`,`384807`,`384812`,`384920`,`385300`,`385303`,`385332`,`385337`,`385390`）を確認。
- 各tradeの `close_request`→`close_ok` 移行が遅延し、長時間同一建玉の再試行が連続。
- `close` パスには `strategy_tag` 必須拒否が無く、欠損時は `strategy_control` が事実上スキップされる経路が残存。
Failure Cause:
1. `order_manager._reject_exit_by_control` が `strategy_tag` 未定義時に即時許可していたため、共通 EXIT ガードが欠損建玉で適用されない。
2. 2/24の阻害再発は、該当戦略の `strategy_control_exit_disabled` 連打と、閉鎖遅延の長期化を伴って発生。
Improvement:
- `execution/order_manager.py` の `_reject_exit_by_control` を、`global_exit`/`global_lock` を先頭で適用する形へ修正。
- `close_trade` に `strategy_tag` 欠損時の明示 `close_reject_missing_strategy_tag` を追加。
Verification:
- VM `orders.db` で `close_reject_missing_strategy_tag` の新規件数を監視。
- VM側で同一期間を再実行し、`close_request=close_reject` の主要 13取引が strategy_control への制御で停止し続けないことを確認。
- `strategy_control_exit_disabled` 連打抑制（閾値到達時の failopen / bypass）導線が再開することを確認。
Status:
- in_progress

### 2026-03-02（追記）position_manager open_positions タイムアウトの再発抑止

Source:
- `logs/journal`（VM）
- `execution/position_manager.py`
- `ops/env/quant-v2-runtime.env`

Fact:
- `open_positions` の呼び出しで `Read timed out`（`read timeout=4.0/6.5`）が継続観測され、シグナルを拾い続ける前提が崩れていた。

Failure Cause:
- タイムアウト設定が `open_positions` 経路で短く、`position_manager` からの応答前に上位が切り戻されるループが発生していた。

Action:
- `POSITION_MANAGER_SERVICE_OPEN_POSITIONS_TIMEOUT: 4.0 -> 8.0`
- `POSITION_MANAGER_HTTP_TIMEOUT: 5.0 -> 8.0`
- `POSITION_MANAGER_OPEN_TRADES_HTTP_TIMEOUT: 2.8 -> 8.0`
- `POSITION_MANAGER_WORKER_OPEN_POSITIONS_TIMEOUT_SEC: 4.0 -> 10.0`

Next check:
- VM 反映後 15〜30 分で `position_manager service call failed` のログ頻度、`entry_probability_reject` 以外の拒否率変化、`trades.db` の filled 再開状況を確認。

## 2026-02-28 23:40 UTC / 2026-02-29 08:40 JST - `orders_snapshot_48h.db` の鮮度差起因監査崩れに対する同梱/監査ロジック同時修正

Period:
- 対象:
  - `scripts/collect_gcs_status.py`
  - `ops/env/quant-core-backup.env`
  - `execution/position_manager.py`
  - `remote_logs_current/core_extract` 同一時点抽出

Fact:
- `core` バックアップ運用は、`QR_CORE_BACKUP_INCLUDE_ORDERS_DB=0` のため監査に `orders.db` が同梱されず、代替の
  `orders_snapshot_48h.db` が長期更新滞留して `trades` 時系列と `orders` 時系列のズレが発生。
- 追加実装で `collect_gcs_status` は `core_*` ファイル名（`.tar`/`.tar.gz`）対応を維持しつつ、`orders_db_source` と
  `orders_snapshot_age_vs_trades_h` / `orders_snapshot_freshness` を出力し、監査時に鮮度劣化を数値化できる状態に変更。
- `execution/position_manager.py` の `_normalize_entry_contract_fields` は、`payload.units` を含む経路を追加。

Failure Cause:
1. 監査再現では注文DB更新タイムラインを `trades` と同居させる同梱方針が不足していた。
2. 過去世代 `entry_thesis` 再構成では payload 欠損が混在し、監査キー注入が不十分な経路が残っていた。

Improvement:
- `ops/env/quant-core-backup.env` を更新し、`QR_CORE_BACKUP_INCLUDE_ORDERS_DB=1` へ固定。
- `collect_gcs_status.py` に注文DB鮮度の警告分類（`warning` / `critical`）を追加。
- `position_manager` の `entry_units_intent` 補完に `payload.units` を追加し、`fallback_units` 経路を維持。

Verification:
- VM同時取得された `core_*.tar(.gz)` を前提に、`collect_gcs_status` が `orders_db_source`、`orders_snapshot_freshness` を返すことを確認。
- `collect_gcs_status` の判定基準:
  - `orders_snapshot_freshness != critical` かつ `orders_db_source` が `orders.db` の場合に監査結果を採用。

Status:
- in_progress

## 2026-02-28 22:10 UTC / 2026-02-29 07:10 JST - リプレイID `sim-*` の閉鎖失敗を Exit 共通拒否ではなく入力ID不正で遮断へ変更

Period:
- 監査対象:
  - `logs/orders.db`（close 関連ログ）
  - `scripts/replay_exit_workers.py` / `execution/order_manager.py`

Fact:
- close 系の再現ログで `trade_id` が `sim-40`,`sim-8`,`sim-37`,`sim-4` のみを対象に `close_request` と `close_failed` が連続。
- 失敗コードは `oanda::rest::core::InvalidParameterException`、`Invalid value specified for 'tradeID'`。
- 同期間の close は共通拒否系（`close_blocked_by_strategy_control`/`close_reject_*`）の増加が確認されず、リプレイ起因の ID 形式不整合が主因と判断。

Failure Cause:
- リプレイ側で `trade_id` が `sim-*` 形式のまま `order_manager.close_trade` に渡され、`client_order_id` が `sim-sim-*` まで二重化されるケースが残存。

Improvement:
- `execution/order_manager.py` に live tradeID 形式バリデーションを追加し、`close_trade` で `sim-*` 等の非数値 ID を `close_reject_invalid_trade_id` で即時停止。
- `scripts/replay_exit_workers.py` で `sim-` 再付与を抑止（既に `sim-*` の場合はそのまま利用）して `client_order_id` の `sim-sim-*` 化を防止。

Verification:
- 直近時点の `orders.db` では `close_reject_invalid_trade_id` 件数と `trade_id` 非数値拒否ログの出現を確認し、`sim-*` への CLOSE_REQUEST が消失しているかを観測。
- 併せて V2 上で `quant-order-manager` 側が同変更を採用した後、同等条件で close が再度失敗しないかを VM 実行で確認。

Status:
- in_progress

## 2026-02-28 21:10 UTC / 2026-02-29 06:10 JST - ローカルDB再集計で契約欠損は `entry`/`trade` の同世代不一致が主因と判定

Period:
- 対象:
  - `logs/trades.db`
  - `logs/orders.db`
  - `tmp/vm_audit_20260226/trades.db`
  - `tmp/vm_audit_20260226/orders.db` (開封不可; malformed)

Fact:
- `logs/trades.db` を `open_time` 基準で再集計（`entry_probability`,`entry_units_intent` の必須キー）:
  - 24h: `rows=101 / missing_any=101`
  - 48h: `rows=168 / missing_any=168`
  - 240h: `rows=275 / missing_any=275`
  - いずれも `entry_thesis` 自体は存在しているが、対象キーは未挿入。
- `logs/orders.db` 時系列:
  - `ts` 範囲: 2025-05-30 ～ 2026-02-24
  - 2026-02-24 時点以降は `close_request/close_failed` が中心で、`submit_attempt` の `side/instrument/request` エントリー路線は含まれず、`trades` 側 2/27 の新規エントリーと非整合。
  - `submit_attempt` (`n=2907`) 内訳:
    - `entry_thesis` あり: 1840件（`entry_probability` と `entry_units_intent` は 1840 件とも欠損）
    - `entry_thesis` なし: 1067件
- `tmp/vm_audit_20260226/orders.db` は開封時点で `sqlite_error: database disk image is malformed`。

Failure Cause:
1. `trades` の監査対象窓と `orders` の有効窓が同世代化されておらず、再構成が成立しない。
2. 2026-02-24 以降の `orders` 側は close 系に偏り、entry 系の同窓口が欠落。
3. VM への接続が `Connection timed out during banner exchange` / `port 65535` で停止し、最新同世代ログを再取得できない。

Improvement:
- `orders.db` と `trades.db` の同世代再取得が先決条件であることを明文化し、`entry_thesis` 欠損の再評価はそれ後に再実行。
- `entry_thesis` は `logs` 側では `entry_probability:1.0` / `entry_units_intent:abs(units)` で補完再構成可能だった領域と、`request_json` 未持ち込み等で不可の領域を分離して監査継続。

Verification:
- ローカル再現コマンド:
  - `python3 - <<'PY'` で `logs/trades.db` の `open_time` 窓別欠損集計を再計測。
  - `python3 - <<'PY'` で `logs/orders.db` の `status/request_json/entry_thesis` 欠損別分解を再計測。
- 本番側同世代再取得成功後:
  - `gcloud compute ssh fx-trader-vm --tunnel-through-iap --command "echo ok"`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --window-hours 240 --limit 20000 --json ...`

Status:
- in_progress

## 2026-02-28 18:55 UTC / 2026-02-28 03:55 JST - `entry_thesis` 監査は同世代再取得未達で確定

Period:
- 再検証対象:
  - `logs/trades.db` / `logs/orders.db`
  - `tmp/vm_audit_20260226/trades.db` / `tmp/vm_audit_20260226/orders.db`
  - `remote_tmp/trades.db` / `remote_tmp/orders.db`
  - `remote_logs_current/core_extract/trades.db` / `remote_logs_current/core_extract/orders.db`
- コマンド:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --window-hours 240 --limit 20000 --json --trades-db ... --orders-db ...`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --window-hours 48 --limit 20000 --json --trades-db logs/trades.db --orders-db logs/orders.db`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --window-hours 24 --limit 20000 --json --trades-db logs/trades.db --orders-db logs/orders.db`

Fact:
- `logs/trades.db`:
  - `window=240h`: `rows_missing_any=274`、`sampled_rows_in_window=275`
  - `window=48h`: `rows_missing_any=100`、`sampled_rows_in_window=101`
  - `window=24h`: `trades_no_samples_in_window`
  - `orders` 側は `window=240h` で `sampled_rows_in_window=344` なのに `evaluated_rows=0`（`submit_attempt` 参照で再構成不能）
- `tmp/vm_audit_20260226`:
  - `trades`: `rows_missing_any=2740`（`window=240h`）
  - `orders`: `sqlite_error: database disk image is malformed`
- `remote_tmp`: `no sample in requested window`
- `remote_logs_current/core_extract`: `trades`/`orders` のどちらも対象窓に未到達（`no samples in window`）
- 追加観測:
  - `df` 空き不足（`df` 結果が 100% 過負荷状態）で `gcloud` 実行やコピー前にクリーンアップが必要
  - VM 接続:
    - `Permission denied (publickey)` が継続
    - `Connection closed by UNKNOWN port 65535`（鍵指定実行でも再現）
    - シリアル出力で `sshd` と Python が OOM kill される痕跡を確認

Failure Cause:
1. `orders` / `trades` の同一時刻同一世代ペアが取れないため、監査の比較軸がずれている。
2. VM への SSH/OAuth 経路（OS Login / key）と VM 側資源状態（sshd OOM/再起動）が不安定で、同時データ取得不能。
3. ローカル側ディスク空き不足（`No space left on device`）で作業環境ノイズが混入。

Improvement:
- VM 接続安定化（OS Login/SSH 鍵経路、sshd メモリ負荷の是正）を優先して同一時刻 `trades.db` + `orders.db` を再取得し、現地同時監査に入る。
- 監査は `raw_missing` と `recoverable_missing` を分離して記録し続ける運用を維持し、同世代再取得後に `rows_missing_contract_fields=0` 到達を再確認。

Verification:
- 同一世代取得が可能になった時点で以下を実行:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --trades-db logs/trades.db --orders-db orders.db --window-hours 240 --limit 20000 --json`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --trades-db logs/trades.db --orders-db orders.db --window-hours 24 --limit 20000 --json`
- VM 障害確認:
  - `gcloud compute ssh fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --tunnel-through-iap --command "echo ok"`
  - `gcloud compute instances get-serial-port-output fx-trader-vm --project=quantrabbit --zone=asia-northeast1-a --port=1`

Status:
- in_progress

## 2026-02-28 20:40 UTC / 2026-02-29 05:40 JST - 過去DB再現で `entry_probability` / `entry_units_intent` は補完で吸収、欠損自体は主に旧世代要因

Period:
- 対象DB（`--window-hours 240`, `--window-hours 100000`, `--limit 20000~30000`）:
  - `logs/trades.db` / `logs/orders.db`
  - `remote_logs_current/core_extract/trades.db` / `remote_logs_current/core_extract/orders.db`
  - `tmp/vm_audit_20260226/trades.db` / `tmp/vm_audit_20260226/orders.db`
  - `tmp/qr_gcs_restore/core_20260227T062401Z/trades.db`
  - `tmp/qr_gcs_restore/multi/core_20260227T054231Z/trades.db`
  - `tmp/qr_gcs_restore/multi/core_20260227T050329Z/trades.db`
  - `tmp/qr_gcs_restore/multi/core_20260227T060325Z/trades.db`

Fact:
- `check_entry_thesis_contract`（厳密監査）での集計:
  - `logs/trades.db` (`240h`) `trades_missing_contract_fields=274`
  - `logs/trades.db` (`100000h`) `trades_missing=12049`, `orders_missing=3350`
  - `remote_logs_current/core_extract` (`100000h`) `trades_missing=6576`, `orders_missing=20392`（`orders` 側は 287,835 件をサンプリング）
  - `tmp/vm_audit_20260226` (`100000h`) `trades_missing=4739`
  - `tmp/qr_gcs_restore/*` (`100000h`) `trades_missing=4740`（`orders` は同DBに未同梱）
- 補完シミュレーション（`entry_probability=1.0`、`entry_units_intent=abs(units)`）を適用すると、上記 `trades` 群はほぼ `raw_missing` から回収可能（`recoverable` 化）と確認。
- `orders` 側は `request_json` 欠損/不完全が支配的で、`remote_logs_current` では `raw_missing 278,814` に対し `recovered 8,813`（`unrecoverable 270,001`）と再現。

Failure Cause:
1. 過去世代の `trades` は、保存時点で必須キー注入がなかったため監査で欠損扱い。
2. `orders` は世代差分や `request_json` 欠損の影響が大きく、再構成で contract 欠損を回収できない行が多数。

Improvement:
- `execution/position_manager.py` の保存前補完ロジック自体は新規データの再発抑制効果が高いため、現行運用は同実装を前提に継続。
- 監査運用として、`raw_missing` と `recoverable_missing` の分離レポートを採用し、真に改善対象が残る行だけを精査。
- 旧世代 `logs` ではなく VM 同期新世代（`logs/trades.db` + `logs/orders.db` 同時更新）で再監査し、監査差分の消失を確認。

Verification:
- 厳密監査:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --trades-db ... --orders-db ... --window-hours 240 --limit 20000 --json`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --trades-db ... --orders-db ... --window-hours 100000 --limit 30000 --json`
- 補完シミュレーション（レガシーデータ検証）を実行済み。

Status:
- done

## 2026-02-28 19:00 UTC / 2026-02-29 04:00 JST - `position_manager` の監査復元で `entry_thesis` 欠損を縮小する対策

Period:
- ローカルVM `logs/orders.db` / `logs/trades.db`
- 検証ウィンドウ: `--window-hours 240`

Fact:
- 直近再検証:
  - `trades` 直近 275 行中 274 行が `entry_probability` / `entry_units_intent` 欠損。
  - `orders` 直近 344 行中、`sampled_rows_in_window` は 344、`evaluated_rows=0`（`submit_attempt` が同窓口に存在せず再構成不能）。
  - `sqlite3 ...` の `ATTACH` 照合では `trades` 直近 `client_order_id` 274 件の `orders` 参照一致は 0。
- `gcloud compute ssh fx-trader-vm --tunnel-through-iap` は依然として `Permission denied (publickey)`。

Failure Cause:
1. `position_manager` の `entry_thesis` は、`orders` 由来の再構成に失敗した経路では永続化時にデフォルト補完が十分に入らず、監査レベルで `trades` が欠損扱いになる構成だった。
2. VM 側 `orders.db` 側の同一世代 `submit_attempt` がローカル窓に入らず、監査再構成が成立しない。

Improvement:
- `execution/position_manager.py`:
  - `_get_trade_details_from_orders` 取得後、`_normalize_entry_contract_fields` を通して `entry_probability` / `entry_units_intent` を補完。
  - `_get_trade_details`（OANDAフォールバック）と `open_positions` 形成経路でも同補完を適用し、保存・監査で最終 `entry_thesis` を契約形に寄せる。
- 併せて、監査再計算対象の指針を `TRADE_FINDINGS` に 1 箇所集約。

Verification:
- 追加後に以下を実行済み:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 240 --limit 20000 --json`
  - `sqlite3 logs/orders.db` / `sqlite3 logs/trades.db` で `client_order_id` 相互照合（`ATTACH`）
- 完全再現は VM 側同一世代取得ができた時点で `rows_missing_any=0` 目標で再確認。

Status:
- in_progress

## 2026-02-28 19:15 UTC / 2026-02-29 04:15 JST - `position_manager` 永続保存時に `entry_thesis` を保存前補完

Period:
- ローカルVM `logs/trades.db` / `logs/orders.db`
- 検証: `--window-hours 240`, `--window-hours 100000`（`check_entry_thesis_contract`）
- 追加検証: トレード行を `entry_probability`/`entry_units_intent` を
  `_normalize_entry_contract_fields` 的規則で再構成した場合のシミュレーション

Fact:
- 変更前後で `execution/position_manager.py` の `_parse_and_save_trades` と `_get_trade_details` に
  保存前補完を追加。
- 追加直後に再実行したスクリプト:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 240 --limit 20000 --json`
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 100000 --limit 20000 --json`
  いずれも既存DBが旧世代であるため `trades_missing_contract_fields` は継続し、`orders` 側も `entry_...` 欠損を保持。
- ただし、`trades` 生データ（12000件）を上記再構成規則で再評価した結果、`entry_probability`/`entry_units_intent` 欠損は 0 件に収束（将来保存レコードの再発防止効果を示唆）。

Failure Cause:
- 直近ローカル `trades.db` は `entry_thesis` 自体が監査前提キーを欠いた履歴を持ち、`orders.db` との同一世代参照が不十分。
- したがって本番実行データの「保存時点」での補完を追加しても、過去データの監査結果は同時に変わらない。

Improvement:
- `execution/position_manager.py`
  - `trades` 永続化ループ内で `details["entry_thesis"]` を契約正規化。
  - OANDAフォールバック取得 (`_get_trade_details`) でも再正規化し、上流入力が不足していても保存前に契約形へ寄せる。
- VM 取得データが更新された後は、同スクリプトで `trades_missing_contract_fields=0` 到達を再確認することを最終目標として追跡。

Verification:
- 同一ファイル内 `saved_records` 生成前後で保存行 `json.dumps(details["entry_thesis"])` が必ず契約キーを含む形になることをローカル確認。
- 運用上は次の `checks` 再実行時（VM新世代）にて `trades_missing_contract_fields` が解消されるかを監視。

Status:
- in_progress

## 2026-02-28 09:02 UTC / 2026-02-28 18:02 JST - `entry_thesis` 欠損監査: ローカル世代断絶と過去断面の全面欠損を再確認

Period:
- ローカルVM `logs/orders.db` / `logs/trades.db`（`--window-hours 240`）
- 過去スナップショット `remote_logs_current/core_extract/orders.db` / `remote_logs_current/core_extract/trades.db`（`--window-hours 100000`、`check_entry_thesis_contract`）

Fact:
- `check_entry_thesis_contract`（ローカル）:
  - `overall=fail`
  - `trades_missing_contract_fields:274`
  - `trades` `evaluated_rows=274`（`sampled_rows_in_window=275`）
  - `orders` `sampled_rows_in_window=344` のうち `evaluated_rows=0`（`orders` 側は `submit_attempt` がローカル内で `window` に入る世代が `trades` と照合不可）
- ローカル `logs/orders.db` と `logs/trades.db` の `client_order_id` 再構成:
  - `client_order_id` を `qr-` で集約した `trades` 全体一致（`105/...`）はありうる一方、`240h` 窓での `submit_attempt` への一致は `0/255`（一致なし）。
  - `logs/orders.db` の `MAX(ts)` は `2026-02-24T02:09:44Z`、`logs/trades.db` の `MAX(open_time)` は `2026-02-27T00:23:46Z` で、3日超の世代ズレ。
- `remote_logs_current/core_extract`（過去断面）を同スクリプトで検証:
  - `trades_missing_contract_fields:6576 / 6576`
  - `orders_missing_contract_fields:22180 / 40000(対象)`
  - `orders` 側 `top_strategies` 上位多数で `entry_probability` / `entry_units_intent` 不在、`entry_...` 欠損は運用開始初期断面の系統的欠落と一致。

Hypothesis:
1. 直近 240h のローカル監査失敗は `trades` 側の新規世代と `orders` 側断面の世代切替が崩れたことが主因で、検証条件として再構成不能。
2. `remote_logs_current/core_extract` が示す長期 `100000h` 断面では `entry_probability` / `entry_units_intent` が未注入状態で生成されており、欠損自体は「新規機能未反映」の履歴的傾向も確認済み。

Failure Cause:
- `entry_thesis` 必須フィールド注入前の断面が複数存在。
- ローカル `orders.db`/`trades.db` が同期世代を跨いでいるため、`trades` → `request_json` 再構成導線で `orders` を使う監査が成立しない。

Improvement:
- 今回の検証結果は、現時点のローカル断面では「監査基準を満たす再構成不能」扱いとし、VM 側で同一世代を取り直した上で再測定する前提で運用に反映。
- `order_manager` / `position_manager` 側の補完ロジックは維持し、`orders` 側が新世代で欠損が続く場合は追加にて全呼び出し経路の `entry_thesis` 注入箇所を再監査する。

Verification:
- 直近実行:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 240 --limit 20000 --json`
  - `sqlite3 logs/orders.db` / `sqlite3 logs/trades.db` の `client_order_id` 世代一致率集計（`ATTACH` で相互照合）
  - 過去断面再現:
    - 上記リポジトリを一時ディレクトリでリンクし、
    `worker` / `execution` と `remote_logs_current/core_extract` を使って同スクリプト再実行

Status:
- done

## 2026-02-28 17:50 UTC / 2026-02-29 02:50 JST - `entry_thesis` 欠損監査: logs 世代不整合確認

Period:
- ローカルVM `logs/orders.db` / `logs/trades.db`（`window_hours=240`）
- 補助検証: `remote_logs_current/core_extract/orders.db` / `remote_logs_current/core_extract/trades.db`

Fact:
- `qr-entry-thesis-contract-check`（`--window-hours 240`）で `trades_missing_contract_fields=274` を再現。
- `logs/trades.db` は `qr-177215...` 世代（2026-02-27付近）を持つ一方、`logs/orders.db` の `qr-` 世代は `qr-176...`（2025-12-10まで）で世代がずれている。
- `logs/orders.db` で `entry_probability`/`entry_units_intent` 同居は 0 件。`remote_logs_current` でも同条件 0 件。

Failure Cause:
1. `orders.db` / `trades.db` の時系列同期ズレにより、trade側再構成の照合対象が欠落。
2. 該当 logs は旧仕様断面で、`entry_thesis` 必須フィールドが履歴に未付与。
3. VM再取得確認は `Permission denied (publickey)` と `GlobalPerProjectConnectivityTests` クォータ上限で遅延。

Improvement:
- まず VM 側で同一世代の `orders.db` / `trades.db` を取得し、同仕様の断面で監査を再実行。
- 本変更分の `order_manager` / `position_manager` 反映後、`rows_missing_any=0` 到達を目標に再計測。

Verification:
- 再監査コマンド:
  - `python3 ~/.codex/skills/qr-entry-thesis-contract-check/scripts/check_entry_thesis_contract.py --repo-root . --window-hours 240 --limit 20000 --json`
  - `gcloud compute ssh fx-trader-vm --tunnel-through-iap --command "..."`（認証確認後）

Status:
- in_progress

## 2026-02-28 08:50 UTC / 2026-02-28 17:50 JST - `entry_thesis` 欠損補完フェイルセーフを拡張（order/position）

Period:
- ローカルVM `logs/orders.db` / `logs/trades.db`（`window_hours=240`）
- 対象: `entry_probability` / `entry_units_intent` 未補完件

Fact:
- `execution/order_manager.py` で `ORDER_MANAGER_ENTRY_PROBABILITY_DEFAULT` を追加し、`_ensure_entry_intent_payload` に `entry_probability` 補完を追加。
- `execution/position_manager.py` に `entry_thesis` 欠損時復元ヘルパーを追加し、`request_json.entry_probability`、`request_json.entry_units_intent`、`request_json.oanda.order.units`、`orders.units` を順に補完。
- 直近240h監査で `trades` 側の欠損は 274 件。`orders` 側は同 window で `submit_attempt` が存在するが（`MAX(ts)=2026-02-24 02:09:44`）、`trades` 連携先 `client_order_id` 参照で `rows_missing` が解消されず再構成不能の状態を確認。

Failure Cause:
- `trades` の多くは `client_order_id` 解決しても当該 `submit_attempt` ログが window 内に存在せず、履歴補完の再構成ソース欠損。

Improvement:
- `order_manager`/`position_manager` の双方でフェイルセーフ補完を追加し、将来の約定履歴に対して `entry_thesis` 要件を持ち越す。
- 監査は VM 実データで `submit_attempt` を再現できる周期へ遷移させ、欠損率を再測定する前提で再実行する。

Verification:
- `python3 -m py_compile execution/order_manager.py execution/position_manager.py` 通過。
- `check_entry_thesis_contract.py --window-hours 240` は現在のローカル再現データ上では `trades_missing_contract_fields=274` を返却。`orders.db` の `submit_attempt` が `2025-11-25` までで、再構成不能区間を含むため、VM 時系列再確認が必要。
- 併せて `gcloud compute ssh fx-trader-vm --tunnel-through-iap --command "sqlite3 /home/tossaki/QuantRabbit/logs/orders.db ..."` で VM 側 `submit_attempt` の時系列を再確認する。

Status:
- in_progress

## 2026-02-28 23:42 UTC / 2026-02-29 08:42 JST - `strategy_control_exit` 同一建玉の再試行キーを `trade_id` 優先へ固定

Period:
- VM実測（`logs/orders.db` / `logs/trades.db`）の 2026-02-24 01:00 UTC〜10:00 UTC を再監査。
- 併せて `execution/order_manager.py` の `strategy_control` 失効導線を確認。

Fact:
- `orders` で `status='strategy_control_exit_disabled'` が 10,277 件確認。
- 集約は 4 つの `trade_id`（`384420`,`384425`,`384430`,`384435`）が中心で、各 `2,044`〜`2,045` 件の連続阻害が発生。
- 同時間帯の `close_request=0` が 11 取引で、阻害後の `close_ok` 取得がほぼ無く、`close_bypassed_strategy_control` / `strategy_control_exit_failopen*` が出力されなかった。
- VM `ops/env/quant-v2-runtime.env` では `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_*` は有効（threshold=3, window=20, reset=180, emergency_only=0）。

Failure Cause:
1. `order_manager.py` の再試行キーが `client_order_id` 優先だったため、同一建玉でも client_id が変化した場合に同一阻害として集約されず、フェイルオープン閾値到達判定が崩れる可能性があった。
2. 連続阻害のキー粒度不足により、`strategy_control_exit_failopen` へ遷移せずに同一tradeのリトライだけが増幅し、損失持続を招いた。

Improvement:
1. `execution/order_manager.py` の `_strategy_control_exit_block_key` を `trade_id` 優先キーに変更し、同一建玉の阻害を 1 つのキーで集約。
2. `request_json` に渡す `block_state` と既存ログ `strategy_tag` / `trade_id` を用いて、後続監査でキー再解釈ができる状態を維持。

Verification:
1. VM反映後に `orders.db` で同一 `trade_id` に対する `strategy_control_exit_disabled` の連続数を再集計し、  
   `strategy_control_exit_failopen_*` か `close_bypassed_strategy_control` が threshold 近傍で登場することを確認。
2. 直近 24h で同一tradeの `close_request=1, close_ok=1` 到達率が維持され、`close_request=0` が集中する trade の再発がないことを確認。
3. 併せて `logs/ops_v2_audit_latest.json` / `journalctl -u quant-order-manager.service` の起動後再起動有無を監査。

Status:
- in_progress

## 2026-02-28 23:10 UTC / 2026-02-29 08:10 JST - `close_reject_no_negative` の保護理由集合を収斂

Period:
- VM直近実データ（`orders.db`）に基づき、`close_reject_no_negative` の `exit_reason` 上位を再審査。
- 同時に `quant-order-manager` の `order_manager.py` 設定値を照合し、allow/force/bypass セットの整合性を確認。

Fact:
- `close_reject_no_negative` の主要原因は `max_adverse` / `no_recovery` / `time_stop` 系が大きく、同時に `fast_cut_time` が上位帯に入り、  
  allow/force/immediate の理由集合で運用上の想定と実際トリガが一致しない箇所が存在していた。
- `fast_cut_time` は保護的な exit reason であるにもかかわらず、共通 allow 集合に欠落する設定が残っていた。

Failure Cause:
1. `ORDER_ALLOW_NEGATIVE_REASONS` / `EXIT_FORCE_ALLOW_REASONS` / `ORDER_STRATEGY_CONTROL_EXIT_IMMEDIATE_BYPASS_REASONS` が
   意図的に保護すべき理由と実時点の reason 仕様でズレており、拒否/通過の一貫性が崩れていた。
2. `config/strategy_exit_protections.yaml` の `defaults` / no-block anchor に `fast_cut_time` が未登録で、
   戦略別許可の下位互換上、`close_reject_no_negative` 化しにくい経路が生じていた。

Improvement:
1. `execution/order_manager.py`
   - 保護系理由の既定トークンを整理し、実運用保護理由（`hard_stop`,`tech_hard_stop`,`max_adverse`,`time_stop`,`no_recovery`,`max_floating_loss`,`fast_cut_time`,`time_cut`,`tech_return_fail`,`tech_reversal_combo`,`tech_candle_reversal`,`tech_nwave_flip`）へ明示集中。
   - `drawdown`,`max_drawdown`,`health_exit`,`hazard_exit`,`margin_health`,`free_margin_low`,`margin_usage_high` を
     `ORDER_ALLOW_NEGATIVE_REASONS` / `EXIT_FORCE_ALLOW_REASONS` / `ORDER_STRATEGY_CONTROL_EXIT_IMMEDIATE_BYPASS_REASONS` 既定から削除。
2. `config/strategy_exit_protections.yaml`
   - `defaults.neg_exit.allow_reasons` と `scalp_ping_5s_no_block_neg_exit_allow_reasons` に `fast_cut_time` を追加。
3. `docs/WORKER_REFACTOR_LOG.md` に同内容を追記。

Verification:
1. 反映後24hで `orders.db` を再集計し、`close_reject_no_negative` の上位理由における
   `fast_cut_time` の即時通過率が改善すること。
2. `close_reject_no_negative` で想定外のトリガが増加しないことを確認。
3. VM反映後、`quant-order-manager` の起動監査（`Application started!`）と設定有効性を確認。

Status:
- in_progress

## 2026-02-28 22:50 UTC / 2026-02-28 07:50 JST - `close_blocked_negative` と `hold_until_profit` の過剰保護を縮小

Period:
- VM直近実データ（`orders.db` / `trades.db` / `metrics.db`）に基づく、`close_blocked_negative` 原因別再審査。

Source:
- VM `logs/metrics.db`（`metric='close_blocked_negative'`, `metric='close_blocked_hold_profit'`）
- VM `logs/orders.db`（`status` と `exit_request` 検索）
- VM `systemctl` / unit稼働状態

Fact:
- `close_blocked_negative` の上位は
  `max_adverse`（`22556`）> `no_recovery`（`9772`）> `m1_rsi_fade`（`6805`）> `max_hold`（`5842`）> `reentry_reset`（`5693`）> `m1_structure_break`（`3006`）で、`__de_risk__` や `time_cut` 由来の件数も増加傾向。
- `close_blocked_hold_profit` では `min_profit_pips=9999.0` + `strict=true` の組み合わせが `20663` 件観測され、事実上の強制保有化に近い挙動。
- `scalp_ping_5s_no_block_neg_exit_allow_reasons` に `time_cut/__de_risk__/momentum_stop_loss/max_hold_loss` が欠けていたため、実績上想定された保護解除が拒否されるケースがあった。

Failure Cause:
1. `close_blocked_negative` の許可集合に対し、運用上顕在化している保護理由の一部が未登録だった。
2. `hold_until_profit` の固定 `trade_ids` + `min_profit_pips: 9999.0` + `strict=true` が、暫定目的を逸脱して常時解消抑止を発生させていた。

Improvement:
1. `config/strategy_exit_protections.yaml`
   - `scalp_ping_5s_no_block_neg_exit_allow_reasons` を拡張：
     - `time_cut`
     - `__de_risk__`
     - `momentum_stop_loss`
     - `max_hold_loss`
   - `hold_until_profit` を無効化寄りに更新し、`trade_ids: []`, `min_profit_pips: 0.0`, `strict: false`。
2. `docs/WORKER_REFACTOR_LOG.md` に同変更内容を追記し、変更履歴を監査可能化。

Verification:
1. 反映後24hで `metrics.db` 上の `close_blocked_negative` 上位理由分布と `close_reject_no_negative` の変化を再集計し、
   追加した理由の受理率が改善されること。
2. `close_blocked_hold_profit` の `9999.0`/`strict=true` 相当件数が収束し、`trade_id` 固定ブロックが消滅すること。
3. VMで `quant-order-manager` / `quant-scalp-ping-5s-*` の起動監査、`journalctl` の `Application started!` が最新化されること。

Status:
- in_progress

## 2026-02-28 05:05 UTC / 2026-02-28 14:05 JST - 市況不確実帯での hard reject 偏重を是正（RangeFader + ping5s + extrema）
Period:
- 観測: 2026-02-27 21:57 UTC までの直近24h（VM `orders.db` / `trades.db` / `metrics.db`）
- 改善対象: `entry_probability_reject` と `perf_block` の過多

Fact:
- 戦略最終ステータス（client_order_id単位）で
  `entry_probability_reject=614 (26.44%)`、`perf_block=588 (25.32%)`、`filled=589 (25.37%)`。
- `entry_probability_reject` の内訳は
  `entry_probability_below_min_units=596` が支配的。
- `perf_block` は
  `scalp_ping_5s_c_live=288`, `scalp_extrema_reversal_live=178`, `scalp_ping_5s_b_live=109`。
- `order_perf_block` reason は
  `hard:hour*:failfast` / `hard:hour*:sl_loss_rate` が主因で、`reduce` 設定でも hard 拒否に寄っていた。

Failure Cause:
1. 不確実帯での preflight が「縮小」ではなく hard reject に倒れ、約定密度を落としていた。
2. `RangeFader` は `entry_probability_below_min_units` が主因で、確率縮小後にロットが最小閾値を割っていた。
3. `scalp_extrema_reversal_live` は PF/win の軽度劣化でも `block` モードで停止しやすかった。

Improvement:
1. `workers/common/perf_guard.py`
   - `PERF_GUARD_HARD_FAILFAST_ENABLED`
   - `PERF_GUARD_HARD_SL_LOSS_RATE_ENABLED`
   - `PERF_GUARD_HARD_MARGIN_CLOSEOUT_ENABLED`
   を追加し、hard判定を戦略prefixで制御可能化。
2. `ops/env/quant-order-manager.env`
   - `SCALP_PING_5S_[B/C]_PERF_GUARD_HARD_FAILFAST_ENABLED=0`
   - `SCALP_PING_5S_[B/C]_PERF_GUARD_HARD_SL_LOSS_RATE_ENABLED=0`
   - `SCALP_EXTREMA_REVERSAL_PERF_GUARD_MODE=reduce`
   - `RangeFader` 向け `ORDER_MIN_UNITS_STRATEGY_RANGEFADER*` を `120` へ新設し、
     preserve-intent 閾値を緩和。
3. `ops/env/quant-scalp-rangefader.env`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW` 緩和、`WEIGHT_RANGE` 引き上げ、
     `WEIGHT_MICRO` 引き下げで range 判定重視へ再配分。

Verification:
1. 反映後30-60分で `orders.db` の
   `entry_probability_reject(entry_probability_below_min_units)` と
   `perf_block` 比率が低下すること。
2. 同期間で `filled` が維持または増加し、`rejected` が急増しないこと。
3. `metrics.db` の `order_perf_block` reason が `hard:failfast/sl_loss_rate`
   から `warn:*` へ遷移すること。

Status:
- in_progress
## YYYY-MM-DD HH:MM UTC / YYYY-MM-DD HH:MM JST - <short title>
Period:
- ...

Fact:
- ...

Failure Cause:
1. ...
2. ...

Improvement:
1. ...
2. ...

Verification:
1. ...
2. ...

Status:
- open | in_progress | done
```

## 2026-02-28 23:10 UTC / 2026-02-29 08:10 JST - strategy_entry で strategy-side net-edge gate を導入

Period:
- 直近実測ログと env 監査
- 対象: `orders.db` / `trades.db` / `metrics.db` と `execution/strategy_entry.py`,
  `ops/env/quant-v2-runtime.env`

Fact:
- strategy_entry に `STRATEGY_ENTRY_NET_EDGE_*` を使う local gate を追加し、
  `market_order` / `limit_order` の main path で
  `analysis_feedback -> forecast_fusion -> strategy_net_edge_gate -> leading_profile -> coordinate_entry_intent`
  の順に通過する実装を入れた。
- `entry_net_edge` と `entry_net_edge_gate` を `entry_thesis` / キャッシュ payload に残す形で監査経路を拡張。
- `ops/env/quant-v2-runtime.env` に  
  `STRATEGY_ENTRY_NET_EDGE_GATE_ENABLED=1`、
  `SCALP_PING_5S_B_ENTRY_NET_EDGE_MIN_PIPS=0.10`、
  `SCALP_PING_5S_C_ENTRY_NET_EDGE_MIN_PIPS=0.12` を追加。

Failure Cause:
1. 戦略ローカルの期待値除外が preflight 前段で体系化されておらず、戦略別最終意図（probability/intent）との整合トレースが弱かった。
2. 単価・スプレッド・見積コストを含む EV 判定が strategy_entry で未集約だったため、拒否理由分析が散在していた。

Improvement:
1. `execution/strategy_entry.py` に `_apply_strategy_net_edge_gate` を追加し、
  pocket ごとの適用可否・strategy prefix 優先 env 解決を実装。
2. net-edge 失格時に `entry_net_edge_gate` を cache status payload に残し、
  coordination 前で確定拒否するフローを追加。
3. `docs/WORKER_REFACTOR_LOG.md` と `docs/ARCHITECTURE.md` へ設計更新を追記。

Verification:
1. 反映後 24h で `entry_net_edge_negative` 系拒否の比率と `coordination_reject` の変化を比較。
2. 同時に `orders.db` の `entry_probability_reject` / `entry_probability_below_min_units` の偏重化有無を検証。
3. 戦略別サンプル 24h で `entry_net_edge_gate` 情報が `entry_thesis` / order status キャッシュに残ることを確認。

Status:
- in_progress

## 2026-02-28 22:50 UTC / 2026-02-28 07:50 JST - `scalp_ping_5s_b/_c` の neg_exit で `allow_reasons` ワイルドカードを廃止

Period:
- 事象確認: 2026-02-28 22:00 UTC 時点の運用設定監査
- 対象: `config/strategy_exit_protections.yaml`, `execution/order_manager.py`

Fact:
- `scalp_ping_5s_b(_live)` / `scalp_ping_5s_c(_live)` の
  `neg_exit.allow_reasons` が `"*"` 指定の構成が残存。
- `order_manager._strategy_neg_exit_policy()` でも戦略 override の `allow_reasons` を
  そのまま上書きしていたため、既定保護へのフォールバック条件が不安定。

Failure Cause:
1. strategy override の `allow_reasons="*"` により、実運用の明示許可ロジックが
   想定より広くなり、`close_reject_no_negative` 期待動作のブレを誘発。
2. 同一設定セットの読み分けが戦略側/共通側で非対称だったため、
   原因切分がしにくく、保護過大設定の実害検知が遅れていた。

Improvement:
1. `execution/order_manager.py`:
   - `_normalize_reason_tokens()` を追加し、`allow_reasons=["*"]` 時は
     `neg_defaults.allow_reasons` を自動採用する。
   - `_strategy_neg_exit_policy()` で strategy override 反映時に上記を適用。
2. `config/strategy_exit_protections.yaml`:
   - `scalp_ping_5s_no_block_neg_exit_allow_reasons` を追加。
   - `scalp_ping_5s_b(_live)` / `scalp_ping_5s_c(_live)` を共通 anchor 参照へ変更。
3. 実装監査を `docs/WORKER_REFACTOR_LOG.md` へ追記。

Verification:
1. 設定監査で対象4戦略の `allow_reasons="*"` が除去され、wildcard 共有アンカーへ統一されていることを確認。
2. `order_manager` の `allow_reasons` 決定経路に fallback が入ることをコード差分で確認。
3. 反映後、VM で `close_reject_no_negative` 連鎖が減るかを `orders/metrics` で追跡。

Status:
- in_progress

## 2026-02-27 15:38 UTC / 2026-02-28 00:38 JST - `scalp_ping_5s_c_live` の `entry_probability_reject` 閾値を再緩和

- Hypothesis Key:
  - `ping5s_c_entry_probability_reject`

- Primary Loss Driver:
  - `entry_probability_reject`

- Mechanism Fired:
  - worker / order-manager の
    `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C[_LIVE]`
    を
    `0.72 -> 0.58`
    へ同期した。

- Do Not Repeat Unless:
  - `entry_probability_reject`
    が again dominant で、
    かつ C の
    `submit_attempt / filled`
    が不足していると確認できた時だけ、
    同系 threshold 緩和を再実施する。

- Period:
- 観測: 2026-02-27 15:34:41-15:37:20 UTC（long leading reject 無効化後）

- Fact:
- `REJECT_BELOW=0.00` 反映後、long 側 `entry_leading_profile_reject` は減少した一方、
  `entry_probability_reject` が主因化。
- Cログは `prob=0.81〜0.89` のシグナルでも
  `err=entry_probability_reject_threshold` を連発。
- 同期間の C は `orders.db` 上で送出が細く、露出回復が不足。
- `quant-order-manager` 実効envは
  `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C[_LIVE]=0.72`。

- Failure Cause:
1. leading reject を外した後も preserve-intent 側の確率閾値 `0.72` が高く、
   C の実効確率帯（0.8前後）で閾値下振れが起きやすい。
2. C worker と order-manager の reject_under が同水準で、同時に拒否寄りへ働いた。

- Improvement:
1. `ops/env/scalp_ping_5s_c.env` の
   `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE`
   を `0.72 -> 0.58`。
2. `ops/env/quant-order-manager.env` の
   `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C[_LIVE]`
   を `0.72 -> 0.58` へ同期。

- Verification:
1. 再デプロイ後 15 分で C の
   `market_order rejected ... entry_probability_reject` 件数が減少すること。
2. 同期間 `orders.db` で `scalp_ping_5s_c_live` の
   `submit_attempt`/`filled` が再開・増加すること。
3. `metrics.db` の `order_perf_block` hard reason 再発がないこと。

- Status:
- in_progress

## 2026-02-28 04:40 UTC / 13:40 JST - 期待値改善を加速するための即時クランプ（ping B/C + MACD RSI div）

Period:
- 直近24h / 直前24h 比較（`datetime(close_time)` 正規化）
- 直近7d（戦略別寄与）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 全体期待値は改善中だが未だ負値:
  - `last24h expectancy=-1.5 JPY` vs `prev24h=-2.8 JPY`（`+1.3 JPY/trade`）
- 直近24hの負け寄与（非manual）:
  - `scalp_ping_5s_b_live`: `292 trades / -180.4 JPY / PF 0.303`
  - `scalp_ping_5s_c_live`: `109 trades / -35.3 JPY / PF 0.166`
- 少数大損:
  - `scalp_macd_rsi_div_b_live` + `scalp_macd_rsi_div_live` は `4 trades / -729.9 JPY`

Failure Cause:
1. `scalp_ping_5s_b/c` で低エッジ約定が残り、回転で負けを積み上げる。
2. `scalp_macd_rsi_div*` で単発大損が期待値を崩す。
3. order-manager の `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_[B/C]_LIVE=1` がノイズ約定を許している。

Improvement:
1. `ops/env/quant-order-manager.env`
   - `FORECAST_GATE_*` を B/C で引き上げ（expected/target/edge block）。
   - `ORDER_ENTRY_NET_EDGE_MIN_PIPS_STRATEGY_SCALP_PING_5S_B_LIVE: 0.02 -> 0.10`
   - `ORDER_ENTRY_NET_EDGE_MIN_PIPS_STRATEGY_SCALP_PING_5S_C_LIVE=0.12` を追加。
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_[B/C]_LIVE: 1 -> 30`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER/MAX_SCALE` を B/C で強化。
   - `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_SCALP_MACD_RSI_DIV_LIVE=2.8`
   - `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_SCALP_MACD_RSI_DIV_B_LIVE=2.4`
2. `ops/env/quant-scalp-macd-rsi-div.env`
   - `BASE_ENTRY_UNITS: 3000 -> 1500`, `MIN_UNITS: 600 -> 300`
   - `MIN_DIV_SCORE/STRENGTH` を引き上げ、`MAX_DIV_AGE_BARS` を短縮。
   - `SL_ATR_MULT: 0.85 -> 0.65`, `TP_ATR_MULT: 1.10 -> 1.00`
3. `ops/env/quant-scalp-macd-rsi-div-b.env`
   - `BASE_ENTRY_UNITS: 3200 -> 1600`, `MIN_UNITS: 1000 -> 400`
   - `COOLDOWN_SEC: 45 -> 90`
   - `MIN_DIV_SCORE/STRENGTH` 引き上げ、`MAX_DIV_AGE_BARS` 短縮、`RANGE_MIN_SCORE` 引き上げ。
   - `SL_ATR_MULT: 0.85 -> 0.65`, `TP_ATR_MULT: 1.15 -> 1.05`
4. `config/strategy_exit_protections.yaml`
   - `scalp_macd_rsi_div_live` の `loss_cut_hard_pips: 7.0 -> 2.6`
   - `scalp_macd_rsi_div_b_live` を追加し `loss_cut_hard_pips=2.3`

Verification:
1. 反映後24hで `expectancy_jpy` が `> 0` へ近づく（最低でも `-1.5` から改善）。
2. `scalp_ping_5s_b/c` の合算 `net_jpy` が前日比で改善し、`filled` がゼロにならない。
3. `scalp_macd_rsi_div*` の `avg_loss_jpy` と `max_loss_jpy` が明確に低下する。
4. `orders.db` で `entry_probability_reject`/`perf_block` の増加が、`net_jpy` 改善を上回る副作用になっていないことを確認する。

Status:
- applied

## 2026-02-28 04:25 UTC / 2026-02-28 13:25 JST - 全戦略共通: strategy-control EXITロック詰まりの恒久対策

Period:
- incident確認窓: 2026-02-24 07:00-07:07 UTC（JST 16:00-16:07）
- 劣化確認窓: 直近24h / 7d（2026-02-28時点）

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- `execution/order_manager.py` close preflight 実装

Fact:
- `strategy_control_exit_disabled` が同一 trade に対して連続発生し、`MicroPullbackEMA` の close 要求が滞留。
- close_reason/exit_reason 集計では、赤字寄与の多くが `STOP_LOSS_ORDER` または `max_adverse/time_stop` 系クローズ由来。
- 既存 fail-open は block回数/経過秒に依存し、保護系 exit reason でも初動で詰まる経路が残っていた。

Failure Cause:
1. `strategy_control.can_exit=false` 時、保護系理由（`max_adverse` / `time_stop` 等）でも即時通過できない。
2. fail-open が閾値到達型のため、急変局面で EXIT 遅延が先に発生する。

Improvement:
1. `execution/order_manager.py` に `ORDER_STRATEGY_CONTROL_EXIT_IMMEDIATE_BYPASS_REASONS` を追加。
2. close preflight で `strategy_control` ブロック時に、上記理由一致なら即時 `CLOSE_BYPASS` で通過。
3. 既存 fail-open（閾値到達後のバイパス）は維持し、理由一致時のみ先回りで詰まりを回避。

Verification:
1. `orders.db` で `status='strategy_control_exit_disabled'` の新規増加と連続回数が低下すること。
2. `close_bypassed_strategy_control` メトリクスで `reason='strategy_control_exit_immediate_reason'` が観測されること。
3. `MARKET_ORDER_MARGIN_CLOSEOUT` の再発率が 7d 比で悪化しないこと。
4. テスト: `tests/execution/test_order_manager_exit_policy.py` の即時バイパスケースを含め pass すること。

Status:
- deployed_pending

## 2026-02-28 03:55 UTC / 2026-02-28 12:55 JST - 「全然稼げてない」RCA（VM実測, 24h + 7d）

Period:
- 24h: `close_time >= datetime('now','-24 hours')`
- 比較窓: 直前7日（`close_time >= datetime('now','-8 days') and < datetime('now','-24 hours')`）
- 参考: last 7d 全体

Source:
- VM `systemctl` / `journalctl`（V2導線稼働確認）
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/metrics.db`
- OANDA summary/open trades (`scripts/check_oanda_summary.py`, `scripts/oanda_open_trades.py`)

Fact:
- 稼働状態:
  - `quant-market-data-feed` / `quant-strategy-control` / `quant-order-manager` / `quant-position-manager` は `active/running`。
  - 直近時点の open trades は `0`（週末クローズ帯: 2026-02-28 JST）。
- 24h損益（manual除外）:
  - `836 trades / win_rate=0.4234 / PF=0.727 / expectancy=-0.9 JPY / net=-737.0 JPY`
- 直前7日比較（manual除外）:
  - `3591 trades / win_rate=0.4152 / PF=0.55 / avg_daily=-4248.5 JPY`
- 7d全体寄与（manual除外）:
  - 総計 `-31346.5 JPY`
  - `MicroPullbackEMA=-15527.3 JPY`, `scalp_ping_5s_c_live=-10732.8 JPY`, `scalp_ping_5s_b_live=-7342.0 JPY`
  - 3戦略除外シミュレーション: `+2255.6 JPY`（`exclude_MicroPullbackEMA_ping_b_c`）
- 拒否/ブロック（orders final status, 24h）:
  - `entry_probability_reject=614 (26.44%)`
  - `perf_block=588 (25.32%)`
  - `filled=589 (25.37%)`
- EXIT詰まり痕跡（last 7d）:
  - `strategy_control_exit_disabled=10277`（全件 2026-02-24 JST）
  - 内訳: `MicroPullbackEMA-short=8177`, `MicroTrendRetest-long=2068`
- Closeout損失:
  - `MARKET_ORDER_MARGIN_CLOSEOUT` のうち `MicroPullbackEMA=4 trades / -16837.4 JPY`

Failure Cause:
1. `strategy_control_exit_disabled` が長時間連続し、EXIT fail-open が遅く `MicroPullbackEMA` の margin closeout を許容した。
2. `MicroPullbackEMA` は 7d 主因（`-15527.3 JPY`）で、base units / margin utilization が高く tail loss が大きい。
3. `scalp_ping_5s_b/c` の高回転低EVが継続し、24h/7dともに負寄与を積み上げた（特に C）。

Improvement:
1. `ops/env/quant-v2-runtime.env`
   - `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_BLOCK_THRESHOLD: 6 -> 3`
   - `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_WINDOW_SEC: 90 -> 20`
   - `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_RESET_SEC: 300 -> 180`
   - `ORDER_STRATEGY_CONTROL_EXIT_FAILOPEN_EMERGENCY_ONLY: 1 -> 0`
2. `ops/env/quant-order-manager.env`
   - `ORDER_ENTRY_MAX_SL_PIPS_STRATEGY_MICROPULLBACKEMA: 4.0 -> 3.0`
   - `PERF_GUARD_MODE_STRATEGY_MICROPULLBACKEMA=block`（明示）
   - `PERF_GUARD_MARGIN_CLOSEOUT_HARD_MIN_TRADES_STRATEGY_MICROPULLBACKEMA: 4 -> 1`
   - `PERF_GUARD_MARGIN_CLOSEOUT_HARD_RATE_STRATEGY_MICROPULLBACKEMA: 0.20 -> 0.05`
3. `ops/env/quant-micro-pullbackema.env`
   - `MICRO_MULTI_BASE_UNITS: 9000 -> 3500`
   - `MICRO_MULTI_MAX_MARGIN_USAGE: 0.72 -> 0.50`
   - `MICRO_MULTI_TARGET_MARGIN_USAGE=0.55`（追加）
   - `MICRO_MULTI_CAP_MAX=0.65`（追加）
4. `ops/env/quant-scalp-macd-rsi-div-b.env`
   - divergence閾値強化 + size縮小（`BASE_ENTRY_UNITS: 5000 -> 3200` など）

Verification:
1. 次回市場オープン後の first 24h で `strategy_control_exit_disabled` の新規累積が `0` に近いこと。
2. `MARKET_ORDER_MARGIN_CLOSEOUT` の件数/損失が 7d 比で有意に減ること（目標: `MicroPullbackEMA` closeout 0件）。
3. `MicroPullbackEMA` / `scalp_macd_rsi_div_b_live` の `net_jpy` と `avg_loss_jpy` が改善すること。
4. `scalp_ping_5s_b/c` は停止せず、`filled` を維持したまま `net_jpy` 改善傾向を確認すること。

Status:
- in_progress

## 2026-02-27 16:13 UTC / 2026-02-28 01:13 JST - `scalp_ping_5s_b/c` 継続赤字 + `MicroPullbackEMA` closeout tail の同時是正

Period:
- 直近24h / 7d（`trades.db`, `orders.db`, `metrics.db`）
- 取得時刻: 2026-02-27 16:13 UTC / 2026-02-28 01:13 JST

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/metrics.db`
- VM `scripts/oanda_open_trades.py`
- OANDA account summary（VMからAPI照会）

Fact:
- 24h戦略別:
  - `scalp_ping_5s_c_live`: `618 trades / -590.2 pips / -3452 JPY`
  - `scalp_ping_5s_b_live`: `682 trades / -431.7 pips / -769 JPY`
- 7d戦略別:
  - `scalp_ping_5s_b_live`: `2226 trades / -1868.5 pips / -8502 JPY`
  - `scalp_ping_5s_c_live`: `1273 trades / -1348.6 pips / -10722 JPY`
  - `MicroPullbackEMA`: `46 trades / -520.3 pips / -15527 JPY`
- 7d close reason:
  - `MARKET_ORDER_MARGIN_CLOSEOUT`: `18 trades / -621.5 pips / -19125 JPY`
- OANDA口座状態:
  - `marginCloseoutPercent=0.90854`
  - `marginUsed=53043.4000`, `marginAvailable=5339.6956`
  - `openTradeCount=1`（`USD_JPY`, `currentUnits=-8500`, SL/TP なし）

Failure Cause:
1. `scalp_ping_5s_b/c` は低品質帯通過と高回転が重なり、期待値が負のまま損失を積み上げている。
2. `MicroPullbackEMA` は base units と margin 使用上限が高く、急変時に margin closeout tail を作りやすい。
3. 口座の `marginCloseoutPercent` が高止まり（0.90超）し、一発損失の再発余地が大きい。

Improvement:
1. `ops/env/scalp_ping_5s_b.env` を高確度・低回転へ再設定（件数/ロット/確率閾値/intent縮小/spread guard有効化）。
2. `ops/env/scalp_ping_5s_c.env` を高確度・低回転へ再設定（件数/ロット/force-exit損失上限/確率閾値/intent縮小）。
3. `ops/env/quant-micro-pullbackema.env` で `BASE_UNITS` と `MAX_MARGIN_USAGE` を大幅に縮小し、同時シグナル数を1へ制限。
4. `ops/env/quant-order-manager.env` で B/C preserve-intent 閾値を worker 側へ同期し、`MicroPullbackEMA` の許容SL幅を `6.0 -> 4.0` へ縮小。

Verification:
1. 反映後2h/24hで `scalp_ping_5s_b/c` の `sum(realized_pl)` と `avg_pips` が改善方向へ転じること。
2. `MARKET_ORDER_MARGIN_CLOSEOUT` の新規発生が抑制されること（特に `MicroPullbackEMA`）。
3. OANDA summary の `marginCloseoutPercent` が低下方向へ向かうこと。
4. `orders.db` で `filled` を維持しつつ `entry_probability_reject` / `perf_block` の異常増がないこと。

Status:
- in_progress

## 2026-02-27 15:29 UTC / 2026-02-28 00:29 JST - `scalp_ping_5s_c_live` long 側 leading reject を無効化（short維持）
Period:
- 観測: 2026-02-27 15:25:36-15:28:54 UTC（前回緩和反映後）

Fact:
- 前回緩和後も `journalctl` で long 側の `entry_leading_profile_reject` が継続し、
  `open mode=...` の直後に reject が多発。
- 直近ログでも `side=long` の reject が連続し、C の露出回復が不十分。
- 同時に `orders.db` では反映直後に `scalp_ping_5s_c_live` の `submit_attempt=3 / filled=3` まで再開しており、
  方向性としては「hard reject をさらに減らせば約定回復が進む」状態。

Failure Cause:
1. `REJECT_BELOW=0.56` でも long 側の `adjusted_probability` が閾値を下回る局面が多く、hard reject が継続。
2. C は long bias 運用なのに long 側も hard reject で止まり、機会損失が残った。

Improvement:
1. `ops/env/scalp_ping_5s_c.env` の
   `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW` を `0.56 -> 0.00` へ変更し、
   long 側 hard reject を無効化。
2. `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.80` は維持し、
   short 側は従来どおり強く抑制。
3. 既存の `PENALTY_MAX=0.14` / `UNITS_MIN_MULT=0.58` / `entry_probability` ガードで
   低品質シグナルは縮小・拒否を継続。

Verification:
1. 再デプロイ後 15 分で `journalctl` の `market_order rejected ... entry_leading_profile_reject`（long）が大幅減少すること。
2. 同期間 `orders.db` で `scalp_ping_5s_c_live` の `submit_attempt`/`filled` が前回より増加すること。
3. `metrics.db` で `order_perf_block` hard reason（failfast/sl_loss_rate）が再発しないこと。

Status:
- in_progress

## 2026-02-27 15:35 UTC / 2026-02-28 00:35 JST - 市況プレイブックを policy_overlay へ自動適用（no-delta抑止付き）

Period:
- 実装時点（単発検証）

Source:
- `scripts/gpt_ops_report.py`
- `analytics/policy_apply.py`
- `tests/scripts/test_gpt_ops_report.py`

Fact:
- 既存の `gpt_ops_report --policy` は `deterministic_playbook_only` の `no_change` を常に出力し、
  `--apply-policy` でも overlay へ実反映しない実装だった。
- これにより、プレイブックの A/B/C シナリオと `order_manager` の entry gate が接続されず、
  「分析は更新されるが執行条件は固定」の状態になっていた。

Failure Cause:
1. policy diff 生成がスタブ固定（`no_change=true`）で、方向バイアス/イベント/データ鮮度が反映されない。
2. `--apply-policy` でも `apply_policy_diff_to_paths` を呼んでいないため、導線が未接続。
3. 同値判定がないため、将来 patch 適用を追加しても毎サイクル version 増加ノイズが発生しやすい。

Improvement:
1. `gpt_ops_report` に deterministic translator を追加し、`short_term.bias`、`direction_confidence_pct`、
   scenario gap、`event_soon/event_active_window`、`factor_stale`、`reject_rate` から
   pocket別 `entry_gates.allow_new` / `bias` / `confidence` を生成。
2. `--apply-policy` で `apply_policy_diff_to_paths` を実行し、
   `policy_overlay` / `policy_latest` / `policy_history` を更新。
3. 現在 overlay と patch の deep-subset 比較で no-delta 判定を入れ、
   同値時は `no_change=true` として不要な version 連番更新を回避。

Verification:
1. `pytest -q tests/scripts/test_gpt_ops_report.py tests/scripts/test_run_market_playbook_cycle.py` -> `13 passed`
2. ローカル実行:
   - `python3 scripts/gpt_ops_report.py ... --policy --apply-policy ...`
   - `INFO [OPS_POLICY] applied=True ...` を確認。
3. no-delta再計算時に `no_change=true` になるユニットテストを追加済み。

Status:
- done

## 2026-02-27 15:50 UTC / 2026-02-28 00:50 JST - policy適用は回っていたが order_manager gate がOFFだったため本番ON

Period:
- 監査: 2026-02-27 15:42-15:49 UTC

Source:
- VM `journalctl -u quant-ops-policy.service`
- VM `/home/tossaki/QuantRabbit/ops/env/quant-v2-runtime.env`
- VM `/home/tossaki/QuantRabbit/ops/env/quant-order-manager.env`
- VM `execution/order_manager.py`（`_POLICY_GATE_ENABLED`）

Fact:
- `quant-ops-policy.service` は `applied=True` で `policy_overlay` を更新（15:42 UTC時点で確認）。
- ただし runtime/order-manager env に `ORDER_POLICY_GATE_ENABLED` が存在せず、
  order_manager の policy gate が default false のまま。
- 直近 orders サマリでも `policy_*` 系 reject は観測されず、
  preflight 適用が実運用に接続されていない状態だった。

Failure Cause:
1. プレイブック→overlay 導線の実装後、order_manager 側の有効化フラグを本番envでONにしていなかった。

Improvement:
1. `ops/env/quant-order-manager.env` に `ORDER_POLICY_GATE_ENABLED=1` を追加。
2. `quant-order-manager.service` 再起動後、process env と journal で有効化を確認する。

Verification:
1. VMで `systemctl restart quant-order-manager.service` 後、`/proc/<pid>/environ` に
   `ORDER_POLICY_GATE_ENABLED=1` が存在すること。
2. `quant-ops-policy.service` の次回 `applied=True` 更新後、order_manager が同overlayを読み込むこと。
3. 直近 orders で `policy_allow_new_false` / `policy_bias_*` が必要局面で発生することを継続監視。

Status:
- in_progress

## 2026-02-27 15:24 UTC / 2026-02-28 00:24 JST - `scalp_ping_5s_c_live` の `entry_leading_profile_reject` 過多を C専用で緩和
Period:
- 観測: 2026-02-27 15:20-15:24 UTC（再起動直後）

Fact:
- `quant-scalp-ping-5s-c.service` は `2026-02-27 15:19:59 UTC` に再起動後 active。
- 同期間ログ集計で `open=59` に対し `entry_leading_profile_reject=56`。
- 同期間 `orders.db` では `scalp_ping_5s_c_live` の `submit_attempt/filled` が 0 件。
- 一方で `metrics.db` の `order_perf_block` は 15:18 UTC 以降 0 件で、主阻害要因が `perf_block` から `entry_leading_profile_reject` へ移行。

Failure Cause:
1. C の leading profile が `REJECT_BELOW=0.64` + `PENALTY_MAX=0.20` で逆風時にゼロ化しやすく、`entry_leading_profile_reject` が多発。
2. reject 条件が先に成立して `order_manager` 送出前に止まり、約定再開に繋がらなかった。

Improvement:
1. `ops/env/scalp_ping_5s_c.env` の `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW` を `0.64 -> 0.56` へ緩和。
2. `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_PENALTY_MAX` を `0.20 -> 0.14` へ緩和。
3. reject 回避後のリスク抑制として `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_UNITS_MIN_MULT` を `0.72 -> 0.58` に下げ、低確度帯は縮小で通す。

Verification:
1. デプロイ後 15 分で `journalctl` 集計の `entry_leading_profile_reject/open` 比率が低下すること。
2. 同期間 `orders.db` で `scalp_ping_5s_c_live` の `submit_attempt` と `filled` が再開すること。
3. `metrics.db` で `order_perf_block` の hard `failfast/sl_loss_rate` 再発がないこと。

Status:
- in_progress

## 2026-02-27 15:15 UTC / 2026-02-28 00:15 JST - `scalp_ping_5s_c_live` の hard `perf_block` 主因を failfast/sl_loss_rate と特定して緩和
Period:
- 集計: 2026-02-27 14:12-15:12 UTC（直近60分）

Fact:
- VM `logs/metrics.db` の `order_perf_block` は 134 件。
- 内訳は `scalp_ping_5s_c_live` が 119 件で、理由は
  `hard:hour14:sl_loss_rate=0.68 pf=0.39 n=41` が 94 件、
  `hard:hour15:failfast:pf=0.12 win=0.28 n=43` が 25 件。
- 同時間帯の `orders.db` では C は `filled=4 / perf_block=119` で、B は `filled=24 / perf_block=15`。

Failure Cause:
1. `PERF_GUARD_FAILFAST_PF/WIN` を下げても、`PERF_GUARD_FAILFAST_HARD_PF` の既定値で hard block が継続していた。
2. `PERF_GUARD_SL_LOSS_RATE_MAX=0.55` が C の時間帯成績（sl_loss_rate=0.68）に対して厳しすぎ、hour14 で hard block が連発した。

Improvement:
1. `ops/env/quant-order-manager.env` と `ops/env/scalp_ping_5s_c.env` に
   `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_HARD_PF=0.00` を追加（fallback `SCALP_PING_5S_*` も同時設定）。
2. 同2ファイルで `SCALP_PING_5S_C_PERF_GUARD_SL_LOSS_RATE_MAX` を `0.55 -> 0.70` へ緩和
   （fallback `SCALP_PING_5S_*` も同時に `0.70` へ更新）。

Verification:
1. デプロイ後、`metrics.db` の `order_perf_block` を15分監視し、Cの `hard:hour*:failfast` / `hard:hour*:sl_loss_rate` が再発しないことを確認。
2. 同期間で `orders.db` の C `filled/perf_block` 比率が改善（`filled` 増・`perf_block` 減）することを確認。

Status:
- in_progress

## 2026-02-27 17:10 UTC / 2026-02-28 02:10 JST - Counterfactual auto-improve を noise+pattern LCB で昇格判定
Period:
- 実装/テスト: 2026-02-27（ローカル）

Source:
- `analysis/trade_counterfactual_worker.py`
- `analysis/replay_quality_gate_worker.py`
- `tests/analysis/test_trade_counterfactual_worker.py`
- `tests/analysis/test_replay_quality_gate_worker.py`

Fact:
- 既存 auto-improve は `policy_hints.block_jst_hours` のみを採用条件にしており、
  ノイズ局面で時間帯ブロックへ寄る導線だった。
- replay→counterfactual の昇格判定に spread/stuck/OOS のノイズ補正と
  pattern book 事前確率が未統合だった。

Failure Cause:
1. 採用条件が時間帯ブロック中心で、reentry 品質の調整（cooldown/reentry距離）へ接続されていなかった。
2. 候補ランクが期待値中心で、ノイズ耐性（LCB）と pattern prior が弱かった。

Improvement:
1. `trade_counterfactual_worker` に `noise_penalty`（spread/stuck/OOS）と
  `pattern_book_deep` 事前確率を統合し、`quality_score` で候補を再ランキング。
2. `policy_hints.reentry_overrides`（tighten/loosen, multiplier, confidence, lcb_uplift）を追加。
3. `replay_quality_gate_worker` は `reentry_overrides` の
  `confidence`/`lcb_uplift_pips` をゲートにして
  `worker_reentry.yaml` の `cooldown_* / same_dir_reentry_pips / return_wait_bias` を更新。
4. `block_jst_hours` の自動適用は `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_BLOCK_HOURS=1`
  を明示した場合のみ許可（既定 0）。

Verification:
1. `pytest -q tests/analysis/test_trade_counterfactual_worker.py tests/analysis/test_replay_quality_gate_worker.py`
   で回帰テストが通過すること。
2. auto-improve 実行後の `logs/replay_quality_gate_latest.json.auto_improve.strategy_runs[*]` で
   `reentry_mode/confidence/lcb` と `accepted_update.reentry_overrides` が記録されること。
3. `worker_reentry.yaml` の更新が時間帯ブロックでなく
   `cooldown_* / same_dir_reentry_pips / return_wait_bias` 中心であること。

Status:
- in_progress

## 2026-02-27 14:20 UTC / 2026-02-27 23:20 JST - `scalp_extrema_reversal_live` の取り残し（SL欠損 + loss_cut未発火）対策
Period:
- 直近7日（orders/trades 集計）
- 2026-02-27 13:49 UTC 時点の open trade

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `PositionManager.get_open_positions()`
- Repo/VM `ops/env/quant-order-manager.env`, `config/strategy_exit_protections.yaml`

Fact:
- `scalp_extrema_reversal_live` の直近7日 `filled=47` のうち、
  `stopLossOnFill` 付きは `4`（約 `8.5%`）。
- 同時点 open trade は `3件` すべて `scalp_extrema_reversal_live` で、
  `stop_loss=null`（TPのみ）。
- `trade_id=408076` は `hold≈526min / -49.5pips` まで逆行保持。

Failure Cause:
1. `quant-order-manager` 実効envで `scalp_extrema_reversal_live` の
   `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_*` が未設定だった。
2. `strategy_exit_protections` に `scalp_extrema_reversal_live` の個別 `exit_profile` がなく、
   defaults（`loss_cut_enabled=false`, `loss_cut_require_sl=true`）へフォールバックしていた。
3. その結果「SLなしで建つと、負け側を deterministic に閉じる条件が弱い」状態が発生した。

Improvement:
1. `ops/env/quant-order-manager.env`
  - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_SCALP_EXTREMA_REVERSAL_LIVE=1`
  - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_SCALP_EXTREMA_REVERSAL=1`
2. `config/strategy_exit_protections.yaml`
  - `scalp_extrema_reversal_live.exit_profile` を追加:
    `loss_cut_enabled=true`, `loss_cut_require_sl=false`,
    `loss_cut_hard_pips=7.0`, `loss_cut_reason_hard=m1_structure_break`,
    `loss_cut_max_hold_sec=900`, `loss_cut_cooldown_sec=4`

Verification:
1. 反映後 24h で `scalp_extrema_reversal_live` の `filled` について
   `stopLossOnFill` 付与率が改善していること（目標: `>=0.90`）。
2. open trade 監査で `scalp_extrema_reversal_live` の `stop_loss=null` が常態化しないこと。
3. 逆行保持（例: `-20pips` 超かつ `hold>20min`）の滞留件数が低下すること。

Status:
- in_progress

## 2026-02-27 15:05 UTC / 2026-02-28 00:05 JST - 市況プレイブックの stale factor 誤判定を是正（外部価格フォールバック）

Period:
- ローカル再現（`run_market_playbook_cycle.py --force`）

Source:
- `tmp/gpt_ops_report_user_now.json`
- `scripts/gpt_ops_report.py`
- `tests/scripts/test_gpt_ops_report.py`

Fact:
- `factor_cache` M1 の時刻が `2025-10-29T23:58:00+00:00` の stale 条件でも、
  従来の `gpt_ops_report` は `snapshot.current_price` と方向スコアに stale 値を直接利用していた。
- 同時に `market_context.pairs.usd_jpy` は外部取得値（当日値）で更新されるため、
  `snapshot` と `market_context` の価格整合が崩れるケースが発生していた。

Failure Cause:
1. `gpt_ops_report` に factor 鮮度判定が無く、`M1 close` を無条件採用していた。
2. stale 時の信頼度減衰ルールがなく、シナリオ確率が過信方向に寄る余地があった。

Improvement:
1. `OPS_PLAYBOOK_FACTOR_MAX_AGE_SEC`（default 900s）で M1 factor 鮮度を判定。
2. stale 時は `snapshot.current_price` を外部 `USD/JPY` にフォールバック。
3. `snapshot.factor_stale/factor_age_m1_sec/current_price_source` を追加し、判定根拠を可視化。
4. stale 時に `direction_score` と `direction_confidence_pct` を減衰し、`break_points/if_then_rules` に鮮度ガードを追加。
5. `execution/order_manager._factor_age_seconds()` で `timestamp/ts/time` を受理し、
   `ENTRY_FACTOR_MAX_AGE_SEC` の stale block が ts系 factor でも確実に発火するよう補正。

Verification:
1. `pytest -q tests/scripts/test_gpt_ops_report.py tests/scripts/test_run_market_playbook_cycle.py` で `11 passed`。
2. 再実行で `factor_stale=true` の場合 `current_price_source=external_snapshot` を確認。
3. 同ケースで `direction_confidence_pct` が自動で低下（過信抑制）することを確認。

Status:
- in_progress

## 2026-02-27 15:20 UTC / 2026-02-28 00:20 JST - `quant-ops-policy` 新導線の依存欠損（bs4）復旧

Period:
- VM 反映直後（`quant-ops-policy.service` 更新後）

Source:
- VM `journalctl -u quant-ops-policy.service`
- `scripts/fetch_market_snapshot.py`
- `requirements.txt`

Fact:
- `quant-ops-policy.service` を `run_market_playbook_cycle.py` 実行に切り替えた直後、
  `ModuleNotFoundError: No module named 'bs4'` で service が失敗した。

Failure Cause:
1. `fetch_market_snapshot.py` が `from bs4 import BeautifulSoup` を使用。
2. `requirements.txt` に `beautifulsoup4` が未定義で、VM venv へ導入されていなかった。

Improvement:
1. `requirements.txt` に `beautifulsoup4==4.12.3` を追加。
2. VM で venv へ依存を導入後、`quant-ops-policy.service` を再起動して復旧する。

Verification:
1. `quant-ops-policy.service` が失敗せず完走すること。
2. `logs/gpt_ops_report.json` が更新され、`snapshot.factor_stale` と `current_price_source` が出力されること。

Status:
- in_progress

## 2026-02-27 15:00 UTC / 2026-02-28 00:00 JST - `scalp_ping_5s_c` 第13ラウンド（failfast hard block の下限を調整）

Period:
- Round12 反映直後（`2026-02-27T14:59:55+00:00` 以降）

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- Repo/VM `ops/env/scalp_ping_5s_c.env`, `ops/env/quant-order-manager.env`

Fact:
- Round12 後も C は `perf_block` が残存（直近で `perf_block=1` を確認）。
- C ログで reject 原因が明示:
  - `note=perf_block:hard:hour15:failfast:pf=0.12 win=0.28 n=43`
- 同時間帯で B は `submit_attempt=2`, `filled=2` と継続約定。

Failure Cause:
1. setup guard は緩和できたが、C の hour15 failfast（PF 下限 0.20）が先に発火。
2. C は `mapped_prefix=SCALP_PING_5S` を使うため、fallback failfast も同時に満たす必要がある。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`:
  - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_PF: 0.20 -> 0.10`
  - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_WIN: 0.20 -> 0.25`
  - `SCALP_PING_5S_PERF_GUARD_FAILFAST_PF: 0.20 -> 0.10`
  - `SCALP_PING_5S_PERF_GUARD_FAILFAST_WIN: 0.20 -> 0.25`
2. `ops/env/quant-order-manager.env`:
  - 上記 C/fallback failfast 値を同値へ同期。

Verification:
1. 反映後 30 分で C の `order_reject:perf_block` が減少し、`submit_attempt/filled` が再出現すること。
2. `failfast:pf=...` 理由の reject が連続しないこと。
3. 24hで C の `sum(realized_pl)` が急悪化しないこと（setup/SL系ガードは維持）。

Status:
- in_progress

## 2026-02-27 14:58 UTC / 2026-02-27 23:58 JST - `scalp_ping_5s_c` 第12ラウンド（setup perf guard を failfast中心へ寄せる）

Period:
- Round11 反映直後（`2026-02-27T14:56:53+00:00` 以降）

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- Repo/VM `ops/env/scalp_ping_5s_c.env`, `ops/env/quant-order-manager.env`

Fact:
- Round11 後、B は約定再開:
  - `submit_attempt=4`, `filled=4`, `avg_units=121.5`
- C は `perf_block` が残存:
  - `perf_block=12`, `slo_block=1`（同期間、strategy tag 抽出）
  - 例: `14:57:40 UTC` の C `entry-skip summary total=31` で `order_reject:perf_block=10`
- C の `RISK multiplier` は `pf=0.47 / win=0.48` で、setup guard の `PF_MIN=0.90` が主にボトルネック。

Failure Cause:
1. C の setup guard が現在性能帯（PF 0.47 前後）より高すぎ、`perf_block` が継続。
2. C は `mapped_prefix=SCALP_PING_5S` を使うため、`SCALP_PING_5S_*` fallback も同時に厳しいとブロックが残る。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`（C + fallback）:
  - `SETUP_MIN_TRADES: 16 -> 24`
  - `SETUP_PF_MIN: 0.90 -> 0.45`
  - `SETUP_WIN_MIN: 0.45 -> 0.40`
2. `ops/env/quant-order-manager.env`（C + fallback）:
  - `SETUP_MIN_TRADES: 16 -> 24`
  - `SETUP_PF_MIN: 0.90 -> 0.45`
  - `SETUP_WIN_MIN: 0.45 -> 0.40`

Verification:
1. 反映後 30 分で C の `order_reject:perf_block` 比率が低下すること。
2. `orders.db` で C の `submit_attempt/filled` が再出現すること。
3. 24h で C の `sum(realized_pl)` が急悪化しないこと（failfastは維持）。

Status:
- in_progress

## 2026-02-27 14:53 UTC / 2026-02-27 23:53 JST - `quant-order-manager` 第11ラウンド（B/C 閾値ドリフト同期）

Period:
- Round10 後の直近 120 分

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- Repo/VM `ops/env/quant-order-manager.env`, `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`

Fact:
- `orders.db` 直近120分（B/C strategy tag）は `perf_block` のみ:
  - `scalp_ping_5s_b_live: 57`
  - `scalp_ping_5s_c_live: 131`
- worker ログ主因:
  - B（14:51:23 UTC）: `no_signal:revert_not_found=17`, `extrema_block=15`, `rate_limited=22`, `order_reject:perf_block=1`
  - C（14:51:43 UTC）: `no_signal:revert_not_found=32`, `extrema_block=18`, `order_reject:entry_leading_profile_reject=7`
- `quant-order-manager.env` が worker env より厳しいまま残存:
  - B preserve-intent: `0.78/0.20/0.32`（worker `0.76/0.40/0.42`）
  - B min units: `10`（worker `1`）
  - B/C setup/hourly guard の `min_trades=6` や `setup pf/win=0.95/0.50` が worker より strict

Failure Cause:
1. order-manager 側 env ドリフトが worker 側緩和を上書きし、`perf_block` が過多。
2. B の `ORDER_MIN_UNITS=10` が縮小後通過を阻害し、送信機会を削減。
3. C/common perf guard の setup/hourly 閾値が高く、短期窓で block 判定が先行。

Improvement:
1. `ops/env/quant-order-manager.env` を worker 現行値へ同期。
2. B 同期:
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.76`
  - `MIN_SCALE/MAX_SCALE=0.40/0.42`
  - `ORDER_MIN_UNITS=1`
  - `PERF_GUARD_HOURLY_MIN_TRADES=10`, `SETUP_MIN_TRADES=10`
  - `SETUP_PF/WIN=0.88/0.44`, `FAILFAST_PF/WIN=0.10/0.27`
3. C + fallback 同期:
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER=0.72`
  - `SCALP_PING_5S[_C]_PERF_GUARD_HOURLY_MIN_TRADES=16`
  - `SCALP_PING_5S[_C]_PERF_GUARD_SETUP_MIN_TRADES=16`
  - `SCALP_PING_5S[_C]_PERF_GUARD_SETUP_PF/WIN=0.90/0.45`
  - `SCALP_PING_5S[_C]_PERF_GUARD_PF/WIN_MIN=0.92/0.49`
  - `SCALP_PING_5S[_C]_PERF_GUARD_SL_LOSS_RATE_MAX=0.55`

Verification:
1. 反映後 30 分/2h で `orders.db` が `perf_block only` から脱し、`submit_attempt/filled` が再出現すること。
2. `entry-skip summary` の `order_reject:perf_block` 比率が低下すること。
3. 24h で B/C の `sum(realized_pl)` が悪化せず、`avg_loss_pips` 再拡大がないこと。

Status:
- in_progress

## 2026-02-27 14:03 UTC / 2026-02-27 23:03 JST - `scalp_ping_5s_c` 第10ラウンド（約定再開後のロット底上げ）

Period:
- Round8b 反映直後: `2026-02-27T14:00:00+00:00` 以降

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `journalctl -u quant-scalp-ping-5s-c.service`

Fact:
- Round8b 後に `perf_block` は解消し、`orders.db` で `submit_attempt=3`, `filled=3` を確認。
- 一方で再開直後の約定は `buy` 偏重かつ `avg_units=1.3`（min=1, max=2）と小口化。
- `filled` 行の `entry_thesis.entry_units_intent` も `1-2` が中心で、long 露出不足が継続。

Failure Cause:
1. C worker の `BASE_ENTRY_UNITS=80` / `MIN_UNITS=1` が、現行の多段縮小下で実効 1-2 units へ収束。
2. `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.85` と `ALLOW_HOURS_OUTSIDE_UNITS_MULT=0.55` がロット回復を抑制。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`
  - `BASE_ENTRY_UNITS 80 -> 140`
  - `MIN_UNITS 1 -> 5`
  - `MAX_UNITS 160 -> 260`
  - `ALLOW_HOURS_OUTSIDE_UNITS_MULT 0.55 -> 0.70`
  - `ENTRY_LEADING_PROFILE_UNITS_MIN_MULT 0.58 -> 0.72`
  - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT 0.85 -> 1.00`

Verification:
1. 反映後30分で C `filled` の `avg(abs(units))` が `>=5` へ上昇すること。
2. `filled` 継続（ゼロ化しない）を維持しつつ、`perf_block` が再燃しないこと。
3. 24hで C long の `sum(realized_pl)` と `avg_units` を併記し、収益立ち上がりを評価すること。

Status:
- in_progress

## 2026-02-27 13:56 UTC / 2026-02-27 22:56 JST - `scalp_ping_5s_b/c` 第9ラウンド（revert/leading の過剰拒否を小幅緩和）

Period:
- Round8 反映後（直近 120 分）
- 24h 集計（`julianday(close_time) >= julianday('now','-24 hours')`）

Source:
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- 24h は依然マイナス:
  - `scalp_ping_5s_b_live`: `582 trades / -670.8 JPY / -352.3 pips / avg_win=1.165 / avg_loss=1.994`
  - `scalp_ping_5s_c_live`: `359 trades / -139.1 JPY / -284.7 pips / avg_win=1.090 / avg_loss=1.849`
- 直近 120 分 `orders.db`（strategy tag filter）は `perf_block` 偏重:
  - B: `perf_block=50`
  - C: `perf_block=110`
- worker ログの skip 主因:
  - B（13:54:26 UTC）: `total=92`, `revert_not_found=32`, `rate_limited=21`, `entry_probability_reject=3`
  - C（13:54:36 UTC）: `total=108`, `revert_not_found=41`, `rate_limited=10`, `entry_leading_profile_reject=9`

Failure Cause:
1. `revert_not_found` が B/C 共通で高止まりし、シグナル段階での取りこぼしが継続。
2. C は `entry_leading_profile_reject` と `entry_probability` 側の閾値が重なり、送信前 reject が多い。
3. B は第5ラウンドで絞った `MAX_ORDERS_PER_MINUTE` の影響が残り、`rate_limited` が高め。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `MAX_ORDERS_PER_MINUTE: 7 -> 8`
   - `REVERT_MIN_TICK_RATE: 0.50 -> 0.45`
   - `REVERT_RANGE_MIN_PIPS: 0.05 -> 0.04`
   - `REVERT_BOUNCE_MIN_PIPS: 0.008 -> 0.006`
   - `REVERT_CONFIRM_RATIO_MIN: 0.18 -> 0.15`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.77 -> 0.76`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.68 -> 0.67`
2. `ops/env/scalp_ping_5s_c.env`
   - `ENTRY_PROBABILITY_ALIGN_FLOOR: 0.72 -> 0.70`
   - `REVERT_MIN_TICK_RATE: 0.50 -> 0.45`
   - `REVERT_RANGE_MIN_PIPS: 0.05 -> 0.04`
   - `REVERT_BOUNCE_MIN_PIPS: 0.008 -> 0.006`
   - `REVERT_CONFIRM_RATIO_MIN: 0.18 -> 0.15`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.74 -> 0.72`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.66 -> 0.64`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT: 0.82 -> 0.80`

Verification:
1. 反映後 30 分/2h で `entry-skip summary` の `revert_not_found` 比率が B/C とも低下すること。
2. C で `entry_leading_profile_reject` が減り、`orders.db` の `submit_attempt/filled` が再出現すること。
3. 24h で B/C の `sum(realized_pl)` が悪化せず、`avg_loss_pips` の再拡大がないこと。

Status:
- in_progress

## 2026-02-27 13:54 UTC / 2026-02-27 22:54 JST - `scalp_ping_5s_c` 第8ラウンド（order-manager env乖離の是正）

Period:
- Round7 反映後: `2026-02-27T13:47:22+00:00` 以降

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM python 実測（`workers.common.perf_guard.is_allowed`）

Fact:
- Round7 後の C 注文状態（strategy tag filter）は `perf_block=39`, `entry_probability_reject=11`, `probability_scaled=9`, `filled=0`。
- ログは `market_order rejected ... reason=perf_block` が連続し、long の送信前段で停止。
- 同時点の `perf_guard` 実測:
  - order-manager 実効env（`quant-v2-runtime` + `quant-order-manager`）では  
    `allowed=False`, `reason='hard:hour13:failfast:pf=0.32 win=0.36 n=22'`
  - worker env（`scalp_ping_5s_c.env`）も加えると  
    `allowed=True`, `reason='warn:margin_closeout_soft...'`
- failfast 同期後の再計測では reject 理由が  
  `hard:sl_loss_rate=0.50 pf=0.32 n=22` へ移行し、`perf_block` は継続。

Failure Cause:
1. `quant-order-manager.service` が読む env 側で C failfast 閾値が旧値（`min_trades=8`, `pf=0.90`, `win=0.48`）のまま残存し、hard block 化。
2. worker 側で緩めた preserve-intent 閾値（`reject_under=0.74`）が order-manager 側へ未同期で、`entry_probability_reject` が過多。

Improvement:
1. `ops/env/quant-order-manager.env` の C preserve-intent を worker 側と同期:
  - `REJECT_UNDER 0.76 -> 0.74`
  - `MIN_SCALE 0.24 -> 0.34`
  - `MAX_SCALE 0.50 -> 0.56`
2. `ops/env/quant-order-manager.env` の `SCALP_PING_5S[_C]_PERF_GUARD_FAILFAST_*` を worker 側と同期:
  - `MIN_TRADES 8 -> 30`
  - `PF 0.90 -> 0.20`
  - `WIN 0.48 -> 0.20`
3. `ops/env/quant-order-manager.env` の `SCALP_PING_5S[_C]_PERF_GUARD_SL_LOSS_RATE_*` を warmup寄りへ更新:
  - `MIN_TRADES 16 -> 30`
  - `MAX 0.55/0.50 -> 0.68`

Verification:
1. 再起動後の `perf_guard.is_allowed(..., env_prefix=SCALP_PING_5S_C)` が `allowed=True` となること。
2. 反映後30分で `orders.db` の C `submit_attempt/filled` が再出現すること。
3. `entry-skip summary` の `order_reject:perf_block` 比率が低下すること。

Status:
- in_progress

## 2026-02-27 13:30 UTC / 2026-02-27 22:30 JST - `scalp_ping_5s_b/c` 第4ラウンド（rate-limit/revert/perf同時緩和）

Period:
- Round3 反映後: `2026-02-27T13:07:28+00:00` 以降（主に `13:21-13:26 UTC`）

Source:
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- Round3 反映後も `entry-skip summary` の上位は `rate_limited` と `no_signal:revert_not_found`。
  - B 例（13:25:30 UTC）: `total=82`, `rate_limited=49`, `revert_not_found=14`
  - C 例（13:25:18 UTC）: `total=106`, `rate_limited=48`, `revert_not_found=25`
- `entry_leading_profile_reject` は依然 long 側にも発生（B/C とも継続）。
- 反映後直近（約 19 分）で `orders.db` は B long のみ約定:
  - `7 fills`, `avg_units=57.9`, `avg_sl=1.23 pips`, `avg_tp=1.17 pips`, `tp/sl=0.95`
  - C は同期間で `filled` が確認できず、long ロット回復が不十分。

Failure Cause:
1. `MAX_ORDERS_PER_MINUTE=6` が高頻度シグナル区間で飽和し、長短とも通過機会を失っている。
2. `REVERT_*` がまだ厳しく、`revert_not_found` による no-signal 落ちが継続。
3. long 側の `entry_leading_profile` / `preserve-intent` / setup perf guard が重なり、C で特に通過率が低い。

Improvement:
1. B/C 共通で `MAX_ORDERS_PER_MINUTE` を `10` へ引き上げ。
2. B/C 共通で `REVERT_*` を追加緩和:
   - `RANGE_MIN 0.08->0.05`, `SWEEP_MIN 0.04->0.02`, `BOUNCE_MIN 0.01->0.008`, `CONFIRM_RATIO_MIN 0.22->0.18`
3. long 通過率とサイズ下限を追加緩和:
   - B: `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.68->0.64`, `UNITS_MIN_MULT 0.70->0.76`,
     `PRESERVE_INTENT_REJECT_UNDER 0.78->0.74`, `MIN_SCALE 0.34->0.40`
   - C: `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.68->0.64`, `UNITS_MIN_MULT 0.68->0.74`,
     `PRESERVE_INTENT_REJECT_UNDER 0.76->0.72`, `MIN_SCALE 0.38->0.44`
4. setup perf guard の早期ブロックを緩和（B/C + Cのfallbackキー）:
   - `HOURLY_MIN_TRADES 6->10`, `SETUP_MIN_TRADES 6->10`, `SETUP_PF_MIN`/`WIN_MIN` を小幅緩和
5. `MIN_UNITS_RESCUE` 閾値を引き下げ、long の極小ロット化を抑制。

Verification:
1. 反映後30分/2hで `entry-skip summary` の `rate_limited` と `revert_not_found` 比率が低下すること。
2. C の `filled` 再開と、B/C long の `avg_units` 上昇を確認すること。
3. `perf_block` の過剰増加がないことを確認しつつ、`tp/sl` 改善方向を維持すること。

Status:
- in_progress

## 2026-02-27 13:32 UTC / 2026-02-27 22:32 JST - `scalp_ping_5s_b/c` 第5ラウンド調整（損失側圧縮 + 低品質約定の抑制）

Period:
- 直近24h（`julianday(close_time) >= julianday('now','-24 hours')`）
- 直近注文ログ（`orders.db` 最新30件）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 24h 戦略別:
  - `scalp_ping_5s_b_live`: `587 trades / -679.0 JPY / -358.8 pips / avg_win=1.158 / avg_loss=1.998 / avg_units=106.1`
  - `scalp_ping_5s_c_live`: `372 trades / -146.5 JPY / -294.6 pips / avg_win=1.075 / avg_loss=1.844 / avg_units=26.7`
- close reason（24h）:
  - B: `STOP_LOSS_ORDER 310 trades / -1094.6 JPY`, `TAKE_PROFIT_ORDER 190 / +336.8 JPY`, `MARKET_ORDER_TRADE_CLOSE 87 / +78.8 JPY`
  - C: `STOP_LOSS_ORDER 175 / -113.5 JPY`, `MARKET_ORDER_TRADE_CLOSE 111 / -89.3 JPY`, `TAKE_PROFIT_ORDER 86 / +56.2 JPY`
- 最新注文ログは `perf_block` が上位で、Cは `units=2-5` の小ロット通過が中心。

Failure Cause:
1. B/C とも `avg_loss_pips > avg_win_pips` が継続し、RR が負のまま。
2. B は `STOP_LOSS_ORDER` 側損失が過大で、勝ちトレードで吸収できていない。
3. C は通過時ユニットが小さく、低品質約定の churn で収益復元が遅い。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - 通過頻度抑制: `MAX_ORDERS_PER_MINUTE 10 -> 5`
   - 入口品質を引き上げ: `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY 0.54 -> 0.58`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE 75 -> 78`
   - RR再設計: `TP_BASE/MAX 0.75/2.2 -> 0.90/2.6`, `SL_BASE/MAX 1.20/1.8 -> 1.00/1.5`, `FORCE_EXIT_MAX_FLOATING_LOSS 1.5 -> 1.2`
   - preserve-intent/leading-profile を厳格化: `REJECT_UNDER 0.74 -> 0.80`, `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.64 -> 0.70`
2. `ops/env/scalp_ping_5s_c.env`
   - 絶対エクスポージャ抑制: `BASE_ENTRY_UNITS/MAX_UNITS 120/260 -> 80/160`, `MAX_ORDERS_PER_MINUTE 10 -> 6`
   - 入口品質を引き上げ: `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY 0.56 -> 0.60`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE 78 -> 82`
   - RR再設計: `TP_BASE/MAX 0.60/1.8 -> 0.85/2.3`, `SL_BASE/MAX 1.05/1.7 -> 0.90/1.4`, `FORCE_EXIT_MAX_FLOATING_LOSS 0.8 -> 0.6`
   - preserve-intent/leading-profile を厳格化: `REJECT_UNDER 0.72 -> 0.82`, `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.64 -> 0.74`

Verification:
1. 反映後2h/24hで B/C の `avg_loss_pips` と `STOP_LOSS_ORDER` の `sum(realized_pl)` が低下すること。
2. `orders.db` で B/C の `perf_block` 比率が維持されつつ、`filled` がゼロ化しないこと。
3. 24hで B/C の `sum(realized_pl)` が改善方向（損失縮小）へ転じること。

Status:
- in_progress

## 2026-02-27 13:40 UTC / 2026-02-27 22:40 JST - `scalp_ping_5s_c` spread guard が約定阻害（第5ラウンド）

Period:
- Round4 反映後: `2026-02-27T13:29:51+00:00` 以降

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- C の反映後ログで skip 主因が `spread_blocked` に集中:
  - `13:30:35 UTC`: `entry-skip summary total=143, spread_blocked=134`
  - `13:31:05 UTC`: `entry-skip summary total=110, spread_blocked=64`
- ガード理由は `spread_med ... >= limit 1.00p` で、実勢例は `med=0.85p, p95=1.16p, max=1.20p`。
- 同期間は `orders.db` で C の `filled` を確認できず、通過不足が継続。

Failure Cause:
1. C の spread guard 閾値 `limit=1.00p` が現行マーケット実勢より低く、入口で継続ブロックされる。
2. `hot_spread_now` と `spread_med` の連鎖でクールダウンが重なり、エントリー機会が枯渇する。

Improvement:
1. `ops/env/scalp_ping_5s_c.env` に C 専用 `spread_guard_*` を追加:
   - `spread_guard_max_pips=1.30`
   - `spread_guard_release_pips=1.05`
   - `spread_guard_hot_trigger_pips=1.50`
   - `spread_guard_hot_cooldown_sec=6`
2. B は `SPREAD_GUARD_DISABLE=1` 運用を維持し、今回の調整対象外とする。

Verification:
1. 反映後30分/2hで C の `entry-skip summary` における `spread_blocked` 比率が低下すること。
2. C の `orders.db status=filled` が再開すること。
3. `spread block remain` ログの連続発生が短縮/減少すること。

Status:
- in_progress

## 2026-02-27 13:46 UTC / 2026-02-27 22:46 JST - `scalp_ping_5s_c` 第6ラウンド（rate-limit/perf_block 縮小）

Period:
- 第5ラウンド反映後: `2026-02-27T13:39:35+00:00` 以降

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 第5ラウンド後、`spread_blocked` はほぼ消失した一方で skip 主因が移行:
  - `13:40:24 UTC`: `total=110`, `revert_not_found=38`, `rate_limited=24`, `perf_block=5`
  - `13:40:54 UTC`: `total=118`, `rate_limited=53`, `revert_not_found=27`
- `orders.db`（同期間）は `perf_block` と `probability_scaled` のみで `filled=0`。

Failure Cause:
1. `MAX_ORDERS_PER_MINUTE=6` で高頻度区間に飽和し、`rate_limited` が先に上位化。
2. perf guard の `*_MIN_TRADES=10` が短期ノイズで発火し、`perf_block` が継続。
3. C の通過閾値（preserve-intent / leading profile）が厳しめで、注文送信まで届きにくい。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`
   - `MAX_ORDERS_PER_MINUTE 6 -> 10`
   - `SCALP_PING_5S_C_PERF_GUARD_HOURLY_MIN_TRADES 10 -> 16`
   - `SCALP_PING_5S_C_PERF_GUARD_SETUP_MIN_TRADES 10 -> 16`
   - fallback `SCALP_PING_5S_PERF_GUARD_HOURLY_MIN_TRADES 10 -> 16`
   - fallback `SCALP_PING_5S_PERF_GUARD_SETUP_MIN_TRADES 10 -> 16`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER 0.82 -> 0.78`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.74 -> 0.70`

Verification:
1. 反映後30分/2hで `rate_limited` 比率が低下すること。
2. C の `orders.db status=filled` が再開すること。
3. `perf_block` が優位理由でなくなること。

Status:
- in_progress

## 2026-02-27 13:49 UTC / 2026-02-27 22:49 JST - `scalp_ping_5s_c` 第7ラウンド（rate_limited 優位の追加緩和）

Period:
- 第6ラウンド反映後: `2026-02-27T13:43:11+00:00` 以降

Source:
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 第6ラウンド後、`spread_blocked` は実質解消したが `rate_limited` が主因で残存:
  - `13:45:22 UTC`: `entry-skip summary total=107, rate_limited=65`
  - 近接窓で `revert_not_found` も継続（`12-42`程度）
- `orders.db`（同期間）は `perf_block/probability_scaled` のみで `filled=0`。

Failure Cause:
1. C の `MAX_ORDERS_PER_MINUTE=10` と `ENTRY_COOLDOWN_SEC=1.6` が高頻度局面で依然ボトルネック。
2. preserve-intent / leading profile の閾値が高く、レート制限解除後も通過率が伸びにくい。
3. `min_units_rescue` 閾値が高めで、long 側の極小シグナル救済が不足。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`
   - `ENTRY_COOLDOWN_SEC 1.6 -> 1.2`
   - `MAX_ORDERS_PER_MINUTE 10 -> 16`
   - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY 0.60 -> 0.56`
   - `MIN_UNITS_RESCUE_MIN_CONFIDENCE 82 -> 78`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER 0.78 -> 0.74`
   - `ENTRY_LEADING_PROFILE_REJECT_BELOW 0.70 -> 0.66`

Verification:
1. 反映後30分/2hで `rate_limited` 比率が低下すること。
2. C の `orders.db status=filled` が再開すること。
3. `entry_probability_reject` と `entry_leading_profile_reject` が急増しないこと。

Status:
- in_progress

## 2026-02-27 08:52 UTC / 2026-02-27 17:52 JST - M1系 spread 閾値を 1.00 に統一
Period:
- Adjustment window: `2026-02-27 17:46` ～ `17:52` JST
- Source: VM `journalctl`, `ops/env/quant-*.env`

Fact:
- 新規3戦略ログで `blocked by spread spread=0.80p` が連続。
- 分離3戦略の spread 上限は `0.35/0.40/0.45` と不統一だった。

Failure Cause:
1. ワーカー追加時の個別チューニングで spread 上限が規定運用から外れた。
2. 実勢スプレッドに対して閾値が低すぎ、entry判定まで進まなかった。

Improvement:
1. M1系4ワーカー（既存M1 + 分離3戦略）を `M1SCALP_MAX_SPREAD_PIPS=1.00` に統一。
2. 分離3戦略に `spread_guard_max_pips=1.00` を追加し、`spread_monitor` 側のガード閾値も同一化。
3. 既存 `quant-m1scalper.env` に spread 上限を明示し、暗黙 default 依存を排除。

Verification:
1. VM反映後に `printenv/EnvironmentFile` と `journalctl` で実効 spread 上限を確認。
2. `blocked by spread` の頻度低下と、`preflight_start -> submit_attempt` 遷移を比較。

Status:
- in_progress

## 2026-02-27 13:07 UTC / 2026-02-27 22:07 JST - `order_manager` の duplicate回復失敗と orders.db スレッド競合を是正（VM実測）

Period:
- 直近24h（`2026-02-26 13:07 UTC` 以降）

Source:
- VM `systemctl status quant-order-manager.service`
- VM `journalctl -u quant-order-manager.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `scripts/oanda_open_trades.py`

Fact:
- 24h損益は `1296 trades / -851.8 pips / -3285.14 JPY`。
- 大きな下押しは `scalp_ping_5s_c_live (-3431.03 JPY)` と `scalp_ping_5s_b_live (-696.34 JPY)`。
- orders最終状態（client_order_id単位）で `perf_block=1535`, `margin_usage_projected_cap=514`, `filled=491`, `entry_probability_reject=392`, `rejected=190`。
- `status='rejected'` の内訳は `CLIENT_TRADE_ID_ALREADY_EXISTS=294`, `LOSING_TAKE_PROFIT=14`, `STOP_LOSS_ON_FILL_LOSS=6`, `TAKE_PROFIT_ON_FILL_LOSS=1`。
- `quant-order-manager` で `SQLite objects created in a thread can only be used in that same thread` が継続発生し、ordersログ永続化が断続的に失敗。

Failure Cause:
1. `orders.db` 接続がプロセス内で単一接続共有になっており、複数スレッド利用で SQLite thread affinity 例外が発生。
2. duplicate復旧（`CLIENT_TRADE_ID_ALREADY_EXISTS`）は `orders.db` の filled レコード依存で、書き込み失敗時に復旧できず reject 化。
3. duplicate復旧のキャッシュ情報に `ticket_id` が保持されず、DB失敗時の回復余地が狭い。

Improvement:
1. `execution/order_manager.py`
   - `orders.db` 接続を global singleton から thread-local へ変更。
   - `_cache_order_status` に `ticket_id` を保持。
   - duplicate復旧で `orders.db` → cache → `trades.db` の順で trade_id を回収するフォールバックを追加。

Verification:
1. `python3 -m py_compile execution/order_manager.py` が成功すること。
2. 反映後VMで `journalctl -u quant-order-manager.service` の SQLite thread例外が減少/消失すること。
3. `orders.db` で `status='rejected' AND error_message='CLIENT_TRADE_ID_ALREADY_EXISTS'` が減少し、`duplicate_recovered` が増えること。
4. 24hで `reject_rate` と `scalp_ping_5s_b/c` の実効約定品質（filled比率）が改善方向になること。

Status:
- in_progress

## 2026-02-27 13:10 UTC / 2026-02-27 22:10 JST - `scalp_ping_5s_b/c` 第3ラウンド（RR再補正 + longロット押上げ）

Period:
- 直近6h（`datetime(ts) >= now - 6 hours`）を主観測

Source:
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`

Fact:
- Round2 後の skip 主因は `entry_leading_profile_reject` に移行し、`rate_limited` と `revert_not_found` は直近集計で優位でない。
  - B: `entry_leading_profile_reject=39`, `entry_probability_reject=5`（直近800行）
  - C: `entry_leading_profile_reject=40`, `entry_probability_reject=3`（直近800行）
- 直近6h `orders.db`（filled）:
  - `scalp_ping_5s_b_live long`: `203 fills`, `avg_units=78.8`, `avg_sl=1.84 pips`, `avg_tp=0.99 pips`, `tp/sl=0.54`
  - `scalp_ping_5s_b_live short`: `117 fills`, `avg_units=130.4`
  - `scalp_ping_5s_c_live long`: `46 fills`, `avg_units=31.5`, `avg_sl=1.31 pips`, `avg_tp=0.96 pips`, `tp/sl=0.74`
  - `scalp_ping_5s_c_live short`: `88 fills`, `avg_units=37.2`
- 直近6h `trades.db`:
  - `B long`: `203 trades`, `sum_realized_pl=-97.4 JPY`, `avg_win=1.017 pips`, `avg_loss=1.859 pips`
  - `C long`: `46 trades`, `sum_realized_pl=-12.1 JPY`, `avg_win=0.859 pips`, `avg_loss=1.952 pips`

Failure Cause:
1. `rate_limit/revert` 問題はほぼ解消した一方、`entry_leading_profile_reject` が強く、long通過ロットが依然不足。
2. B/C long とも `tp/sl < 1` が継続し、`avg_loss_pips > avg_win_pips` の負け非対称が残存。
3. preserve-intent と leading profile の下限設定が、long側のサイズ回復を抑制している。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - long RR補正: `TP_BASE/MAX 0.55/1.9 -> 0.75/2.2`, `SL_BASE/MAX 1.35/2.0 -> 1.20/1.8`
   - net最小利幅引上げ: `TP_NET_MIN 0.45 -> 0.65`, `TP_TIME_MULT_MIN 0.55 -> 0.72`
   - lot押上げ: `BASE_ENTRY_UNITS 260 -> 300`, `MAX_UNITS 900 -> 1000`
   - 通過下限緩和: `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE 0.30 -> 0.34`
   - leading profile: `REJECT_BELOW 0.70 -> 0.68`, `UNITS_MIN/MAX 0.64/0.95 -> 0.70/1.00`
2. `ops/env/scalp_ping_5s_c.env`
   - long RR補正: `TP_BASE/MAX 0.45/1.5 -> 0.60/1.8`, `SL_BASE/MAX 1.15/1.9 -> 1.05/1.7`
   - net最小利幅引上げ: `TP_NET_MIN 0.40 -> 0.55`, `TP_TIME_MULT_MIN 0.55 -> 0.70`
   - lot押上げ: `BASE_ENTRY_UNITS 95 -> 120`, `MAX_UNITS 220 -> 260`
   - 通過下限緩和: `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE 0.34 -> 0.38`
   - leading profile: `REJECT_BELOW 0.70 -> 0.68`, `UNITS_MIN/MAX 0.62/0.90 -> 0.68/0.95`

Verification:
1. 反映後2h/24hで B/C long の `avg_tp/avg_sl` が上昇し、`tp/sl` が改善すること。
2. 反映後2h/24hで `avg_units(long)` が増加し、`entry_leading_profile_reject` の件数比率が低下すること。
3. 24hで `scalp_ping_5s_b/c long` の `sum(realized_pl)` が改善方向へ向かうこと。

Status:
- in_progress

## 2026-02-27 09:06 UTC / 2026-02-27 18:06 JST - split worker の spread 二重判定を解消（single-source化）

Period:
- Log validation window: 2026-02-27 08:45 UTC - 09:06 UTC

Source:
- VM `journalctl -u quant-scalp-{trend-breakout,pullback-continuation,failed-break-reverse}.service`
- VM `/home/tossaki/QuantRabbit/ops/env/quant-scalp-*.env`
- Repo `workers/scalp_*_*/worker.py`, `market_data/spread_monitor.py`

Fact:
- 08:45-08:55 UTC に split 3 worker で
  `blocked by spread spread=0.80p reason=guard_active` が連続発生。
- env は `M1SCALP_MAX_SPREAD_PIPS=1.00` / `spread_guard_max_pips=1.00` に揃っていたが、
  worker 側に `spread_pips > M1SCALP_MAX_SPREAD_PIPS` の別判定があり、
  設定ズレ時に entry skip が再発しうる構造だった。
- 08:57 UTC 以降の直近ブロック要因は spread ではなく
  `skip_nwave_long_late` / `tag_filter_block` / `reversion_range_block`。

Failure Cause:
1. spread 入口判定が worker と spread_monitor で重複し、構成差分時の挙動が不透明化。
2. 判定 reason の一元性が不足し、監査時に「どちらで止まったか」を切り分けにくい。

Improvement:
1. split/m1 worker で spread 判定を整理し、既定は `spread_monitor` を単一ソース化。
2. `M1SCALP_LOCAL_SPREAD_CAP_ENABLED` を追加し、
   必要時のみ local cap を有効化する運用へ変更。
3. `SPREAD_GUARD_DISABLE=1` 運用（quant-m1scalper）は local cap を自動fallbackで維持。
4. runtime env へ `M1SCALP_LOCAL_SPREAD_CAP_ENABLED` を明示し、新規worker追加時のズレを防止。

Verification:
1. split 3 worker で `blocked by spread` の件数を反映後 2h/24h で監視。
2. 同期間の block reason 内訳で spread 以外（tag/range/timing）へ収束することを確認。
3. `M1SCALP_LOCAL_SPREAD_CAP_ENABLED` が split=0 / m1=1 でロードされていることを確認。

Status:
- in_progress

## 2026-02-27 08:45 UTC / 2026-02-27 17:45 JST - 3分離戦略を既存M1と同時起動へ切替
Period:
- Activation window: `2026-02-27 17:40` ～ `17:45` JST
- Source: `ops/env/quant-scalp-*.env`, VM `systemctl` / `journalctl`

Fact:
- 直前のVM状態は、`quant-m1scalper*` のみ active、新規3戦略は disabled/inactive。
- ユーザー指定は「既存停止なしで全部起動」。

Failure Cause:
1. 新規3戦略は `M1SCALP_ENABLED=0` の安全初期値のまま。
2. unit は導入済みでも enable/start 未実施だった。

Improvement:
1. 3戦略envの `M1SCALP_ENABLED=1` に更新。
2. `quant-scalp-{trend-breakout,pullback-continuation,failed-break-reverse}` と各 exit の計6 unit を `enable --now`。
3. 既存 `quant-m1scalper*` は停止せず同時稼働を維持。

Verification:
1. `systemctl is-active` で既存M1 + 新規6unit が `active`。
2. `systemctl is-enabled` で新規6unit が `enabled`。
3. `journalctl -u quant-scalp-*-*.service` で起動直後ログを確認。

Status:
- in_progress

## 2026-02-27 08:40 UTC / 2026-02-27 17:40 JST - M1シナリオ3戦略の独立性をロジック単位まで引き上げ
Period:
- Implementation window: `2026-02-27 17:20` ～ `17:40` JST
- Source: `workers/scalp_{trend_breakout,pullback_continuation,failed_break_reverse}/*`, `tests/workers/test_m1scalper_split_workers.py`

Fact:
- 先行版は service 分離済みでも、entry/exit ロジック本体が `workers.scalp_m1scalper` 依存のラッパー構成だった。
- そのため、`m1scalper` の単一変更が3戦略へ同時波及する構造だった。

Failure Cause:
1. 「独立ワーカー」の定義が process 分離止まりで、ロジック独立まで達していなかった。
2. 戦略別のデフォルト（タグ・allowlist）が wrapper の `os.environ.setdefault` に依存していた。

Improvement:
1. 3戦略へ `m1scalper` の entry/exit 実体モジュールを複製し、パッケージ内で完結させた。
2. 各戦略 `config.py` に戦略別 default（tag/side policy）を埋め込み、wrapper依存を削除。
3. 各戦略 `exit_worker.py` の `_DEFAULT_ALLOWED_TAGS` を専用タグへ変更。
4. テストを更新し、`workers.scalp_m1scalper` 直importなし・戦略別 default 反映を検証。

Verification:
1. `pytest -q tests/workers/test_m1scalper_split_workers.py tests/workers/test_m1scalper_config.py tests/workers/test_m1scalper_quickshot.py`
2. `rg \"workers\\.scalp_m1scalper\" workers/scalp_trend_breakout workers/scalp_pullback_continuation workers/scalp_failed_break_reverse`
3. `python3 -m py_compile` で新規/更新モジュールを検証。

Status:
- in_progress

## 2026-02-27 05:35 UTC / 2026-02-27 14:35 JST - 全体監査で B/C 損失源を再圧縮し、勝ち筋へ再配分（timeout 再劣化の再発防止込み）
Period:
- VM実測: `2026-02-26 17:35 UTC` 〜 `2026-02-27 05:35 UTC`（直近12h）
- 参考窓: 24h（`julianday(close_time) >= julianday('now','-24 hours')`）
- Source: `trades.db`, `orders.db`, `metrics.db`, `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `journalctl -u quant-order-manager.service`, `scripts/oanda_open_trades.py`

Fact:
- 24h realized P/L:
  - `scalp_ping_5s_c_live`: `473 trades / -1455.4 JPY`
  - `scalp_ping_5s_b_live`: `425 trades / -592.3 JPY`
  - `WickReversalBlend`: `7 trades / +332.3 JPY`
  - `scalp_extrema_reversal_live`: `14 trades / +30.9 JPY`
- 12h side別（`b_live/c_live`）:
  - `b_live buy`: `189 trades / -309.7 JPY`
  - `b_live sell`: `111 trades / -71.3 JPY`
  - `c_live buy`: `136 trades / -34.3 JPY`
  - `c_live sell`: `88 trades / -24.3 JPY`
- 12h orders（`scalp_fast/scalp/micro`）では、`filled=565` に対して
  `margin_usage_exceeds_cap=217`, `margin_usage_projected_cap=210`, `slo_block=109` が同時多発。
- `quant-v2-runtime.env` の実効値は `ORDER_MANAGER_SERVICE_TIMEOUT=60.0`。
  同期間の B/C ログには `order_manager service call failed ... Read timed out (45.0)` が `165件/12h`（`85件/2h`）発生。
- 直近稼働中の open trade でも、`B/C` と `Extrema` が混在し、損失源と勝ち筋の配分最適化余地が継続。

Failure Cause:
1. `b_live` が高頻度・高比率で負け寄与を継続（特に buy 側）。
2. `c_live` はサイズが小さい一方で頻度が高く、合算で負けを積み上げる構造が残存。
3. service timeout が長いため、order-manager 遅延時に worker が長時間ブロックされ、取りこぼしと偏りを増幅。
4. 正寄与戦略（Wick/Extrema）の配分が相対的に不足。

Hypothesis:
- B/C の頻度・上限倍率を追加圧縮し、Wick/Extrema の発火枠を増やすことで、
  `STOP_LOSS_ORDER` 優位の負け勾配を下げつつ、純益の期待値を上げられる。

Improvement:
1. B 圧縮（`ops/env/scalp_ping_5s_b.env`）:
   - `MAX_ORDERS_PER_MINUTE: 8 -> 6`
   - `BASE_ENTRY_UNITS: 380 -> 300`
   - `MAX_UNITS: 1400 -> 1100`
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT: 0.58 -> 0.45`
   - `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT: 0.68 -> 0.55`
   - `SIDE_BIAS_BLOCK_THRESHOLD: 0.00 -> 0.12`
   - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT: 0.95 -> 0.82`
2. C 圧縮（`ops/env/scalp_ping_5s_c.env`）:
   - `MAX_ORDERS_PER_MINUTE: 8 -> 6`
   - `BASE_ENTRY_UNITS: 140 -> 110`
   - `MAX_UNITS: 320 -> 240`
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT: 0.62 -> 0.56`
   - `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT: 0.72 -> 0.62`
   - `SIDE_BIAS_BLOCK_THRESHOLD: 0.10 -> 0.16`
   - `ENTRY_PROBABILITY_BAND_ALLOC_SIDE_METRICS_MAX_MULT: 1.00 -> 0.88`
   - fallback local 整合:
     `ORDER_MANAGER_PRESERVE_INTENT_(REJECT_UNDER/MIN_SCALE/MAX_SCALE)` を
     `0.68/0.35/0.72` へ同期
3. 共通 preflight 圧縮（`ops/env/quant-order-manager.env`）:
   - B: `REJECT_UNDER 0.68 -> 0.72`, `MAX_SCALE 0.50 -> 0.42`
   - C: `REJECT_UNDER 0.66 -> 0.68`, `MIN_SCALE 0.40 -> 0.35`, `MAX_SCALE 0.78 -> 0.72`
   - `ops/env/scalp_ping_5s_b.env` 側の fallback local 値も
     `REJECT_UNDER 0.68 -> 0.72`, `MAX_SCALE 0.55 -> 0.42` へ同期
4. service timeout 再発防止（`ops/env/quant-v2-runtime.env`）:
   - `ORDER_MANAGER_SERVICE_TIMEOUT: 60.0 -> 12.0`
   - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_WAIT_SEC: 10.0 -> 4.0`
5. 勝ち筋再配分:
   - `ops/env/quant-scalp-wick-reversal-blend.env`
     - `MAX_OPEN_TRADES: 3 -> 4`
     - `UNIT_BASE_UNITS: 10200 -> 11200`
   - `ops/env/quant-scalp-extrema-reversal.env`
     - `COOLDOWN_SEC: 35 -> 30`
     - `MAX_OPEN_TRADES: 2 -> 3`
     - `BASE_UNITS: 12000 -> 13000`
     - `MIN_ENTRY_CONF: 57 -> 54`
6. B unit override 競合を解消（`systemd/quant-scalp-ping-5s-b.service`）:
   - unit直書きの `BASE_ENTRY_UNITS` / `MAX_UNITS` /
     `ORDER_MANAGER_PRESERVE_INTENT_*` を削除し、
     `ops/env/scalp_ping_5s_b.env` と `ops/env/quant-order-manager.env` を唯一の実効値に統一。

Verification:
1. デプロイ後、VMで `HEAD == origin/main` と `quant-order-manager`, `quant-scalp-ping-5s-{b,c}`, `quant-scalp-{wick-reversal-blend,extrema-reversal}` の再起動完了を確認。
2. 30-120分窓で `b_live/c_live` の `realized_pl`、`STOP_LOSS_ORDER` 比率、`filled`/`submit_attempt` を反映前と比較。
3. 同窓で `order_manager service call failed` と `slow_request` の再発有無を監査。
4. 同窓で Wick/Extrema の `filled` と `realized_pl` が増加/維持し、全体損益勾配が改善することを確認。

Status:
- in_progress

## 2026-02-27 17:50 JST - quickshot 判定のローカル回帰テストを追加
Period:
- Validation window: `2026-02-27 17:40` ～ `17:50` JST
- Source: `tests/workers/test_m1scalper_config.py`, `tests/workers/test_m1scalper_quickshot.py`

Fact:
- quickshot 判定の主要分岐（allow / JSTメンテ時間 block / side mismatch block）をユニットテスト化した。
- quickshot 設定値（`M1SCALP_USDJPY_QUICKSHOT_*`）の env 読込を config テストで監査可能にした。

Improvement:
1. 判定ロジックの改修時に、回帰で「誤って全拒否/全通過」になるリスクを抑制。
2. `target_jpy` 逆算ロジックの単位崩れ（pips換算）をテストで即検出できる状態へ固定。

Verification:
1. `pytest -q tests/workers/test_m1scalper_config.py tests/workers/test_m1scalper_quickshot.py`
   - `7 passed`

Status:
- done (local)

## 2026-02-27 18:20 JST - 場面別3戦略（Trend/Pullback/FailedBreak）を専用ワーカーへ分離
Period:
- Design window: `2026-02-27 18:00` ～ `18:20` JST
- Source: `workers/scalp_*`, `systemd/quant-scalp-*.service`, `ops/env/quant-scalp-*.env`

Fact:
- 既存 `M1Scalper` は single worker 内で複数シグナルを扱うため、戦略単位で on/off・監査・exit閉域化が難しかった。
- EXIT 側は固定 allowlist（`M1Scalper,m1scalper,m1_scalper`）で、strategy_tag分離に追従できなかった。

Improvement:
1. `TrendBreakout` / `PullbackContinuation` / `FailedBreakReverse` を ENTRY/EXIT ペアで新設。
2. 各 ENTRY は signal タグを固定し、strategy_tag を専用名へ分離。
3. EXIT は `M1SCALP_EXIT_TAG_ALLOWLIST` で対象タグを閉域化。
4. 既存ワーカーとの競合回避のため、新規 env は `M1SCALP_ENABLED=0` を初期値にした。

Verification:
1. 新規ラッパー module の import と env 固定化をユニットテストで確認。
2. `M1SCALP_EXIT_TAG_ALLOWLIST` の反映をユニットテストで確認。

Status:
- in_progress

## 2026-02-27 17:35 JST - M1Scalper quickshot（M5 breakout + M1 pullback + 100円逆算）を導入
Period:
- Design/implementation window: `2026-02-27 16:55` ～ `17:35` JST
- Source: `workers/scalp_m1scalper/*`, `ops/env/quant-m1scalper.env`

Fact:
- 既存 `M1Scalper` は `breakout_retest` シグナルを持つが、最終執行側で
  「100円目標のロット逆算」や「JSTメンテ時間の quickshot block」は持っていなかった。
- `entry_probability` / `entry_units_intent` 契約は既に worker 側で維持されている。

Failure Cause:
1. シグナル品質が良くても、ロットが目標利益に対して過大/過小になりやすかった。
2. 即時トレード用の追加条件（spread上限、M5方向一致、pullback成立）が統合されていなかった。

Improvement:
1. `M1SCALP_USDJPY_QUICKSHOT_*` を追加し、`M5 breakout + M1 pullback` を機械判定化。
2. `tp/sl` を ATR 連動で決定し、`target_jpy` を `tp_pips` で逆算した `target_units` を採用。
3. `entry_thesis.usdjpy_quickshot` に `setup_score/entry_probability/target_units` を保存し、監査可能化。

Verification:
1. ユニットテストで allow/block（JST7時/side mismatch）を確認。
2. VM反映後に `orders.db` の `entry_thesis.usdjpy_quickshot` と block 理由を集計。
3. 24h 集計で `avg_win_jpy` / `avg_loss_jpy` のバランス悪化がないことを確認。

Status:
- in_progress

## 2026-02-27 07:46 UTC / 2026-02-27 16:46 JST - 早利確/ロット偏りの深掘り（M1 lock_floor + B payoff是正）
Period:
- 直近48h/2h（VM live `logs/trades.db`, `logs/orders.db`）
- Source: VM `fx-trader-vm`（`sudo -u tossaki` で直接照会）

Fact:
- `scalp_ping_5s_b_live`（48h）:
  - long: `win avg_units=436.2`, `loss avg_units=632.5`（勝ち側のロットが相対的に小さい）
  - close reason別:
    - `long TAKE_PROFIT_ORDER win=118 avg_pips=+0.888 avg_units=216.2`
    - `long STOP_LOSS_ORDER loss=438 avg_pips=-2.025 avg_units=642.1`
  - 直近2hでも `STOP_LOSS_ORDER loss=119 avg_pips=-2.102` に対し
    `TAKE_PROFIT_ORDER win=99 avg_pips=+1.033` で、Rが負側に偏る。
- `M1Scalper-M1`（直近2h）:
  - `MARKET_ORDER_TRADE_CLOSE`: `loss=11 avg_pips=-1.936`, `win=9 avg_pips=+1.244`
  - `tp_pips` 比率（`pl_pips/tp_pips`）は勝ちでも `0.218` と低く、TP到達前のクローズが主体。
  - `orders.db` の `close_request.exit_reason` は `lock_floor`/`m1_rsi_fade` が主。
    - `lock_floor win=6 avg_pips=+0.983`
    - `m1_rsi_fade loss=8 avg_pips=-1.25`
- 直近2hの B/M1 発注では `filled/preflight_start` は大差なし
  （B: win `0.888`, loss `0.899`; M1: win/loss とも `1.0`）。

Failure Cause:
1. `M1Scalper` は `lock_floor` 発火が早く、`tp_hint` まで伸ばす前に利を確定しやすい。
2. `M1Scalper` の `m1_rsi_fade` が逆行初期で多発し、反発余地のある局面も早期クローズする。
3. `scalp_ping_5s_b_live` は TPが浅く（+1p台）SL側が重い（-2p台）ため、勝率が維持されても期待値が伸びにくい。

Improvement:
1. `workers/scalp_m1scalper/exit_worker.py`
  - lock/trail 関連を env で調整可能化:
    - `M1SCALP_EXIT_LOCK_TRIGGER_FROM_TP_RATIO`
    - `M1SCALP_EXIT_LOCK_TRIGGER_MIN_PIPS`
    - 既存 hard-coded の `profit_take/trail/lock` も env 化。
2. `ops/env/quant-m1scalper-exit.env`
  - `M1SCALP_EXIT_RSI_FADE_LONG=40`
  - `M1SCALP_EXIT_RSI_FADE_SHORT=60`
  - `M1SCALP_EXIT_LOCK_FROM_TP_RATIO=0.70`
  - `M1SCALP_EXIT_LOCK_TRIGGER_FROM_TP_RATIO=0.55`
  - `M1SCALP_EXIT_LOCK_TRIGGER_MIN_PIPS=1.00`
3. `ops/env/scalp_ping_5s_b.env`
  - TP/SL再調整: `TP_BASE/TP_MAX` を上げ、`SL_BASE` と force-exit loss を圧縮。
  - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT` を `0.72 -> 0.80` に引き上げ、良化局面の過小サイズを緩和。

Verification:
1. 反映後 2h/24h で `M1Scalper-M1` の `exit_reason` 分布を再集計し、
   `lock_floor` 比率低下と `take_profit` 比率上昇を確認。
2. 反映後 2h/24h で `scalp_ping_5s_b_live` の
   `avg_win_pips / avg_loss_pips` と `TAKE_PROFIT_ORDER` 平均pipsを前窓比較。
3. `orders.db` で B/M1 の `filled/preflight_start` 比率を監視し、約定率の悪化がないことを確認。

Status:
- in_progress

## 2026-02-27 14:20 UTC / 2026-02-27 23:20 JST - duplicate CID + exit disable 連鎖の実装対策（order_manager）
Period:
- Analysis window: `2026-02-20` ～ `2026-02-27`
- Source: VM `orders.db`, `trades.db`, `strategy_control.db`

Fact:
- `strategy_control_exit_disabled` が短時間に集中し、同一 trade/client で close reject が連鎖。
- `CLIENT_TRADE_ID_ALREADY_EXISTS` が多発し、filled 復元不能時に同一CID再送の再拒否ループが残っていた。
- `entry_probability` が欠損/不正な entry_thesis 経路でも、order-manager 側で reject せず通る余地があった。

Failure Cause:
1. close preflight で `strategy_control` 拒否が続くと、緊急状態でも fail-open 経路が無かった。
2. duplicate CID reject 時、filled 復元不可のケースで CID を更新せず次の reject を誘発。
3. `entry_thesis` の必須意図項目（`entry_probability`, `entry_units_intent`, `strategy_tag`）の order-manager 側検証が弱い。

Improvement:
1. `order_manager.close_trade` に連続ブロック監視 + emergency fail-open 条件を追加。
2. market/limit の duplicate CID reject で再採番リトライを追加（filled復元不可時）。
3. entry-intent guard を追加し、必須項目欠損を `entry_intent_guard_reject` で拒否。
4. reject ログへ `request_payload` を必須付与して追跡精度を上げた。

Verification:
1. ユニットテスト:
   - `tests/execution/test_order_manager_log_retry.py`
   - `tests/execution/test_order_manager_exit_policy.py`
   - `tests/workers/test_scalp_ping_5s_worker.py`
2. 反映後VM監査（予定）:
   - `orders.status='strategy_control_exit_disabled'` と `close_bypassed_strategy_control` の推移
   - `orders.status='rejected' and error_code='CLIENT_TRADE_ID_ALREADY_EXISTS'` の再発率
   - `orders.status='entry_intent_guard_reject'` の戦略別件数

Status:
- in_progress

## 2026-02-27 07:33 UTC / 2026-02-27 16:33 JST - `scalp_ping_5s_b_live` の `close_reject_no_negative` 連発停止
Period:
- 24h: `datetime(ts) >= datetime('now','-24 hours')`
- 6h: `datetime(ts) >= datetime('now','-6 hours')`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `journalctl -u quant-strategy-control.service`

Fact:
- 稼働状態:
  - `quant-strategy-control`, `quant-order-manager`, `quant-position-manager`, `quant-market-data-feed` は全て `active`。
  - `journalctl` heartbeat は `global(entry=True, exit=True, lock=False)` を継続。
- `strategy_control.db`:
  - `strategy_control_flags` は全行 `entry_enabled=1` かつ `exit_enabled=1`（`entry=1 & exit=0` は 0 件）。
- `orders.db`（24h）:
  - `close_reject_no_negative=37`
  - `strategy_control_exit_disabled=0`（ステータス上位に非出現）
  - `client_order_id LIKE '%scalp_ping_5s_b_live%'` が `35` 件、`wick` 系が `2` 件。
- 直近拒否サンプルでは `exit_reason=candle_*` / `take_profit` で
  `status=close_reject_no_negative` が反復し、EXITシグナルが通過していない。

Failure Cause:
1. `scalp_ping_5s_b(_live)` が `scalp_ping_5s` の `neg_exit.strict_no_negative=true` を継承していた。
2. B系の exit_reason（`candle_*`, `take_profit`）が strict allow と整合せず、
   no-negative ガードが実運用で EXIT 詰まりを発生させた。
3. `strategy_control_exit_disabled` は解消済みで、今回の主因は strategy-control ではなく `neg_exit` ポリシー側。

Improvement:
1. `config/strategy_exit_protections.yaml`:
   - `scalp_ping_5s_b` / `scalp_ping_5s_b_live` に
     `neg_exit.strict_no_negative=false`
     `neg_exit.allow_reasons=["*"]`
     `neg_exit.deny_reasons=[]`
     を追加し、B系を no-block 運用へ統一。

Verification:
1. デプロイ後、`orders.db` 1h/6h で `close_reject_no_negative` の総数と
   `LIKE '%scalp_ping_5s_b_live%'` 件数が連続減少すること。
2. `close_ok` が維持され、`strategy_control_exit_disabled` が 0 を維持すること。
3. 24hで B系の負け玉平均保有時間（close遅延）が短縮すること。

Status:
- in_progress

## 2026-02-27 06:41 UTC / 2026-02-27 15:41 JST - 方向精度劣化（B/C）+ 単発大損（M1/MACD）を同時圧縮
Period:
- 集計時刻: `2026-02-27 06:41 UTC`（`15:41 JST`）
- 期間: 直近 `6h` / `24h`
- Source: VM `fx-trader-vm` (`/home/tossaki/QuantRabbit/logs/trades.db`, `orders.db`, `metrics.db`) + `scripts/oanda_open_trades.py`

Fact:
- 24h（主要赤字）:
  - `scalp_ping_5s_c_live`: `444 trades`, `-3394.6 JPY`, `-434.7 pips`
  - `scalp_ping_5s_b_live`: `264 trades`, `-519.9 JPY`, `-186.5 pips`
- 6h（主要赤字）:
  - `scalp_macd_rsi_div_live`: `1 trade`, `-279.4 JPY`, `-6.4 pips`
  - `scalp_ping_5s_b_live`: `223 trades`, `-129.2 JPY`, `-115.9 pips`
  - `M1Scalper-M1`: `20 trades`, `-90.1 JPY`, `-10.1 pips`
  - `scalp_ping_5s_c_live`: `122 trades`, `-36.1 JPY`, `-99.7 pips`
- close reason（6h）:
  - B: `STOP_LOSS_ORDER 112 trades / -265.1 JPY`、`TAKE_PROFIT_ORDER 94 / +116.5 JPY`
  - C: `STOP_LOSS_ORDER 64 / -28.7 JPY`、`MARKET_ORDER_TRADE_CLOSE 25 / -20.9 JPY`
  - M1: `MARKET_ORDER_TRADE_CLOSE 20 / -90.1 JPY`
  - MACD: `MARKET_ORDER_TRADE_CLOSE 1 / -279.4 JPY`
- 執行系:
  - `order_success_rate(avg)=0.952`, `reject_rate(avg)=0.048`
  - `decision_latency_ms(avg)=194.357`, `data_lag_ms(avg)=2032.046`
- open trades:
  - extrema short 3本のみ（`-1122/-187/-433 units`）、3本とも `stopLoss=null`。

Failure Cause:
1. B/C は勝ち負け混在でも `SL側の損失幅` が優位で、方向ミス時のpayoff非対称が継続。
2. M1（long固定）とMACD（大ロット）の単発逆行で、短時間に資産毀損を増幅。
3. preflight は動作しているが、低品質エントリーの通過ロットがなお大きい。

Improvement:
1. B: `max orders/min`, `base/max units`, `conf floor`, `entry_probability_align floor`, `preserve-intent` を厳格化。
2. C: 同様に `頻度/ロット/確率閾値` を引き上げ、`force-exit hold/loss` を短縮・厳格化。
3. order-manager: B/C の `preserve-intent` と `forecast gate` を強化し、service実効値を同期。
4. M1: `SIDE_FILTER=none` に戻しつつ `PERF_GUARD_ENABLED=1`、`base/min units` を圧縮。
5. MACD: `range必須化`, `divergence閾値強化`, `spread上限厳格化`, `base/min units` を縮小。

Verification:
1. デプロイ後 `quant-order-manager` / B / C / M1 / MACD 各serviceの実効envを `/proc/<pid>/environ` で照合。
2. 直近2h/6hで B/C の `STOP_LOSS_ORDER 比率` と `avg_loss_jpy` が低下するか確認。
3. M1/MACD の単発損失（`realized_pl`）が縮小し、`-200 JPY` 超級の再発頻度が下がるか監査。

Status:
- in_progress

## 2026-02-27 05:55 UTC / 14:55 JST - B/C 継続赤字に対する追加圧縮（損失幅優先）
Period:
- Audit window: 24h / 6h（VM `trades.db` / `orders.db` / `metrics.db`）
- Post-deploy spot check: `2026-02-27T05:46:00+00:00` 以降（`fe400c8` 反映後）

Purpose:
- B/C を停止せず稼働継続したまま、期待値の主因である「平均損失過大」をまず縮小する。

Hypothesis:
1. 執行コスト（spread/slip/submit latency）は許容範囲で、主因は entry 品質と損切り幅。
2. B は `avg_win < avg_loss` が明確なので、SL/force-exit の縮小でマイナス勾配を圧縮できる。
3. C は低品質エントリー通過が残っているため、prob/conf floor と頻度を絞ると赤字を抑えられる。

Fact:
- 24h realized:
  - `scalp_ping_5s_c_live`: `453 trades / -1077.6`
  - `scalp_ping_5s_b_live`: `454 trades / -607.3`
  - `WickReversalBlend`: `7 trades / +332.3`
  - `scalp_extrema_reversal_live`: `14 trades / +30.9`
- 6h realized:
  - `scalp_ping_5s_b_live`: `199 trades / -121.6`（`avg_win=+1.28`, `avg_loss=-2.52`）
  - `scalp_ping_5s_c_live`: `124 trades / -36.0`（`avg_win=+0.40`, `avg_loss=-0.68`）
- 24h orders status:
  - `margin_usage_projected_cap=548`, `margin_usage_exceeds_cap=368`, `rejected=314`, `entry_probability_reject=233`, `slo_block=136`
- execution precision（`analyze_entry_precision.py`）:
  - `spread_pips mean ≈ 0.80`, `slip p95 ≈ 0.20`, `latency_submit p50 ≈ 203-206ms`
  - コストよりも strategy-side の期待値設計がボトルネック。
- post-deploy spot（`05:46+00:00` 以降）:
  - B fill 3件で `avg_preflight_ms=125`, `avg_submit_to_fill_ms=213`（timeout起因の長遅延は未再発）

Failure Cause:
1. B/C とも pay-off が負（特に B は損失幅が利益幅を大きく上回る）。
2. B/C の margin cap 多発は「意図サイズ過大」を示し、期待値の低い試行が多い。
3. B/C の perf guard が disable のままで、悪化局面の縮小が効いていない。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_MAX_ACTIVE_TRADES=6`（from `10`）
   - `SCALP_PING_5S_B_MAX_PER_DIRECTION=4`（from `6`）
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=1`（from `0`）
   - `SCALP_PING_5S_B_SL_BASE_PIPS=1.8`（from `2.2`）
   - `SCALP_PING_5S_B_SL_MAX_PIPS=2.4`（from `3.0`）
   - `SCALP_PING_5S_B_SHORT_SL_BASE_PIPS=1.7`（from `2.0`）
   - `SCALP_PING_5S_B_SHORT_SL_MAX_PIPS=2.2`（from `2.8`）
   - `SCALP_PING_5S_B_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=2.0`（from `2.6`）
   - `SCALP_PING_5S_B_SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=1.8`（from `2.2`）
   - `SCALP_PING_5S_B_FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC=14`（from `20`）
   - `SCALP_PING_5S_B_FORCE_EXIT_RECOVERY_WINDOW_SEC=55`（from `75`）
   - `SCALP_PING_5S_B_FORCE_EXIT_RECOVERABLE_LOSS_PIPS=0.80`（from `1.05`）
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=5`（from `6`）
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=90`（from `110`）
   - `SCALP_PING_5S_C_MAX_UNITS=200`（from `240`）
   - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=1`（from `0`）
   - `SCALP_PING_5S_PERF_GUARD_ENABLED=1`（from `0`、fallback local同期）
   - `SCALP_PING_5S_C_CONF_FLOOR=78`（from `76`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.74`（from `0.70`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.64`（from `0.61`）

Impact Scope:
- 対象: `quant-scalp-ping-5s-b.service`, `quant-scalp-ping-5s-c.service`
- 非対象: V2導線（order-manager/position-manager/strategy-control）の共通ロジック変更なし。

Verification:
1. VMで env 反映確認（`/proc/<pid>/environ`）:
   - B/C の `PERF_GUARD_ENABLED=1`, `B SL/force-exit`, `C units/prob floor` を確認。
2. 反映後 2h/6h 監査:
   - B/C の `avg_loss` 縮小（B目標 `>-2.0`, C目標 `>-0.60`）
   - B/C の `realized_pl` 勾配改善（赤字幅の縮小）
   - `margin_usage_projected_cap`, `margin_usage_exceeds_cap` の件数低下
3. 執行品質監査:
   - `analyze_entry_precision.py` で `spread/slip/latency_submit` が悪化していないことを確認。

Status:
- in_progress

## 2026-02-27 06:05 UTC / 15:05 JST - order-manager 実効設定ズレ是正（B/C perf guard 未適用の修正）
Period:
- Audit window: deploy後 spot check（`2026-02-27T05:57:00+00:00` 以降）
- Source: VM `/proc/<order-manager-pid>/environ`, `orders.db`, `trades.db`

Purpose:
- 直近チューニングが service 経路で実効化されていることを保証し、B/C の悪化局面で縮小運転を確実に発火させる。

Hypothesis:
1. `quant-order-manager` 側の `PERF_GUARD_ENABLED=0` が残っていると、worker側で有効化しても実際の preflight で効かない。
2. preserve-intent 閾値を service 側と worker 側で同値にしないと、経路差で通過品質がブレる。

Fact:
- VM 実測（`quant-order-manager` process env）で以下を確認:
  - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
  - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=0`
  - `SCALP_PING_5S_PERF_GUARD_ENABLED=0`
  - preserve-intent も旧閾値（B `0.72/0.25/0.42`, C `0.68/0.35/0.72`）が残存
- 一方で worker env では `PERF_GUARD_ENABLED=1` へ更新済みで、service/local で実効値が不一致だった。

Improvement:
1. `ops/env/quant-order-manager.env`
   - B preserve-intent: `REJECT_UNDER=0.74`, `MAX_SCALE=0.38`
   - C preserve-intent: `REJECT_UNDER=0.72`, `MIN_SCALE=0.30`, `MAX_SCALE=0.64`
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=1`
   - `SCALP_PING_5S_C_PERF_GUARD_MODE=reduce`, `SCALP_PING_5S_C_PERF_GUARD_ENABLED=1`
   - `SCALP_PING_5S_PERF_GUARD_MODE=reduce`, `SCALP_PING_5S_PERF_GUARD_ENABLED=1`
2. `ops/env/scalp_ping_5s_b.env` / `ops/env/scalp_ping_5s_c.env`
   - preserve-intent 閾値を service 側と同値に同期（B `0.74/0.25/0.38`, C `0.72/0.30/0.64`）

Impact Scope:
- 対象: `quant-order-manager.service` preflight（B/C）
- 非対象: 共通導線仕様（V2役割分離）は変更なし

Verification:
1. デプロイ後 `quant-order-manager` の `/proc/<pid>/environ` で上記キーを確認。
2. `orders.db` で B/C の `entry_probability_reject` 増加と `margin_usage_*cap` 減少傾向を確認。
3. 2h/6h の B/C `avg_loss` が改善することを継続監査。

Status:
- in_progress

## 2026-02-27 01:30 UTC / 2026-02-27 10:30 JST - B/C 非エントリーの直接因子を解除（revert復帰 + rate limit緩和 + service timeout短縮）
Period:
- VM実測: `2026-02-27 00:56-01:26 UTC`
- Source: `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `journalctl -u quant-order-manager.service`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/ops/env/*.env`

Fact:
- B/C worker の startup ログで `SCALP_PING_5S_REVERT_ENABLED is OFF` が継続。
- `ops/env/scalp_ping_5s_{b,c}.env` の実値が
  - `SCALP_PING_5S_{B,C}_REVERT_ENABLED=0`
  - `SCALP_PING_5S_{B,C}_MAX_ORDERS_PER_MINUTE=4`
- 直近10分ログカウント:
  - B: `order_manager_none=39`, `revert_disabled=17`, `rate_limited=39`
  - C: `order_manager_none=49`, `revert_disabled=27`, `rate_limited=46`
- `quant-v2-runtime.env` で `ORDER_MANAGER_SERVICE_TIMEOUT=20.0`（fallback local有効）を確認。

Failure Cause:
1. `REVERT_ENABLED=0` により `no_signal:revert_disabled` が恒常化し、シグナル生成が大きく欠損。
2. `MAX_ORDERS_PER_MINUTE=4` が過抑制となり、候補シグナルの大半が `rate_limited` で棄却。
3. order-manager service timeout が長く、応答遅延時に `order_manager_none` を誘発してエントリー密度をさらに低下。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_REVERT_ENABLED: 0 -> 1`
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE: 4 -> 24`
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_REVERT_ENABLED: 0 -> 1`
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE: 4 -> 24`
3. `ops/env/quant-v2-runtime.env`
   - `ORDER_MANAGER_SERVICE_TIMEOUT: 20.0 -> 8.0`
4. `execution/strategy_entry.py`
   - 協調/パターンゲートで `coordinated_units=0` になった場合に、
     `client_order_id` へ reject理由を `order_status` キャッシュ記録するよう変更。
   - `order_manager_none` に潰れていた reject 内訳（`coordination_*` / `pattern_gate_*`）を可視化。

Verification:
1. デプロイ後ログで `revert_enabled=1` を確認し、`revert_disabled` 件数が減少していること。
2. 直近10分の `rate_limited` 件数が B/C ともに低下していること。
3. `orders.db` で `submit_attempt` と `filled` の発生密度が維持/改善していること。
4. `order_manager_none` と `CLIENT_TRADE_ID_ALREADY_EXISTS` が逓減していること。
5. B/C ログの reject reason が `order_manager_none` から実理由（`coordination_*` 等）へ置換されること。

Status:
- in_progress

## 2026-02-27 05:55 UTC / 2026-02-27 14:55 JST - 発火頻度不足の補正
Period:
- Post-adjust short window: `2026-02-27 05:06` 以降
- Source: VM `orders.db`, `journalctl`

Fact:
- B/C の圧縮後、timeout/none は 0 を維持。
- 一方で短時間窓の約定が薄く、Wick/Extrema の寄与立ち上がりが不足。

Failure Cause:
1. 収益側戦略の監視間隔/クールダウンが相対的に長く、短期機会を拾い切れていない。

Improvement:
1. Extrema:
   - `LOOP_INTERVAL_SEC=1.5`
   - `COOLDOWN_SEC=35`
   - `LOOKBACK=24`
   - `HIGH/LOW_BAND_PIPS=0.9`
2. Wick:
   - `LOOP_INTERVAL_SEC=2.0`
   - `COOLDOWN_SEC=4`
   - `WICK_BLEND_BBW_MAX=0.0018`

Verification:
1. 再反映後 10-20 分窓で Wick/Extrema の `submit_attempt` と `filled` 増加を確認。
2. 同窓で timeout/none が再増加していないことを確認。

Status:
- in_progress

## 2026-02-27 05:40 UTC / 2026-02-27 14:40 JST - 即時収益寄せの第2段（発火不足解消）
Period:
- Analysis window: 直近60分（VM `trades.db` / `orders.db`）
- Source: `trades.db`, `orders.db`, `quant-scalp-{ping_5s_b,ping_5s_c,wick,extrema}` env

Fact:
- 直近60分:
  - `scalp_ping_5s_b_live`: `39 trades / -20.7 JPY`
  - `scalp_ping_5s_c_live`: `21 trades / -11.0 JPY`
- Wick/Extrema は稼働中だが、直近窓で新規寄与が薄く、利益側の回転不足。

Failure Cause:
1. B/C の頻度がまだ高く、短期の負け寄与を削り切れていない。
2. Wick/Extrema の閾値・クールダウンが相対的に厳しく、相場適合時の発火数が不足。

Improvement:
1. B/C 頻度を追加圧縮:
   - `MAX_ORDERS_PER_MINUTE: 12 -> 8`（B/C）
2. Extrema 発火緩和:
   - `COOLDOWN_SEC=45`, `MAX_OPEN_TRADES=2`, `MIN_ENTRY_CONF=57`
   - spread/range/rsi/leading-profile 閾値を緩和。
3. Wick 発火緩和:
   - `COOLDOWN_SEC=5`, `MAX_OPEN_TRADES=3`
   - range/adx/tick/leading-profile 閾値を緩和。

Verification:
1. デプロイ後 10-30 分窓で
   - B/C の `submit_attempt` 減少
   - Wick/Extrema の `submit_attempt` / `filled` 増加
   - 合算 realized P/L の改善
2. timeout 系 (`Read timed out`, `order_manager_none`) が増えていないことを確認。

Status:
- in_progress

## 2026-02-27 05:20 UTC / 2026-02-27 14:20 JST - B/C負け寄与の即圧縮 + Wick再配分
Period:
- Analysis window: 24h (`datetime(close_time) >= now - 24 hours`)
- Source: VM `trades.db`, `orders.db`, worker env/systemd overrides

Fact:
- 24h realized P/L:
  - `scalp_ping_5s_c_live`: `493 trades / -1984.2 JPY`
  - `scalp_ping_5s_b_live`: `415 trades / -588.6 JPY`
  - `WickReversalBlend`: `7 trades / +332.3 JPY`
- `orders.db` 24h では B/C の試行が高く、`submit_attempt` は
  `b=600`, `c=608`。高頻度・低EVの積み上げが継続していた。

Failure Cause:
1. B/C が no-stop のまま高頻度運転（`MAX_ORDERS_PER_MINUTE=24`）で負け寄与を増幅。
2. preserve-intent / leading-profile の閾値が緩く、低品質通過が残存。
3. B は systemd override（`BASE_ENTRY_UNITS=520`）が env より強く、圧縮意図とズレていた。

Improvement:
1. B/Cの頻度・サイズを即圧縮:
   - `MAX_ORDERS_PER_MINUTE: 24 -> 12`
   - `BASE_ENTRY_UNITS: B 450->380, C 170->140`
2. B/Cの通過閾値を引き上げ:
   - preserve-intent reject: `B 0.68`, `C 0.66`
   - leading profile reject: `B 0.68/0.74`, `C 0.66/0.72`
   - confidence / align floor も引き上げ。
3. B service override を同値化:
   - `BASE_ENTRY_UNITS=420`, `MAX_UNITS=780`,
     `ORDER_MANAGER_PRESERVE_INTENT_*` を env 同値へ同期。
4. 勝ち寄与へ小幅再配分:
   - `WickReversalBlend` base units `9500 -> 10200`
   - cooldown `8 -> 7`

Verification:
1. デプロイ後に `HEAD == origin/main` と各 service `active` を確認。
2. 10-30分窓で以下を比較:
   - B/C `submit_attempt` と `filled` の絶対数（過剰頻度が落ちること）
   - B/C realized P/L の損失勾配
   - `WickReversalBlend` の約定寄与

Status:
- in_progress

## 2026-02-27 03:20 UTC / 2026-02-27 12:20 JST - order-manager API の event-loop 詰まり対策
Period:
- Analysis window: 直近の `Read timed out (45.0)` 多発区間（`2026-02-27` UTC）
- Source: `workers/order_manager/worker.py`, `execution/order_manager.py`, `orders.db` 集計ログ

Fact:
- strategy worker 側で `order_manager service call failed ... Read timed out (45.0)` が継続。
- `ORDER_MANAGER_SERVICE_WORKERS=6` に増やした後も timeout 警告が残った。
- `workers/order_manager/worker.py` は `async def` endpoint で
  `execution.order_manager.*` を直接 await していた。
- `execution.order_manager` の実処理は OANDA API / SQLite への同期I/Oを含むため、
  endpoint event loop 占有が発生しやすい。

Failure Cause:
1. service worker の event loop が同期I/O処理を抱え、同時リクエスト時に head-of-line blocking が起きる。
2. RPC 応答遅延が client 側 timeout（45秒）を引き起こし、fallback/再送連鎖のトリガーになる。

Improvement:
1. `workers/order_manager/worker.py` に `_run_order_manager_call` を追加し、
   `execution.order_manager.*` の実行を `asyncio.to_thread(... asyncio.run(...))` に統一。
2. `cancel_order / close_trade / set_trade_protections / market_order / coordinate_entry_intent / limit_order`
   の全 endpoint を同ヘルパー経由へ切替。
3. `ORDER_MANAGER_SERVICE_SLOW_REQUEST_WARN_SEC`（default 8秒）を追加し、
   遅いリクエストを `slow_request` ログで監査可能化。

Verification:
1. ローカル回帰:
   - `python3 -m py_compile workers/order_manager/worker.py`（OK）
   - `pytest -q tests/execution/test_order_manager_safe_json.py tests/execution/test_order_manager_preflight.py`（29 passed）
2. VM確認（実施予定）:
   - `quant-order-manager.service` 再起動後 10分窓で
     `Read timed out` 件数と `order_manager_none` 件数の減少を確認。
   - `orders.db` の `filled/rejected/duplicate_recovered` 比率を同一窓で比較。

Status:
- in_progress（この時点では VM SSH が `Permission denied (publickey)` で未検証）

## 2026-02-27 04:55 UTC / 2026-02-27 13:55 JST - timeout 閾値を実測遅延に合わせて再調整
Period:
- Post-deploy check: `2026-02-27 04:47` ～ `04:55` UTC
- Source: VM `quant-order-manager.service` journald, `orders.db`, strategy worker logs

Fact:
- `quant-order-manager.service` 再起動後に
  `slow_request op=market_order elapsed=49.047s` を確認。
- 同時点で strategy worker 側に
  `order_manager service call failed ... Read timed out` が短時間で残存。

Failure Cause:
1. service client timeout `45.0s` が、49秒級の正常処理を timeout 扱いしていた。
2. timeout 判定が local fallback/再送を誘発し、reject ノイズと遅延を増幅しうる。

Improvement:
1. `ops/env/quant-v2-runtime.env` で
   `ORDER_MANAGER_SERVICE_TIMEOUT=60.0`（from `45.0`）へ更新。
2. `slow_request` 監査ログは継続し、閾値超過の実数を追跡する。

Verification:
1. デプロイ後に `quant-order-manager.service` 再起動と health 応答 (`/health=200`) を確認。
2. `orders.db` 直近窓で `filled` 継続・`rejected` 0 を確認（短時間窓）。
3. 次の監視ポイント:
   - `Read timed out` 件数の減少
   - `duplicate_recovered` / `rejected` 比率の改善

Status:
- in_progress

## 2026-02-27 03:25 UTC / 2026-02-27 12:25 JST - `coordination_reject` 誤ラベルと短期 sell 偏重の同時是正
Period:
- VM確認: `2026-02-27 02:55` ～ `03:20` UTC
- Source: VM `orders.db` / `trades.db` / `journalctl` (`quant-scalp-ping-5s-b/c`, `quant-order-manager`)

Fact:
- 直近30分の `orders.db` では `scalp_ping_5s_b/c` は `sell` 約定のみ（`filled=23`）だったが、2時間窓では `buy/sell` 両側の約定履歴あり。
- `journalctl` 側では `order_reject:coordination_reject` が多発していた一方、`orders.db` には同時刻の `rejected` が乖離しており、拒否理由の可観測性にズレがあった。
- `strategy_entry.market_order/limit_order` は `forecast_fusion` / `entry_leading_profile` で `units=0` になっても coordination へ進み、最終的に `coordination_reject` として記録されうる実装だった。

Failure Cause:
1. 前段拒否（forecast/leading/feedback）と coordination拒否の原因ラベルが混線し、実際の方向判定失敗点が見えない。
2. C の momentum trigger が `long=0.18 / short=0.08` と非対称で、短期的に sell 偏重へ寄りやすい設定だった。

Improvement:
1. `execution/strategy_entry.py`
   - `units=0` を前段で検知した時点で即 return し、`strong_contra_forecast` / `entry_leading_profile_reject` などの実理由を `_cache_order_status` へ記録。
   - side 記録を `units==0` 時でも要求方向（requested units）基準に統一。
   - coordination拒否と前段拒否を分離し、`coordination_reject` の過大計上を抑制。
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_SHORT_MOMENTUM_TRIGGER_PIPS=0.09`（from `0.08`）
3. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_SHORT_MOMENTUM_TRIGGER_PIPS=0.10`（from `0.08`）
   - `SCALP_PING_5S_C_LONG_MOMENTUM_TRIGGER_PIPS=0.12`（from `0.18`）

Verification:
1. Unit test: `pytest -q tests/execution/test_strategy_entry_forecast_fusion.py`（17 passed）
2. VM反映後、`journalctl` の `order_reject:*` が `strong_contra_forecast` / `entry_leading_profile_reject` などに分解されること。
3. 反映後30～60分で B/C の side 分布（buy/sell）と `pl_pips` 偏りを再集計し、sell固定化が緩和しているか監査する。

Status:
- in_progress

## 2026-02-27 01:35 UTC / 2026-02-27 10:35 JST - order_manager timeout起点の重複CIDを回収し、エントリー取りこぼしを削減
Period:
- 調査窓: `2026-02-27 01:00` ～ `01:33` UTC（`10:00` ～ `10:33` JST）
- Source: VM `journalctl`（`quant-order-manager`, `quant-scalp-ping-5s-b/c`, `quant-scalp-extrema-reversal`）, `orders.db`, `trades.db`

目的:
- `Read timed out (20s)` で service call が落ち、同一 `client_order_id` 再送から
  `CLIENT_TRADE_ID_ALREADY_EXISTS` が連鎖する経路を解消する。

仮説:
1. 20秒RPC timeout が短く、order_manager 側で実際に約定済みでも caller が失敗扱いして再送している。
2. 同一CIDの「filled済み」を再送側で回収できれば、reject扱いを成功に変換できる。

Fact:
- 直近30分 `orders.db` の `rejected=11` は全て `CLIENT_TRADE_ID_ALREADY_EXISTS`。
- 同一CIDで `filled` の後に `rejected` が発生する実例を確認:
  - `qr-1772155441497-scalp_ping_5s_c_live-l4d5f062d`
    - `01:24:51 filled ticket=406075`
    - `01:25:11 rejected CLIENT_TRADE_ID_ALREADY_EXISTS`
- B/C/Extrema で `order_manager service call failed ... Read timed out. (read timeout=20.0)` を継続観測。

Improvement:
1. `execution/order_manager.py`
   - `CLIENT_TRADE_ID_ALREADY_EXISTS` 発生時に、同一 `client_order_id` の既存 `filled` 行を `orders.db` から逆引きし、`trade_id` を回収して `duplicate_recovered` として成功返却する導線を追加。
   - service timeout 後の local fallback 前に同一CIDの `orders.db` 状態を最大10秒ポーリングし、`filled/rejected` など終端状態を先に回収して二重送信を抑止。
2. `ops/env/quant-v2-runtime.env`
   - `ORDER_MANAGER_SERVICE_TIMEOUT=45.0`（from `8.0`）
   - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_WAIT_SEC=10.0`
   - `ORDER_MANAGER_SERVICE_TIMEOUT_RECOVERY_POLL_SEC=0.5`
3. `ops/env/quant-order-manager.env`
   - `ORDER_MANAGER_SERVICE_WORKERS=6`（from `4`）

影響範囲:
- order_manager の新規注文導線（`market_order`）のみ。
- EXIT共通ロジックや戦略ローカル判定には非侵襲。

検証手順:
1. デプロイ後15分で `orders.db` の `status='rejected'` うち `CLIENT_TRADE_ID_ALREADY_EXISTS` 件数を前窓比較。
2. 同一CIDに `filled + rejected` が残っても、worker側で `order_manager_none` が減ることを journal で確認。
3. 30分窓で `scalp_ping_5s_b/c` の `realized_pl` と `filled/submit_attempt` を再集計して改善有無を確認。

Status:
- in_progress

## 2026-02-26 12:19 UTC / 2026-02-26 21:19 JST - PDCA深掘り（`perf_block` 固定化 + `orders.db` ロック + SLO劣化の重畳）
Period:
- 監査期間: 直近24h（`2026-02-25 12:11` ～ `2026-02-26 12:19` UTC）
- ロック集中窓: `2026-02-26 11:26` ～ `11:46` UTC（`20:26` ～ `20:46` JST）
- Source: VM `journalctl -u quant-order-manager.service`, `journalctl -u quant-bq-sync.service`, `journalctl -u quant-position-manager.service`, `/home/tossaki/QuantRabbit/logs/{orders,metrics}.db`, `lsof`, `systemctl cat quant-bq-sync.service`

Fact:
- `orders.db` 24h集計（`rows=44862`）:
  - `preflight_start=17121`, `perf_block=16389`, `probability_scaled=7348`, `entry_probability_reject=594`
  - `submit_attempt=545`, `filled=541`（filled/submit `=99.3%`）
  - 戦略別 `perf_block`: `scalp_ping_5s_c_live=7032`, `scalp_ping_5s_b_live=4577`, `scalp_ping_5s_flow_live=3577`, `M1Scalper-M1=577`
- `perf_block` の実ログ内訳（`[ORDER][OPEN_REJECT]` 行）:
  - 合計 `191`
  - `scalp_ping_5s_b_live=157`, `M1Scalper-M1=34`
  - 主因ノート: `perf_block:hard:hour9:failfast:pf=0.56`（88件）, `hour11:failfast:pf=0.15`（36件）, `failfast:pf=0.38`（34件）
- `entry_probability` skip は `RangeFader` 系中心（sell/buy/neutral 合計74件）、`scalp_ping_5s_c_live=5`, `scalp_ping_5s_b_live=2`。
- `database is locked` は 24hで `83` 件。発生分は `35` 分に集中し、最多は `11:31/11:34`（各5件）。
- lock集中分（例 `11:26-11:42`）は `submit_attempt/filled=0` かつ `manual_margin_pressure` 併発分が多く、発注可否判定の再試行だけが増える形になっていた。
- 同時点で `lsof /home/tossaki/QuantRabbit/logs/orders.db` は PID `3400`（`run_sync_pipeline.py --interval 60 --bq-interval 300 --limit 1200`）が `40+` FD を保持。加えて戦略ワーカー PID `682/706` が保持。
- `orders.db` 自体は `6.7G` / `921,528 rows`（`2025-12-29` ～ `2026-02-26`）まで増大。
- `quant-bq-sync.service` は 60秒周期で常時 `sync_trades start`。env は `POSITION_MANAGER_SERVICE_ENABLED=1`, `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=1`, `PIPELINE_DB_READ_TIMEOUT_SEC=2.0`。
- `quant-position-manager.service` は 24hで timeout/busy 警告が複数（`fetch_recent_trades timeout`, `sync_trades timeout`, `position manager busy`）。
- `metrics.db` 24h:
  - `data_lag_ms`: `p50=768.9`, `p90=1958.7`, `p95=3603.5`, `p99=13325.1`, `max=225164.5`、閾値超過（`>1500ms`）`16.21%`
  - `decision_latency_ms`: `p50=24.4`, `p90=99.4`, `p95=678.8`, `p99=11947.6`, `max=37865.4`、閾値超過（`>1200ms`）`3.98%`
- `manual_margin_pressure=36` 件（24h）で、`scalp_ping_5s_b_live=29`, `scalp_ping_5s_c_live=7`。サンプルに `manual_net_units=-8500`, `margin_available_jpy=4677.279` を確認。

Failure Cause:
1. `perf_block` は一時的ノイズではなく、`scalp_ping_5s_b_live` と `M1Scalper-M1` の failfast 判定が時間帯別に固定化している。
2. `orders.db` への高頻度書き込み（order-manager）と、高頻度読み取り（bq-sync + position-manager fallback local）が重なり、ロック競合を増幅している。
3. `data_lag_ms` のテール悪化と `manual_margin_pressure` が同時に存在し、通るべき注文の密度を下げて改善ループが遅れる。

Hypothesis:
- `run_sync_pipeline.py` の orders 参照導線（毎分複数クエリ）と `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL=1` の併用で、`orders.db` ハンドル保持が増えやすく、ロック競合を誘発している可能性が高い（A/B確認が必要）。
- `preflight_start` 行の `request_json` が `{"note":"preflight_start",...}` のみで戦略情報を持たず、ボトルネック時間帯の戦略別分解を難化させている。

Improvement:
1. P0: `quant-bq-sync` の `POSITION_MANAGER_SERVICE_FALLBACK_LOCAL` を無効化し、service timeout 時は stale キャッシュ優先でローカルDBフォールバックを抑制する。
2. P0: `orders.db` へ `preflight_start/perf_block` 記録時の `strategy_tag` と `reject_note` 永続化を必須化する（監査盲点の解消）。
3. P1: `run_sync_pipeline.py` の SQLite 読み取りを明示 close に統一し、read-only URI（`mode=ro`）・短timeout・接続数上限を導入する。
4. P1: `replay_quality_gate` / `trade_counterfactual` の worker 対象を `TrendMA/BB_RSI` 以外（`scalp_ping_5s_b_live`, `M1Scalper-M1` など）へ拡張し、failfast固定化へ直接フィードバックする。
5. P1: `manual_margin_pressure` 発火時の段階的建玉縮退（manual併走含む）を優先し、`margin_available_jpy` の下限回復を先に実行する。

Verification:
1. `journalctl -u quant-order-manager.service --since "1 hour ago" | grep -c "database is locked"` が `<=5`。
2. `lsof /home/tossaki/QuantRabbit/logs/orders.db` で PID `3400` の FD 本数が `<=10` に低下。
3. `metrics.db` で `data_lag_ms p95 < 1500`, `decision_latency_ms p95 < 1200` を継続達成。
4. `perf_block` 主因（`hour9/hour11 failfast`）の件数が日次で逓減し、`submit_attempt`/`filled` 密度が回復する。
5. `manual_margin_pressure` が 24h 連続で `0`（または明確な逓減）になる。

Status:
- in_progress

## 2026-02-27 01:25 UTC / 2026-02-27 10:25 JST - `position_manager sync_trades` 過負荷を設定で緩和
Period:
- VM実測: `2026-02-27 00:55-01:20 UTC`
- Source: `journalctl -u quant-position-manager.service`, `journalctl -u quant-scalp-wick-reversal-blend.service`

Fact:
- `position_manager` で `sync_trades timeout (8.0s)` / `position manager busy` が高頻度発生。
- WickBlend 側にも `/position/sync_trades` 失敗警告が継続し、処理遅延を誘発。

Failure Cause:
1. `sync_trades` の取得上限・呼び出し間隔・キャッシュ窓が短く、負荷集中時に timeout 連鎖していた。

Improvement:
1. `ops/env/quant-v2-runtime.env`
   - `POSITION_MANAGER_MAX_FETCH=600`（new）
   - `POSITION_MANAGER_SYNC_MIN_INTERVAL_SEC=4.0`（from `2.0`）
   - `POSITION_MANAGER_SYNC_CACHE_WINDOW_SEC=4.0`（from `1.5`）
   - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC=12.0`（from `8.0`）
   - `POSITION_MANAGER_WORKER_SYNC_TRADES_CACHE_TTL_SEC=3.0`（new）
   - `POSITION_MANAGER_WORKER_SYNC_TRADES_STALE_MAX_AGE_SEC=120.0`（from `60.0`）
   - `POSITION_MANAGER_WORKER_SYNC_TRADES_MAX_FETCH=600`（from `1000`）

Verification:
1. 再起動後に `quant-position-manager` の `sync_trades timeout` / `position manager busy` 件数が減少。
2. WickBlend の `position_manager service call failed path=/position/sync_trades` が減少。
3. `orders.db` の `submit_attempt -> filled` 変換率が維持/改善。

Status:
- in_progress

## 2026-02-27 01:15 UTC / 2026-02-27 10:15 JST - B/Cを追加圧縮（停止なしで損失勾配をさらに低減）
Period:
- VM実測: 直近30分 `scalp_ping_5s_b_live=-35.7 JPY`, `scalp_ping_5s_c_live=-18.4 JPY`
- Source: `logs/trades.db`

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=450`（from `600`）
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=4`（from `5`）
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=170`（from `220`）
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=4`（from `5`）

Verification:
1. 反映後15分の strategy_tag 付き損益で B/C 合算損失が前窓より縮小すること。
2. `filled` 件数を維持しつつ `rejected` 比率が悪化しないこと。

Status:
- in_progress

## 2026-02-27 01:20 UTC / 2026-02-27 10:20 JST - B/C 方向精度リセット（sell固定解除 + 低確率遮断強化）
Period:
- 直近24h（`close_time >= now-24h`）
- post-check（`close_time >= 2026-02-27T00:36:34Z`）
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- post-check で `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` は実質 `sell` のみ。
  - `B sell: 27 trades / acc 37.0% / -11.8 pips`
  - `C sell: 22 trades / acc 40.9% / -10.3 pips`
- 24h の `entry_probability` 帯別では、低確率帯が大量通過して負け寄与。
  - `B [0.55,0.60): 57 trades / acc 40.4% / -46.4 pips`
  - `C [0.00,0.55): 185 trades / acc 38.4% / -129.9 pips`
  - `C [0.55,0.60): 38 trades / acc 28.9% / -56.2 pips`
- 稼働中プロセス環境（`/proc/<pid>/environ`）で
  `SCALP_PING_5S_{B,C}_SIDE_FILTER=sell`,
  `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER` が
  `B=0.48 / C=0.46` だった。

Failure Cause:
1. B/C の side が `sell` 固定になっており、方向選択の自由度を失っていた。
2. `REJECT_UNDER` が緩く、低 edge の entry が継続通過していた。
3. `entry_leading_profile` が無効で、strategy_entry 側の追加フィルタが働いていなかった。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
   - `SIDE_FILTER=none`
   - `ALLOW_NO_SIDE_FILTER=1`
2. 低確率遮断を引き上げ
   - `B: ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...=0.64`
   - `C: ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER...=0.62`
   - 反映先を `scalp_ping_5s_{b,c}.env` と `quant-order-manager.env` の両方で同値化
3. strategy_entry の追加ゲート有効化
   - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_ENABLED=1`
   - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_ENABLED=1`
   - `REJECT_BELOW` を B/C で引き上げ（B: `0.64/0.70`, C: `0.62/0.68`）

Verification:
1. VM反映後に `/proc/<pid>/environ` で上記キーが新値へ更新されていること。
2. `orders.db` の `entry_probability_reject` 件数が増え、`probability_scaled` の低帯通過が減ること。
3. post-check で `buy/sell` の両方向が再出現し、B/C の方向一致率が `>50%` へ回復すること。

Status:
- in_progress

## 2026-02-27 01:12 UTC / 2026-02-27 10:12 JST - WickBlendを`StageTracker`初期化失敗時も継続稼働に変更
Period:
- VM実測: `2026-02-27 00:59 UTC` で `quant-scalp-wick-reversal-blend.service` が再停止
- Source: `journalctl -u quant-scalp-wick-reversal-blend.service`

Fact:
- `stage_tracker` の `sqlite3.OperationalError: database is locked` が再発し、WickBlend が `failed`。
- 例外は `StageTracker()` 初期化時に発生し、worker プロセス自体が終了していた。

Failure Cause:
1. `stage_state.db` のロック競合が強い瞬間に、`StageTracker` 初期化例外がプロセス致命化していた。

Improvement:
1. `workers/scalp_wick_reversal_blend/worker.py`
   - `StageTracker()` を `try/except` 化。
   - 初期化失敗時は `_NoopStageTracker` にフォールバックし、worker 本体は稼働継続。

Verification:
1. `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py` が pass。
2. VM反映後に `quant-scalp-wick-reversal-blend.service` の `active` と `Application started!` 継続を確認。

Status:
- in_progress

## 2026-02-27 01:10 UTC / 2026-02-27 10:10 JST - order_manager API詰まり緩和（service workerを2並列化）
Period:
- VM実測: `2026-02-27 00:57-00:58 UTC`
- Source: `journalctl -u quant-scalp-ping-5s-b.service`, `journalctl -u quant-scalp-ping-5s-c.service`, `journalctl -u quant-order-manager.service`

Fact:
- B/C worker で `order_manager service call failed ... Read timed out (read timeout=20.0)` が発生。
- 同時間帯の `quant-order-manager` は active だが、`ORDER_MANAGER_SERVICE_WORKERS=1` で単一処理。
- timeout 後の再試行で `CLIENT_TRADE_ID_ALREADY_EXISTS` reject が混在し、約定効率が低下。

Failure Cause:
1. order_manager が単一workerのため、OANDA API待ちが重なると localhost API 応答が遅延。
2. strategy worker 側の service timeout 到達で request 経路が不安定化し、重複リクエストが発生しやすい。

Improvement:
1. `ops/env/quant-order-manager.env` の `ORDER_MANAGER_SERVICE_WORKERS` を `1 -> 4` へ段階引き上げ。
2. order_manager の同時処理能力を増やし、localhost API timeout と reject の発生頻度を下げる。

Verification:
1. `quant-order-manager.service` 再起動後に active 維持。
2. 直後ウィンドウで `Read timed out` 警告件数が減少することを `journalctl` で確認。
3. `orders.db` の `filled/submit_attempt` 比率をデプロイ前後で比較する。

Status:
- in_progress

## 2026-02-27 01:00 UTC / 2026-02-27 10:00 JST - `StageTracker` 起動時ロック再発をテーブル存在確認で抑止
Period:
- VM実測: `2026-02-27 00:46-00:50 UTC`
- Source: `journalctl -u quant-scalp-wick-reversal-blend.service`, `logs/stage_state.db`

Fact:
- `quant-scalp-wick-reversal-blend.service` が起動直後に
  `sqlite3.OperationalError: database is locked` で連続停止。
- 例外位置は `execution/stage_tracker.py` の初期DDL（`CREATE TABLE IF NOT EXISTS ...`）。
- 同時に `stage_state.db` は複数 worker (`order_manager`, `scalp_ping_5s`, `tick_imbalance` 等) が共有。

Failure Cause:
1. `StageTracker.__init__` が毎回スキーマDDLを書き込み実行し、共有DBのスキーマロック競合時に起動失敗する。
2. 既存テーブルでも DDL/ALTER を実行するため、起動時ロック競合の露出面が広い。

Improvement:
1. `execution/stage_tracker.py` に `_table_exists` / `_column_exists` を追加。
2. DDLは `_ensure_table` / `_ensure_column` 経由で「不足時のみ」実行へ変更。
3. 既存テーブル環境では起動時DDLを書き込まないため、スキーマロック競合を回避。

Verification:
1. `python3 -m py_compile execution/stage_tracker.py` が pass。
2. `pytest -q tests/test_stage_tracker.py` が `3 passed`。
3. VM反映後に `quant-scalp-wick-reversal-blend.service` の `Application started!` と連続稼働を確認する。

Status:
- in_progress

## 2026-02-27 00:48 UTC / 2026-02-27 09:48 JST - `WickReversalBlend` が `stage_tracker` ロックで停止する障害を修正
Period:
- Incident window: `2026-02-27 00:46` ～ `00:48` UTC
- Source: VM `journalctl -u quant-scalp-wick-reversal-blend.service`, repo `execution/stage_tracker.py`

Fact:
- `quant-scalp-wick-reversal-blend.service` が起動直後に `failed` へ遷移。
- 直近ログで `execution/stage_tracker.py` 初期化中に
  `sqlite3.OperationalError: database is locked` を確認。
- 同時に B/C ワーカーは稼働継続しており、勝ち筋戦略だけが停止していた。

Failure Cause:
1. `StageTracker.__init__` の schema 作成が単発 `execute` で、ロック競合時に即例外終了していた。
2. `stage_state.db` に `busy_timeout` / `WAL` / retry の耐性が不足していた。

Improvement:
1. `execution/stage_tracker.py`
   - `STAGE_DB_BUSY_TIMEOUT_MS` / `STAGE_DB_LOCK_RETRY` /
     `STAGE_DB_LOCK_RETRY_SLEEP_SEC` を追加。
   - 接続を `busy_timeout + WAL + autocommit` で初期化。
   - schema作成 SQL を lock retry 付き `_execute_with_lock_retry()` 経由へ変更。

Verification:
1. `python3 -m py_compile execution/stage_tracker.py` が pass。
2. `pytest -q tests/test_stage_tracker.py` が pass（`3 passed`）。
3. 反映後 `quant-scalp-wick-reversal-blend.service` が `active` を維持すること。

Status:
- in_progress

## 2026-02-27 00:33 UTC / 2026-02-27 09:33 JST - 勝ち筋寄せ再配分（WickBlend増量 + B/C縮小）
Period:
- Snapshot window: `2026-02-27 00:30` ～ `00:33` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- 自動戦略24h（strategy_tag あり）:
  - `WickReversalBlend`: `3 trades / +114.7 JPY`
  - `scalp_ping_5s_b_live`: `292 trades / -540.2 JPY`
  - `scalp_ping_5s_c_live`: `466 trades / -3403.0 JPY`
- 直近15分（自動のみ）は `-91.9 JPY` で、依然マイナス。

Failure Cause:
1. 損失寄与の大きい B/C の約定量が、勝ち筋に対して過大。
2. WickBlend は勝っているが発火頻度と配分が低く、寄与不足。

Improvement:
1. `ops/env/quant-scalp-wick-reversal-blend.env`
   - `SCALP_PRECISION_UNIT_BASE_UNITS=9500`（from `7000`）
   - `SCALP_PRECISION_UNIT_CAP_MAX=0.65`（from `0.55`）
   - `SCALP_PRECISION_COOLDOWN_SEC=8`（from `12`）
   - `SCALP_PRECISION_MAX_OPEN_TRADES=2`（from `1`）
   - `WICK_BLEND_RANGE_SCORE_MIN=0.40`（from `0.45`）
   - `WICK_BLEND_ADX_MIN/MAX=14/28`（from `16/24`）
   - `WICK_BLEND_BB_TOUCH_RATIO=0.18`（from `0.22`）
   - `WICK_BLEND_TICK_MIN_STRENGTH=0.30`（from `0.40`）
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=600`（from `720`）
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE=5`（from `6`）
3. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS=220`（from `260`）
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=5`（from `6`）

Verification:
1. 反映後10分で `WickReversalBlend` の `filled` 件数が増えること。
2. 反映後15分で自動損益（strategy_tagあり）が 0 以上へ改善すること。
3. 同窓で B/C の損失寄与（JPY）が縮小すること。

Status:
- in_progress

## 2026-02-27 01:05 UTC / 2026-02-27 10:05 JST - `margin_usage_projected_cap` 誤拒否（side cap と net-reducing の不整合）
Period:
- VM `journalctl -u quant-order-manager.service`（00:11〜00:20 UTC）
- VM `/home/tossaki/QuantRabbit/logs/orders.db`（`margin_usage_projected_cap` 行）

Fact:
- `quant-order-manager` で `margin_usage_projected_cap` が連発し、B/C の `sell` シグナルが約定前に落ちていた。
- 同時刻ログに `projected margin scale ... usage=0.921~0.946` が記録され、cap 付近として扱われていた。
- 一方で口座スナップショット（同VM実測）は `usage_total` が低位（例: 約 `0.03` 台）で、総ネット余力とは乖離していた。
- 乖離は「総ネット使用率」は低いが「同方向 side 使用率」だけが高い局面で顕在化した。

Failure Cause:
1. `MARGIN_SIDE_CAP_ENABLED=1` 経路で `usage/projected_usage` を side ベースへ上書きした後、net-reducing 例外も同じ side 値で判定していた。
2. そのため「総ネット使用率を下げる注文」でも `projected_usage < usage` 条件を満たせず、`margin_usage_projected_cap` として誤拒否されていた。
3. 拒否ログに total/side の両指標が十分残っておらず、現場切り分けコストが高かった。

Improvement:
1. `execution/order_manager.py` に `_is_net_reducing_usage(...)` を追加し、例外判定を常に total usage（netting）基準へ固定。
2. side-cap 経路で `usage/projected_usage` を side 用に使っても、拒否可否は `usage_total` と `projected_usage_total` で評価するよう修正（market/limit 両経路）。
3. `margin_usage_projected_cap` ログ payload に `projected_usage_total / margin_usage_total / side_usage / side_projected` を追加し、再発時の即時判別を可能化。

Verification:
1. ローカル回帰:
   - `pytest -q tests/execution/test_order_manager_preflight.py tests/execution/test_order_manager_log_retry.py`
   - `35 passed`
2. 新規テスト:
   - `test_is_net_reducing_usage_*`（純粋判定）
   - `test_limit_order_allows_net_reducing_under_side_cap`（side cap 高負荷でも net-reducing を許可）
3. VM検証（次段）:
   - 反映後 `margin_usage_projected_cap` の連発が収束し、同条件シグナルで `filled/submitted` が再開することを確認。

Status:
- implemented_local_verified

## 2026-02-27 00:25 UTC / 2026-02-27 09:25 JST - 自動損益の実測確認と `scalp_ping_5s_b_live` 即効デリスク
Period:
- Snapshot window: `2026-02-27 00:22` ～ `00:25` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 直近15分:
  - 全体: `17 trades / +1311.9 JPY`
  - 自動のみ（`strategy_tag != null`）: `16 trades / -65.1 JPY`
  - `scalp_ping_5s_b_live + scalp_ping_5s_c_live`: `14 trades / +6.9 JPY`
- JST当日（`2026-02-27 00:00 JST` 以降）:
  - 全体: `455 trades / +731.4 JPY`
  - 自動のみ: `454 trades / -645.6 JPY`
  - B/C 内訳:
    - `scalp_ping_5s_b_live`: `244 trades / -428.1 JPY`
    - `scalp_ping_5s_c_live`: `204 trades / -97.4 JPY`
- 全体プラスの主因は `strategy_tag=null` の単発決済
  （ticket `400470`, `+1377.0 JPY`, `MARKET_ORDER_TRADE_CLOSE`）。
- 直近2時間の B は long 側損失偏重:
  - long `34 trades / -41.1 JPY`
  - short `3 trades / -6.2 JPY`

Failure Cause:
1. 「全体損益」は手動/タグ欠損の単発利益に引っ張られ、自動戦略の実態が見えにくい。
2. B は long 側で低品質エントリーが残り、stop 系損失が先行している。

Improvement:
1. `ops/env/scalp_ping_5s_b.env` を即時デリスク:
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=720`（from `900`）
   - `SCALP_PING_5S_B_CONF_FLOOR=75`（from `72`）
   - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.74`（from `0.70`）
   - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_FLOOR=0.60`（from `0.54`）
2. 停止なし方針を維持しつつ、B の低確度 long 発火を抑制して損失勾配を圧縮する。

Verification:
1. 反映後30分で `scalp_ping_5s_b_live` の `realized_pl` がゼロ超へ改善すること。
2. `orders.db` で B の `probability_scaled` / `rejected` 比率が低下すること。
3. JST当日の自動損益（`strategy_tag != null`）がマイナス幅縮小に転じること。

Status:
- in_progress

## 2026-02-26 13:30 UTC / 2026-02-26 22:30 JST - `SIDE_FILTER=none` が wrapper で `sell` 強制され、B/C entry が詰まる問題を修正
Period:
- Analysis/patch window: `2026-02-26 13:12` ～ `13:30` UTC
- Source: `workers/scalp_ping_5s_b/worker.py`, `workers/scalp_ping_5s_c/worker.py`, `tests/workers/test_scalp_ping_5s_b_worker_env.py`, `ops/env/scalp_ping_5s_{b,c}.env`

Fact:
- `ops/env/scalp_ping_5s_b.env` は `SCALP_PING_5S_B_SIDE_FILTER=none` だったが、
  wrapper 側の fail-closed 実装により不正値扱いで `sell` に上書きされていた。
- `ops/env/scalp_ping_5s_c.env` の `SCALP_PING_5S_C_SIDE_FILTER=none` と
  `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` も同様に `sell` へ上書きされていた。
- このため `SIDE_FILTER=none` を設定しても実効せず、`side_filter_block` が残る状態だった。

Failure Cause:
1. wrapper で `ALLOW_NO_SIDE_FILTER` が実装されておらず、`none` が常に invalid 扱いだった。
2. env の意図値と実効値が乖離し、skip要因の切り分けを難しくしていた。

Improvement:
1. `workers/scalp_ping_5s_b/worker.py`, `workers/scalp_ping_5s_c/worker.py`
   - `ALLOW_NO_SIDE_FILTER=1` かつ `SIDE_FILTER in {"", "none", "off", "disabled"}` のとき、
     `SCALP_PING_5S_SIDE_FILTER=""` を許可する正規化を追加。
   - 上記以外の未設定/不正値は従来どおり `sell` へ fail-closed。
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=1` を追加。
3. `tests/workers/test_scalp_ping_5s_b_worker_env.py`
   - B/C の `ALLOW_NO_SIDE_FILTER=1` で空 side filter が通る検証を追加/更新。

Verification:
1. `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py -k "side_filter"` → `8 passed`
2. `python3 -m py_compile workers/scalp_ping_5s_b/worker.py workers/scalp_ping_5s_c/worker.py` → pass
3. VM 反映後に `entry-skip summary` の `side_filter_block` 比率が低下することを確認する。

Status:
- in_progress

## 2026-02-26 13:05 UTC / 2026-02-26 22:05 JST - 方向精度再崩れの根本対策（C no-side-filter封鎖 + side-filter復元）
Period:
- Analysis/patch window: `2026-02-26 12:40` ～ `13:05` UTC
- Source: repository `workers/scalp_ping_5s*`, `ops/env/scalp_ping_5s_c.env`, unit tests

Fact:
- `workers/scalp_ping_5s_c/worker.py` には `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` かつ `SIDE_FILTER=none` で
  side filter を未設定扱いにできる分岐が存在した。
- `ops/env/scalp_ping_5s_c.env` は実際に `SCALP_PING_5S_C_SIDE_FILTER=none`,
  `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1` だった。
- `workers/scalp_ping_5s/worker.py` は初段で side_filter を通した後に
  ルーティングで side が反転した場合、最終 `side_filter_final_block` で no-entry になる設計だった。

Failure Cause:
1. C に no-side-filter 例外があり、方向固定の前提が運用設定で破れる。
2. side_filter を通過したシグナルでも、後段flipで反転すると発注前に消失し、エントリー密度が不安定化する。

Improvement:
1. C ラッパーで no-side-filter 例外を廃止し、invalid/missing を常に `sell` へ fail-closed。
2. 本体ワーカーへ `_resolve_final_signal_for_side_filter()` を追加し、
   後段反転時は初段の side-filter 適合シグナルへ復元して発注経路を維持。
3. 運用envを `SCALP_PING_5S_C_SIDE_FILTER=sell`, `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=0` に固定。
4. ロット計算は途中丸めを廃止して最終丸めへ統一し、
   `units_below_min` による 0 化でシグナルが消える経路を縮小。
5. `MIN_UNITS_RESCUE`（確率/信頼度/リスクcap条件付き）を導入し、
   最終段だけ 1unit 救済して実約定導線を維持。

Verification:
1. 対象テスト（10件）:
   - `tests/workers/test_scalp_ping_5s_b_worker_env.py`
   - `tests/workers/test_scalp_ping_5s_worker.py -k resolve_final_signal_for_side_filter`
2. 結果: `10 passed`

Status:
- in_progress

## 2026-02-26 12:35 UTC / 2026-02-26 21:35 JST - quote 問題の実測再監査と執行層ハードニング
Period:
- 直近24h / 7d（VM実DB）
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `journalctl -u quant-order-manager.service`

Fact:
- 24h `orders` 合計: `50365`
- 24h status 上位:
  - `preflight_start=18629`
  - `perf_block=16971`
  - `probability_scaled=7719`
  - `entry_probability_reject=1101`
- 24h error:
  - `TRADE_DOESNT_EXIST=8`、`OFF_QUOTES/PRICE_*` は 0
- 7d `quant-order-manager` journal でも `quote_unavailable`/`quote_retry`/`OFF_QUOTES`/`PRICE_*` は 0

Failure Cause:
1. 現在の損失・機会損失の主因は quote 不足ではなく、`perf_block` と確率/余力ガード側。
2. ただし再クオート要求が急増する局面では既定 `FETCH=2 + RETRY=1` が薄く、将来的な取り逃しリスクは残る。

Improvement:
1. `ops/env/quant-order-manager.env` で quote 再取得耐性を強化。
2. `ORDER_SUBMIT_MAX_ATTEMPTS=1` は維持し、quote 専用リトライだけ増やして戦略判定を汚さない。

Verification:
1. 反映後24hで `quote_unavailable`/`quote_retry`/`OFF_QUOTES`/`PRICE_*` の件数を再計測。
2. 同期間で `filled / submit_attempt` 比率の悪化がないことを確認。
3. `perf_block` が依然主因なら quote ではなく戦略側改善を優先継続。

Status:
- in_progress

## 2026-02-26 12:25 UTC / 2026-02-26 21:25 JST - no-stop維持で「無約定化」を解消する再配線
Period:
- Incident window: `2026-02-26 11:40` ～ `12:25` UTC
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, `journalctl -u quant-scalp-ping-5s-{b,c}.service`

Fact:
- `orders.db` の `orders` は直近30分で `0 rows`（`datetime(substr(ts,1,19)) >= now-30m`）。
- `quant-order-manager` は `coordinate_entry_intent` 受信を継続しているが、`preflight_start` 以降の新規発注イベントが停止。
- `entry_intent_board` の直近45分は `1件` のみで、`scalp_extrema_reversal_live` が `below_min_units_after_scale`（`raw=45`, `min_units=1000`）で reject。
- B/C ワーカーは稼働継続しているが、`entry-skip summary` は
  `no_signal:revert_not_found` が最多、次点で `no_signal:side_filter_block` / `units_below_min` が継続。

Failure Cause:
1. no-stop方針の下で `SIDE_FILTER=sell` と逆風ドリフト縮小が重なり、B/C が local 判定段階で枯渇。
2. 共通 `POLICY_HEURISTIC_PERF_BLOCK` が有効のままで、他戦略側の再起動余地も狭い。
3. 最小ロット閾値が小口シグナルの通過率を削り、intent が `order_manager` まで届かない局面が残る。

Improvement:
1. 共通 hard reject の解除:
   - `ops/env/quant-v2-runtime.env`
   - `POLICY_HEURISTIC_PERF_BLOCK_ENABLED=0`（from `1`）
2. B の通過率回復:
   - `SCALP_PING_5S_B_MIN_UNITS=1`（from `5`）
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=1`（from `5`）
   - `SHORT_MOMENTUM_TRIGGER_PIPS=0.08`（from `0.10`）
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.58`（from `0.42`）
   - `SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`（from `0.50/0.18`）
3. C の無約定解消（停止ではなく両方向縮小運転）:
   - `SCALP_PING_5S_C_SIDE_FILTER=none`（from `sell`）
   - `SCALP_PING_5S_C_ALLOW_NO_SIDE_FILTER=1`
   - `SCALP_PING_5S_C_MIN_UNITS=1`（from `5`）
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`（from `5`）
   - `SHORT/LONG_MOMENTUM_TRIGGER_PIPS=0.08/0.18`（from `0.10/0.10`）
   - `DIRECTION_BIAS_SHORT_OPPOSITE_UNITS_MULT=0.62`（from `0.45`）
   - `SIDE_BIAS_SCALE_GAIN/FLOOR=0.35/0.28`（from `0.50/0.18`）
4. C の reject/rate-limit 緩和:
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE=3`（from `1`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN=0.68`（from `0.76`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR=0.58`（from `0.70`）
   - `SCALP_PING_5S_C_ENTRY_PROBABILITY_ALIGN_FLOOR_MAX_COUNTER=0.38`（from `0.24`）
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.52`（from `0.62`）
   - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_MIN_TRADES=30`（from `6`）
   - `SCALP_PING_5S_C_PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（from `0.90/0.48`）
   - `SCALP_PING_5S_PERF_GUARD_FAILFAST_MIN_TRADES=30`（from `6`）
   - `SCALP_PING_5S_PERF_GUARD_FAILFAST_PF/WIN=0.20/0.20`（from `0.90/0.48`）
   - `SCALP_PING_5S_C_PERF_GUARD_ENABLED=0`
   - `SCALP_PING_5S_PERF_GUARD_ENABLED=0`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.46`（from `0.52`）
   - `ops/env/quant-order-manager.env` の C上書き値を同値同期
     - `REJECT_UNDER=0.46`, `MIN/MAX_SCALE=0.40/0.85`, `BOOST_PROBABILITY=0.85`
     - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=1`
     - `SCALP_PING_5S(_C)_PERF_GUARD_MODE=off`, `SCALP_PING_5S(_C)_PERF_GUARD_ENABLED=0`
   - `scalp_extrema_reversal_live` の協調拒否解消:
     - `ORDER_MIN_UNITS_STRATEGY_SCALP_EXTREMA_REVERSAL(_LIVE)=30`（runtime/order-manager 両env）

Verification:
1. 反映後 30分で `orders.db` の `preflight_start` と `filled` が再出現すること。
2. `entry-skip summary` の `units_below_min` 比率が低下すること。
3. 反映後 60分で `trades.db` の `realized_pl` 増分が `scalp_ping_5s_c_live` 単独で急悪化しないこと（損失勾配監視）。

Status:
- in_progress

## 2026-02-26 12:20 UTC / 2026-02-26 21:20 JST - `scalp_ping_5s_b_live/c_live` 方向劣化の再発防止（side filter fail-closed）
Period:
- Direction audit window: `datetime(close_time) >= now - 24 hours`
- Runtime env check: `quant-scalp-ping-5s-b.service` MainPID/child PID
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, systemd process env

Fact:
- 24h集計（`trades.db`）で `scalp_ping_5s_b_live` の buy 側が劣化:
  - buy `72 trades`, win rate `20.83%`, avg `-0.936 pips`
  - `entry_probability>=0.75` の buy でも win rate `18.84%`, avg `-1.051 pips`
- 実行中プロセス（起動 `2026-02-26 12:10:55 UTC`）では
  `SCALP_PING_5S_B_SIDE_FILTER=sell` と
  `SCALP_PING_5S_SIDE_FILTER=sell` が有効。
- ただし過去履歴には buy 発注が残るため、env 欠落/不正時に再発しうる。

Failure Cause:
1. B/C variant の方向制御が env 設定依存で、設定欠落時に fail-open になり得る。
2. 高確率帯 buy の実績悪化と整合しない方向シグナルが通過した履歴が存在する。

Improvement:
1. `workers/scalp_ping_5s_b/worker.py` と `workers/scalp_ping_5s_c/worker.py` で side filter を fail-closed 化。
2. `SCALP_PING_5S_SIDE_FILTER` が未設定/不正値なら `sell` を強制。
3. 起動ログへ `side_filter` を明示出力し、監査を容易化。
4. `tests/workers/test_scalp_ping_5s_b_worker_env.py` に
   B/C それぞれの `missing/invalid/valid` ケースを追加。

Verification:
1. `pytest -q tests/workers/test_scalp_ping_5s_b_worker_env.py` が全緑（14 passed）。
2. VMで `quant-scalp-ping-5s-b` の子プロセス環境に
   `SCALP_PING_5S_SIDE_FILTER=sell` が存在することを確認。

Status:
- in_progress

## 2026-02-26 12:09 UTC / 2026-02-26 21:09 JST - B側の `units_below_min` 残りを削るため最小ロット閾値を再調整
Period:
- Snapshot: `2026-02-26 12:06:59` ～ `12:08:10` UTC（`21:06:59` ～ `21:08:10` JST）
- Source: VM `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `/home/tossaki/QuantRabbit/ops/env/{scalp_ping_5s_b.env,quant-order-manager.env}`

Fact:
- 12:06:59 UTC 再起動後、`C` は `units_below_min=0` まで低下。
- 同条件で `B` は `entry-skip summary side=short total=4 units_below_min=4` が残存。
- `RISK multiplier` は引き続き `mult=0.40` で、縮小局面の最終ユニットが閾値を割り込みやすい。

Failure Cause:
1. Bは strategy 側と order-manager 側の min units が `20` で揃っており、縮小ロットの通過余地が不足していた。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_MIN_UNITS: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B: 20 -> 10`
2. `ops/env/quant-order-manager.env`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE: 20 -> 10`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B: 20 -> 10`

Verification:
1. 反映後15分で `journalctl -u quant-scalp-ping-5s-b.service` の `units_below_min` が 0 になること。
2. `orders.db` で `scalp_ping_5s_b_live` の `submit_attempt` / `filled` が再出現すること。
3. `manual_margin_pressure` / `slo_block` が再増加しないこと。

Status:
- in_progress

## 2026-02-26 12:20 UTC / 2026-02-26 21:20 JST - short-only後の無約定ボトルネック（revert_not_found + short units_below_min）
Period:
- Observation window: `2026-02-26 12:00:00` 以降（VM journal）
- Source: `journalctl -u quant-scalp-ping-5s-b.service`, `journalctl -u quant-scalp-ping-5s-c.service`, `journalctl -u quant-order-manager.service`

Fact:
- `SCALP_PING_5S_B/C_SIDE_FILTER=sell` は機能し、long は `no_signal:side_filter_block` / `side_filter_final_block` で継続遮断。
- 一方で約定が止まり、`quant-order-manager` の `OPEN_REJECT/OPEN_FILLED` は同窓で実質発生なし。
- B/C の skip 内訳は `no_signal:revert_not_found` が最大（各30秒集計で概ね `40-94` 件）。
- short 側は `units_below_min` が継続（B: `3-17`, C: `1-6` / 30秒集計）。

Failure Cause:
1. short-only化後、短期反転検知（revert）が成立せず `revert_not_found` に集中。
2. 成立した short シグナルも、動的縮小後ユニットが最小ロット未満になり通過不能。
3. long遮断は効いているが、short化の再配線が不足し、取引密度が0近傍に落ちた。

Improvement:
1. B/C の `revert` 閾値を同時緩和（`REVERT_RANGE/SWEEP/BOUNCE/CONFIRM_RATIO`, `REVERT_SHORT_WINDOW`）。
2. short 最小通過ロットを引き下げ（`SCALP_PING_5S_{B,C}_MIN_UNITS`, `ORDER_MIN_UNITS_STRATEGY_*` を `5`）。
3. C は short 発火側を追加緩和（`SHORT_MIN_TICKS`, `SHORT_MIN_SIGNAL_TICKS`）。
4. long→short 変換を有効化（B/C `EXTREMA_GATE_ENABLED=1`, `EXTREMA_REVERSAL_ALLOW_LONG_TO_SHORT=1`, `LONG_TO_SHORT_MIN_SCORE` 緩和）。
5. C は flip系を再稼働（`SIDE_METRICS_DIRECTION_FLIP_ENABLED=1`）し、short側への再配線を強化。

Verification:
1. 反映後30分で `entry-skip summary side=short` の `units_below_min` 比率が低下すること。
2. `orders.db` で `filled` が再発し、`strategy in {scalp_ping_5s_b_live, scalp_ping_5s_c_live}` の short約定が出ること。
3. `trades.db` で B/C の新規closeにおける `realized_pl` の負勾配が反転または鈍化すること。

Status:
- in_progress

## 2026-02-26 12:06 UTC / 2026-02-26 21:06 JST - 損失主因の再監査（執行品質より制御異常が支配）
Period:
- 24h監査: `orders.db` / `trades.db` / `metrics.db`（`datetime(ts/close_time) >= now - 24 hours`）
- 事故窓監査: `2026-02-24 02:20:16` ～ `09:13:14` UTC（`11:20:16` ～ `18:13:14` JST）
- Source: VM `systemd`, `journalctl`, `/home/tossaki/QuantRabbit/logs/{orders,trades,metrics}.db`, `oanda_*_live.json`

Fact:
- V2主導線は稼働中（`quant-market-data-feed/order-manager/position-manager/strategy-control` は active/running）。
- 直近24hの `orders`:
  - `preflight_start=18629`, `perf_block=16971`, `probability_scaled=7719`, `submit_attempt=1221`, `filled=1202`
  - `rejected=18`, `slo_block=12`, `margin_usage_projected_cap=190`, `manual_margin_pressure=36`
- 直近24hの `trades`: `1283件`, `-4417.54 JPY`。
  - close reason: `STOP_LOSS_ORDER=390件/-5232.48 JPY`, `MARKET_ORDER_MARGIN_CLOSEOUT=3件/-2163.76 JPY`
  - 戦略別下位: `scalp_ping_5s_c_live=-7025.83 JPY`, `scalp_ping_5s_b_live=-3144.31 JPY`
- 執行品質（`analyze_entry_precision.py --limit 1200`）:
  - `slip p95=0.300 pips`, `cost_vs_mid p95=0.700 pips`, `submit latency p95=276.9 ms`, `missing quote=0`
- レイテンシ/SLO:
  - `data_lag_ms p95=4485.9`, `decision_latency_ms p95=8835.9`
  - 閾値超過率: `data_lag_ms>1500ms = 22.8%`, `decision_latency_ms>2000ms = 10.71%`
  - `journalctl quant-order-manager` に `slo_block:data_lag_p95_exceeded` を確認（2026-02-26 11:42～11:45 UTC）
- 事故窓の確定事実:
  - `strategy_control_exit_disabled=10277`（2026-02-24 JST日単位）
  - `micro-micropul*` 4注文で各 `2044～2045` 回の exit disabled 後、`MARKET_ORDER_MARGIN_CLOSEOUT`
  - 上記4件合計 `-16837.43 JPY`
- 直近口座スナップショット（2026-02-26 12:00:35 UTC）:
  - `nav=56970.299`, `margin_used=53060.740`, `margin_available=3943.559`, `health_buffer=0.06918`
  - `USD_JPY short_units=8500`
- `entry_thesis` 必須欠損:
  - 直近24hで `entry_probability/entry_units_intent` 欠損 `80/1283` 件（主に `scalp_ping_5s_*` 派生タグ行）

Failure Cause:
1. 大損の一次原因は、戦略予測精度より `exit不能（strategy_control_exit_disabled）` と `margin closeout tail` のシステム異常。
2. manual併走を含む高マージン使用状態で、`margin_*` ガードと closeout が発火しやすい口座状態が継続。
3. `data_lag_ms` スパイクにより `slo_block` が増え、良い局面のエントリーが欠落して収益回復を阻害。
4. `entry_thesis` 欠損により、意図協調/監査の完全性が崩れ、異常時の切り分けと制御精度を落としている。
5. B/Cは執行コスト劣化より `STOP_LOSS_ORDER` と手仕舞い設計由来の期待値悪化が支配的。

Improvement:
1. `strategy_control` ガード強化:
   - `entry=1 & exit=0` を検知即時で自動修正（`entry=0 & exit=1`）し、監査ログとアラートを必須化。
2. exit不能の再発防止:
   - `strategy_control_exit_disabled` が同一 `client_order_id` で閾値超過したら強制エスカレーション（exit優先モード）。
3. マージン防衛:
   - manual玉込みで `health_buffer` / `margin_available` 連動の新規抑制を強化し、先に建玉縮退を実行。
4. SLO復旧:
   - `data_lag_ms` スパイク時間帯の `market-data-feed` / DB遅延要因を切り分け、`slo_block` は hard reject 連打ではなく段階縮小中心へ。
5. `entry_thesis` スキーマ強制:
   - 非manual注文は `entry_probability` と `entry_units_intent` 欠損時に reject（`status=missing_entry_thesis_fields`）。
6. 収益回復の優先順:
   - B/Cは方向・手仕舞いロジックを再調整し、`STOP_LOSS_ORDER` 依存を減らす。

Verification:
1. `orders.db` で `strategy_control_exit_disabled` が 24h連続 `0`。
2. `trades.db` で `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続 `0`。
3. `metrics.db` で `data_lag_ms p95 < 1500`, `decision_latency_ms p95 < 2000` を継続達成。
4. `orders.db` で `slo_block` の連続発火（同一時間帯クラスタ）が解消。
5. `trades.db` で `entry_thesis` 必須2項目の欠損が `0`。
6. B/Cの 24h `jpy` と `PF` が改善方向（少なくとも連続悪化が停止）。

Status:
- in_progress

## 2026-02-26 12:02 UTC / 2026-02-26 21:02 JST - B/C sell限定運用で `units_below_min` が主因化し、発注ゼロ化したため最小ロット閾値を緩和
Period:
- Snapshot: `2026-02-26 11:59:57` ～ `12:01:39` UTC（`20:59:57` ～ `21:01:39` JST）
- Source: VM `journalctl -u quant-scalp-ping-5s-{b,c}.service`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/ops/env/scalp_ping_5s_{b,c}.env`

Fact:
- `quant-scalp-ping-5s-b.service` / `quant-scalp-ping-5s-c.service` は `11:59:57 UTC` に再起動済み（active/running）。
- しかし `orders.db` は `ts >= 2026-02-26T11:59:57Z` で `0件`（新規発注フロー未到達）。
- ワーカーログで共通して以下を確認:
  - `SCALP_PING_5S_{B,C}_SIDE_FILTER=sell` により long 候補が `side_filter_block`。
  - short 候補は `units_below_min` が発生（B: `10件`, C: `8件`）。
  - `RISK multiplier` は `mult=0.40` で推移し、縮小後ユニットが閾値未満へ落ちやすい。

Failure Cause:
1. long を遮断する side filter 自体は意図どおりだが、short 側の最終ユニットが `min_units=30` を下回り、注文生成まで到達できない。
2. strategy 側 min units と order 側 min units の閾値差（Cは 30 固定）が、縮小運転時の失効を助長している。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_MIN_UNITS: 30 -> 20`
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_MIN_UNITS: 30 -> 20`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C_LIVE: 30 -> 20`
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C: 30 -> 20`

Verification:
1. 反映後30分で `orders.db` に `submit_attempt` / `filled` が B/C で再出現すること。
2. `journalctl -u quant-scalp-ping-5s-{b,c}.service` の `units_below_min` 件数が減少すること。
3. `orders.db` で `manual_margin_pressure` / `slo_block` の再悪化がないこと。

Status:
- in_progress

## 2026-02-26 11:50 UTC / 2026-02-26 20:50 JST - PDCA導線の実運用監査（稼働中だが改善ループに断点あり）
Period:
- Snapshot: `2026-02-26 11:39` ～ `11:50` UTC（`20:39` ～ `20:50` JST）
- Source: VM `systemd`, `journalctl`, `/home/tossaki/QuantRabbit/logs/{orders,trades,metrics}.db`, `/home/tossaki/QuantRabbit/logs/*_latest.json`, OANDA account summary/open positions

Fact:
- V2主導線は稼働中:
  - `quant-market-data-feed`, `quant-strategy-control`, `quant-order-manager`, `quant-position-manager`, `quant-forecast` は `active(running)`。
  - `quantrabbit.service` は存在せず、VMリポジトリは `HEAD == origin/main == 0c0caae2c05295cbccd7113454fb24cb7f8afda3`。
- 分析/改善タイマーは起動:
  - `quant-pattern-book`, `quant-dynamic-alloc`, `quant-policy-guard`, `quant-replay-quality-gate`, `quant-trade-counterfactual`, `quant-forecast-improvement-audit` が schedule され実行履歴あり。
- ただし改善ループは市場オープン中に停止:
  - `quant-replay-quality-gate.service` は `skipped: market_open` を連続出力。
  - `quant-trade-counterfactual.service` も `skipped: market_open` を連続出力。
- 発注導線の品質劣化:
  - `quant-order-manager` 直近1時間で `database is locked` が `67` 回。
- 24hの orders 状態:
  - `preflight_start=18629`, `submit_attempt=1221`, `filled=1202`, `perf_block=16971`, `entry_probability_reject=1101`。
- 監査ログ差分:
  - `logs/ops_v2_audit_latest.json` で `POLICY_HEURISTIC_PERF_BLOCK_ENABLED expected=0 actual=1` の `warn`。

Failure Cause:
1. `replay` と `counterfactual` が market open 時に止まるため、改善施策の入力が日中に更新されない。
2. replay quality gate の対象が `TrendMA/BB_RSI` 中心で、現行の主要 scalp/micro 群を十分に監査できていない。
3. auto-improve が `block_jst_hours` を `worker_reentry.yaml` へ自動反映する設定で、方針「停止より改善優先」と衝突しやすい。
4. `orders.db` lock 競合が残り、preflight->submit の実効通過率を毀損している。
5. pattern gate の実効キー (`ORDER_MANAGER_PATTERN_GATE_ENABLED`) と運用キー (`ORDER_PATTERN_GATE_ENABLED`) が分岐し、設定意図と実動作がずれる余地がある。

Improvement:
1. `orders.db` 競合の再抑制を最優先（busy timeout/retry/lock制御と write 経路の再点検）。
2. `REPLAY_QUALITY_GATE_SKIP_WHEN_MARKET_OPEN=0` と `COUNTERFACTUAL_SKIP_WHEN_MARKET_OPEN=0` へ変更し、改善ループを 24/7 化。
3. `config/replay_quality_gate_main.yaml` の `workers` を現行稼働戦略へ拡張し、PDCA対象を本番導線へ合わせる。
4. `REPLAY_QUALITY_GATE_AUTO_IMPROVE_APPLY_REENTRY=0` にして、自動時間帯ブロック反映を停止（提案のみ運用）。
5. pattern gate の env キーを一本化し、order-manager 側の参照名に統一。
6. `POLICY_HEURISTIC_PERF_BLOCK_ENABLED` の期待値と実運用値を監査基準に合わせる。

Verification:
1. `journalctl -u quant-order-manager` の `database is locked` が 1h あたり 0 件へ収束。
2. `replay_quality_gate_latest.json` と `trade_counterfactual_latest.json` の `generated_at` が市場オープン中も更新される。
3. replay report の対象戦略に、稼働中 scalp/micro 群が含まれる。
4. `worker_reentry.yaml` への自動 `block_jst_hours` 書き換えが停止し、改善提案だけが残る。
5. `orders.db` の `preflight_start -> submit_attempt -> filled` の通過率が改善する。

Status:
- in_progress

## 2026-02-26 10:12 UTC / 2026-02-26 19:12 JST - 停止なし・時間帯停止なし運用へ再構成
Period:
- Snapshot: `2026-02-26 09:52` ～ `10:12` UTC（`18:52` ～ `19:12` JST）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `config/worker_reentry.yaml`, OANDA `openTrades`

Fact:
- 24h は `1120 trades`, `-11510.6 JPY`, `PF=0.455`（主損失: `scalp_ping_5s_c_live`, `scalp_ping_5s_b_live`, `M1Scalper-M1`）。
- 直近監査で `entry=1 & exit=0` は `0` 件、`strategy_control_exit_disabled` も直近1hで `0` 件。
- ただし運用設定には「停止相当」の制約が残存:
  - `scalp_ping_5s_{b,c}` の `ALLOW_HOURS_JST`（時間帯限定）
  - `worker_reentry` の `block_jst_hours`（`M1Scalper`, `MicroPullbackEMA`, `MicroLevelReactor`, `scalp_ping_5s_{b,c,d}_live`）

Failure Cause:
1. 収益改善を「停止/時間帯限定」に寄せると、運用方針（常時動的トレード）と矛盾し、再現性が崩れる。
2. `scalp_ping_5s_c_live` は `close_reject_no_negative` が残ると損失玉の解放が遅れやすい。

Improvement:
1. `ops/env/scalp_ping_5s_{b,c}.env`:
   - `ALLOW_HOURS_JST=`（時間帯停止を撤去）
   - `PERF_GUARD_MODE=reduce`（停止ではなく縮小）
2. `config/worker_reentry.yaml`:
   - `M1Scalper`, `MicroPullbackEMA`, `MicroLevelReactor`, `scalp_ping_5s_{b,c,d}_live` の `block_jst_hours` を空配列化。
3. `config/strategy_exit_protections.yaml`:
   - `scalp_ping_5s_{c,c_live}.neg_exit` に `strict_no_negative: false`, `deny_reasons: []` を追加し、EXIT詰まりを解消。
4. `strategy_control_flags` は `entry=1 & exit=1` を基本状態として維持し、停止は緊急安全時のみ許可。

Verification:
1. `orders.db` で `strategy_control_entry_disabled` / `strategy_control_exit_disabled` が発生しないこと。
2. `orders.db` で `close_reject_no_negative` が `scalp_ping_5s_c_live` で減少すること。
3. 24hで `B/C` の `jpy PF` が改善方向（`>=1.0` へ接近）すること。
4. `worker_reentry` 由来の時間帯ブロック理由で待機しないこと。

Status:
- in_progress

## 2026-02-26 11:50 UTC / 2026-02-26 20:50 JST - `slo_block(data_lag_p95_exceeded)` を緩和し、M1配分を引き上げ
Period:
- 30m: `datetime(ts) >= now - 30 minutes`
- 24h: `datetime(close_time) >= now - 24 hours`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- 30m `orders.db`:
  - `preflight_start=110`, `probability_scaled=33`
  - `slo_block=10`（latest `2026-02-26T11:45:09Z`）
  - `manual_margin_pressure=25`（ただし latest `2026-02-26T11:34:06Z` で新規発生停止）
- `slo_block` の reason は全件 `data_lag_p95_exceeded`。
  - `data_lag_p95_ms` は `~7152ms`、現行閾値 `5000ms` を超過。
- 24h損益:
  - `M1Scalper-M1` は直近で `-3.1 JPY`（ほぼ横ばい、下振れ縮小）
  - B/C は side_filter=sell 反映後、B/C の `buy` 注文は 0 件。

Failure Cause:
1. strategy-control 側の data lag p95 スパイクが SLO 閾値を超え、scalp_fast が連続 reject。
2. M1 は止めずに残しているが、配分が低く利益寄与の伸びが不足。

Improvement:
1. `ops/env/quant-order-manager.env`
   - `ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS: 5000 -> 9000`
2. `ops/env/quant-v2-runtime.env`（worker local `order_manager` 経路）
   - `ORDER_SLO_GUARD_*` を明示追加し、`ORDER_SLO_GUARD_DATA_LAG_P95_MAX_MS=9000`
3. `ops/env/quant-m1scalper.env`
   - `M1SCALP_BASE_UNITS: 3000 -> 4500`

Verification:
1. 反映後 30m で `slo_block` 最新時刻が更新されないこと（再発停止）。
2. `orders.db` の `preflight_start -> probability_scaled/filled` の遷移比率が改善すること。
3. `M1Scalper-M1` の 6h / 24h `realized_pl` が正方向へ改善すること。

Status:
- in_progress

## 2026-02-26 11:45 UTC / 2026-02-26 20:45 JST - B/C の side 偏りを実損で是正（long 逆風遮断）
Period:
- 24h: `datetime(close_time) >= now - 24 hours`
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`

Fact:
- `scalp_ping_5s_c_live`:
  - long `347 trades / -5380.5 JPY / -353.4 pips`
  - short `75 trades / -88.8 JPY / -52.2 pips`
- `scalp_ping_5s_b_live`:
  - long `128 trades / -1300.8 JPY / -141.3 pips`
  - short `26 trades / -7.7 JPY / -0.5 pips`
- 直近損失の主因は B/C ともに long 側へ偏在。

Failure Cause:
1. no-stop 継続を優先して `SIDE_FILTER` を空に戻したことで、逆風側（long）の負けが継続拡大。
2. B/C の短期ローカル判定は機能しているが、方向バイアスの切替が遅く実損に追随できていない。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_SIDE_FILTER=sell`
2. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_SIDE_FILTER=sell`

Verification:
1. 反映後 30m で `trades.db` の B/C long 新規クローズ件数が 0 であること。
2. B/C の short 側 `avg_lose_jpy` と `total_jpy` が long 全開時より改善すること。
3. `orders.db` の `preflight_start -> probability_scaled/filled` の遷移を継続確認し、`perf_block` 再発がないこと。

Status:
- in_progress

## 2026-02-26 11:30 UTC / 2026-02-26 20:30 JST - no-stop阻害点を `perf_block` + `manual_margin_pressure` に限定して除去
Period:
- 15m: `datetime(ts) >= now - 15 minutes`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, live account snapshot via `execution.order_manager.get_account_snapshot()`

Fact:
- 直近15分の status 集計:
  - `preflight_start=68`
  - `perf_block=52`
  - `probability_scaled=33`
  - `manual_margin_pressure=8`
  - `entry_probability_reject=4`
  - `slo_block=1`
- strategy別（同窓）:
  - `M1Scalper-M1 | perf_block=25`
  - `scalp_ping_5s_b_live | perf_block=24`
  - `scalp_ping_5s_b_live | manual_margin_pressure=10`
- 口座スナップショット:
  - `nav=57,556.799`, `margin_used=53,037.280`, `margin_available=4,553.519`, `health_buffer=0.07907`
  - manual 建玉: `-8500 units`, `1 trade`

Failure Cause:
1. no-stop向けに failfast を緩めても、`perf_guard` の hard reason 判定で B/M1 が preflight reject を継続。
2. manual 併走時の `manual_margin_guard` が小ロット再開局面でも `manual_margin_pressure` を発火させ、B の通過を削る。

Improvement:
1. `ops/env/quant-order-manager.env`:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `M1SCALP_PERF_GUARD_ENABLED=0`
2. `ops/env/quant-v2-runtime.env`（worker local `order_manager` 経路へ実効反映）:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER=0.00`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY=0`
   - `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `M1SCALP_PERF_GUARD_ENABLED=0`
3. 戦略env側も整合:
   - `ops/env/scalp_ping_5s_b.env`: `SCALP_PING_5S_B_PERF_GUARD_ENABLED=0`
   - `ops/env/quant-m1scalper.env`: `M1SCALP_PERF_GUARD_ENABLED=0`

Verification:
1. `orders.db` 15分窓で `status in ('perf_block','manual_margin_pressure')` が B/M1 で減少すること。
2. 同窓で `filled` と `probability_scaled` の増加が確認できること。
3. `MARKET_ORDER_MARGIN_CLOSEOUT` が増えないこと（24h監査継続）。

Status:
- in_progress

## 2026-02-26 11:02 UTC / 2026-02-26 20:02 JST - `manual_margin_pressure` が B エントリー再開の最終ボトルネック
Period:
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, OANDA account snapshot/openTrades
- Window: `datetime(ts) >= now - 15 minutes`（UTC）

Fact:
- 全体（15分）: `entry_probability_reject=13`, `preflight_start=7`, `probability_scaled=3`, `manual_margin_pressure=3`, `perf_block=1`
- `scalp_ping_5s_b_live`（15分）: `preflight_start=6`, `probability_scaled=3`, `manual_margin_pressure=3`
- `manual_margin_pressure` 3件はすべて B の `scalp_fast` エントリー（`units=139/140/181`）
- 口座実測（2026-02-26 11:01 UTC）:
  - `NAV=57,930.80`, `margin_used=53,022.32`, `margin_available=4,942.48`, `health_buffer=0.0853`
  - open trade: `USD_JPY -8500`（`id=400470`, `TP/SLなし`）

Failure Cause:
1. failfast/forecast/leading-profile を緩和した後、B の最終拒否が `manual_margin_pressure` に収束。
2. guard閾値（`free_ratio>=0.05`, `health_buffer>=0.07`, `available>=3000`）が、手動玉併走時の小ロット意図まで遮断。
3. その結果、B の `probability_scaled` 後の通過意図が実発注に到達しない。

Improvement:
1. `ops/env/quant-order-manager.env` の manual margin guard を no-stop 方針向けに再調整:
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_FREE_RATIO: 0.05 -> 0.01`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_HEALTH_BUFFER: 0.07 -> 0.02`
   - `ORDER_MANUAL_MARGIN_GUARD_MIN_AVAILABLE_JPY: 3000 -> 500`
2. guard 自体は維持（`ORDER_MANUAL_MARGIN_GUARD_ENABLED=1`）し、極端な near-closeout だけを継続遮断。

Verification:
1. `quant-order-manager` 再起動後に process env で3閾値の反映を確認する。
2. 反映後 15 分で `manual_margin_pressure` 件数が減少し、`submit_attempt/filled` が増えることを確認する。
3. `margin_usage_projected_cap` と `MARKET_ORDER_MARGIN_CLOSEOUT` が増加しないことを同時監視する。

Status:
- in_progress

## 2026-02-26 11:12 UTC / 2026-02-26 20:12 JST - no-stop 維持のまま「負け源圧縮 + 勝ち源増量」へ再配分
Period:
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`
- Window:
  - 拒否分析: `datetime(ts) >= now - 30 minutes`
  - 損益分析: `datetime(close_time) >= now - 7 days`（補助で 6h）

Fact:
- 直近30分の拒否内訳:
  - `entry_probability_reject=21`（全件 `rangefader`）
  - `preflight_start=7`, `probability_scaled=3`, `manual_margin_pressure=3`, `perf_block=1`
- `rangefader` 拒否理由は `entry_probability_below_min_units` に収束。
  - 直近サンプルの `entry_probability` は約 `0.40`
  - 確率スケール後ユニットが pocket 最小ユニット未満で落ちる状態。
- 7日損益（strategy別、主なもの）:
  - `MomentumBurst`: `+1613.7 JPY`（n=7）
  - `MicroRangeBreak`: `+662.3 JPY`（n=32, PF=3.05）
  - `scalp_ping_5s_b_live`: `-9475.8 JPY`（n=2422, PF=0.43）
  - `scalp_ping_5s_c_live`: `-2735.5 JPY`（n=894, PF=0.86）
  - `M1Scalper-M1`: `-1627.3 JPY`（n=284, PF=0.64）
- 直近6hでも `scalp_ping_5s_c_live` は `-1859.9 JPY`（n=125）。

Failure Cause:
1. 発注経路は稼働しているが、`rangefader` が最小ユニット条件で連続 reject され、約定機会が失われた。
2. B/C と M1 が数量面で重く、no-stop 運用時に損失寄与が勝ち寄与を上回る配分になっていた。

Improvement:
1. `RangeFader` の通過回復（停止ではなく通過条件調整）:
   - `ORDER_MIN_UNITS_STRATEGY_SCALP_RANGEFAD=300` を `quant-order-manager` に追加。
2. 負け源の即時圧縮（B/C/M1 を継続稼働のまま減速）:
   - B: `BASE 1800->900`, `MAX 3600->1800`, `MAX_ORDERS_PER_MINUTE 6->4`, `CONF_FLOOR 74->78`
   - C: `BASE 400->220`, `MAX 900->500`, `MAX_ORDERS_PER_MINUTE 2->1`, `CONF_FLOOR 82->86`
   - M1: `BASE 10000->6000->3000`, `MAX_OPEN_TRADES 2->1`
   - B 追加ホットフィックス: `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE) 30->20`
     （縮小後の確率スケールで `below_min_units` 連発したため）
   - B/M1 追加ホットフィックス: `hard:failfast` 連続拒否を避けるため
     `PERF_GUARD_FAILFAST_*` を soft 警告側へ再設定（Bは `FAILFAST_PF=0.10`, `HARD_PF=0.00`。
     M1は `MODE=reduce`, `FAILFAST_PF/WIN=0.30/0.35`, `HARD_PF=0.20`）。
3. 勝ち源の増量:
   - `MicroRangeBreak` と `MomentumBurstMicro` の `MICRO_MULTI_BASE_UNITS 42000->52000`
   - 上記2戦略の breakout 発火閾値を緩和:
     - `MIN_ADX 20.0->16.0`, `MIN_RANGE_SCORE 0.42->0.34`, `MIN_ATR 1.2->0.9`
     - `LOOP_INTERVAL_SEC 4.0->3.0`
   - `RangeFader` は過大化を避けるため `RANGEFADER_BASE_UNITS 13000->11000`

Verification:
1. 反映後30分で `rangefader` の `entry_probability_below_min_units` が減少し、`submit_attempt/filled` が発生すること。
2. 反映後2時間で B/C/M1 の `realized_pl` ドローダウン勾配が低下すること。
3. 同時に `MicroRangeBreak` / `MomentumBurst` の filled 数と `realized_pl` を増分監視すること。

Status:
- in_progress

## 2026-02-26 10:31 UTC / 2026-02-26 19:31 JST - Bがhard failfastで全面停止して約定不足
Period:
- `datetime(ts) >= now - 30 minutes`
- Source: VM `/home/tossaki/QuantRabbit/logs/orders.db`, `journalctl -u quant-order-manager.service`

Fact:
- 直近30分の `orders.db`:
  - `scalp_ping_5s_b_live`: `perf_block=32`
  - `scalp_ping_5s_c_live`: `perf_block=4`
  - `RangeFader-sell-fade`: `entry_probability_reject=23`
- `quant-order-manager` 拒否ログ（B）:
  - `perf_block:hard:hour10:failfast:pf=0.62 win=0.29 n=191` が連続発生
- 口座状態:
  - open trades はフラット（新規約定不足）

Failure Cause:
1. `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF/WIN` が現状実績より高く、`reduce` 設定でも hard 理由で新規を全面拒否。
2. `SCALP_PING_5S_B_SIDE_FILTER=buy` / `SCALP_PING_5S_C_SIDE_FILTER=buy` で方向固定が残り、相場方向とのズレ時に機会損失が拡大。

Improvement:
1. Bの hard failfast 閾値を実績連動に下げる:
   - `quant-order-manager` 内では `ORDER_MANAGER_SERVICE_ENABLED=0` に固定（自己HTTP再入を停止）
   - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_PF: 0.88 -> 0.58`
   - `SCALP_PING_5S_B_PERF_GUARD_FAILFAST_WIN: 0.48 -> 0.27`
   - `SCALP_PING_5S_B_PERF_GUARD_SL_LOSS_RATE_MAX: 0.52 -> 0.75`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER(B/C): 0.64/0.76 -> 0.48/0.62`
   - 反映先: `ops/env/quant-order-manager.env`, `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
2. B/C の方向固定を解除:
   - `SCALP_PING_5S_B_SIDE_FILTER=`
   - `SCALP_PING_5S_C_SIDE_FILTER=`
   - 反映先: `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`
3. strategy_entry の forecast 逆行一律拒否を解除:
   - `STRATEGY_FORECAST_FUSION_STRONG_CONTRA_REJECT_ENABLED=0`
   - `STRATEGY_FORECAST_FUSION_WEAK_CONTRA_REJECT_ENABLED=0`
   - 反映先: `ops/env/quant-v2-runtime.env`
4. B/C の strategy_entry leading profile 拒否を解除:
   - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_ENABLED=0`
   - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_ENABLED=0`
   - 反映先: `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`

Verification:
1. `orders.db` 30分窓で `scalp_ping_5s_b_live` の `perf_block` 比率が低下すること。
2. 同窓で `submit_attempt` / `filled` が再出現すること。
3. `trades.db` の当日 `scalp_ping_5s_b_live` / `scalp_ping_5s_c_live` 実現JPYが改善方向に転じること。

Status:
- in_progress

## 2026-02-26 10:06 UTC / 2026-02-26 19:06 JST - 即日止血（C停止 + strategy_control 衝突解消）
Period:
- Snapshot: `2026-02-26 09:52` ～ `10:06` UTC（`18:52` ～ `19:06` JST）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, OANDA `openTrades`

Fact:
- 24h: `1120 trades`, `-1009.3 pips`, `-11510.6 JPY`, `PF=0.455`
- 7d: `4189 trades`, `-2127.9 pips`, `-25067.5 JPY`, `PF=0.656`
- 24h 主損失は `scalp_ping_5s_c_live=-5633.2 JPY`, `scalp_ping_5s_b_live=-3102.6 JPY`, `M1Scalper-M1=-1265.8 JPY`
- `entry=1 & exit=0` の残骸は是正前に複数残存（`micropullbackema*`, `microtrendretest*`, `scalp_ping_5s_flow_live`）
- `openTradeCount=0`（OANDA、手動玉含めフラット）

Failure Cause:
1. C/B が負EVのまま高回転し、日次損失の主因を継続している。
2. `strategy_control` の stale flag（`entry=1 & exit=0`）が再発時の closeout テール要因になりうる。
3. 直近は `close_reject_no_negative` も C 側で多発し、EXIT遅延を誘発していた。

Improvement:
1. VM運用ガードを即時適用:
   - `strategy_control_flags` の `entry=1 & exit=0` を全解消（`entry=0 & exit=1` に補正）。
2. 即日止血:
   - `scalp_ping_5s_c` を `entry=0 & exit=1` として新規停止（EXITは許可）。
3. 収益機会の残し方:
   - `microtrendretest` / `microtrendretest-long` は `entry=1 & exit=1` へ再有効化。
4. 高リスク再発系は停止維持:
   - `micropullbackema*` と `scalp_ping_5s_flow*` は `entry=0 & exit=1` を維持。

Verification:
1. `strategy_control.db` 現在値: `entry=1 & exit=0` が `0` 件。
2. 直近1h: `orders.db` の `strategy_control_exit_disabled=0`、`close_reject_no_negative=0`。
3. OANDA `openTrades=[]` を確認（含み損ポジション持越しなし）。

Status:
- in_progress

## 2026-02-26 09:36 UTC / 2026-02-26 18:36 JST - 改善策統合プラン（P0/P1/P2）
Period:
- Synthesis window: `2026-02-24` ～ `2026-02-26`（既存3エントリ統合）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/metrics.db`, OANDA account/openTrades

Fact:
- 24h: `1211 trades`, `-5155.7 JPY`, `PF=0.773`
- 7d: `4277 trades`, `-25750.3 JPY`, `PF=0.650`
- 7d 主損失は `MARKET_ORDER_MARGIN_CLOSEOUT=-19124.7 JPY` と `STOP_LOSS_ORDER=-18393.1 JPY`
- `strategy_control_exit_disabled=10277`（`2026-02-24` 単日）
- 24h 主損失戦略: `scalp_ping_5s_c_live=-6094.2 JPY`, `scalp_ping_5s_b_live=-3141.3 JPY`
- core unit churn: `quant-order-manager starts=62`, `quant-market-data-feed starts=22`, `quant-scalp-ping-5s-c starts=30`（24h）

Failure Cause:
1. `entry=1 & exit=0` と broker `stopLossOnFill` 欠損が重なると、EXIT封鎖から closeout テールへ直結する。
2. `scalp_ping_5s_b/c` は勝率寄り運用で、`avg_win/avg_loss` と `jpy PF` が負のまま件数が先行している。
3. 手動玉込み余力監視と 1-trade loss cap の運用が弱く、大損1発の耐性が不足している。
4. core unit の再起動多発で、執行品質（reject/timeout/latency）の再現性が落ちる。

Improvement:
1. P0（当日）安全不変条件:
   - `entry=1 & exit=0` を自動補正で禁止（検知時は `entry=0` へ強制 + alert）。
   - 非manual新規注文は broker `stopLossOnFill` 必須化（欠損は `missing_broker_sl` reject）。
   - 手動玉込み余力で新規抑制を先行適用し、`MARKET_ORDER_MARGIN_CLOSEOUT` を即時遮断。
2. P1（24-48h）収益反転:
   - `scalp_ping_5s_c_live` は停止ではなく「低頻度・低サイズ運用」に固定し、`jpy PF>=1` まで上限解除しない。
   - `scalp_ping_5s_b_live` も同一KPI（`jpy PF`, `avg_win/avg_loss`, `avg_loss_jpy`）でロット再配分。
   - `dynamic_alloc` は pips優先を避け、`realized_jpy_per_1k_units` と `jpy_downside_share` 主導に固定。
3. P1（24-48h）EXIT詰まり解消:
   - `close_reject_no_negative` を strategy tag × exit_reason で棚卸しし、拒否条件の過剰部分を除去。
   - 負け玉の平均保有時間を短縮するため、reject系ステータスの即時監査を追加。
4. P2（72h）運用品質:
   - core unit ごとの restart budget を設定し、閾値超過時は新規改善より先に安定化を実施。
   - 日次で `orders/trades/strategy_control` を定型監査し、本台帳へ同フォーマット追記。

Verification:
1. `strategy_control.db` で `entry=1 & exit=0` が 0 件、`orders.db` で `strategy_control_exit_disabled` が 24h 連続 0 件。
2. 非manual filled の broker `stopLossOnFill` 設定率 `>= 99%`。
3. `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続で件数 0 / JPY損失 0。
4. `scalp_ping_5s_b/c` の 24h `jpy PF >= 1.00` かつ `avg_win/abs(avg_loss) >= 1.00`。
5. core unit の start/stop が各 24h で `<= 5`。
6. 全体 `all_24h JPY` が 3日連続でプラス。

Status:
- in_progress

## 2026-02-26 09:25 UTC / 2026-02-26 18:25 JST - 回転点まとめ（資産曲線を反転させる条件）
Period:
- 24h/7d の実績再集計: `datetime(close_time) >= now - 1 day / 7 day`
- Exit封鎖イベント確認: `2026-02-24`（`orders.db` / `strategy_control.db`）
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/metrics.db`, OANDA account summary

Fact:
- 24h 合計: `1211 trades`, `+525.6 pips`, `-5155.7 JPY`, `PF=0.773`
- 7d 合計: `4277 trades`, `-2287.7 pips`, `-25750.3 JPY`, `PF=0.650`
- 24h の主損失戦略:
  - `scalp_ping_5s_c_live`: `545 trades`, `-6094.2 JPY`（勝率 `53.0%` でも `avg_win=+20.5 JPY < avg_loss=-48.1 JPY`）
  - `scalp_ping_5s_b_live`: `480 trades`, `-3141.3 JPY`
- 24h の反事実:
  - `all_24h = -5155.7 JPY`
  - `exclude C = +938.5 JPY`
  - `exclude B/C = +4079.7 JPY`
  - `exclude margin_closeout = -2991.9 JPY`
- 7d の損失内訳:
  - `MARKET_ORDER_MARGIN_CLOSEOUT=-19124.7 JPY`（負け総額の `51.0%`）
  - `STOP_LOSS_ORDER=-18393.1 JPY`（同 `49.0%`）
- `2026-02-24` 単日に `strategy_control_exit_disabled=10277`。同日 `micropullbackema` で `margin closeout 4件 = -16837.4 JPY`。
- 現在も `strategy_control.db` に `exit_enabled=0` が残存:
  - `micropullbackema`, `micropullbackema-short`, `microtrendretest`, `microtrendretest-long`, `scalp_ping_5s_flow_live`
- 24h の unit churn（journal）:
  - `quant-order-manager starts=62 / stops=30`
  - `quant-market-data-feed starts=22 / stops=20`
  - `quant-scalp-ping-5s-c starts=30 / stops=28`

Failure Cause:
1. C/B の期待値が負のまま件数が多く、日次損失のベースを作っている。
2. Exit封鎖（`entry=1 & exit=0`）が実ポジに刺さると、損失玉が閉じられず closeout テールが発生する。
3. 再起動多発で order/position 連携が不安定化し、執行品質とリスク制御の再現性が落ちる。
4. `perf_block` と `probability_scaled` が大きく、良い局面の約定密度が不足する。

Improvement:
1. **回転点A（資産反転の最低条件）**: `entry=1 & exit=0` を本番で禁止（自動修正 + アラート）。
2. **回転点B（収益反転の主条件）**: `scalp_ping_5s_c_live` の EV を 0 以上にするまで、件数/サイズを強制的に抑える。
3. **回転点C（ドローダウン抑制）**: `MARKET_ORDER_MARGIN_CLOSEOUT` を連続 0 件に固定（manual玉込み余力ガード）。
4. **回転点D（実運用品質）**: core unit の restart/stop を抑制し、連続運転で検証する。

Verification:
1. `orders.db` で `strategy_control_exit_disabled` が 24h 連続で 0 件。
2. `trades.db` で `MARKET_ORDER_MARGIN_CLOSEOUT` が 7日連続 0 件。
3. `scalp_ping_5s_c_live` の 24h `ev_jpy >= 0` かつ `avg_loss_jpy` の絶対値縮小。
4. 日次反事実で `exclude C` が不要な状態（実績 `all_24h` が単独でプラス）へ移行。

Status:
- in_progress

## 2026-02-26 12:58 UTC / 2026-02-26 21:58 JST - `stage_state.db` ロックで market_order が失敗しエントリー停止
Period:
- Window: `2026-02-26 11:42` ～ `12:56` UTC
- Source: VM `journalctl -u quant-order-manager.service`, `logs/orders.db`, `logs/trades.db`

Fact:
- `quant-order-manager` は active だが、`[ORDER_MANAGER_WORKER] request failed: database is locked` が連続発生。
- `orders.db` 直近2h: `preflight_start=237`, `probability_scaled=99`, `perf_block=78`, `entry_probability_reject=19`, `slo_block=11`。
- `trades.db` 直近2h の新規 `entry_time` は `0` 件、open trades も `0` 件。
- 失敗直前ログは `OPEN_SCALE` / `projected margin scale` まで進み、その後に `database is locked` で abort している。

Failure Cause:
1. `execution/strategy_guard.py` が `logs/stage_state.db` に対してロック耐性なし（busy_timeout/retryなし）でアクセス。
2. 同DBを `stage_tracker` 系と共有するため、短時間の書き込み競合で `sqlite OperationalError: database is locked` が発生。
3. 例外が `market_order` 経路まで伝播し、preflight通過後でも発注まで到達しない。

Improvement:
1. `strategy_guard` に busy_timeout + WAL + lock retry を追加。
2. lock時は fail-open（`is_blocked=False`）で返し、エントリーを DB 競合で止めない。
3. `set_block` / `clear_expired` / expired削除も lock耐性化して例外伝播を防止。

Verification:
1. `pytest -q tests/test_stage_tracker.py` pass（3 passed）。
2. デプロイ後、`journalctl -u quant-order-manager.service` で `request failed: database is locked` が再発しないこと。
3. `orders.db` で `preflight_start` のみ増える状態が解消し、`filled` が復帰すること。

Status:
- in_progress

## 2026-02-26 13:10 UTC / 2026-02-26 22:10 JST - B/C エントリー枯渇に対する no-signal 緩和（revert依存解除 + side filter開放）
Period:
- Window: `2026-02-26 13:08` ～ `13:10` UTC
- Source: VM `journalctl -u quant-scalp-ping-5s-b.service`, `quant-scalp-ping-5s-c.service`

Fact:
- `quant-order-manager` の lockエラーは再発していないが、B/C ワーカーは `entry-skip` が継続。
- 主因は `no_signal:revert_not_found` と `units_below_min`（B/C とも short 側で 8〜15件/集計周期）。
- `side_filter_block` も残り、sell固定のままでは通過率が伸びにくい。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - `SIDE_FILTER=none`, `REVERT_ENABLED=0`
   - `MAX_ORDERS_PER_MINUTE=6`, `BASE_ENTRY_UNITS=900`
   - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.45`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE=65`
   - `CONF_FLOOR=72`
2. `ops/env/scalp_ping_5s_c.env`
   - `SIDE_FILTER=none`, `ALLOW_NO_SIDE_FILTER=1`, `REVERT_ENABLED=0`
   - `MAX_ORDERS_PER_MINUTE=6`, `BASE_ENTRY_UNITS=260`
   - `MIN_UNITS_RESCUE_MIN_ENTRY_PROBABILITY=0.45`, `MIN_UNITS_RESCUE_MIN_CONFIDENCE=70`
   - `CONF_FLOOR=74`

Verification:
1. デプロイ後 10〜20 分で `orders.db` の `preflight_start` だけでなく `filled` が増えること。
2. `entry-skip summary` の `revert_not_found` 比率が低下すること。
3. `units_below_min` が `min_units_rescue` に置換され、0件へ収束すること。

Status:
- in_progress

## 2026-02-26 13:20 UTC / 2026-02-26 22:20 JST - B/C side filter を sell 再固定（精度優先）
Period:
- Analysis/retune window: `2026-02-26 13:10` ～ `13:20` UTC
- Source: repository `ops/env/scalp_ping_5s_b.env`, `ops/env/scalp_ping_5s_c.env`

Fact:
- 最新 `main` では B/C とも `SIDE_FILTER=none` + `ALLOW_NO_SIDE_FILTER=1` になっていた。
- 同時に `MIN_UNITS_RESCUE` は導入済みで、no-entry 側は rescue で緩和可能な状態だった。

Failure Cause:
1. side filter 開放により、方向精度劣化の再発リスク（buy 再流入）を残していた。

Improvement:
1. B/C を `SIDE_FILTER=sell`, `ALLOW_NO_SIDE_FILTER=0` へ再固定。
2. `MIN_UNITS_RESCUE` は維持しつつ閾値を再引き上げ
   - B: `prob>=0.58`, `conf>=78`
   - C: `prob>=0.60`, `conf>=82`

Verification:
1. デプロイ後ログで `env mapped ... side_filter=sell` を確認。
2. restart 後の B/C journal で `side_filter_fallback:long->short` が継続して出ることを確認。

Status:
- in_progress

## 2026-02-26 09:15 UTC / 2026-02-26 18:15 JST - EXIT封鎖とSL未実装の複合で margin closeout が連鎖
Period:
- Incident window: `2026-02-24 02:20:16` ～ `09:13:14` UTC（`2026-02-24 11:20:16` ～ `18:13:14` JST）
- P/L window: `datetime(close_time) >= now - 14 day`
- Source: VM `/home/tossaki/QuantRabbit/logs/strategy_control.db`, `/home/tossaki/QuantRabbit/logs/orders.db`, `/home/tossaki/QuantRabbit/logs/trades.db`, OANDA account/openTrades

Fact:
- `strategy_control_flags` に `entry=1 & exit=0` が残存（`note=manual_hold_current_positions_20260224`）:
  - `micropullbackema`, `micropullbackema-short`
  - `microtrendretest`, `microtrendretest-long`
  - `scalp_ping_5s_flow_live`
- `orders.db` の `strategy_control_exit_disabled` は合計 `10,277` 件。
  - 戦略別: `MicroPullbackEMA-short=8,177`, `MicroTrendRetest-long=2,068`, `scalp_ping_5s_flow_live=32`
- `MicroPullbackEMA` の margin closeout 4件（`384420/384425/384430/384435`）は、
  各 ticket で `strategy_control_exit_disabled` が `2,044～2,045` 回連続した後に強制クローズ。
  - 合計 `-16,837.4 JPY` / `-582.6 pips`
- 上記4件の filled 注文は `takeProfitOnFill` はある一方、`stopLossOnFill` が未設定（broker SLなし）。
- 14日合計: `-93,081.6 JPY`
  - `MARKET_ORDER_MARGIN_CLOSEOUT=-49,370.5 JPY`
  - 非closeoutでも `-43,711.0 JPY`
- 直近口座状態（2026-02-26 09:04 UTC）:
  - `NAV=57,731.21`, `margin_used=53,020.96`, `margin_available=4,744.25`
  - open trade に `TP/SL なし` の `USD_JPY -8500` が残存

Failure Cause:
1. `entry=1 & exit=0` の危険状態を作る運用が残り、保有調整を詰まらせた。
2. close経路が `strategy_control` により拒否され続け、損失玉が放置された。
3. broker側 `stopLossOnFill` 未設定の玉は、ローカルEXIT依存となり封鎖時に脆弱。
4. margin closeout を除いても `scalp_ping_5s_b/c` を中心に期待値が負で、資産減少が継続。

Improvement:
1. 運用ルール固定: `entry=1 & exit=0` を禁止し、保有維持が必要なときは `entry=0 & exit=1` のみ許可。
2. 自動ガード追加: `strategy_control` に `entry=1 & exit=0` が入った時点でアラート + 自動修正（`entry=0` へ強制）。
3. 発注必須化: 非manualの新規注文は `stopLossOnFill` 必須。欠損時は reject して `status=missing_broker_sl` を記録。
4. 既存残骸の解消: `manual_hold_current_positions_20260224` の stale flag を全解除し、原因ノート付きで再設定履歴を残す。
5. 平常時改善: `scalp_ping_5s_b/c` は `win_rate` ではなく `jpy PF` と `avg_win/avg_loss` を主指標に再設計。

Verification:
1. `strategy_control.db` で `entry=1 & exit=0` 行が 0 件。
2. `orders.db` の `strategy_control_exit_disabled` が 24h で 0 件。
3. 7日連続で `MARKET_ORDER_MARGIN_CLOSEOUT` の件数 0 / JPY損失 0。
4. 非manual filled の `stopLossOnFill` 設定率が `>= 95%`（戦略別に監査）。
5. `scalp_ping_5s_b/c` の14日 `jpy PF >= 1.0` かつ `avg_loss_jpy` の絶対値縮小を確認。

Status:
- in_progress

## 2026-02-26 08:40 UTC / 2026-02-26 17:40 JST - Margin Closeout Tail が資産毀損の主因
Period:
- 24h: `datetime(close_time) >= now - 1 day`
- 7d: `datetime(close_time) >= now - 7 day`
- Source: VM `/home/tossaki/QuantRabbit/logs/trades.db`, `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 24h 合計: `1213 trades`, `+555.4 pips`, `-4652 JPY`
- 7d 合計: `4283 trades`, `-2281.8 pips`, `-25242 JPY`
- 7d close reason:
  - `STOP_LOSS_ORDER`: `1752`, `-3980.5 pips`, `-18436 JPY`
  - `MARKET_ORDER_MARGIN_CLOSEOUT`: `17`, `-602.5 pips`, `-17415 JPY`
  - `MARKET_ORDER_TRADE_CLOSE`: `2217`, `+1805.6 pips`, `+315 JPY`
- 7d 戦略別（下位）:
  - `MicroPullbackEMA`: `46`, `-520.3 pips`, `-15527 JPY`（`margin_closeout 4件 = -16837 JPY`）
  - `scalp_ping_5s_c_live`: `855`, `-1006.7 pips`, `-10554 JPY`
  - `scalp_ping_5s_b_live`: `2093`, `-1879.5 pips`, `-9448 JPY`
- 口座スナップショット（VM実測）:
  - `nav=56813.21`, `margin_used=53057.68`, `margin_available=3789.53`, `health_buffer=0.06666`
  - open positions: `manual short -8500 units`
- 稼働確認:
  - `quant-scalp-ping-5s-b.service`: running, `SCALP_PING_5S_B_ENABLED=1`
  - `quant-scalp-ping-5s-c.service`: running, `SCALP_PING_5S_C_ENABLED=1`
  - `quant-scalp-ping-5s-d.service`: running, `SCALP_PING_5S_D_ENABLED=0`

Failure Cause:
1. `Margin closeout` を許す残余証拠金状態で稼働を継続し、少数の巨大損失が累積。
2. `scalp_ping_5s_c_live` は勝率が一定でもペイオフ負け（平均損失が平均利益を上回る）。
3. `close_reject_no_negative` が多発し、負け玉解放が遅れる経路が残る。
4. DD判定は存在するが、エントリー停止ガードへの直接連携が弱く、資産保全の最終防衛線になっていない。
5. 1トレード損失上限（loss cap）機能はあるが、運用値未設定で未活用。

Improvement:
1. 最優先は `margin closeout 回避`:
   - 余力逼迫時の新規抑制条件を先に適用（手動玉込みで判定）。
2. 次に `1トレード損失上限` を明示有効化:
   - `ORDER_ENTRY_LOSS_CAP_*` 系を strategy/pocket 単位で設定。
3. `close_reject_no_negative` の原因別棚卸し:
   - `exit_reason` と `neg_exit policy` の不整合を strategy tag 単位で解消。
4. `scalp_ping_5s_b/c` は勝率ではなく `avg_win/avg_loss` と `jpy PF` を第一指標に調整。
5. 24hごとに同一フォーマットで本台帳へ追記し、改善の効き/副作用を継続監査。

Verification:
1. `MARKET_ORDER_MARGIN_CLOSEOUT` の 24h 件数・JPY損失が連続減少。
2. 24h 合計で `pips` と `JPY` の符号乖離（`+pips / -JPY`）が解消。
3. `close_reject_no_negative` の件数が減少し、負け玉の平均保有時間が短縮。
4. `scalp_ping_5s_b/c` の `jpy PF` と `avg_loss_jpy` が改善。

Status:
- in_progress

## 2026-02-27 08:28 UTC / 2026-02-27 17:28 JST - `WickReversalBlend` の成功例へ寄せる閾値更新（entry probability + exit buffer）

Period:
- Analysis window: 24h / 7d / 14d（`trades.db`）
- Tick validate window: 2026-02-27 UTC（`USD_JPY_ticks_20260227.jsonl`）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `/home/tossaki/QuantRabbit/logs/replay/USD_JPY/USD_JPY_ticks_20260227.jsonl`
- `~/.codex/skills/qr-tick-entry-validate/scripts/tick_entry_validate.py`

Fact:
- `WickReversalBlend` 24h は `16 trades / +403.5 JPY`、14d は `24 trades / +255.8 JPY`。
- `entry_probability` 閾値シミュレーション:
  - 現状相当（`>=0.70`）: `24h +403.5 JPY`, `14d +362.8 JPY`
  - `>=0.78`: `24h +497.8 JPY`, `14d +457.1 JPY`（件数は `16 -> 12` / `20 -> 16`）
  - `>=0.80`: `24h +490.9 JPY`, `14d +450.2 JPY`
- 24h の損失 3件はいずれも `MARKET_ORDER_TRADE_CLOSE` で、`hold_sec=349/446/794` と長期化。
- `orders.db` 実測:
  - `ticket 408599` で `close_reject_profit_buffer`（`est_pips=0.5 < min_profit_pips=0.6`）が発生後、`max_adverse` で `-322.0 JPY` へ拡大。
- Tick照合（同ticket）:
  - `sl_hit_s=132`、`hold_sec=794`、`MAE_300s=3.9 pips` と逆行継続。

Failure Cause:
1. `entry_probability 0.75-0.80` 帯の期待値が悪く、勝ちパターンへの集中度が不足。
2. `min_profit_pips=0.6` が lock系 close を弾き、`max_adverse` まで保持して損失が拡大。
3. `loss_cut_max_hold_sec=900` が長く、逆行ポジの解放が遅い。

Improvement:
1. `ops/env/quant-scalp-wick-reversal-blend.env`
   - `SCALP_PRECISION_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.78`（from `0.46`）
2. `config/strategy_exit_protections.yaml`（`WickReversalBlend`）
   - `min_profit_pips: 0.45`（from `0.6`）
   - `loss_cut_max_hold_sec: 420`（from `900`）

Verification:
1. 反映後 2h/24h で `WickReversalBlend` の `entry_probability` 分布が `>=0.78` に収束すること。
2. `orders.db` の `close_reject_profit_buffer`（WickReversalBlend）が減少すること。
3. `trades.db` で `hold_sec>=420` かつ負けの件数が減ること。
4. `WickReversalBlend` の 24h `sum(realized_pl)` が改善または維持されること。

Status:
- in_progress

## 2026-02-27 09:20 UTC / 2026-02-27 18:20 JST - `scalp_ping_5s_b/c` の「SL過大・利幅不足・long lot圧縮」是正（VM実測ベース）

Period:
- 直近24h（`julianday(now, '-24 hours')`）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- side別（24h）:
  - `long`: `755 trades / -416.9 pips / -730.1 JPY / avg_units=185.8`
  - `short`: `277 trades / -129.5 pips / +1448.6 JPY / avg_units=338.9`
- `scalp_ping_5s_b_live`:
  - `long`: `426 trades / -565.8 JPY / avg_win=1.165 pips / avg_loss=2.035 pips / sl_reason_rate=0.507`
  - `filled long` の実効距離: `avg_sl=2.03 pips`, `avg_tp=0.99 pips`, `tp/sl=0.49`
- `scalp_ping_5s_c_live`:
  - `long`: `266 trades / -117.8 JPY / avg_win=1.159 pips / avg_loss=1.857 pips / sl_reason_rate=0.444`
  - `filled long` の実効距離: `avg_sl=1.30 pips`, `avg_tp=0.90 pips`, `tp/sl=0.69`

Failure Cause:
1. B/C とも long 側で `avg_loss_pips > avg_win_pips` が継続し、RR が負け越し。
2. B/C の lot 圧縮（base units + preserve-intent max scale + leading profile units上限）が重なり、収益復元が遅延。
3. `TP_ENABLED=0` 運用下でも virtual target が短く、実効 `tp/sl` が 1.0 未満に張り付く局面が多い。

Improvement:
1. `ops/env/scalp_ping_5s_b.env`
   - long RR改善: `SL_BASE 1.6 -> 1.35`, `SL_MAX 2.4 -> 2.0`, `TP_BASE 0.35 -> 0.55`, `TP_MAX 1.4 -> 1.9`, `TP_NET_MIN 0.35 -> 0.45`
   - short維持: `SHORT_TP_BASE=0.35`, `SHORT_TP_MAX=1.4` を明示
   - lot復元: `BASE_ENTRY_UNITS 220 -> 260`, `MAX_UNITS 750 -> 900`
   - 圧縮緩和: `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE 0.32 -> 0.42`, `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT 0.80 -> 0.95`
2. `ops/env/scalp_ping_5s_c.env`
   - long RR改善: `SL_BASE 1.3 -> 1.15`, `SL_MIN=0.85`, `SL_MAX=1.9`, `TP_BASE 0.20 -> 0.45`, `TP_MAX 1.0 -> 1.5`, `TP_NET_MIN 0.25 -> 0.40`
   - short維持: `SHORT_TP_BASE=0.20`, `SHORT_TP_MAX=1.0`, `SHORT_SL_BASE=1.30`, `SHORT_SL_MIN=0.95`, `SHORT_SL_MAX=2.0`
   - lot復元: `BASE_ENTRY_UNITS 70 -> 95`, `MAX_UNITS 160 -> 220`
   - 圧縮緩和: `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE 0.50 -> 0.62`, `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT 0.75 -> 0.90`

Verification:
1. 反映後2h/24hで `scalp_ping_5s_b/c long` の `avg_loss_pips` が低下し、`tp/sl` が改善すること。
2. `orders.db` で `scalp_ping_5s_b/c` の `filled avg_units` が増加しつつ、`perf_block` と `rejected` が急増しないこと。
3. 24hで `long` 側 `sum(realized_pl)` が改善方向へ転じること。

Status:
- in_progress

## 2026-02-27 10:50 UTC / 2026-02-27 19:50 JST - `scalp_ping_5s_b/c` 反映後の無約定化を追加補正（VMログ）

Period:
- 反映後（`>= 2026-02-27T09:29:00+00:00`）

Source:
- VM `journalctl -u quant-scalp-ping-5s-b.service`
- VM `journalctl -u quant-scalp-ping-5s-c.service`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`

Fact:
- 反映後ログ主因（直近300行）:
  - B: `no_signal:revert_not_found=278`, `rate_limited=123`
  - C: `no_signal:revert_not_found=289`, `rate_limited=92`
- 反映後注文は C long の小ロット約定が散発（`units=18`）で、Bの約定密度が戻っていない。

Failure Cause:
1. `revert` 判定が厳しすぎて signal 化前に落ちる。
2. `MAX_ORDERS_PER_MINUTE=4` でレート制限が先に効き、通過機会を失う。
3. long側は side-metrics と preserve-intent の下限が低く、最終unitsが縮み過ぎる。

Improvement:
1. B/C 共通で `MAX_ORDERS_PER_MINUTE` を `6` へ引き上げ。
2. B/C 共通で `REVERT_*` 閾値を小幅緩和（`MIN_TICKS=1`, `RANGE/SWEEP/BOUNCE` 最小幅を低減）。
3. B/C 共通で long圧縮下限を引き上げ（`SIDE_METRICS_MIN_MULT` と `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE` を上方修正）。

Verification:
1. 反映後2hで `entry-skip` に占める `revert_not_found` と `rate_limited` の比率が低下すること。
2. `orders.db` で `scalp_ping_5s_b/c` の `filled` 件数と `avg_units(long)` が回復すること。
3. 24hで `scalp_ping_5s_b/c long` の `sum(realized_pl)` が改善方向へ向かうこと。

Status:
- in_progress

## 2026-02-27 16:03 UTC / 2026-02-28 01:03 JST - `scalp_ping_5s_b/c` 収益悪化に対する高確度化（VM実測）

Period:
- 直近24h（`close_time >= datetime('now','-24 hour')`）
- 直近7d（`close_time >= datetime('now','-7 day')`）

Source:
- VM `/home/tossaki/QuantRabbit/logs/trades.db`
- VM `/home/tossaki/QuantRabbit/logs/orders.db`
- VM `journalctl -u quant-order-manager.service`
- OANDA openTrades (`scripts/oanda_open_trades.py`)

Fact:
- 24h合計: `-3023.8 JPY / -983.5 pips / 1433 trades`
- 7d合計: `-22169.4 JPY / -2031.4 pips / 4310 trades`
- 24h戦略別:
  - `scalp_ping_5s_c_live`: `605 trades / -3444.2 JPY / -575.1 pips`
  - `scalp_ping_5s_b_live`: `677 trades / -756.8 JPY / -420.4 pips`
- side別（24h）:
  - `B long`: `547 trades / -666.3 JPY / avg_win=1.159 / avg_loss=1.863`
  - `B short`: `130 trades / -93.1 JPY / avg_win=1.156 / avg_loss=2.031`
  - `C long`: `493 trades / -3389.2 JPY / avg_win=1.419 / avg_loss=2.300`
  - `C short`: `112 trades / -54.9 JPY / avg_win=0.946 / avg_loss=1.858`
- `orders.db` 24h status:
  - `perf_block=2943`, `entry_probability_reject=683`, `rejected=317`, `filled=1422`
  - `STOP_LOSS_ON_FILL_LOSS` reject が継続。

Failure Cause:
1. `scalp_ping_5s_c_live` は `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.00` で低品質シグナル通過が過多。
2. B/C とも `avg_loss_pips > avg_win_pips` が続き、高回転で負けを積み上げる構造。
3. B は同時保有・発注回転が高く、逆行局面で負け玉を増幅。

Improvement:
1. `ops/env/scalp_ping_5s_c.env`
   - `SCALP_PING_5S_C_ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.00 -> 0.74`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE: 0.58 -> 0.66`
   - `SCALP_PING_5S_C_CONF_FLOOR: 80 -> 83`
   - `SCALP_PING_5S_C_MAX_ORDERS_PER_MINUTE: 16 -> 10`
   - `SCALP_PING_5S_C_BASE_ENTRY_UNITS: 140 -> 110`
2. `ops/env/scalp_ping_5s_b.env`
   - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW: 0.67 -> 0.72`
   - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE: 0.76 -> 0.80`
   - `SCALP_PING_5S_B_CONF_FLOOR: 80 -> 82`
   - `SCALP_PING_5S_B_MAX_ACTIVE_TRADES: 6 -> 4`
   - `SCALP_PING_5S_B_MAX_PER_DIRECTION: 4 -> 3`
   - `SCALP_PING_5S_B_MAX_ORDERS_PER_MINUTE: 8 -> 6`
   - `SCALP_PING_5S_B_BASE_ENTRY_UNITS: 300 -> 260`

Verification:
1. 反映後2h/24hで `scalp_ping_5s_b/c` の `sum(realized_pl)` が改善方向へ転じること。
2. `orders.db` で `filled` を維持しつつ `rejected` と `STOP_LOSS_ON_FILL_LOSS` が減ること。
3. 24h side別で `avg_loss_pips` が `avg_win_pips` に接近または逆転すること。
4. `MARKET_ORDER_MARGIN_CLOSEOUT` が増えないこと（特に micro系 tail 監視を継続）。

Status:
- in_progress

## 2026-02-28  (UTC) / 2026-02-28 09:00 JST - EXIT阻害原因分解（trade_id整合性）

Period:
- 本調査対象: `logs/orders.db`（ローカル）
- 補助: `logs/trades.db`（ローカル）
- 補足: VM直読はIAP SSHでコマンドが確立しないため保留（ローカル実DBで継続監査）

Source:
- `logs/orders.db`
- `logs/trades.db`
- `execution/order_manager.py`
- `scripts/replay_exit_workers.py`

Fact:
- `orders` のEXIT関連は `close_request=1598`, `close_ok=1399`, `close_failed=199`。
- `close_failed` 内訳:
  - `oanda::rest::core::InvalidParameterException / Invalid value specified for 'tradeID'` = `172`
  - `TRADE_DOESNT_EXIST / The Trade specified does not exist` = `24`
  - `CLOSE_TRADE_UNITS_EXCEED_TRADE_SIZE` = `3`
- `close_failed` の `172` 件は `ticket_id` が `sim-*`。
  - うち `sim-40:65`, `sim-8:63`, `sim-37:43`, `sim-4:1`
  - 時間帯: `2026-02-24T02:07:21.605365+00:00`〜`2026-02-24T02:09:44.063590+00:00` の集中発生
- `close_failed` 172件の `ticket_id` が同じ `sim-` 系で、`sim-sim-` という client/order-id由来の疑い（`client_order_id` も同系統）を示唆。
- 同期間の `close_request`/`close_failed` 差分 `199` は、共通一律EXITガード系の拒否ステータス（`close_reject_*` 相当）ではなく、OANDA API拒否での `InvalidParameterException` が主因。

Failure Cause:
1. `order_manager.close_trade` が受けた `ticket_id` が実トレードIDではなく `sim-` 系の擬似ID（最終的に `sim-sim-*`）になったことによる `tradeID` 形式不正。
2. 追加で `TRADE_DOESNT_EXIST` が少数残るため、既に決済済み/不整合建玉に対する遅延closeの再送も混在。

Action (already partially applied):
1. `scripts/replay_exit_workers.py` の `_create_trade` を `sim-` 重複付与なしで正規化。
   - `trade_id` が既に `sim-` で始まる場合は再付与しない。
   - `client_order_id`/`clientExtensions.id` へ `sim-<id>` を一貫投入。
2. `execution/order_manager.py` の `close_trade()` は `_is_valid_live_trade_id` で数値ID以外を早期拒否し、`log_metric("close_reject_invalid_trade_id")` で観測可能化。

Verification:
1. `close_failed` の `InvalidParameterException` が減衰し、`close_reject_invalid_trade_id` へ寄与遷移するかをVM `orders.db` で要再確認。
2. VM `orders.db` で `ticket_id LIKE 'sim-%'` の `close_failed` 再集計。
3. `close_request` から `close_ok` への通過率、`strategy_control` 拒否系の有無を同期間で再監査。

Status:
- in_progress

## 2026-03-02  (UTC) / 2026-03-02 06:00 JST - 5秒スキャッパー D/Flow 総力復帰

Source:
- local config audit (`ops/env/scalp_ping_5s_d.env`, `ops/env/scalp_ping_5s_flow.env`)
- preflight/env audit (`ops/env/quant-order-manager.env`)

Hypothesis:
- `scalp_ping_5s_d` が `SCALP_PING_5S_D_ENABLED=0` のままで、`scalp_ping_5s_flow` も停止状態だったことが、シグナル欠落の主要因。
- D/Flow 起動後は、`scalp_ping_5s_d` 側の過剰ブロック（ticks/align/leading_profile）を緩和し、`order_manager` の D/Flow 下限を緩く設定してエントリー量を回復させる。

Action:
- `ops/env/scalp_ping_5s_d.env`
  - `SCALP_PING_5S_D_ENABLED: 0 -> 1`
  - `SCALP_PING_5S_D_MAX_ORDERS_PER_MINUTE: 6 -> 12`
  - `SCALP_PING_5S_D_MIN_TICKS: 4 -> 3`
  - `SCALP_PING_5S_D_MIN_SIGNAL_TICKS: 3 -> 2`
  - `SCALP_PING_5S_D_CONF_FLOOR: 74 -> 72`
  - `SCALP_PING_5S_D_ENTRY_PROBABILITY_ALIGN_PENALTY_MAX: 0.55 -> 0.24`
  - `SCALP_PING_5S_D_ENTRY_PROBABILITY_ALIGN_FLOOR_RAW_MIN: 0.70 -> 0.64`
  - `SCALP_PING_5S_D_ENTRY_LEADING_PROFILE_REJECT_BELOW(_SHORT): 0.50/0.62 -> 0.44/0.54`
- `ops/env/scalp_ping_5s_flow.env`
  - `SCALP_PING_5S_FLOW_ENABLED: 0 -> 1`
  - `SCALP_PING_5S_FLOW_MIN_TICKS: 4 -> 3`
  - `SCALP_PING_5S_FLOW_MIN_SIGNAL_TICKS: 3 -> 2`
  - `SCALP_PING_5S_FLOW_SHORT_MIN_TICKS: 3 -> 2`
  - `SCALP_PING_5S_FLOW_SHORT_MIN_SIGNAL_TICKS: 3 -> 2`
  - `SCALP_PING_5S_FLOW_MIN_TICK_RATE: 0.50 -> 0.45`
  - `SCALP_PING_5S_FLOW_SHORT_MIN_TICK_RATE: 0.50 -> 0.45`
  - `SCALP_PING_5S_FLOW_SIGNAL_WINDOW_ADAPTIVE_ENABLED: 0 -> 1`
  - `SCALP_PING_5S_FLOW_ENTRY_LEADING_PROFILE_REJECT_BELOW(_SHORT): 0.45/0.55 -> 0.40/0.52`
- `ops/env/quant-order-manager.env`
  - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_D_LIVE: 1.00 -> 0.80`
  - Flow 用の `ORDER_MANAGER_PRESERVE_INTENT_*` / `ORDER_MIN_UNITS` 設定を新規追加。
- `ops/env/quant-scalp-ping-5s-d.env`, `ops/env/quant-scalp-ping-5s-flow.env`
  - 各戦略のサービス共通 `SCALP_PING_5S_*_ENABLED` を `1` に更新

Verification:
1. VM反映後2時間で `scalp_ping_5s_d_live` と `scalp_ping_5s_flow_live` の `filled` が 0 以外に戻ること。
2. `orders.db` の `entry_probability_reject + rate_limited + revert_not_found` 比率を比較し、合計が前日比で低下すること。
3. `filled` と `stop_loss` の増加バランスが極端化しない（短期で `MARKET_ORDER_MARGIN_CLOSEOUT` が増加しない）こと。

Status:
- in_progress

## 2026-03-03 (JST) / bot entry再開対応（manualポジ非干渉）

Source:
- OANDA summary (`scripts/check_oanda_summary.py`): `openTradeCount=1`（manualのみ継続）
- VM serial (`gcloud compute instances get-serial-port-output`): 戦略起動後も新規約定増加なし
- config audit (`ops/env/quant-v2-runtime.env`, `ops/env/quant-order-manager.env`, `ops/env/scalp_ping_5s_{b,c,d,flow}.env`)

Hypothesis:
1. ping5s系のローカル閾値（`CONF_FLOOR` / `ENTRY_PROBABILITY_ALIGN_FLOOR` / reject_under）が高すぎ、`market_order` 呼び出し前に skip される。
2. `BLOCK_MANUAL_NETTING` が有効だと手動建玉と逆方向のbot entryが抑止されるため、明示的に無効化して manual 非干渉を担保する。

Action:
- `ops/env/quant-v2-runtime.env`
  - `BLOCK_MANUAL_NETTING=0` を追加
- `ops/env/quant-order-manager.env`
  - `BLOCK_MANUAL_NETTING=0` を追加
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B(_LIVE)=0.15`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C(_LIVE)=0.15`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_D(_LIVE)=0.12`
- `ops/env/scalp_ping_5s_b.env`
  - `CONF_FLOOR=60`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.35`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT=0`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_B_LIVE=0.15`
- `ops/env/scalp_ping_5s_c.env`
  - `CONF_FLOOR=60`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR=0.35`
  - `ENTRY_PROBABILITY_ALIGN_FLOOR_REQUIRE_SUPPORT=0`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_C_LIVE=0.15`
- `ops/env/scalp_ping_5s_d.env`
  - `PERF_GUARD_MODE=reduce`
  - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER_STRATEGY_SCALP_PING_5S_D_LIVE=0.12`

Impact:
- manualポジション自体の操作・クローズ処理は未変更（botのENTRY通過条件のみ緩和）。
- V2導線（strategy worker → order_manager）と `entry_thesis` 契約は非変更。

Verification (post-deploy):
1. OANDA `lastTransactionID` が更新され、`openTradeCount` が manual分以外で増えること。
2. `orders.db` の ping5s系 `entry_probability_reject` 比率が低下し、`filled` が復帰すること。
3. `manual_netting_block` ステータス増加がないこと（manual非干渉の維持確認）。

Status:
- in_progress

## 2026-03-04 (JST) / ローカルV2 PDCA導線追加前の市況確認

Source:
- OANDA v3 Pricing (`USD_JPY`) and Candles (`M1`, 80本) をローカル実行で取得。

Snapshot (2026-03-04 JST):
- bid/ask: `157.668 / 157.676`（mid `157.672`）
- spread: `0.8 pips`
- ATR14 (M1): `5.29 pips`
- 直近20本 M1 平均レンジ: `4.96 pips`
- OANDA API応答遅延（3サンプル）: `min 303.6ms / max 336.0ms / avg 318.4ms`

Interpretation:
- スプレッド・短期レンジ・API遅延はいずれも極端な悪化は観測せず、ローカル開発導線追加作業は継続可能と判断。

Action:
- `scripts/local_v2_stack.sh` を追加し、V2サービスをローカルで `up/down/restart/status/logs` 管理できるようにした。
- `ops/env/local-v2-stack.env` を追加し、ローカル上書きテンプレート（local LLMゲート設定例付き）を用意した。

Status:
- done

## 2026-03-04 (JST) / scalp_ping_5s_b 売り側復帰 + ローカルV2安定化

Source:
- `logs/orders.db` / `logs/trades.db` / `logs/metrics.db`
- `logs/local_v2_stack/quant-scalp-ping-5s-b.log`
- `logs/local_v2_stack/quant-order-manager.log`
- `logs/local_v2_stack/quant-position-manager.log`

Hypothesis:
1. 売り限定ではなく両方向判定は有効だが、short 側は `units_below_min` で事実上失注していた。
2. `position-manager`/`order-manager` の起動・停止の不安定さが、entry/exit 評価の連続性を壊していた。

Action:
- `ops/env/scalp_ping_5s_b.env`
  - `SCALP_PING_5S_B_SIDE_BIAS_SCALE_FLOOR: 0.40 -> 0.55`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX: 0.32 -> 0.22`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT: 0.45 -> 0.60`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_BAND_ALLOC_UNITS_MIN_MULT: 0.40 -> 0.55`
  - `SCALP_PING_5S_B_SHORT_MIN_SIGNAL_TICKS: 3 -> 2`
  - `SCALP_PING_5S_B_SHORT_MIN_TICK_RATE: 0.50 -> 0.42`
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_MIN_MULT: 0.60 -> 0.75`
- `ops/env/quant-order-manager.env`
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE): 10 -> 6`
- `scripts/local_v2_stack.sh`
  - `quant-scalp-ping-5s-b` のPID照合パターンに `workers.scalp_ping_5s.worker` を追加し、`status/down` の誤判定を抑制。

Verification (post-tune, 2026-03-04 16:03 JST 以降):
- `orders.db`（`status in preflight_start/filled/rejected`）
  - `buy: attempts=15 / filled=15 / rejected=0`
  - `sell: attempts=5 / filled=5 / rejected=0`
- `quant-scalp-ping-5s-b.log`
  - short open が復帰（例: `units=-100, -94, -85, -42`）
  - short skip 理由の主成分が `units_below_min` から `rate_limited/cooldown/max_active_cap` へ移行
- `metrics.db`（`account.nav`）
  - `07:03:34 UTC -> 07:09:48 UTC` で `+812.554`（含み評価）

Interpretation:
- 売り限定化は解消し、short 実約定は再開した。
- ただし直後は open ポジションが残る時間帯があり、`realized_pl` の評価窓はもう少し必要。

Next:
1. 同設定で 30-60分連続観測し、`close_reason` 別の実現損益（PF/勝率）を再評価。
2. `rate_limited` が過多なら `ENTRY_COOLDOWN_SEC` / `MAX_ORDERS_PER_MINUTE` を微調整。
3. `risk_mult_total=0.4` 固定が続く場合は、劣化要因（SL連打区間）を切り分けて別途改善。

Status:
- in_progress

### 追記: 2026-03-04 16:20 JST / OANDA ReadTimeoutでのworker停止対策

Issue:
- `scalp_ping_5s_b` が `requests.exceptions.ReadTimeout`（`get_account_snapshot`）でプロセス終了する事象を確認。

Action:
- `workers/scalp_ping_5s/worker.py`
  - `get_account_snapshot(cache_ttl_sec=0.5)` を `try/except` で包み、
    - キャッシュ済み snapshot があればそれを継続利用
    - キャッシュが無い場合は `account_snapshot_unavailable` として当該ループのみ skip
  - 例外で worker 全体が落ちないように変更。

Verification:
- `python3 -m py_compile workers/scalp_ping_5s/worker.py` : OK
- patch反映後に `quant-scalp-ping-5s-b` を再起動し、`status` で running を確認。
- 直近2分観測で新規 `Traceback/ReadTimeout` 出力なし（既存ログの旧トレースは除外）。

Status:
- in_progress

## 2026-03-04 18:21 JST / scalp_ping_5s_b short参加拡大 + long損失抑制チューニング

Evidence summary (直近3h):
- fill は long 側優勢で偏り、`buy/sell fill ratio` が高止まり。
- 損失は `STOP_LOSS_ORDER` に集中し、long 側のマイナス寄与が目立つ。
- `units_below_min` は short 側で多発し、売り参加率を押し下げた。

Hypothesis:
- `BASE_ENTRY_UNITS` と各 `*_UNITS_MIN_MULT` の引き上げで short の `units_below_min` を減らし、`SIDE_BIAS_*`/`DIRECTION_BIAS_*`/`*_MOMENTUM_TRIGGER_PIPS`/`ENTRY_CHASE_MAX_PIPS` の再配分で long の逆風局面エントリーを抑制すれば、短期PFの毀損を抑えつつ売り参加を回復できる。

Recheck KPIs (next 60-90 min):
- `buy/sell fill ratio <= 5.0`
- `STOP_LOSS_ORDER share <= 60%`
- `PF >= 0.90`
- `short units_below_min` を現状比 `>=30%` 削減

## 2026-03-04 18:31 JST / restart後フォローアップ（long過大抑制）

- 再起動直後の事実: fill は buy-only、`STOP_LOSS_ORDER` は1件、short 側 `units_below_min` は依然高止まり。
- 施策意図: 両方向ロジックは維持しつつ、long の過大ロット増幅を抑えて long 損失寄与を下げる。
- 45-60分後の再確認KPI: `avg filled buy units` 前run比低下、`STOP_LOSS_ORDER share` 低下、`net_jpy` 改善、short `units_below_min` トレンド非増加。

## 2026-03-04 19:42 JST / short `units_below_min` 第2チューニング

- 再起動直後の事実: short 側 `units_below_min` は高止まりのままで、初期 fill は buy-only だった。
- 第2チューニング仮説: opposite-side 時のロット縮退（lot collapse）を抑え、動的トレードを維持したまま short 参加を回復させる。
- 30-60分の再確認KPI: short fill 増加、short `units_below_min` を現状比 `>=30%` 削減、`buy/sell fill ratio` のトレンド改善。

## 2026-03-04 20:02 JST / short `units_below_min` 第3チューニング

- 第2段反映後の3分窓で `short units_below_min=75`、fill は依然 buy-only だったため、counter-side の縮退下限を追加で引き上げた。
- 変更方針: `MIN_UNITS_RESCUE` 条件緩和、`MTF/HORIZON/M1` の opposite 側ユニット下限引き上げ、`band_alloc` の最小倍率底上げ、同時に long 閾値（`LONG_MOMENTUM_TRIGGER_PIPS`）を上げて long 過多を抑制。
- 再確認KPI（30-60分）: `short fills > 0`、`short units_below_min` を第2段比でさらに `>=30%` 減、`buy/sell fill ratio` の改善継続。

## 2026-03-04 20:08 JST / 実行詰まり（cooldown/rate_limit）緩和

- 第3段後の集計で short 側は `rate_limited` / `cooldown` が主要スキップに浮上し、`OPEN_REQ` は buy のみだった。
- 追加対策: `ENTRY_COOLDOWN_SEC` を短縮し、`MAX_ORDERS_PER_MINUTE` を引き上げて short の通過機会を回復させる。
- 再確認KPI（30-45分）: short `OPEN_REQ` 発生、`rate_limited(short)` の減少、`buy/sell fill ratio` の改善。

## 2026-03-04 20:16 JST / short最小ロット救済ロジック追加

- env調整後も `short units_below_min` が高止まりし、short fill 未発生の窓が継続した。
- 追加実装: `workers/scalp_ping_5s/worker.py` に short 側限定の `short_probe_rescued` を追加し、`fast/sl_streak/side_metrics` の反転根拠がある場合のみ、緩和閾値で `MIN_UNITS` 救済を許可。
- 期待効果: 動的判定を維持したまま、counter-side の 0lot 化を抑制して short の実発注化率を改善する。
- 再確認KPI（30-60分）: short `OPEN_REQ` / `OPEN_FILLED` 発生、`short units_below_min` 低下、`buy/sell fill ratio` 改善。

## 2026-03-04 20:22 JST / short救済条件の再緩和

- 初回実装後の観測で `short_probe_rescued` が未発火だったため、条件を「shortかつprob/conf/risk_cap充足」へ簡素化。
- 目的: short 側の `units_below_min` を直接減らし、buy-only状態からの離脱を優先する。
- 再確認KPI（30-45分）: `short_probe_rescued` ログ発生、short `OPEN_REQ` 発生、`short units_below_min` 低下。

## 2026-03-04 20:28 JST / short救済の最終整合（worker + order_manager）

- 判明事項: worker 側で short を `MIN_UNITS` まで救済しても、`ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B(_LIVE)=4` により order_manager 側で拒否され得る。
- 修正:
  - `workers/scalp_ping_5s/worker.py`: short `units_below_min` は `units_risk >= MIN_UNITS` を満たす限り `short_probe_rescued` で `MIN_UNITS` へ救済。
  - `ops/env/quant-order-manager.env`: `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B_LIVE=1`, `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_B=1` に変更。
- 再確認KPI（30-45分）: `short OPEN_REQ/OPEN_FILLED` 発生、`entry_probability_below_min_units` 減少、`buy/sell fill ratio` 改善。

## 2026-03-04 20:36 JST / short救済を強制発火へ変更

- 観測で `short_probe_rescued` 未発火が継続したため、short の `units < MIN_UNITS` は `MIN_UNITS` へ強制救済する実装に更新。
- 意図: counter-side のシグナルが0lotで消える経路を遮断し、まず short 発注を発生させる。
- 監視: `short_probe_rescued` ログ件数、short `OPEN_REQ/OPEN_FILLED`、短期 `STOP_LOSS_ORDER` 増加有無を同時監視。

## 2026-03-04 20:44 JST / short復帰確認後のlongサイズ圧縮

- 観測結果: short は `OPEN_REQ/OPEN_FILLED` が発生し復帰したが、buy 平均ユニットが大きく（約160）短期損失寄与が拡大。
- 追加対策: `SCALP_PING_5S_B_BASE_ENTRY_UNITS` を `120 -> 70` に圧縮し、long 側の損失インパクトを即時低減。
- 再確認KPI（30-45分）: buy 平均ユニット低下、short fill 維持、短期 net_jpy 改善。

## 2026-03-04 21:35 JST / local-v2 `STOP_LOSS_ON_FILL_LOSS` 初回拒否低減（scalp_ping_5s_b）

- Evidence（`logs/orders.db` 直近24h）:
  - `status='rejected' and error_message='STOP_LOSS_ON_FILL_LOSS'` は 23 件。
  - 同一 `client_order_id` 追跡で、ほぼ全件が `attempt=1 rejected -> attempt=2 filled` の回復パターン。
  - `attempt=1` の `stopLossOnFill` ギャップは概ね `1.3~1.5 pips`。
- Hypothesis:
  - `scalp_ping_5s_b` の SL 上限がタイトで、短時間の価格変位時に初回 `stopLossOnFill` が拒否されやすい。
- Action（ローカル運用上書き）:
  - `ops/env/local-v2-stack.env` と `ops/env/local-v2-full.env` に以下を追加。
  - `SCALP_PING_5S_B_SL_MAX_PIPS=1.60`
  - `SCALP_PING_5S_B_SHORT_SL_MAX_PIPS=1.80`
- Recheck KPI（次の60分）:
  - `STOP_LOSS_ON_FILL_LOSS` の `attempt=1` 発生率低下。
  - `submit_attempt(1) -> filled` までの中央値レイテンシ短縮。
  - `STOP_LOSS_ORDER` 比率が急増していないことを同時確認。

## 2026-03-04 21:34 JST / OANDA実発注確認 + `scalp_ping_5s_b` long偏損対策

- OANDA API実測（`USD_JPY`）:
  - 価格/流動性: `bid=157.196` `ask=157.204` `spread=0.8 pips`
  - ボラ: `ATR14=3.2 pips`, `ATR60=3.083 pips`, `range_60m=20.2 pips`
  - API品質: pricing ping `5/5` 成功, 平均 `269ms`
- 実発注トレース:
  - `order_manager.market_order` 経路（manual最小試験）は内部スケールで `units=-986` となり `INSUFFICIENT_MARGIN` reject。
  - 直接 OANDA REST で `REDUCE_ONLY 1 unit` を実行し約定を確認。
    - `orderCreateTransaction=417417`
    - `orderFillTransaction=417418`
    - `tradeReduced.tradeID=413001`, `realizedPL=+0.1720`
- 直近24hの収益悪化ポイント（`trades.db`, `strategy_tag=scalp_ping_5s_b_live`）:
  - 全体: `n=543`, `PF=0.428`, `winrate=20.3%`, `realized=-166.25`
  - side別: `long n=399 sum=-170.11 PF=0.412` / `short n=147 sum=+2.87 PF=2.155`
  - `STOP_LOSS_ORDER` が損失の大半（特に long 側）
- 反映（long偏損時の縮小/反転を強化）:
  - `SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_MIN_CURRENT_SL_RATE=0.48` (from `0.52`)
  - `SCALP_PING_5S_B_SIDE_METRICS_DIRECTION_FLIP_CONFIDENCE_ADD=6` (from `4`)
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.22` (from `0.12`)
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.72` (from `0.95`)
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_DD_MIN_MULT=0.78` (from `0.92`)
- 監視KPI（次の30-90分）:
  - `long STOP_LOSS_ORDER` 件数/損失寄与の低下
  - `scalp_ping_5s_b_live` の `PF` 改善（0.428 -> 0.8+ を目標）
  - `buy/sell` の fillバランス維持（short優位を殺さないこと）

## 2026-03-04 21:43 JST / ローカル運用: lookahead有効化 + side-adverse強化

- 市況チェック（OANDA live, USD/JPY）:
  - `bid=157.206` `ask=157.214` `spread=0.8 pips`
  - `ATR14=3.2 pips`, `ATR60=3.083 pips`, `range_60m=20.2 pips`
  - pricing応答 `5/5`, 平均レイテンシ `283ms`（p95近似 `294ms`）
- 直近実績（local `logs/*.db`）:
  - `scalp_ping_5s_b_live` 24h: `n=546`, `PF=0.387`, `winrate=20.3%`, `avg=-0.721 pips`
  - side別: `long PF=0.347` / `short PF=0.514`（long側の悪化が主因）
  - `orders.db` 24h reject: `STOP_LOSS_ON_FILL_LOSS=23`, `api_error(502)=1`
- 実発注確認（本番キー・最小往復）:
  - OANDA REST で `USD_JPY -1 unit` を約定→3秒後に全決済
  - `trade_id=417472`, `open=157.158`, `close=157.174`, `realized=-0.0160`
- 反映（`ops/env/scalp_ping_5s_b.env`）:
  - `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=1`
  - `SCALP_PING_5S_B_LOOKAHEAD_ALLOW_THIN_EDGE=0`
  - `SCALP_PING_5S_B_LOOKAHEAD_EDGE_MIN_PIPS=0.14`
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_ACTIVE_START=3`（`4 -> 3`）
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_STEP_MULT=0.28`（`0.22 -> 0.28`）
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_UNITS_MIN_MULT=0.45`（`0.60 -> 0.45`）
  - `SCALP_PING_5S_B_SIDE_ADVERSE_STACK_DD_MIN_MULT=0.55`（`0.65 -> 0.55`）
- 目的:
  - 薄いエッジのエントリーを lookahead で遮断
  - 損失側サイドが続く局面でロット縮小を早期/強度高めに適用

## 2026-03-04 22:05 JST / `scalp_ping_5s_b` 逆行耐性とpreserve_intentレンジ再調整

- 根因:
  - `scalp_ping_5s_b` は逆行時の早期 `force_exit` と対向トレンド側の縮小不足が重なり、ノイズ帯で損失確定が先行。
  - 併せて `preserve_intent` 下限が低く、意図ロットが過小化しやすい局面が残存。
- 変更値（2026-03-04）:
  - `SCALP_PING_5S_B_SL_BASE_PIPS=1.15`
  - `SCALP_PING_5S_B_SHORT_SL_BASE_PIPS=1.30`
  - `SCALP_PING_5S_B_SL_MAX_PIPS=2.00`
  - `SCALP_PING_5S_B_SHORT_SL_MAX_PIPS=2.10`
  - `SCALP_PING_5S_B_FORCE_EXIT_FLOATING_LOSS_MIN_HOLD_SEC=3`
  - `SCALP_PING_5S_B_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.75`
  - `SCALP_PING_5S_B_SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.70`
  - `SCALP_PING_5S_B_M1_TREND_OPPOSITE_UNITS_MULT=0.70`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_COUNTER_EXTRA_PENALTY_MAX=0.18`
  - `SCALP_PING_5S_B_ENTRY_PROBABILITY_ALIGN_UNITS_MIN_MULT=0.88`
  - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=0.60`
  - `ORDER_MANAGER_PRESERVE_INTENT_MAX_SCALE_STRATEGY_SCALP_PING_5S_B_LIVE=1.00`
- 検証KPI:
  - `30m`: `STOP_LOSS_ORDER` 件数、`force_exit` 理由内訳、`OPEN_REQ -> OPEN_FILLED` 成功率。
  - `2h`: `scalp_ping_5s_b_live` の side別 PF/勝率、平均保持秒、平均実効units。
  - `24h`: 総合 PF・実現損益・最大DD、`STOP_LOSS_ON_FILL_LOSS` reject率、entry sideバランス。

## 2026-03-04 23:42 JST / ローカルRCA再実施とB戦略の再調整（buy偏損再発防止）

- 市況実測（OANDA API, USD/JPY）:
  - `bid=157.294`, `ask=157.302`, `spread=0.8 pips`
  - `ATR14(M1)=3.336 pips`, `range60(M1)=26.8 pips`
  - pricing API 応答: 平均 `466.82ms`（max `1385.88ms`）
- 収益分解（`logs/trades.db`, 直近24h, `strategy_tag=scalp_ping_5s_b_live`）:
  - 総計: `n=608`, `net_jpy=-175.6`, `win_rate=19.6%`, `PF=0.416`
  - side別: `buy n=412 net=-178.0 avg_units=62.9`, `sell n=196 net=+2.4 avg_units=1.65`
  - close_reason別: `STOP_LOSS_ORDER n=470 net=-280.7`, `MARKET_ORDER_TRADE_CLOSE n=138 net=+105.1`
- 根因:
  - 損失のほぼ全量が `buy + STOP_LOSS_ORDER`（`n=317 net=-277.6`）に集中。
  - `sell` はほぼ建て値圏だが、`spread 0.8` に対して薄いエッジのエントリーが残り、微損を積む。
- 反映（`ops/env/scalp_ping_5s_b.env`）:
  - sell固定維持: `SCALP_PING_5S_B_SIDE_FILTER=sell`, `...ALLOW_NO_SIDE_FILTER=0`
  - 薄利エントリー抑制: `MAX_SPREAD_PIPS=0.85`, `LOOKAHEAD_EDGE_MIN_PIPS=0.22`, `LOOKAHEAD_SAFETY_MARGIN_PIPS=0.12`
  - SL拒否低減: `SL_BASE/MIN=1.25/1.20`, `SHORT_SL_BASE/MIN/MAX=1.45/1.20/2.20`
  - 確率帯ロット再配分: `LOW/HIGH_THRESHOLD=0.65/0.80`, `HIGH_REDUCE_MAX=0.70`, `LOW_BOOST_MAX=0.16`, `UNITS_MIN_MULT=0.55`
  - shortの強制最小ロット停止: `SCALP_PING_5S_B_SHORT_PROBE_RESCUE_ENABLED=0`
- 実装追補:
  - `workers/scalp_ping_5s/config.py` に `SCALP_PING_5S_SHORT_PROBE_RESCUE_ENABLED` を追加。
  - `workers/scalp_ping_5s/worker.py` で short救済を関数化し、envでON/OFF可能化。
- 追加監査:
  - `close_failed` の大半は `ticket_id=sim-*` 由来（直近12000件中 `172/182`）で、実運用ノイズとして分離。
- 再検証KPI（次の30-90分）:
  - `buy` の `preflight_start/submit_attempt/filled` が 0 を維持すること。
  - `STOP_LOSS_ON_FILL_LOSS` 比率の低下。
  - `sell` の `expectancy_jpy > 0` への復帰、および `PF > 1.0`。

## 2026-03-04 23:46 JST / ローカル自動復帰（launchd）導入

- 目的:
  - PC再起動/ログイン後、スリープ復帰後、ネット復帰後に `local_v2_stack` を自動再開し、
    手動起動なしで運用継続できるようにする。
- 作業前市況確認（OANDA API）:
  - `bid=157.304`, `ask=157.312`, `spread=0.8 pips`
  - `ATR14(M1)=4.5 pips`, `range60(M1)=23.1 pips`
  - pricing応答: 平均 `230.28ms`, max `246.18ms`
  - 作業継続判定: 通常レンジ内として実施
- 実装:
  - `scripts/local_v2_autorecover_once.sh`（新規）
    - `local_v2_stack status` で `stopped/stale` 検知時のみ `up` を実行
    - ネット未接続時は待機（`api-fxtrade.oanda.com:443` 到達確認）
    - `parity` 競合（exit code `3`）時は安全にスキップ
    - 重複実行防止のロック (`logs/local_v2_autorecover.lock`)
  - `scripts/install_local_v2_launchd.sh`（新規）
    - LaunchAgent を `~/Library/LaunchAgents` に生成・`bootstrap`
    - `RunAtLoad + StartInterval + KeepAlive(NetworkState)` を設定
  - `scripts/uninstall_local_v2_launchd.sh`（新規）
  - `scripts/status_local_v2_launchd.sh`（新規）
  - `docs/OPS_LOCAL_RUNBOOK.md`
    - 自動復帰セクション（install/status/uninstall、ログ位置）を追加
- 監視ログ:
  - `logs/local_v2_autorecover.log`
  - `logs/local_v2_autorecover.launchd.out`
  - `logs/local_v2_autorecover.launchd.err`

## 2026-03-05 00:06 JST / 自動復帰の実働安定化（launchd 126/子プロセス回収/lock詰まり修正）

- 症状:
  - LaunchAgent の `last exit code=126`（`Operation not permitted`）で自動復帰が動作しない。
  - `up` 実行直後にワーカーが落ちる（launchd が子プロセスを回収）。
  - ロックディレクトリ残骸で autorecover が無反応になる。
- 根因:
  - macOS `launchd` から `~/Documents` 実体へのスクリプト読み取り制約。
  - plist に `AbandonProcessGroup` 未設定で、ジョブ終了時に spawned worker が終了。
  - ロックが `mkdir` のみで stale 判定が無い。
- 対応:
  - リポジトリ実体を `/Users/tossaki/App/QuantRabbit` へ移動し、
    `/Users/tossaki/Documents/App/QuantRabbit` は互換 symlink 化。
  - `scripts/install_local_v2_launchd.sh`
    - 実行コマンドを絶対パス化し `bash -lc` 経由へ統一。
    - `AbandonProcessGroup=true` を付与。
  - `scripts/local_v2_autorecover_once.sh`
    - lock に owner PID を保存。
    - stale lock 自動除去と再取得を実装。
- 検証（ローカル実測）:
  - `scripts/status_local_v2_launchd.sh` で `last exit code=0` を確認。
  - `local_v2_stack down` 後、20〜30秒で `recover` 実行と全8サービス `running` を確認。
  - `quant-micro-rangebreak` を手動 kill 後、約25秒で自動再起動（PID更新）を確認。

## 2026-03-04 17:07 UTC / 2026-03-05 02:07 JST - no-entry緩和（`scalp_ping_5s_b` / `micro_rangebreak`）

- 目的:
  - `entry-skip` 偏重を緩和し、`scalp_ping_5s_b` と `micro_rangebreak` の約定再開余地を増やす。
- 仮説:
  - `scalp_ping_5s_b` の `lookahead_block` / `no_signal:revert_not_found` と、`micro_rangebreak` の trend-flip 偏重を局所緩和すると no-entry を減らせる。
- 変更値:
  - `ops/env/scalp_ping_5s_b.env`
    - `SCALP_PING_5S_B_LOOKAHEAD_ALLOW_THIN_EDGE=1`
    - `SCALP_PING_5S_B_LOOKAHEAD_COUNTER_PENALTY=0.24`
    - `SCALP_PING_5S_B_REVERT_MIN_TICKS=2`
    - `SCALP_PING_5S_B_REVERT_CONFIRM_RATIO_MIN=0.07`
    - `SCALP_PING_5S_B_SHORT_MIN_SIGNAL_TICKS=3`
  - `ops/env/quant-micro-rangebreak.env`
    - `MICRO_MULTI_TREND_FLIP_STRATEGY_BLOCKLIST=MicroLevelReactor,MicroCompressionRevert,MicroRangeBreak`
    - `MICRO_RANGEBREAK_ENTRY_RATIO=0.38`
    - `MICRO_RANGEBREAK_MIN_RANGE_SCORE=0.32`
    - `MICRO_RANGEBREAK_REVERSION_MAX_ADX=27.0`
    - `MICRO_MULTI_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.38`
    - `MICRO_MULTI_MIN_UNITS=500`
    - `MICRO_MULTI_MAX_MARGIN_USAGE=0.95`
- 作業前市況チェック（ローカルV2 + OANDA API）:
  - `USD_JPY bid=156.930 ask=156.938 spread=0.8p`
  - `ATR14=2.764p / ATR60=2.997p / range60=22.0p`
  - `orders.db` 直近24h: `filled=662`, `reject_like=29`
  - OANDA API応答: pricing 平均 `228.5ms`（max `232.3ms`）、candles `217.3ms`
- 検証手順（ローカルV2）:
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env`
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
  - `tail -n 120 logs/local_v2_stack/quant-scalp-ping-5s-b.log`
  - `tail -n 120 logs/local_v2_stack/quant-micro-rangebreak.log`

## 2026-03-05 09:10 JST / `scalp_ping_5s_b` 逆期待値抑制（sell固定 + 薄エッジ遮断）

- 目的:
  - 「全然稼げてない」状態に対し、`scalp_ping_5s_b_live` の逆期待値エントリーを即時抑制する。
- 市況実測（作業前チェック, ローカルV2 + OANDA API）:
  - `USD/JPY mid=156.977`
  - `spread`（tick_cache直近300）`avg=0.8 pips / p95=0.8 pips`
  - `ATR14(M1)=1.743 pips`, `range60=14.2 pips`
  - OANDA summary API 応答: 平均 `218.46ms`（max `238.77ms`, 3/3 success）
- 収益分解（`logs/trades.db`, 直近24h, `strategy_tag=scalp_ping_5s_b_live`）:
  - 総計: `n=611`, `net_realized=-175.658`, `PF=0.416`
  - side別: `buy n=412 net=-177.957`, `sell n=199 net=+2.299`
  - `STOP_LOSS_ORDER` 偏重で損失が集中。
- 原因:
  - 実運用値が緩和モード（`SCALP_PING_5S_B_SIDE_FILTER=none`）のままで、buy側逆期待値を通していた。
  - lookahead/net-edge 閾値が実コスト（`cost_vs_mid ~0.4p`, spread ~`0.8p`）に対して低かった。
- 反映（`ops/env/scalp_ping_5s_b.env`）:
  - `SCALP_PING_5S_B_SIDE_FILTER=sell`
  - `SCALP_PING_5S_B_ALLOW_NO_SIDE_FILTER=0`
  - `SCALP_PING_5S_B_MAX_SPREAD_PIPS=0.90`（1.15→0.90）
  - `SCALP_PING_5S_B_LOOKAHEAD_ALLOW_THIN_EDGE=0`
  - `SCALP_PING_5S_B_LOOKAHEAD_EDGE_MIN_PIPS=0.35`（0.10→0.35）
  - `SCALP_PING_5S_B_LOOKAHEAD_SAFETY_MARGIN_PIPS=0.16`（0.08→0.16）
  - `SCALP_PING_5S_B_ENTRY_NET_EDGE_MIN_PIPS=0.35`（0.20→0.35）
  - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=24`（35→24）
- 再検証KPI（30m / 2h / 24h）:
  - 30m: `buy` の `submit_attempt/filled` が 0 維持、`rejected/api_error` 増加なし。
  - 2h: `scalp_ping_5s_b_live` の `PF >= 0.8`、`expectancy_realized` の改善。
  - 24h: `strategy_tag=scalp_ping_5s_b_live` の `net_realized` 改善、`STOP_LOSS_ORDER` 比率低下。

## 2026-03-05 11:35 JST / scalp_ping_5s_b no-signal緩和（可変パラメータ拡張）

Period:
- 調査時刻: 2026-03-05 11:30〜11:35 JST
- 対象: `ops/env/local-v2-stack.env`, `logs/local_v2_stack/quant-scalp-ping-5s-b.log`, `logs/orders.db`

Fact:
- `SCALP_PING_5S_B_LOOKAHEAD_GATE_ENABLED=0` は実効反映済み（process env確認）。
- ただし最新 skip は `no_signal:revert_not_found` / `no_signal:momentum_tail_failed` が主因で、`filled` 最終時刻は `2026-03-05T02:22:07+00:00` のまま更新停滞。

Improvement:
- `ops/env/local-v2-stack.env` へ以下を追加し、シグナル生成を可変で緩和。
  - `SCALP_PING_5S_B_MIN_SIGNAL_TICKS=3`
  - `SCALP_PING_5S_B_LONG_MIN_SIGNAL_TICKS=3`
  - `SCALP_PING_5S_B_SHORT_MIN_SIGNAL_TICKS=3`
  - `SCALP_PING_5S_B_SIGNAL_MODE_BLOCKLIST=`
  - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.72`
  - `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.78`

Verification:
- 再起動後に `entry-skip summary` の `signal_mode_blocked` と `momentum_tail_failed` 比率が低下すること。
- `orders.db` の `submit_attempt/filled` 最終時刻が更新されること。

Status:
- in_progress

## 2026-03-05 11:40 JST / units_below_min対策（risk floor + base units）

Fact:
- `entry-skip summary` は `units_below_min` が継続（short側で 4〜10 件）。
- `orders` 最終 `filled` は `2026-03-05T02:22:07+00:00` で更新停滞。

Improvement:
- `ops/env/local-v2-stack.env` へ追加。
  - `SCALP_PING_5S_B_BASE_ENTRY_UNITS=32`
  - `RISK_PERF_MIN_MULT=0.55`

Verification:
- 再起動後に `entry-skip summary` の `units_below_min` 比率が低下すること。
- `orders.db` の `submit_attempt` / `filled` 最終時刻が更新されること。

Status:
- in_progress

## 2026-03-05 13:05 JST / 全戦略ワーカー可変調整 + 停止系復旧（local_v2 trade_all）

- 目的:
  - 「全戦略ワーカーの停止解消」と「収益悪化戦略の即時可変チューニング」を同時実施し、trade_all を安定稼働へ戻す。
- 作業前市況（ローカル実測 / OANDA API）:
  - `USD/JPY bid=156.990 ask=156.998 spread=0.8p`
  - `ATR14(M1)=2.921p`, `range60=19.3p`
  - API遅延（pricing 6サンプル）: `avg=251.42ms`, `p95=260.10ms`, `max=271.72ms`
- 停止要因（ログ根拠）:
  - `quant-position-manager(8301)` への `Connection refused` が exit worker で連鎖。
  - `local_v2_autorecover` の旧 `trade_min` 復旧履歴が残り、core restart連鎖を誘発。
- 実施変更:
  - `workers/scalp_macd_rsi_div_b/exit_worker.py` を追加（B exit import欠損解消）。
  - `ops/env/local-v2-stack.env` を可変調整:
    - B: `SIDE_FILTER=sell`, `ALLOW_NO_SIDE_FILTER=0`, `MAX_ACTIVE_TRADES=4`, `MAX_PER_DIRECTION=2`,
      `MIN/LONG/SHORT_MIN_SIGNAL_TICKS=5`, `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.80/0.86`,
      `ENTRY_PROBABILITY_ALIGN_FLOOR=0.62`, `BASE_ENTRY_UNITS=18`。
    - C: `LOOKAHEAD_GATE_ENABLED=1`, `MIN_SIGNAL_TICKS=3`,
      `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.74/0.80`, `ENTRY_PROBABILITY_ALIGN_FLOOR=0.52`, `MAX_SPREAD_PIPS=1.20`。
    - D: `LOOKAHEAD_GATE_ENABLED=1`, `MIN_SIGNAL_TICKS=3`,
      `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.66/0.74`, `ENTRY_PROBABILITY_ALIGN_FLOOR=0.55`,
      `BASE_ENTRY_UNITS=4200`, `MAX_ACTIVE_TRADES=1`, `MAX_PER_DIRECTION=1`,
      `DIRECTION_BIAS_LONG_OPPOSITE_UNITS_MULT=0.30`,
      `FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.65`, `SHORT_FORCE_EXIT_MAX_FLOATING_LOSS_PIPS=0.55`, `MAX_SPREAD_PIPS=1.20`。
    - Flow: `LOOKAHEAD_GATE_ENABLED=1`, `MIN_SIGNAL_TICKS=3`,
      `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.40/0.52`, `BASE_ENTRY_UNITS=700`,
      `MAX_ACTIVE_TRADES=6`, `MAX_PER_DIRECTION=3`。
- 反映:
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-scalp-ping-5s-b,quant-scalp-ping-5s-c,quant-scalp-ping-5s-d,quant-scalp-ping-5s-flow`
  - 直後に `up --services core4` を再実行して PID/health を再確定。
- 検証:
  - `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で `stopped=0`。
  - 60秒後再確認でも core + B/C/D/Flow はすべて `running` 維持。
  - `logs/local_v2_autorecover.log` 最新で `2026-03-05 13:01:51 JST [recover] ... profile=trade_all` を確認。
  - `scripts/collect_local_health.sh` 成功（`logs/health_snapshot.json` 更新）。
- 反映後の初期観測（`datetime(close_time) >= 2026-03-05 04:01:00 UTC`）:
  - closed trades: `1`, realized `+0.21`, win_rate `1.0`（サンプル小）。
  - order status は B/C/D/Flow で `lookahead block` / `entry_probability_reject` / `strategy_cooldown` が主となり、逆期待値シグナルの通過が抑制。

## 2026-03-05 14:40 JST / ローカルLLM常時運用化（停止回避→改善優先）

- 目的:
  - 「トレード停止」ではなく、ローカルLLMで判定改善・パラメータ改善を回しながらエントリーを維持する。

- 作業前市況チェック（ローカル実測）:
  - 取得元: `logs/orders.db`, `logs/brain_state.db`, `logs/local_v2_stack/quant-market-data-feed.log`, `logs/health_snapshot.json`
  - 直近90分 `orders.db`:
    - `mid range: 156.980 - 157.164`
    - `spread avg/min/max: 0.801 / 0.8 / 1.0 pips`
    - `atr_m1 avg: 2.346 pips`, `atr_m5 avg: 5.738 pips`
  - OANDA API品質:
    - `quant-market-data-feed.log` 直近800行のHTTP集計: `HTTP/1.1 200 = 129`, `non-200 = 0`
  - 実行品質:
    - `health_snapshot`: `data_lag_ms ~403`, `decision_latency_ms ~21`
    - 直近90分 `orders.db`: `filled=162`, `rejected=2`（STOP_LOSS_ON_FILL_LOSS）
  - 判定:
    - 通常レンジ内として作業継続（保留条件には非該当）。

- ローカルLLM比較ベンチ（2026-03-05）:
  - レポート:
    - `logs/brain_local_llm_benchmark_multimodel_20260305.json`
    - `logs/brain_local_llm_benchmark_multimodel_4way_20260305.json`
  - 4モデル比較（10 samples, outcome prioritize）:
    - `gpt-oss:20b` score `1.245`, parse `1.0`, p95 `27791ms`
    - `qwen2.5:7b` score `1.231`, parse `1.0`, p95 `3641ms`
    - `llama3.1:8b` score `1.217`, parse `1.0`, p95 `4250ms`
    - `gemma3:4b` score `1.182`, parse `1.0`, p95 `2682ms`
  - 運用選定:
    - preflightは遅延制約を優先して `qwen2.5:7b`
    - async autotuneは品質優先で `gpt-oss:20b`

- 反映内容:
  - `ops/env/local-v2-stack.env`
    - `STRATEGY_CONTROL_ENTRY_*` を再有効化（B/C/D/Flow/MicroPullbackEMA/MicroTrendRetest = 1）
    - `LOCAL_V2_EXTRA_ENV_FILES=ops/env/profiles/brain-ollama.env`
  - `ops/env/profiles/brain-ollama.env`
    - `BRAIN_OLLAMA_MODEL=qwen2.5:7b`
    - `BRAIN_TIMEOUT_SEC=8`
    - `BRAIN_PROMPT_AUTO_TUNE_MODEL=gpt-oss:20b`
    - `BRAIN_RUNTIME_PARAM_AUTO_TUNE_MODEL=gpt-oss:20b`
  - `config/brain_prompt_profile.json`
    - 初期プロファイルを `REDUCE優先・参加率維持` 方針へ更新
  - `config/brain_runtime_param_profile.json`
    - `activity_rate_floor=0.55`, `block_rate_soft_limit=0.78` 等へ更新

- 次の監視KPI:
  - `strategy_control_entry_disabled` の減少
  - `filled / submit_attempt` 比率の回復
  - `brain_prompt_autotune_latest.json`, `brain_runtime_param_autotune_latest.json` の更新継続
  - `block_rate` と `activity_rate` が `runtime_profile` の目標帯に収束すること

## 2026-03-05 15:22 JST / Brain autoPDCA実行欠損の修正（改善ループ常時化）

- 作業前市況（ローカル実測 + OANDA API）:
  - `USD/JPY bid=157.240 ask=157.248 spread=0.8p`
  - `ATR14(M1)=2.613p`, `ATR14(M5)=6.676p`, `M1 range60=21.9p`
  - `orders_60m_total=1522`, `reject-like=282 (18.53%)`
  - `quant-market-data-feed` の HTTP 応答は直近サンプルで `200 OK` のみ
  - 判定: 通常レンジ内のため改善作業を継続

- 原因:
  - `local_v2_autorecover_once.sh` は `run_brain_autopdca_cycle.sh --interval-sec` を呼ぶが、
    受け側が `--interval-sec` 非対応で即終了し、LLMベンチ→モデル反映→再起動のPDCAが回っていなかった。

- 改善:
  - `run_brain_autopdca_cycle.sh` に以下を追加。
    - `--interval-sec` / `--force`
    - lock/state による多重起動・短周期連打防止
    - `env_changed=true` の時だけ core再起動
    - 市況ガード（spread/reject-rate）で異常時は skip
    - `latest + history(jsonl)` 監査ログ出力
  - `test_run_brain_autopdca_cycle.py` を契約整合（`env_changed`）へ修正し、
    interval skip ケースを追加。

- 期待効果:
  - 「停止寄り」ではなく、ローカルLLMの判断履歴を使ったモデル・プロンプト・タイムアウト改善を定期実行できる。
  - 不要な再起動を抑えつつ、改善が出た時だけ即時反映してトレード品質を上げる。

- 追補（15:33 JST）:
  - 初回実運用で `market_snapshot: null` を検知。原因は market取得出力に警告行が混在し、JSON全文パースが失敗していたこと。
  - `run_brain_autopdca_cycle.sh` を修正し、末尾JSON行を優先抽出する方式に変更。
  - 確認: `logs/brain_autopdca_cycle_latest.json` で `market_snapshot.status=ok` と実測 spread/ATR/reject-rate が記録されることを確認。

## 2026-03-05 17:40 JST / trade_min を `M1Scalper + MicroRangeBreak` に固定（ping5s停止）+ launchd追随（local V2）

- 目的:
  - `scalp_ping_5s_b_live` の「取引数だけ増えてSLで負ける」状態から脱し、trade_min を `M1Scalper + MicroRangeBreak` に寄せて収益性を回復する。

- 作業前市況（ローカル実測 / OANDA API, 2026-03-05 17:17 JST）:
  - `USD/JPY bid=157.281 ask=157.289 spread=0.8p`
  - `ATR14(M1)=3.700p`, `ATR14(M5)=7.193p`, `range60=42.2p`
  - API遅延: `pricing=241ms`, `candles(M1)=253ms`, `candles(M5)=271ms`, `summary=244ms`
  - 判定: spreadは通常、レンジは拡大気味だが流動性悪化（スプレッド拡大/応答劣化）は見られないため作業継続。

- 直近の損益分解（`logs/trades.db`, pocket<>manual, close_time>=now-24h）:
  - NOTE: `close_time` は `2026-03-05T05:13:34+00:00` のようなISO文字列のため、時間窓のSQLは `datetime(substr(close_time,1,19))` で正規化して集計する。
  - `n=618`, `win_rate=0.2104`, `PF=0.407`, `expectancy_jpy=-0.8`, `net_jpy=-489.2`
  - 寄与（pocket×strategy, net_jpy上位の赤字）:
    - `scalp_fast / scalp_ping_5s_b_live`: `n=518`, `net=-103.9`, `win_rate=0.189`
    - `micro / MicroPullbackEMA`: `n=25`, `net=-133.9`, `win_rate=0.28`

- 口座リスク（OANDA, 2026-03-05 17:39 JST）:
  - NAV `49,928.68 JPY`, margin_used `44,768.80`, margin_available `5,188.10`, health_buffer `0.1038`
  - openTrades: `USD_JPY -6998`（2026-03-02 open, clientExtensionsなし）, `USD_JPY -120`（tag=codex_bi_hf）
  - 直近の重大損失: `pocket=manual` の `MARKET_ORDER_MARGIN_CLOSEOUT`（2026-03-02, ticket `412993`, realized `-7696`）

- 実施（ローカルV2導線のみ）:
  - `scripts/local_v2_stack.sh` の `PROFILE_trade_min` を更新（`scalp_ping_5s_b(+exit)` を外し、`quant-m1scalper(+exit)` を追加）。
    - commit: `5fb475eb chore(local_v2): trade_min add m1scalper`
  - 既存の launchd が `--services` 固定（core+microのみ）で動いており、profile更新が反映されず `quant-m1scalper` が起動しない状態だった。
    - `scripts/install_local_v2_launchd.sh --interval-sec 20 --profile trade_min --env ops/env/local-v2-stack.env` を再実行し、launchd を「profile追随（--services無し）」へ戻した。

- 反映確認:
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    - `quant-m1scalper` / `quant-m1scalper-exit` が `running`（ppid=1）で常駐することを確認。

- Pattern Gate（opt-in）確認:
  - `ops/env/local-v2-stack.env`: `ORDER_MANAGER_PATTERN_GATE_ENABLED=1`
  - `ops/env/quant-v2-runtime.env`: `ORDER_PATTERN_GATE_GLOBAL_OPT_IN=0`（全戦略強制はしない）
  - `orders.db` 直近2hで `request_json LIKE '%pattern_gate%'` の行が `1210`（pattern gate payload が request に注入されていることを確認）

- 次に見るKPI（再検証条件）:
  - 直近60m/24hの `M1Scalper-M1` と `MicroRangeBreak` の `PF>1.0`、`expectancy>=0` へ回復すること
  - `health_buffer>=0.10` を維持（下回る場合は「自動戦略追加で押さない」方向へ即時縮小）

## 2026-03-05 18:20 JST / Pattern Gate マッチ率改善（MicroRangeBreak canonical tag）+ entry_thesis backfill診断強化

- 背景:
  - `orders.db` の `MicroRangeBreak-*` 注文は `entry_thesis.strategy_tag` も suffix 付きのままで、
    Pattern book 側（`logs/patterns.db`）の `st:microrangebreak` と一致せず gate が no-op になりうる。
  - `scripts/backfill_entry_thesis_from_orders.py` は `submit_attempt.request_json` 前提だが、
    過去世代は `orders` 行が存在しても `request_json` が残っていないケースがあり、復元不能が混在する。

- 変更:
  - `workers/micro_runtime/worker.py`
    - `MicroRangeBreak`/`MicroVWAPBound` の `entry_thesis.strategy_tag` を base tag へ正規化し、raw は `strategy_tag_raw` へ退避。
  - `scripts/backfill_entry_thesis_from_orders.py`
    - `orders.db` を `ATTACH` して参照し、`submit_attempt -> preflight_start -> other` 優先で `request_json` を拾う方式へ更新。
    - `orders_matched / orders_with_request / recovered_from_orders` を出力して「一致はするが復元ソース無し」を判別可能にした。

- 再検証:
  - Pattern book 更新後（`scripts/pattern_book_worker.py`）、`MicroRangeBreak` の新規注文で `orders.status IN ('pattern_scaled','pattern_block','pattern_scale_below_min')` が出ること。

## 2026-03-05 19:40 JST / M1Scalper: quickshot hard-gate で signal を全drop（local V2）

- 症状:
  - `logs/orders.db` の `max(ts)` が `2026-03-05T05:53:30Z`（`14:53 JST`）以降更新されず、`trade_min` 起動中でも新規注文が出ない。

- 原因（RCA）:
  - `workers/scalp_m1scalper/worker.py` の quickshot 判定が `quickshot_allow=False` の場合、ログ後に無条件 `continue` しており quickshot が「必須ゲート」になっていた。
  - quickshot は `M5 breakout + M1 pullback` を要求するため、レンジ局面では成立しにくく、結果として signal をほぼ全drop していた。

- 対応（local V2）:
  - `M1SCALP_USDJPY_QUICKSHOT_HARD_GATE`（default=1）を追加し、`0` のときは quickshot 不成立でも通常フローで entry を継続（quickshot plan は適用しない）。
  - `ops/env/local-v2-stack.env` で `M1SCALP_USDJPY_QUICKSHOT_HARD_GATE=0`、`M1SCALP_USDJPY_QUICKSHOT_MAX_SPREAD_PIPS=1.20` を設定（retest 要件は維持）。

- 検証手順:
  - `./scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env --services quant-m1scalper,quant-m1scalper-exit`
  - `sqlite3 logs/orders.db 'select max(ts) from orders;'` が `2026-03-05T05:53:30Z` より新しい
  - `scripts/local_v2_stack.sh logs --service quant-order-manager --tail 200` で preflight の通過ログを確認

## 2026-03-05 20:19 JST / 11:19 UTC - no-entry 継続の暫定復旧（M1Scalper 強トレンドflip + MicroRangeBreak hist_block 緩和, local V2）

- 症状:
  - `logs/orders.db` の `max(ts)=2026-03-05T05:53:30Z` 以降更新が止まり、`trade_min` 起動中でも新規注文が出ない。
  - `logs/local_v2_stack/quant-micro-rangebreak.log` に `hist_block ... score=0.266 n=27` が継続。
  - `logs/local_v2_stack/quant-m1scalper.log` で `range_hold_reversion_*` と `trend_block_*` が交互に出て signal が返らず無風になりやすい。

- 作業前市況（ローカル実測 / OANDA API, 2026-03-05 20:18 JST）:
  - `USD/JPY bid=157.240 ask=157.248 spread=0.8p`
  - `ATR14(M1)=2.434p`, `range60(M1)=18.8p`
  - API遅延: `pricing=334ms`, `candles(M1)=244ms`
  - 判定: 通常レンジ、流動性悪化は顕著でないため作業継続。

- 対応（local V2 / main 反映）:
  - `strategies/scalping/m1_scalper.py`
    - `range_reversion_only==True` でも `strong_up/strong_down` のときは `OPEN_LONG/OPEN_SHORT` へflipし、`trend_block_*` で全dropしないようにした。
    - 監視ログ: `range_flip_to_trend_long` / `range_flip_to_trend_short`
  - `ops/env/local-v2-stack.env`
    - `MICRO_MULTI_HIST_SKIP_SCORE=0.20`（`hist_block` の hard skip を緩和）
    - `MICRO_MULTI_HIST_LOT_MIN=0.25`（低スコア時は縮小運転）
    - `M1SCALP_ENTRY_GUARD_BYPASS=1` は暫定維持（BB/projection reject の可視化/復旧用）。`entry_guard_bypass` が常態化する場合は閾値チューニングへ戻す。

- 影響範囲:
  - M1Scalper のシグナル方向が強トレンドで順張りに寄る（range_reversion_only の freeze 回避）。
  - micro runtime の hist skip を緩和し、低品質戦略はロット縮小で継続。

- 検証:
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env --services quant-m1scalper,quant-m1scalper-exit,quant-micro-rangebreak,quant-micro-rangebreak-exit`
  - `sqlite3 logs/orders.db 'select max(ts) from orders;'` が `2026-03-05T05:53:30Z` より新しい
  - `logs/local_v2_stack/quant-m1scalper.log` に `range_flip_to_trend_*` または `entry_guard_bypass` が出て、preflight が流れること
  - `logs/local_v2_stack/quant-micro-rangebreak.log` の `hist_block` 頻度が減り、entry が流れること
  - `orders.db` で `error_code=INSUFFICIENT_MARGIN` / `margin_*` 系の拒否が急増しないこと（増えるならロット縮小へ即応）

## 2026-03-05 21:34 JST / ping5s の dyn alloc sizing 適用 + dyn alloc soft-participation 安全化 + order-manager Brain import fail-open

- 作業前市況（ローカル実測 / OANDA API, 2026-03-05 21:29 JST）:
  - `USD/JPY bid=157.388 ask=157.396 spread=0.8p`
  - `ATR(M1)=2.388p`, `ATR(M5)=5.170p`
  - 判定: 通常レンジ、流動性悪化は顕著でないため作業継続。

- 狙い / 仮説:
  - `trade_all` 等で「未観測 strategy が dyn alloc 未適用のまま full size」になると損益・マージンが悪化しやすい → soft-participation では未観測も `min_lot_multiplier` へ寄せる。
  - ping5s 系（B/C/D/flow）は `config/dynamic_alloc.json` の score/lot_multiplier を sizing に取り込めておらず、悪化戦略の縮小が効かない → dyn alloc multiplier を entry の `units` へ反映する。
  - Brain ゲートは default disabled だが、依存 import 失敗で `quant-order-manager` が起動不能になると no-trade を再発する → import を fail-open にして起動継続。

- 対応（main反映 / local V2）:
  - `execution/order_manager.py`: `workers.common.brain` を optional import に変更し、Brain gate enabled でも module 不在時は warning+metric を出して skip（fail-open）。
  - `workers/common/dynamic_alloc.py`: `allocation_policy.soft_participation=true` のとき、`dynamic_alloc.json` に無い strategy は `min_lot_multiplier` をデフォルト適用（1.0固定を回避）。
  - `workers/scalp_ping_5s/config.py` / `workers/scalp_ping_5s/worker.py`: dyn alloc profile を読み、`lot_multiplier` を entry `units` に反映。`entry_thesis.dynamic_alloc` を付与（found時）。

- 検証手順:
  - `python3 -m compileall execution/order_manager.py workers/common/dynamic_alloc.py workers/scalp_ping_5s/config.py workers/scalp_ping_5s/worker.py`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-order-manager,quant-scalp-ping-5s-b,quant-scalp-ping-5s-b-exit`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-order-manager,quant-scalp-ping-5s-b,quant-scalp-ping-5s-b-exit`
  - `logs/local_v2_stack/quant-scalp-ping-5s-b.log` で `dynamic_alloc` が `entry_thesis` に付与されていること（found時）

## 2026-03-05 22:08 JST / trade_all の worker 大半が停止（stale pid）+ OM timeout 緩和 + strategy_entry dyn alloc trim-only（local V2）

- 症状:
  - `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で `[running] 10 / [stopped] 54`（停止側は `stale_pid_file` 付き）。
  - worker ログに `order_manager` 呼び出しの `Read timed out. (read timeout=8.0)` が散発（例: `logs/local_v2_stack/quant-micro-momentumburst.log`）。

- 作業前市況（ローカル実測 / OANDA pricing, 2026-03-05 21:44 JST）:
  - `USD/JPY bid=157.445 ask=157.453 spread=0.8p`
  - `pricing latency=260ms`
  - `ATR(M1)=2.520p`, `ATR(M5)=5.559p`

- 狙い / 仮説:
  - trade_all で停止 worker が多い状態だと、戦略が走っておらず機会損失 → 起動/復旧導線を標準化して観測できる状態に戻す。
  - `ORDER_MANAGER_SERVICE_TIMEOUT=8.0` だと負荷時に service call timeout が出やすく、skip が増える → local override を `12.0` に上げて false-timeout を減らす。
  - dyn alloc 未対応 worker が raw_units を full size で通すと risk/margin が悪化しやすい → `execution/strategy_entry.py` 側で dyn alloc を「trim-only（縮小のみ）」として適用し、かつ worker 側で `entry_thesis.dynamic_alloc` が付与済みなら二重適用しない。

- 対応（main反映 / commit=`a296316d`）:
  - `execution/strategy_entry.py`
    - `STRATEGY_DYNAMIC_ALLOC_*` を追加し、`entry_thesis.dynamic_alloc` が無い注文に限り `lot_multiplier` で units を trim（デフォルトは up-scale しない）。
  - `workers/common/dynamic_alloc.py`
    - `allocation_policy.soft_participation=true` かつ unknown strategy の場合でも `found=true` の fallback profile を返し、metadata 付与/二重適用回避をしやすくした。
  - `ops/env/local-v2-stack.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=12.0`（runtime 8.0 → local override 12.0）。

- 検証（local V2）:
  - `python3 -m compileall execution/strategy_entry.py workers/common/dynamic_alloc.py`
  - `pytest -q tests/workers/common/test_dynamic_alloc.py tests/execution/test_strategy_entry_dynamic_alloc_trim.py` → `6 passed`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-scalp-ping-5s-b,quant-scalp-ping-5s-b-exit,quant-micro-rangebreak,quant-micro-rangebreak-exit,quant-m1scalper,quant-m1scalper-exit`
  - `sqlite3 logs/orders.db 'select max(ts) from orders;'` が更新継続（例: `2026-03-05T13:06:52Z`）。

- 注記:
  - trade_all の全 worker 常時起動はホスト負荷が大きい可能性がある（load avg が急上昇）。維持できない場合は「走らせたい戦略のみ worker を残す」方向で profile を再設計する。

## 2026-03-05 21:55 JST / strategy_entry dyn alloc trim-only（未対応戦略のfull-size抑制）+ order-manager timeout上書き

- 狙い / 仮説:
  - `trade_all` 等で「dyn alloc 未対応 strategy が full size のまま走る」→ 損益悪化/マージン圧迫で no-entry に見える状況を作りやすい。
  - `execution/strategy_entry.py` で dyn alloc を trim-only（<=1.0）適用して、未対応戦略を自動縮小しつつ「一律停止」には寄せない。
  - 既に strategy 側で `entry_thesis.dynamic_alloc` を付与しているケースは二重適用しない（`dynamic_alloc` があれば skip）。
  - `trade_all` は `quant-order-manager` 呼び出しの read timeout（既定 8s）で worker 側 skip が増えやすい → timeout を上書きして false skip を減らす。

- 対応（main反映 / local V2）:
  - `execution/strategy_entry.py`
    - `STRATEGY_DYNAMIC_ALLOC_*` を追加し、coordinate 前に `lot_multiplier` を trim-only 適用。
    - 適用時は `entry_thesis.dynamic_alloc.source=strategy_entry` を付与し、監査可能にした。
  - `workers/common/dynamic_alloc.py`
    - soft-participation 時、`dynamic_alloc.json` に無い strategy でも `min_lot_multiplier` をデフォルト返却し、trim-only が必ず効くようにした。
  - `ops/env/local-v2-stack.env`
    - `ORDER_MANAGER_SERVICE_TIMEOUT=12.0` を追加（trade_all で timeout skip が増えやすいため）。

- 検証:
  - Unit test:
    - `pytest -q tests/workers/common/test_dynamic_alloc.py tests/execution/test_strategy_entry_dynamic_alloc_trim.py`
  - local V2:
    - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env`
    - `logs/health_snapshot.json` の `git_rev` が `a296316d` で、`orders_last_ts` が更新され続けること。
  - 監査:
    - `orders.db` の `request_json.entry_thesis.dynamic_alloc.source=strategy_entry` が（dyn alloc 未実装戦略で）付与されること。

## 2026-03-05 22:18 JST / ping5s: scalp_fast protection fallback 縮小 + mode blocklist + flow SL有効化（local V2）

- 背景（ローカル実測 / `logs/trades.db` + `logs/orders.db`）:
  - `scalp_ping_5s_b_live` の損失は `Trend + long` に集中（直近3日 `n=444 / -182.1 JPY`）。
  - ping5s B は `STOP_LOSS_ON_FILL_LOSS` の reject 後に protection fallback が走り、SL gap が `8p+` になる filled が存在（直近3日 `28/761`）。
  - `scalp_ping_5s_flow_live` は `entry_thesis.sl_pips=null` かつ `disable_entry_hard_stop=1` のまま取引が入り、平均損失pipsが大きくなりやすい（例: avg win `+0.83p` vs avg loss `-4.82p`）。

- 対応（local override / commit=`45a5fb18`）:
  - `ops/env/local-v2-stack.env`
    - `ORDER_PROTECTION_FALLBACK_PIPS_SCALP_FAST=0.02`（USDJPYで約2p。既定 `0.12` は12p相当でscalp_fastに広すぎ）
    - ping5s B/D/flow の `*_SIGNAL_MODE_BLOCKLIST` を設定（直近負けモードを遮断）
    - `SCALP_PING_5S_FLOW_USE_SL=1`（flowのSL/entry hard stop を復帰）

- 検証（再起動後）:
  - `sqlite3 logs/orders.db 'select count(*) from orders where status=\"rejected\" and error_message=\"STOP_LOSS_ON_FILL_LOSS\" and datetime(ts) >= datetime(\"now\",\"-1 day\") and client_order_id like \"%scalp_ping_5s_b%\";'` が減る
  - `sqlite3 logs/orders.db 'with o as (select executed_price, sl_price from orders where status=\"filled\" and datetime(ts) >= datetime(\"now\",\"-1 day\") and client_order_id like \"%scalp_ping_5s_b%\" and executed_price is not null and sl_price is not null) select sum(case when abs(executed_price - sl_price)/0.01 >= 8.0 then 1 else 0 end), count(*) from o;'` の `>=8p` 比率が下がる
  - flow を走らせた場合: `sqlite3 logs/trades.db 'select json_extract(entry_thesis,\"$.sl_pips\"), json_extract(entry_thesis,\"$.disable_entry_hard_stop\") from trades where strategy_tag like \"scalp_ping_5s_flow_live%\" order by close_time desc limit 5;'` で `sl_pips` が埋まる

## 2026-03-05 22:45 JST / trade_all 起動不全の主要クラッシュ修正（ping5s C/D/flow wrapper・TrendBreakout・micro runtime）

- 作業前市況（ローカル実測 / OANDA pricing, 2026-03-05 22:12 JST）:
  - `USD/JPY bid=157.516 ask=157.524 spread=0.8p`
  - `pricing latency=243ms`
  - `ATR(M1)=2.358p`（直近120本の complete M1）

- 症状（local V2）:
  - `scripts/local_v2_stack.sh up --profile trade_all --env ops/env/local-v2-stack.env` が失敗しやすく、`status` で `stale_pid_file` が多数出て「戦略が走っていない」状態になる。
  - ping5s C/D/flow: `logs/local_v2_stack/quant-scalp-ping-5s-*.log` に `CalledProcessError ... died with SIGTERM`（wrapperが子プロセス死亡で終了）。
  - TrendBreakout: `logs/local_v2_stack/quant-scalp-trend-breakout.log` で `TypeError: _log() got multiple values for argument 'reason'`。
  - micro: `logs/local_v2_stack/quant-micro-momentumburst.log` で `UnboundLocalError: bb_style referenced before assignment`。

- 原因:
  - ping5s C/D/flow wrapper が `workers.scalp_ping_5s.worker` を `subprocess.run(check=True)` で起動しており、`local_v2_stack` 側の cleanup（ping5s-b が汎用 module を pattern に含める）で子プロセスが巻き込み SIGTERM → wrapper が例外で落ちる。
  - `strategies/scalping/m1_scalper.py` の `_log(reason, **kwargs)` に対して `reason=` を kwargs で渡していた（TrendBreakoutが M1Scalper を参照するため worker ごと落ちる）。
  - `workers/micro_runtime/worker.py` の `bb_style` が未初期化のまま `_bb_entry_allowed(...)` に渡され得る。

- 対応（main反映）:
  - ping5s C/D/flow: wrapper を **1プロセス**に変更し、`workers.scalp_ping_5s.worker:scalp_ping_5s_worker()` を `asyncio.run()` で直接実行（commit=`dc7751f2`）。
  - M1Scalper: `_log` の kwargs `reason` を `flip_reason` へ変更（commit=`105b15f2`）。
  - micro runtime: `bb_style` を `reversion` で初期化し、`_TREND_STRATEGIES` を `trend` 判定に追加（commit=`71b9dbd2`）。

- 検証:
  - `pytest -q tests/workers/test_micro_multistrat_trend_flip.py tests/workers/test_m1scalper_nwave_tolerance_override.py tests/workers/test_m1scalper_config.py`
  - `scripts/collect_local_health.sh` で `orders_last_ts` が更新され続けること（例: `filled` が直近1hで継続）

## 2026-03-05 14:34 UTC / 2026-03-05 23:34 JST - `close_reject_no_negative` 抑制の allow negative reason 追補（local V2）

- 事実（ローカル実測: `logs/orders.db` / `logs/trades.db` / `logs/metrics.db`）:
  - 直近3h `orders.db`: `close_reject_no_negative=525`, `close_ok=816`, `filled=961`。
  - 直近24h `close_reject_no_negative` の `exit_reason` は `reentry_reset=323`, `__de_risk__=202` に集中。
  - 直近3h `trades.db` の flow系（`strategy|strategy_tag LIKE '%flow%'`）は `n=14`, `avg_pips=-1.55`, `sum_realized_pl=-55.3`。
  - 直近3h `metrics.db` の `account.margin_usage_ratio` は `avg=0.8297`, `max=1.0003`。

- 仮説:
  - `reentry_reset` / `__de_risk__` が `EXIT_ALLOW_NEGATIVE_REASONS` 未登録のため `close_reject_no_negative` が反復し、負け玉解放遅延を通じて flow の平均pips悪化と margin usage 高止まりを誘発している。

- 実施変更:
  - `ops/env/quant-order-manager.env` に `EXIT_ALLOW_NEGATIVE_REASONS` を明示設定。
  - 既定トークン（`hard_stop,tech_hard_stop,max_adverse,time_stop,no_recovery,max_floating_loss,fast_cut_time,time_cut,tech_return_fail,tech_reversal_combo,tech_candle_reversal,tech_nwave_flip`）を維持したまま、`reentry_reset` と `__de_risk__` を追加。

- 検証観点（反映後 1-3h）:
  1. `orders.db` の `status='close_reject_no_negative'` 件数が減少すること（総数と reason 内訳）。
  2. `trades.db` の flow系 `avg(pl_pips)` が改善すること。
  3. `metrics.db` の `account.margin_usage_ratio` が高止まりせず、`max` が改善方向に向かうこと。

## 2026-03-05 15:30 UTC / 2026-03-06 00:30 JST - flow系: `close_reject_no_negative` 再発防止（strict neg_exit allow + closeログ改善）

- 追加確認（ローカル実測: `logs/orders.db`）:
  - `close_reject_no_negative` の request_json に `exit_reason=__de_risk__/reentry_reset` が含まれている。
  - `strategy_exit_protections.yaml` の `scalp_ping_5s` は `neg_exit.strict_no_negative=true` のため、未許可 reason は worker 側 `allow_negative=true` でも close が拒否され得る。
  - `close_reject_no_negative` を orders.db で棚卸しする際に `pocket/instrument` が欠けており、戦略別の集計がしづらい。

- 対応（main反映）:
  - `config/strategy_exit_protections.yaml`（`scalp_ping_5s`）:
    - `neg_exit.strict_allow_reasons` / `allow_reasons` に `reentry_reset` / `__de_risk__` を追補（commit=`efd2f83e`）。
  - `execution/order_manager.py`:
    - `close_reject_no_negative` の orders.db ログに `pocket/instrument/strategy_tag` を付与し、`est_pips` も記録（commit=`efd2f83e`）。

- 検証観点（反映後 1-3h）:
  1. `orders.db`: `status='close_reject_no_negative'` が `reentry_reset/__de_risk__` で反復しないこと。
  2. `orders.db`: `close_reject_no_negative` 行に `pocket/instrument` が入ること（戦略別に集計できること）。

## 2026-03-05 15:45 UTC / 2026-03-06 00:45 JST - MicroRangeBreak: reversion 全敗の強レンジ絞り込み + ping5s D/flow neg_exit no-block + Brain fast/micro の stall 対策

- 事実（ローカル実測: `logs/trades.db` / `logs/orders.db`）:
  - 直近6h `trades.db`（MicroRangeBreak）: `n=32`, `wins=0`, `avg_pips=-1.2531`, `sum_jpy=-32.9`（全敗）。
  - entry_thesis: `signal_mode=reversion (range_scalp)` に偏り、`range_score=0.356..0.382` と「弱レンジ」でも short リバが走っている。`trend_snapshot(H4).adx≈24.9` でも同様。
  - 直近6h `orders.db`: `close_reject_no_negative=475`（`exit_reason='__de_risk__'` 等が起点になり得る）。

- 市況スナップショット（ローカル実測: `logs/tick_cache.json` / `logs/factor_cache.json`）:
  - `USD/JPY bid=157.676 ask=157.684 spread=0.8p`
  - `ATR(M1)=2.81p` / `ADX(H4)=24.87`

- 仮説:
  - MicroRangeBreak の reversion が「弱レンジ」でも発火し、トレンド寄り局面で `m1_structure_break` / `max_adverse` 由来の早期損切りが連発している。
  - ping5s D/flow は B/C と neg_exit 設定が非対称で、`__de_risk__` / `reentry_reset` が `close_reject_no_negative` になりやすい。
  - Brain は過去に `brain_latency_ms` が平均 ~6s に張り付いた時間帯があり、将来有効化しても fast/micro を stall させない設計が必要。

- 対応（main反映 / commit=`48716111`）:
  - MicroRangeBreak（reversionを“強いレンジ”へ絞る）:
    - `MICRO_RANGEBREAK_MIN_RANGE_SCORE=0.44`（`0.32` → `0.44`）
    - `MICRO_RANGEBREAK_REVERSION_MAX_ADX=23.0`（`27.0` → `23.0`）
    - `MICRO_RANGEBREAK_ENTRY_RATIO=0.25`（`0.38` → `0.25`）
    - ※ `local_v2_stack` は base env → service env の順で上書きされるため、`ops/env/quant-micro-rangebreak.env` も同値へ更新。
  - ping5s D/flow: `config/strategy_exit_protections.yaml` に `neg_exit` を付与し、B/C と同じ `no-block` 方針へ（`strict_no_negative=false`, `allow_reasons=*SCALP_PING_5S_NO_BLOCK_NEG_EXIT_ALLOW_REASONS`）。
  - Brain（有効化時のみ）: `workers/common/brain.py` に pocket別 override を追加（`BRAIN_TIMEOUT_SEC_MICRO` / `BRAIN_TIMEOUT_SEC_SCALP_FAST` / `BRAIN_FAIL_POLICY_MICRO` / `BRAIN_FAIL_POLICY_SCALP_FAST`）。

- 検証観点（反映後 3-6h）:
  1. `trades.db`: MicroRangeBreak の `signal_mode=reversion` が `range_score>=0.44` 帯に寄り、全敗が止まること。
  2. `orders.db`: ping5s D/flow の `close_reject_no_negative` が `__de_risk__/reentry_reset` 起点で減ること。
  3. Brain 有効化時: micro/scalp_fast のエントリーが timeout で stall しないこと（fail-open）。

- 反映後実測（local V2, 2026-03-06 01:25〜01:30 JST）:
  - 市況（OANDA pricing / candles, 01:03 JST）:
    - `USD/JPY mid=157.552 spread=0.80p pricing latency=303ms`
    - `ATR14(M1)=3.76p / 60m range=38.0p candles latency=203ms`
  - 反映:
    - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-micro-rangebreak,quant-micro-rangebreak-exit,quant-order-manager,quant-position-manager`
  - ping5s flow_live の close_reject_no_negative（直近30分 / `orders.db`）:
    - `close_reject_no_negative=0`
    - `close_ok=7`（`exit_reason`: `max_adverse=5`, `__de_risk__=2`）
  - 建玉:
    - `position-manager` の `open_positions` で `scalp_fast` が空（flow_live の含み損玉が解消されている）
    - MicroRangeBreak の open positions は無し
  - 収益（直近30分 / `trades.db`）:
    - 全体: `n=76 / PF=2.449 / net_jpy=+185.187`
    - flow_live: `n=4 / net_jpy=+61.840`
  - メモ:
    - `trades.db` が `orders.db` より遅延するため、必要に応じて `/position/sync_trades` を叩いて追いつかせてから評価する（例: `trades_last_close` が `13:30 UTC` → `16:26 UTC` へ更新）。

## 2026-03-05 18:30 UTC / 2026-03-06 03:30 JST - M1Scalper: buy-dip 停止（戦略内ブロック） + sell-rally は projection flip 時のみ許可

- 市況スナップショット（OANDA pricing / candles, 03:29 JST）:
  - `USD/JPY mid=157.818 spread=0.8p`
  - `ATR14(M1)=2.94p / 60m range=26.5p`
  - `pricing latency=226ms / candles latency=300ms`

- 事実（ローカル実測: `logs/trades.db`）:
  - 直近24h（M1Scalper signal別）:
    - `M1Scalper-buy-dip`: `n=64 / net_jpy=-372.9`（継続的に負け寄与）
    - `M1Scalper-sell-rally`: `n=209 / net_jpy=+633.5`
    - `M1Scalper-trend-long`: `n=73 / net_jpy=+251.9`
  - 直近24h（sell-rally の exec_side 別）:
    - `exec_side=long`: `n=169 / net_jpy=+1097.1 / win_rate=0.941`
    - `exec_side=short`: `n=40 / net_jpy=-463.6 / win_rate=0.125`

- 仮説:
  - `buy-dip` は現行の市場状態で逆期待値になっており、継続稼働は資産減少を誘発する。
  - `sell-rally` は「projection による side flip（signal_side と逆）」のときのみ強い期待値があり、signal_side のまま（short）実行すると大きく負ける。

- 対応（main反映予定）:
  - `workers/scalp_m1scalper/worker.py`:
    - `buy-dip` は戦略内でブロック（`buy_dip_block` ログ）。
    - `sell-rally` は projection 適用後に `side != signal_side`（flip）を満たすときのみ許可し、非flipはブロック（`sell_rally_no_flip_block` ログ）。

- 検証観点（反映後 1-3h）:
  1. `trades.db`: `source_signal_tag='M1Scalper-buy-dip'` が 0 件に収束すること。
  2. `trades.db`: `source_signal_tag='M1Scalper-sell-rally'` の `exec_side='short'` が 0 件に収束すること。
  3. `trades.db`: `M1Scalper-M1` の直近1h `net_jpy` / `PF` が改善方向で安定すること。

## 2026-03-05 20:07 UTC / 2026-03-06 05:07 JST - order-manager: `scalp_ping_5s_flow_live` / `M1Scalper-M1` の stopLossOnFill 欠損（broker SLなし）→ targeted allowlist 修正

- 事実（ローカル実測: OANDA openTrades / `logs/orders.db`）:
  - `scalp_ping_5s_flow_live` / `M1Scalper-M1` の一部エントリーで `takeProfitOnFill` は付くが `stopLossOnFill` が付かず、建玉が broker SLなしで残る（`stopLossOrder=null`）。
  - 一方で `logs/orders.db` には同一 `client_order_id` の `sl_price` が記録されており、SL価格の算出自体は行われていた。

- 原因:
  - `ORDER_FIXED_SL_MODE=0` かつ strategy override 未設定時、`execution/order_manager.py:_allow_stop_loss_on_fill()` の family override が `scalp_ping_5s_b/c/d` のみで、`scalp_ping_5s_flow*` と `M1Scalper-M1` が許可対象外になっていた。

- 対応（local V2, 予防的/後方互換）:
  - `execution/order_manager.py`: `scalp_ping_5s_flow*` を family override に追加し、`ORDER_ALLOW_STOP_LOSS_ON_FILL_SCALP_PING_5S_FLOW` を参照（未設定時の既定は `False`）。
  - `execution/order_manager.py`: `_disable_hard_stop_by_strategy()` が `scalp_ping_5s_flow*` を `ORDER_DISABLE_ENTRY_HARD_STOP_SCALP_PING_5S_FLOW` で解決できるようにし、env既定は後方互換で `True`（＝hard stop 無効）を維持。
  - `ops/env/quant-order-manager.env`: `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_M1SCALPER_M1=1` を追加（strategy 単位で stopLossOnFill を許可）。

## 2026-03-05 21:36 UTC / 2026-03-06 06:36 JST - position-manager: ORDER_FILL close の trades.db 欠損（収益RCA誤差）→ clientTradeID fallback + watermark hole防止 + 24h backfill

- Period:
  - 集計窓: 直近24h（OANDA transactions from/to）
  - 市況確認（OANDA実測）:
    - `USD_JPY bid/ask=157.527/157.535`、`spread=0.8 pips`
    - `ATR14(M5)=7.607 pips`、`range_60m=18.2 pips`
    - API応答: pricing `225ms` / openTrades `220ms` / candles(M5) `228ms`

- Fact:
  - `logs/metrics.db` の `account.balance` が直近24hで大きく減少（約 `-5.48k JPY`）。
  - OANDA transactions 真値では、
    - `ORDER_FILL(pl+financing+commission)` 合計が約 `-5.13k JPY`
    - `DAILY_FINANCING` 合計が約 `-0.35k JPY`
    - 上記合計が `balance delta` と整合。
  - backfill 後の `logs/trades.db`（直近24h）:
    - pocket別: `scalp=-4.78k` / `scalp_fast=-1.64k` / `micro=+0.91k` / `manual=+0.38k`（合計 `-5.13k`）
    - loss寄与TOP: `M1Scalper-M1=-4.78k`, `scalp_ping_5s_flow_live=-1.54k`
  - 一方 `logs/trades.db` は close の `ORDER_FILL` が大量欠損し、同窓で
    - `missing_pairs=1265`
    - `missing_realized_sum=-5670.674 JPY`
    - 欠損 reason: `MARKET_ORDER_TRADE_CLOSE` が大半（`1157`）

- Failure Cause:
  - `execution/position_manager.py:_parse_and_save_trades()` が
    `details = _get_trade_details(trade_id)` 失敗時に `continue` して close を保存しない。
  - 同時に `_last_tx_id` を `max(processed_tx_ids)` へ進めるため、
    一部トランザクションが失敗すると水位が穴を飛び越え、永続欠損になる。

- Improvement:
  - `execution/position_manager.py`:
    - `_parse_and_save_trades()`:
      - `closed_trade.clientTradeID` を使い `orders.db (client_order_id)` から entry meta を復元する fallback を追加。
      - details が取れない場合でも tx/closed_trade から最小限の details を組み立て、**close を必ず保存**。
      - `_last_tx_id` は連続区間でのみ進め、hole を飛び越えないようにした（idempotent 再処理は許容）。
  - `scripts/backfill_trades_from_oanda_idrange.py`:
    - OANDA transactions `idrange` を指定レンジで取得し、`PositionManager._parse_and_save_trades()` へ流し込む backfill/repair ツールを追加（`uniq(transaction_id, ticket_id)` により冪等）。

- Verification:
  - 直近24h window を再処理し、OANDA `ORDER_FILL close` の `(transaction_id, tradeID)` が `logs/trades.db` に欠損なく保存される（`missing_pairs=0`）こと。
  - `logs/trades.db` の 24h `sum(realized_pl)` が OANDA `ORDER_FILL` 真値（約 `-5.13k JPY`）と整合すること。

- Status:
  - done（`quant-position-manager` restart + `python scripts/backfill_trades_from_oanda_idrange.py --last-n 5000` 実行、`missing_pairs=0` を確認）
  - 追記: windowズレで `--last-n 5000` が不足し得るため、`python scripts/backfill_trades_from_oanda_idrange.py --from-id 418699` を実行し、`[418699, 439517]` の `missing_pairs=0` を再確認。

## 2026-03-05 21:54 UTC / 2026-03-06 06:54 JST - order-manager: micro pocket の stopLossOnFill 欠損（TPのみ/SLなし）→ strategy allowlist 追補

- Fact（ローカル実測: OANDA openTrades / `logs/orders.db`）:
  - `USD_JPY` の openTrades `7/7` が `takeProfit` は付くが `stopLoss` が `null`（broker SLなし）で残存。全て `pocket=micro`。
  - `logs/orders.db` には同一 `client_order_id` の `sl_price` が記録されており、SL価格の算出自体は行われていた。

- Failure Cause:
  - `ORDER_FIXED_SL_MODE=0` 既定では stopLossOnFill が無効（`stop_loss_policy` の baseline）。
  - `execution/order_manager.py:_allow_stop_loss_on_fill()` の strategy override が micro の主要 strategy tag（`MomentumBurst*` / `MicroLevelReactor*` / `MicroTrendRetest*` / `MicroRangeBreak*` / `MicroVWAPRevert*` / `MicroVWAPBound*`）に未設定だった。

- Improvement（local V2）:
  - `ops/env/quant-order-manager.env`:
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MOMENTUMBURST=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROLEVELREACTOR=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROTRENDRETEST=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICRORANGEBREAK=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROVWAPREVERT=1`
    - `ORDER_ALLOW_STOP_LOSS_ON_FILL_STRATEGY_MICROVWAPBOUND=1`
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-order-manager`

- Verification（反映後 次の micro entry で確認）:
  1. `logs/orders.db` の `submit_attempt.request_json` に `stopLossOnFill` が含まれること。
  2. OANDA openTrades の `stopLossOrder` が `null` ではなくなること。
  3. reject率が悪化しないこと（`orders.db status='rejected'` / `STOP_LOSS_ON_FILL_LOSS` など）。

- Status:
  - in_progress（order-manager restart 済み。次の micro entry で broker SL 付与を実測確認）

## 2026-03-06 04:30 UTC / 2026-03-06 13:30 JST - position-manager: sync_trades が backlog>MAX_FETCH で newest 側にジャンプし欠損を作る → forward paging + /summary lastTransactionID

- Fact:
  - `execution/position_manager.py:_fetch_closed_trades()` は backlog が大きいとき
    `fetch_from=min_allowed=max(1,last_tx_id-_MAX_FETCH+1)` へジャンプし、
    `self._last_tx_id+1..min_allowed-1` の範囲を取得しない（＝決済 tx を永続欠損させ得る）。

- Improvement（local V2）:
  - `execution/position_manager.py:_fetch_closed_trades()`:
    - `fetch_from=self._last_tx_id+1` を維持し、`fetch_to=min(last_tx_id, fetch_from+_MAX_FETCH-1)` までの **前進型ページング** に変更（hole を飛び越えない）。
    - `lastTransactionID` の取得を `/v3/accounts/{ACCOUNT}/summary` に変更し、巨大な `transactions` payload を避ける。

- Verification（ローカル実測: OANDA API）:
  - `POSITION_MANAGER_MAX_FETCH=50`、`pm._last_tx_id=remote_last-200` で `_fetch_closed_trades()` を呼ぶと、
    `fetch_from=remote_last-199` から `fetch_to=remote_last-150` を取得する（newest 側 `remote_last-49` へジャンプしない）。

## 2026-03-06 04:35 UTC / 2026-03-06 13:35 JST - scalp_ping_5s_flow: 直近24hで大幅マイナスのため緊急リスク縮小（破滅サイズ停止）

- Fact（ローカル実測: `logs/trades.db`、backfill反映後）:
  - 直近24h（`2026-03-05T04:28Z..2026-03-06T04:28Z`）の `scalp_ping_5s_flow_live`:
    - `net=-6427.87 JPY`, `trades=379`, `win=0.325`

- Decision:
  - 時間帯ブロックではなく、まず **破滅サイズ（過剰ユニット/過剰同時建玉）** を止めるための緊急縮小を入れる。
  - 目的: 連続損失時の口座急落を抑えつつ、引き続き原因分析と改善を進める。

- Change（local V2 / worker-local env）:
  - `ops/env/scalp_ping_5s_flow.env`（strategy-local）:
    - `SCALP_PING_5S_FLOW_MAX_ACTIVE_TRADES: 16 -> 2`
    - `SCALP_PING_5S_FLOW_MAX_PER_DIRECTION: 8 -> 1`
    - `SCALP_PING_5S_FLOW_BASE_ENTRY_UNITS: 1200 -> 120`
    - `SCALP_PING_5S_FLOW_MAX_UNITS: 3500 -> 600`
    - `SCALP_PING_5S_FLOW_MAX_SPREAD_PIPS: 1.3 -> 0.9`
  - `ops/env/quant-scalp-ping-5s-flow.env`（service overlay）:
    - `SCALP_PING_5S_FLOW_MAX_ACTIVE_TRADES: 16 -> 2`
    - `SCALP_PING_5S_FLOW_MAX_PER_DIRECTION: 8 -> 1`

## 2026-03-06 06:30 UTC / 2026-03-06 15:30 JST - flow short-probe rescue が dynamic trim を打ち消していたため修正（local V2）

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY bid=157.784 ask=157.792 spread=0.8p`
  - `tick_cache 直近15m`: `spread mean=0.8p / max=1.0p / mid range=12.8p`
  - `factor_cache`: `ATR(M1)=3.44p / ATR(M5)=6.20p / ATR(H1)=18.36p`
  - OANDA API応答: `pricing=246.6ms / account_summary=319.7ms / openTrades=233.6ms`
  - 判定: メンテ時間帯（JST 7-8時）外で、流動性異常ではなく作業継続可。

- 事実（ローカル実測: `logs/trades.db` / `logs/orders.db` / `logs/local_v2_stack/quant-scalp-ping-5s-flow.log`）:
  - 直近24h `trades.db`: 全体 `net=-10892.5 JPY / PF=0.549`。
  - 同24h 赤字寄与上位:
    - `scalp_ping_5s_flow_live`: `n=420 / net=-7131.1 JPY / PF=0.295 / avg_units=1842.3`
    - `M1Scalper-M1`: `n=2001 / net=-5725.5 JPY / PF=0.552`
  - 直近3hでも `scalp_ping_5s_flow_live` は `n=273 / net=-5593.4 JPY / PF=0.193 / avg_units=2120.2` と継続悪化。
  - `entry_thesis.dynamic_alloc.lot_multiplier=0.25` が記録されている一方、flow worker log に
    `min_units_rescue applied mode=short_probe_rescued units=2000 ... risk_cap=1200`
    が連発し、縮小後サイズが `MIN_UNITS=2000` rescue で押し戻されていた。
  - `ops/env/scalp_ping_5s_flow.env` の緊急縮小値（`120 / 2 / 1`）も、
    `ops/env/local-v2-stack.env` の `700 / 6 / 3` override で上書きされていた。

- 仮説:
  - flow の主因は市況ではなく、`short_probe_rescue` が non-B/C clone でも既定有効で、
    dynamic alloc と risk mult による縮小を戦略側で打ち消していたこと。
  - 併せて local-v2 override の食い違いが、flow の破滅サイズ継続を許していた。

- 対応:
  - `workers/scalp_ping_5s/config.py`
    - `SHORT_PROBE_RESCUE_ENABLED` の既定を `B/C clone のみ true` へ変更。
  - `workers/scalp_ping_5s/worker.py`
    - `_maybe_rescue_short_probe()` が `units_risk < MIN_UNITS` の場合に rescue しないよう修正。
  - `ops/env/local-v2-stack.env`
    - flow override を緊急縮小値へ統一:
      - `BASE_ENTRY_UNITS=120`
      - `MAX_ACTIVE_TRADES=2`
      - `MAX_PER_DIRECTION=1`
      - `MAX_SPREAD_PIPS=0.90`
      - `MIN_UNITS_RESCUE_ENABLED=0`
      - `SHORT_PROBE_RESCUE_ENABLED=0`
    - `M1SCALP_DYN_ALLOC_MULT_MIN=0.45` として、最新 `dynamic_alloc.json` の縮小値を worker 側で潰さないよう修正。
  - `config/dynamic_alloc.json`
    - `scripts/dynamic_alloc_worker.py --limit 0 --lookback-days 7 --min-trades 12 --pf-cap 2.0 --target-use 0.88 --half-life-hours 36 --min-lot-multiplier 0.45 --max-lot-multiplier 1.65 --soft-participation 1`
      を再実行し、`as_of=2026-03-06T06:26:55Z` へ更新。
    - 主な更新:
      - `scalp_ping_5s_flow_live lot_multiplier=0.45`
      - `M1Scalper-M1 lot_multiplier=0.50`

- 再検証条件（反映後 30-90分）:
  1. `quant-scalp-ping-5s-flow.log` に `short_probe_rescued units=2000` が再発しないこと。
  2. `orders.db/trades.db` の flow 約定ユニットが `<= 600` 帯へ収まること。
  3. `trades.db` の直近1-3h で `scalp_ping_5s_flow_live` の PF が `0.193` から改善すること。
  4. `M1Scalper-M1` の実トレードサイズが更新後 `dynamic_alloc` に従って縮小すること。

## 2026-03-06 08:55 UTC / 2026-03-06 17:55 JST - position-manager sync_trades が timeout/busy 競合で遅延し trades.db 反映が詰まる件を緩和（local V2）

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY bid=157.746 ask=157.754 spread=0.8p`
  - `tick_cache 直近300`: `spread mean=0.8p / max=0.8p / mid range=4.1p`
  - `factor_cache`: `ATR(M1)=3.13p / ATR(M5)=6.95p / ATR(H1)=18.36p`
  - OANDA account summary: `openTradeCount=21`
  - 判定: メンテ時間帯外、spread/ATR は異常ではなく、作業継続可。

- 事実（ローカル実測: `logs/trades.db` / `logs/orders.db` / `logs/health_snapshot.json` / `logs/local_v2_stack/quant-position-manager.log`）:
  - `health_snapshot` では `orders_last_ts=2026-03-06T06:47:57Z` に対し `trades_last_close=2026-03-06T06:11:02Z` で、監視上は trades 側が止まって見えていた。
  - `quant-position-manager.log` には
    - `2026-03-06 17:53:15 JST [POSITION_MANAGER_WORKER] request failed: position manager busy`
    - `2026-03-06 17:53:21 JST [POSITION_MANAGER_WORKER] request failed: sync_trades timeout (8.0s)`
    が出ていた。
  - その直後、`trades.db` には `updated_at >= 2026-03-06T08:53:00Z` で `304` 件が一括反映され、
    `transaction_id` は `443041 -> 446546` まで前進した。
  - つまり `trades.db` は永久停止ではなく、`sync_trades` が timeout 後も裏で完走し、反映が後ろ倒しになっていた。

- 仮説:
  - `workers/position_manager/worker.py` では `sync_trades` / `performance_summary` / `fetch_recent_trades` が同じ `position_manager_db_call_lock` を共有しており、read系呼び出しや再試行と競合すると `position manager busy` が返る。
  - さらに `sync_trades` は `asyncio.wait_for(..., 8.0s)` で包まれているため、backlog catch-up（今回 304件規模）が 8秒を超えると表では timeout 扱いになり、監視上は未反映に見える。

- 対応:
  - `workers/position_manager/worker.py`
    - `sync_trades` 専用の `position_manager_sync_trades_call_lock` を追加し、read系 API と lock を分離。
  - `ops/env/local-v2-stack.env`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_TIMEOUT_SEC=20.0`
    - `POSITION_MANAGER_WORKER_SYNC_TRADES_MAX_FETCH=200`
    - backlog catch-up 時の false timeout を減らし、1回の同期処理量も抑える。

- 再検証条件:
  1. `quant-position-manager.log` で `sync_trades timeout` / `position manager busy` の再発頻度が下がること。
  2. `health_snapshot` の `orders_last_ts` と `trades_last_close` の差が縮むこと。
  3. `trades.db` の `updated_at` が数分単位で追随し、まとめ書きの塊が減ること。

## 2026-03-06 09:10 UTC / 2026-03-06 18:10 JST - quant-position-manager を常時 background sync 化し、監視 blind を構造的に解消

- 事実:
  - 手動 `pm_sync_trades` では backlog が大きいと API 側は timeout を返すが、裏では `Saved 304 new trades.` が完走した。
  - つまり監視 blind の主因は「保存不能」ではなく、「position-manager 自身が backlog を平常時から削り続けていない」ことだった。

- 対応:
  - `workers/position_manager/worker.py`
    - worker lifespan で background `sync_trades` loop を追加。
    - 既定値は `start_delay=1s / interval=5s / max_fetch=120`。
    - 成功時は worker cache を更新して、後続 `/position/sync_trades` の stale 返却も改善。
  - `ops/env/local-v2-stack.env`
    - background sync 系 env を明示して local V2 で固定化。

- 期待効果:
  - `orders.db` と `trades.db` の差分を小さい backlog に保ち、PF/net の監視が「タイムアウト後にまとめて追いつく」状態から、「ほぼ追随」へ寄る。

## 2026-03-06 09:20 UTC / 2026-03-06 18:20 JST - M1Scalper が `MAX_OPEN_TRADES=1` を無視して積み上がる実装漏れを修正

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY bid=157.664 ask=157.672 spread=0.8p`
  - `ATR(M1)=2.05p / ATR(M5)=5.34p / ATR(H1)=18.24p`
  - `openTradeCount=15` で、その大半が `M1Scalper-M1` ロングに偏っていた。

- 事実:
  - OANDA open trades では `M1Scalper-M1` ロングが数秒間隔で `~390-480 units` ずつ積み上がっていた。
  - `logs/local_v2_stack/quant-m1scalper.log` でも同 tag の連続 `OPEN_FILLED` が確認でき、`M1SCALP_MAX_OPEN_TRADES=1` が機能していなかった。
  - 一部は `position_manager service call failed` / `order_manager service call failed` の直後でも継続発注されており、fail-open が過剰エクスポージャを拡大していた。

- 対応:
  - `M1Scalper` worker に `PositionManager` ベースの open-trades guard を追加。
  - `strategy_tag` 単位で open trade 数が `MAX_OPEN_TRADES` 以上なら新規 entry を拒否。
  - `position_manager` 不達時は `M1SCALP_FAIL_CLOSED_ON_POSITIONS_ERROR=1` で fail-closed。

- 期待効果:
  - M1 の「同方向ナンピン的な積み上がり」を止め、利益より先に損失側の tail risk を削る。

## 2026-03-06 09:35 UTC / 2026-03-06 18:35 JST - live の積み増し主因は `TrendBreakout` 派生 worker だったため、M1 family 派生 worker へ同じ fail-closed guard を展開

- 市況確認（ローカルV2実測）:
  - `USD/JPY close=157.730`
  - `ATR(M1)=2.15p / ATR(M5)=5.44p / ATR(H1)=18.24p`
  - `quant-order-manager / quant-position-manager` health は `200`、応答は `9-14ms`
  - `position/open_positions` は `stale=true age_sec~4s` を返す瞬間があり、worker 側が fail-open だと積み増しを止められない条件だった

- 事実:
  - `quant-m1scalper.log` 側では `open_trades_block` が出ていた一方、`quant-scalp-trend-breakout.log` では 2026-03-06 18:06 JST 台に
    `TrendBreakout` が `source_signal_tag=M1Scalper-breakout-retest-long` を受けて `447144 / 447167 / 447174` を連続送信していた。
  - `position/open_positions` でも当該 open trades は `strategy_tag=TrendBreakout`、`entry_thesis.source_signal_tag=M1Scalper-breakout-retest-long` で確認できた。
  - `workers/scalp_trend_breakout/config.py` と `workers/scalp_pullback_continuation/config.py` は `MAX_OPEN_TRADES` を持っていたが、worker 実装側では評価していなかった。

- 対応:
  - `TrendBreakout` / `pullback_continuation` worker に `PositionManager` ベースの `_passes_open_trades_guard()` を追加。
  - `M1Scalper` / `TrendBreakout` / `pullback_continuation` すべてで `entry_thesis.env_prefix=M1SCALP` を同一 family とみなし、別 `strategy_tag` でも同方向積み上がりを block。
  - `M1SCALP_FAIL_CLOSED_ON_POSITIONS_ERROR` を両 config でも読むようにし、`position_manager` 不達時は fail-open せず reject。
  - local override の `M1SCALP_ENTRY_GUARD_BYPASS=1` を `0` に戻し、`bb_entry_reject` を無視して送る経路を止めた。
  - それぞれに targeted test を追加して、limit 到達時 block / family alias block / position-manager error 時 fail-closed を固定化。

- 期待効果:
  - `M1Scalper` 本体だけでなく、同一シグナル系列の派生 worker が別 strategy tag で同方向に積み上がる経路を止める。
  - `position_manager` が stale/busy の瞬間でも、M1 family は「見えないから建てる」ではなく「見えないから建てない」に寄る。
  - `bb_entry_reject` long の bypass を止め、直近3hで続いていた M1 long の赤字送信をさらに削る。

## 2026-03-06 11:10 UTC / 2026-03-06 20:10 JST - 収益悪化の主因は `flow` / `M1` の赤字単価と OANDA 応答劣化だったため、サイズ縮退と timeout 緩和を優先

- 市況確認（ローカルV2実測 + OANDA）:
  - `USD/JPY mid=157.892 spread=0.8p`
  - `ATR(M1)=2.04p / ATR(M5)=5.39p / ATR(H1)=18.05p`
  - `tail300 range=3.6p / tail1000 range=4.5p`
  - `orderbook latency ~=166ms`
  - `health_snapshot`: `data_lag_ms=717ms`, `decision_latency_ms=12.4ms`

- 異常条件:
  - `quant-order-manager.log` で `ORDER_OANDA_REQUEST_TIMEOUT_SEC=8.0` の read timeout と `503 Service unavailable` を確認。
  - `quant-market-data-feed.log` で stream reconnect が断続し、`[Errno 28] No space left on device` により `tick_cache/orderbook/factor_cache` の persist 失敗が発生。
  - `df -h .` は `/System/Volumes/Data` 空き `115-116MiB`、容量 `100%`。ローカル runtime 自体が収益評価を歪める水準だった。

- 収益分解:
  - `scripts/pdca_profitability_report.py --instrument USD_JPY`
    - 24h: `net_jpy=-11176.8 / net_pips=-1754.9 / PF=0.65`
  - 主損失:
    - `scalp_ping_5s_flow_live: -6921.7 JPY`
    - `M1Scalper-M1: -6171.2 JPY`
  - 主利益:
    - `MomentumBurst: +1856.9 JPY / PF=5.99`
  - `M1Scalper-M1` の損失は `MARKET_ORDER_TRADE_CLOSE` 主体、`flow` は `STOP_LOSS_ORDER` と `MARKET_ORDER_TRADE_CLOSE` の複合赤字だった。

- 対応方針:
  - 戦略停止ではなく、`dynamic_alloc` 側で `MARKET_ORDER_TRADE_CLOSE` の負け寄与を新しい縮退シグナルとして扱い、重赤字戦略の `lot_multiplier` を `0.45` 未満へ落とせるようにする。
  - `local-v2-stack.env` では `M1` の dynamic alloc floor を `0.25`、`flow` clone を `0.18` へ下げ、同じ負け方を繰り返す戦略の赤字単価を先に落とす。
  - OANDA read timeout は `10s`、order-manager service timeout は `14s` に緩和し、8秒超の submit/close 失敗を減らす。
  - disk 逼迫で patch すら失敗したため、`logs/replay`, `logs/archive`, `logs/local_vm_parity`, `logs/reports/forecast_improvement`, `logs/oanda` の古い生成物を整理して空きを `310MiB` まで回復してから修正に入る。

## 2026-03-06 11:30 UTC / 2026-03-06 20:30 JST - OANDA `/summary` 503 で entry が全面停止していたため、`order_manager` のみ 15 秒以内 stale snapshot を許可する bounded fallback を追加

- 市況確認（ローカルV2実測）:
  - `USD/JPY pricing/stream` は `200 OK` を維持。
  - 一方で `/summary` と `/openTrades` は `503 Service unavailable` が連発。
  - `health_snapshot`: `data_lag_ms=292-1129ms`, `decision_latency_ms=19-110ms`
  - `orders.db` では `2026-03-06 20:16:55 JST` が直近 `filled` で、その後は `preflight_start -> margin_snapshot_failed` が続いた。

- 事実:
  - `execution/order_manager.py` は `market_order` / `limit_order` / `_preflight_units` の3経路で `/summary` 失敗を即 fail-closed していた。
  - `utils/oanda_account.py` には共有 cache が既にあったが、`order_manager` 側は stale age/source を見られず、`openPositions` 要約も request failure 時に `(0,0)` へ落ちて free margin を過大評価しうる余地があった。

- 対応:
  - `utils/oanda_account.py` に snapshot の `source / age_sec / stale / error_kind` を返す state helper を追加。
  - `order_manager` では `market_order` / `limit_order` / `_preflight_units` の3箇所だけ、`503/timeout/connection_error` かつ `15s` 以内・`free_margin_ratio>=0.30`・`health_buffer>=0.25` の stale snapshot を許可する bounded fallback を実装。
  - `get_position_summary()` は request failure 時でも usable な stale cache を優先再利用し、`margin_used>0` なのに `(0,0)` へ落ちた時に side free margin ratio を `1.0` へ誤って押し上げないよう修正。
  - `ops/env/local-v2-stack.env` に `ORDER_MARGIN_STALE_ALLOW_SEC=15`、`ORDER_MARGIN_STALE_MIN_FREE_RATIO=0.30`、`ORDER_MARGIN_STALE_MIN_HEALTH_BUFFER=0.25` を明示。

- 期待効果:
  - 数秒級の OANDA `/summary` flap では entry を無駄に止めず、長めの outage では従来どおり fail-closed を維持する。
  - stale fallback は `order_manager` の margin/preflight 導線に限定し、全体を fail-open にしない。

## 2026-03-06 13:39 UTC / 2026-03-06 22:39 JST - 通信回復後の `scalp_fast` reject は `STOP_LOSS_ON_FILL_LOSS` に偏っていたため、protection fallback gap を 2p→3p へ戻し過ぎない範囲で拡大

- 市況確認（ローカルV2実測）:
  - `check_oanda_summary.py` は `200` 復帰。`openTrades=3`、`margin used=3169JPY / avail=34803JPY`。
  - `orders.db` では `2026-03-06 22:37 JST` 台に `filled` が再開。
  - `USD/JPY mid=157.51 spread=0.8p`

- 事実:
  - `2026-03-06T13:30:00Z` 以降の `orders.db` は `filled=33`, `rejected=7`, `entry_probability_reject=24`。
  - `rejected` 7件はすべて `scalp_fast` 系で、`quant-order-manager.log` でも `STOP_LOSS_ON_FILL_LOSS` と `protection fallback applied ... gap=0.0200` が並んでいた。
  - 既存の `ORDER_PROTECTION_FALLBACK_PIPS_SCALP_FAST=0.02` は、3/5 の再調整では有効だったが、3/6 22:30 JST 台の回復局面では再び tight 側に寄っていた。

- 対応:
  - `ops/env/local-v2-stack.env` の `ORDER_PROTECTION_FALLBACK_PIPS_SCALP_FAST` を `0.02 -> 0.03` へ変更。
  - 12p 既定値へ戻すのではなく、scalp_fast の fallback だけを 3p に限定して reject 低減を狙う。

- 期待効果:
  - `STOP_LOSS_ON_FILL_LOSS` を減らし、回復直後の `submit_attempt -> rejected` を `filled` 側へ寄せる。
  - fallback SL を広げ過ぎず、scalp_fast の損失尾を増やさない範囲で執行成立率を改善する。

## 2026-03-06 13:58 UTC / 2026-03-06 22:58 JST - OANDA は回復維持だが `scalp_ping_5s_d_live` の期待値が明確に負で、直近クローズのほぼ全件が `STOP_LOSS_ORDER` だったため、D variant の entry 条件を局所的に強化

- 市況確認（ローカルV2実測 + OANDA）:
  - `check_oanda_summary.py` は `200` を維持。`openTrades=4`
  - `USD/JPY mid=157.865 spread=0.8p`
  - `health_snapshot`: `data_lag_ms≈793`, `decision_latency_ms≈47`
  - core services と `quant-scalp-ping-5s-b` / `-exit` は稼働中

- 事実:
  - `pdca_profitability_report.py --instrument USD_JPY`:
    - 24h `scalp_ping_5s_d_live`: `30 trades / win 0.0% / PF 0.00 / -196.6 JPY`
    - 7d `scalp_ping_5s_d_live`: `42 trades / win 4.8% / PF 0.08 / -280.7 JPY`
  - `trades.db` の直近 24h は `STOP_LOSS_ORDER=39 trades / -321.9 JPY / -64.1 pips`、`MARKET_ORDER_TRADE_CLOSE=3 trades / +41.2 JPY`。負けの中心は exit ではなく entry quality。
  - `analyze_entry_precision.py --limit 220` では `scalp_ping_5s_d_live` が `slip_mean=0.314p / slip_p95=1.600p` と、同時間帯の `B` より明確に悪い。
  - `orders.db` では `D` の fills は `407-686 units` 帯でも連続し、その多くが数十秒以内に `STOP_LOSS_ORDER` で閉じていた。

- 対応:
  - `ops/env/local-v2-stack.env`
    - `SCALP_PING_5S_D_ENTRY_LEADING_PROFILE_REJECT_BELOW=0.72` (`0.66` から引き上げ)
    - `SCALP_PING_5S_D_ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.80` (`0.74` から引き上げ)
    - `SCALP_PING_5S_D_BASE_ENTRY_UNITS=3000` (`4200` から縮小)
    - `SCALP_PING_5S_D_MAX_SPREAD_PIPS=0.90` (`1.20` から圧縮)

- 期待効果:
  - `D` の弱いシグナルと広めスプレッド帯だけを削り、`scalp_fast` 全体は止めずに赤字単価を圧縮する。
  - `STOP_LOSS_ON_FILL_LOSS` fallback に依存する前段の low-edge entry を減らし、fills 後の即 SL を抑える。

## 2026-03-06 14:05 UTC / 2026-03-06 23:05 JST - `M1Scalper-M1` は negative expectancy のまま取引回数が圧倒的に多いため、shared path を触らず `base units` だけを 40% 縮小

- 市況確認（ローカルV2実測 + OANDA）:
  - `check_oanda_summary.py` は `200` 維持、`openTrades=4`
  - `USD/JPY spread=0.8p`
  - `local_v2_stack` の core services と `quant-m1scalper` / `-exit` は稼働中

- 事実:
  - `pdca_profitability_report.py --instrument USD_JPY`
    - 24h `M1Scalper-M1`: `1775 trades / win 56.5% / PF 0.55 / -5750.4 JPY`
    - 7d `M1Scalper-M1`: `2290 trades / win 58.5% / PF 0.59 / -6172.5 JPY`
  - `trades.db` 直近 24h の `M1Scalper-M1` は `avg abs(units)=352.8`、負けの内訳は `MARKET_ORDER_TRADE_CLOSE=-3611.1 JPY`、`STOP_LOSS_ORDER=-2283.8 JPY`。
  - すでに dynamic alloc と open-trades guard は効いているため、ここで一番境界の小さいレバーは `base units` のみ。

- 対応:
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_BASE_UNITS=1800` (`3000` から縮小)

- 期待効果:
  - `M1` の挙動や exit 判断を変えず、負けトレードの赤字単価だけを先に 35-40% 程度圧縮する。
  - shared protection や order_manager を再度触らず、strategy ローカルのサイズだけで loss drag を落とす。

## 2026-03-06 14:22 UTC / 2026-03-06 23:31 JST - 24h赤字主因を再点検し、`main` 最新の local-v2 調整が active `trade_min` に未反映だったため反映確認を優先

- 市況確認（ローカルV2実測 + OANDA API）:
  - `pricing`: `bid=158.010 ask=158.018 spread=0.8p status=200 latency=310ms`
  - `summary`: `status=200 latency=244ms openTradeCount=4`
  - `openTrades`: `status=200 latency=281ms`
  - OANDA `M1` 直近120本: `ATR14=4.74p / 15m range=18.7p / 60m range=64.5p`
  - 判定: spread は通常帯、ATR/range はやや高めだが異常域ではなく、作業継続可。

- 24h収益分解（ローカル `logs/trades.db` / `logs/orders.db` / `logs/metrics.db`）:
  - 全体: `3368 trades / win_rate=50.0% / PF=0.552 / expectancy=-3.4 JPY / net=-11595.8 JPY`
  - 赤字寄与上位:
    - `scalp_ping_5s_flow_live: 420 trades / -7131.1 JPY`
    - `M1Scalper-M1: 2290 trades / -6172.5 JPY`
    - `scalp_ping_5s_d_live: 42 trades / -280.7 JPY`
  - reject: `STOP_LOSS_ON_FILL_LOSS=25`, `INSUFFICIENT_MARGIN=6`, `TRADE_DOESNT_EXIST=3`
  - 執行品質: `spread_mean=0.807p / slip_mean=0.008p / latency_submit_p50=190ms / latency_preflight_p50=227ms`
  - つまり主因は「執行遅延」より「負け戦略の件数・赤字単価」で、特に `flow` と `M1` が支配的。

- 直近反映確認で分かったこと:
  - `local-v2-stack.env` は `2026-03-06 23:27:38 JST` に、`ee476feb tune: tighten local-v2 m1 and flow profitability` 相当の値へ更新済みだった。
  - しかし `quant-m1scalper.log` の `2026-03-06 23:20:34 JST` 起動行は `tag_filter=-` で、`M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest-long,nwave-long` が live プロセスへ入っていなかった。
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env` を実行し、`2026-03-06 23:29-23:31 JST` に active services を再起動。
  - 再起動後の確認:
    - `quant-m1scalper.log`: `worker start ... tag_filter=breakout-retest-long,nwave-long`
    - 直後に `tag_filter_block tag=M1Scalper-sell-rally`
    - `quant-scalp-ping-5s-b.log`: `side_filter=sell`
    - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`: core + `B / micro-rangebreak / M1` が running

- 判断:
  - 反映直後は `trade_min` の post-restart sample がまだ薄く、ここで追加の speculative tune を重ねるより、`main` 最新の tighten を live へ載せて効果を見る方が筋が良い。
  - 現在の `trade_min` active services は `B / micro-rangebreak / M1` で、`flow / D` は反映後 profile では動いていない。次の追加調整は post-restart 実績を見てからにする。

- 再検証条件:
  1. `quant-m1scalper.log` に `tag_filter_block tag=M1Scalper-sell-rally` が継続し、`tag_filter=-` が再発しないこと。
  2. `orders.db` の post-restart で `STOP_LOSS_ON_FILL_LOSS` reject が再拡大しないこと。
  3. 30-60分後の `trades.db` で `M1Scalper-M1` と `scalp_ping_5s_b_live` の追加 net が、再起動前の時間帯より悪化しないこと。

## 2026-03-06 14:33 UTC / 2026-03-06 23:36 JST - open trade が決済されなかった直接原因は「exit owner 不在」

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY mid=157.915 spread=0.8p`
  - `pricing/openTrades/summary` は取得継続可
  - `openTrades=4` で、内訳は `scalp_ping_5s_d_live` 1本と `MicroVWAPRevert-long-trendflip` 3本

- 事実:
  - `logs/orders.db` 上、4本とも open 後に `close_request` が一度も発行されていなかった。
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-scalp-ping-5s-d,quant-scalp-ping-5s-d-exit,quant-micro-vwaprevert,quant-micro-vwaprevert-exit`
    - `quant-scalp-ping-5s-d`: `running`
    - `quant-scalp-ping-5s-d-exit`: `stopped`
    - `quant-micro-vwaprevert`: `stopped`
    - `quant-micro-vwaprevert-exit`: `stopped`
  - `workers/micro_momentumburst/exit_worker.py` は `MICRO_MULTI_EXIT_TAG_ALLOWLIST=MomentumBurstMicro` 固定で、`strategy_tag=MicroVWAPRevert-*` の建玉は閉じない。
  - `workers/micro_vwaprevert/exit_worker.py` は `MICRO_MULTI_EXIT_TAG_ALLOWLIST=MicroVWAPRevert` 固定で、該当3本の正しい exit owner は `quant-micro-vwaprevert-exit` だった。
  - つまり:
    - `scalp_ping_5s_d_live` は profile外で entry worker だけが残存し、exit owner が不在
    - `MicroVWAPRevert-*` 3本は exit owner が停止中
    - 結果として close 判定自体が回らず、決済されなかった

- 一次対応:
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-scalp-ping-5s-d-exit,quant-micro-vwaprevert-exit`
  - `scripts/local_v2_stack.sh down --env ops/env/local-v2-stack.env --services quant-scalp-ping-5s-d`

- 結果:
  - `quant-micro-vwaprevert-exit` 復旧直後に
    - `438381 close_request -> close_ok`
    - `438379 close_request -> close_ok`
    - `438377 close_request -> close_ok`
  - `quant-scalp-ping-5s-d-exit` 復旧後に
    - `447946 close_request -> close_ok`
  - `scripts/oanda_open_trades.py` は `[]` となり、未決済は解消

- 残課題:
  - `trade_min` profile外 worker の残存監査が弱く、entry だけ残ると orphan trade を作れる。
  - `position_manager` の `/position/open_positions` は `position manager busy` と `int too large to convert to float` を散発しており、exit owner 復旧後の判定遅延要因として別途修正が必要。

## 2026-03-06 14:45 UTC / 2026-03-06 23:45 JST - 収益改善: `trade_min` の micro 枠を loser `MicroRangeBreak` から winner `MicroLevelReactor` へ差し替え

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY mid=157.811 spread=0.8p`
  - `openTrades=0`
  - `M1` 直近120本は通常帯の値動きで、極端な流動性悪化は確認せず

- 24h/7d 収益分解:
  - 全体 24h: `2396 trades / win 49.7% / PF 0.70 / net -8912.1 JPY`
  - 24h 主損失: `scalp_ping_5s_flow_live=-6883.7`, `M1Scalper-M1=-4408.0`
  - micro 勝敗比較（7d）:
    - `MicroLevelReactor: +259.5 JPY / 101 trades / win 65.3% / PF 1.67`
    - `MicroRangeBreak: -66.9 JPY / 119 trades / win 16.8% / PF 0.74`

- 補助事実:
  - `trade_min` active services は `B / MicroRangeBreak / M1Scalper`
  - `config/dynamic_alloc.json` は
    - `MicroLevelReactor lot_multiplier=1.566`
    - `MicroRangeBreak lot_multiplier=0.28`
    と、active profile と逆方向の配分を示していた
  - `logs/local_v2_stack/quant-micro-levelreactor.log` は `allowlist applied: MicroLevelReactor` を継続出力
  - 一方 `logs/local_v2_stack/quant-micro-momentumburst.log` は `allowlist empty; using all strategies` を出しており、
    `MomentumBurst` は直近成績こそ強いが immediate profile 追加先としては unsafe
  - `logs/orders.db` の直近2000 fill と `metrics.db` では
    - `spread_mean=0.805p`
    - `latency_submit_p50=190ms`
    - `reject_rate avg=0.032`
  - 執行コストは劣化しているが、主因はなお strategy expectancy 側

- 判断:
  - `M1` は直近 restart 後のログで `tag_filter_block` / `trend_block_long` が継続しており、追加 tighten を即重ねるより現設定の観測を続ける方がよい
  - いま一番境界が小さく、かつ期待値改善が大きい変更は、trade_min の micro 枠を `MicroRangeBreak` から `MicroLevelReactor` へ振り替えること
  - `MomentumBurst` は allowlist 崩れを直してから別タスクで採用判断する

- 対応:
  - `scripts/local_v2_stack.sh`
    - `PROFILE_trade_min`
      - `quant-micro-rangebreak(+exit)` を外し
      - `quant-micro-levelreactor(+exit)` を追加

- 期待効果:
  - 同じ trade_min リソース枠のまま loser かつ inactive な micro を外し、winner を入れる
  - `flow` は profile外のまま止め、`B` と `M1` の tighten は維持し、`MomentumBurst` は別途 allowlist 修正後に再投入する

- 再検証条件:
  1. `quant-micro-levelreactor.log` に dedicated worker 起動行が出て、`allowlist applied: MicroLevelReactor` が継続すること
  2. 30-60分後の `trades.db` で `MicroRangeBreak` の新規約定が止まり、`MicroLevelReactor` の増分 net がプラス圏で推移すること
  3. 24h `pdca_profitability_report.py` で `trade_min` active 群の net が現状より改善すること

## 2026-03-06 14:44 UTC / 2026-03-06 23:53 JST - 収益改善: 勝ち筋 micro を active 化し、`MomentumBurst` の dedicated allowlist と dynamic alloc 過小評価を修正

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY mid=157.826 spread=0.8p`
  - `openTrades=0`
  - `order-manager/position-manager health = ok`
  - `position/open_positions?include_unknown=false` は `stale=false`

- 事実:
  - `pdca_profitability_report_latest.md`
    - 24h: `2396 trades / PF=0.70 / net=-8912.1 JPY`
    - 7d 勝ち筋: `MomentumBurst +1856.9`, `MicroLevelReactor +259.5`
    - 7d 負け筋: `M1Scalper-M1 -6236.4`, `scalp_ping_5s_b_live -189.6`, `scalp_ping_5s_flow_live -7131.1`
  - `analyze_entry_precision.py --limit 2000`
    - `spread_mean=0.805p`, `latency_submit_p50=190ms`, `latency_preflight_p50=228ms`
    - 執行コストの悪化はあるが、主因は strategy expectancy 側
  - `config/dynamic_alloc.json` 再計算前:
    - `MomentumBurst lot_multiplier=0.50`
    - `MicroLevelReactor lot_multiplier=1.566`
    - `M1Scalper-M1 lot_multiplier=0.28`
  - `MomentumBurst` は `margin_closeout_rate=0.12` を含むが、`31 trades / win 87.1% / sum_realized_jpy=+1856.9` で、
    過小評価の方が支配的だった
  - `workers/micro_momentumburst/worker.py` / env は strategy 名 `MomentumBurst` ではなく `MomentumBurstMicro` を allowlist に使っており、
    実ログでも `allowlist empty; using all strategies` が出ていた

- 対応:
  - `scripts/local_v2_stack.sh`
    - `PROFILE_trade_min` に `quant-micro-momentumburst(+exit)` を追加
    - `PROFILE_trade_min` に `quant-micro-levelreactor(+exit)` を追加
  - `scripts/dynamic_alloc_worker.py`
    - strong winner が軽微な margin closeout ノイズだけで過度縮小されない補正を追加
  - `config/dynamic_alloc.json`
    - full 7d lookback で再計算し、
      `MomentumBurst=0.85`, `MicroLevelReactor=1.566`, `M1Scalper-M1=0.28`, `scalp_ping_5s_b_live=0.45`
  - `workers/micro_momentumburst/worker.py`
  - `workers/micro_momentumburst/exit_worker.py`
  - `ops/env/quant-micro-momentumburst*.env`
    - allowlist / exit tag を `MomentumBurst` へ統一
  - local V2 再起動:
    - `quant-micro-momentumburst.log`: `allowlist applied: MomentumBurst`
    - `quant-micro-levelreactor.log`: `allowlist applied: MicroLevelReactor`

- 判断:
  - 「件数を増やす」は loser を増やすことではなく、勝ち筋 micro の active 枠と実効ロットを増やすのが正解
  - `M1` と `B` は現設定のまま継続しつつ、dynamic alloc で強く縮小したまま観測する
  - `flow` は赤字寄与が大きすぎるため、今回の増量対象にはしない

- 再検証条件:
  1. 30-60分後に `quant-micro-momentumburst.log` / `quant-micro-levelreactor.log` で `allowlist applied` が継続すること
  2. 次回 `pdca_profitability_report.py` で active micro 群の `net_jpy` が現状より改善すること
  3. `orders.db` で `MomentumBurst` / `MicroLevelReactor` の `filled` が増えつつ、`reject_rate` が悪化しないこと

## 2026-03-06 15:00 UTC / 2026-03-07 00:05 JST - entry細り対策: `MicroLevelReactor` strict range gate を dedicated env で局所緩和

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY mid=157.56 spread=0.8p`
  - `openTrades=0`
  - `order-manager/position-manager health = ok`

- 事実:
  - `logs/orders.db` では `2026-03-06T14:59Z` を最後に新規 `filled` が止まり、ユーザー体感どおり直近数分はエントリーが細っていた
  - `logs/local_v2_stack/quant-micro-levelreactor.log`
    - `mlr_range_gate_block active=False score=0.049 adx=31.24 ma_gap=6.95`
    - `mlr_range_gate_block active=False score=0.067 adx=32.60 ma_gap=6.47`
    - `mlr_range_gate_block active=False score=0.073 adx=34.34 ma_gap=5.74`
    が継続し、勝ち筋 `MicroLevelReactor` の entry 候補が dedicated strict gate で落ちていた
  - gate 条件は
    - `MICRO_MULTI_MLR_MIN_RANGE_SCORE=0.62`
    - `MICRO_MULTI_MLR_MAX_ADX=20.0`
    - `MICRO_MULTI_MLR_MAX_MA_GAP_PIPS=2.2`
    で、直近 live 実測に対して厳しすぎた

- 対応:
  - `ops/env/quant-micro-levelreactor.env`
    - `MICRO_MULTI_MLR_MIN_RANGE_SCORE=0.05`
    - `MICRO_MULTI_MLR_MAX_ADX=36.0`
    - `MICRO_MULTI_MLR_MAX_MA_GAP_PIPS=6.5`
    を追加

- 判断:
  - `M1` や `B` を緩めると loser 側の回転まで増えるため、まずは winner `MicroLevelReactor` の dedicated gate だけを局所緩和するのが妥当
  - strict gate 自体は残し、極端な trend 追従相場を完全には開放しない

- 再検証条件:
  1. `quant-micro-levelreactor.log` で `mlr_range_gate_block` の頻度が下がること
  2. `orders.db` で `MicroLevelReactor` 系の `preflight_start` / `filled` が再開すること
  3. `reject_rate` と `perf_block` が悪化しないこと

## 2026-03-06 15:04 UTC / 2026-03-07 00:04 JST - 「全然エントリーされない」追加切り分け: MLR 緩和値は main 済みで、原因は stale worker

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY mid=157.61 spread=0.8p`
  - `ATR14(M1)=5.24p`, `ATR60(M1)=5.06p`, `range30(M1)=43.90p`
  - `openTrades=0`
  - `order-manager/position-manager health = ok`

- 事実:
  - `ops/env/quant-micro-levelreactor.env` と `HEAD` には既に
    - `MICRO_MULTI_MLR_MIN_RANGE_SCORE=0.05`
    - `MICRO_MULTI_MLR_MAX_ADX=36.0`
    - `MICRO_MULTI_MLR_MAX_MA_GAP_PIPS=6.5`
    が入っていた
  - それでも `logs/local_v2_stack/quant-micro-levelreactor.log` は UTC `15:05:22` まで
    `mlr_range_gate_block active=False score=0.081 adx=34.77 ma_gap=5.37`
    を継続しており、worker が stale 設定で動いていた
  - `quant-m1scalper.log` は
    - `worker start (... tag_filter=breakout-retest-long,nwave-long ...)`
    - `trend_block_long ...`
    - `tag_filter_block tag=M1Scalper-buy-dip ...`
    で、loser 側を意図的に絞っていた
  - `logs/orders.db` では UTC `15:05:03`, `15:05:08`, `15:06:03`, `15:06:20`, `15:08:08` に
    `scalp_ping_5s_b_live` の `filled` を確認し、全体停止ではなかった

- 対応:
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-micro-levelreactor,quant-micro-levelreactor-exit`
    で MLR dedicated worker を明示的に再起動

- 判断:
  - 直近の entry 細りは「設定不足」ではなく「winner dedicated worker が stale state のまま」だった
  - 先に `M1` / `B` のガードを緩めるより、main 済みの MLR 緩和値を live に読ませる方が安全で正しい

## 2026-03-06 15:16 UTC / 2026-03-07 00:16 JST - 収益改善: stray loser を止め、`M1Scalper` は `vshape-rebound-long` だけ再開

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY mid=157.668 spread=0.8p`
  - `openTrades` は OANDA `503` が断続したが、account snapshot は取得できており local health も fresh

- 事実:
  - `pdca_profitability_report.py`
    - 24h: `2314 trades / PF=0.68 / net=-8603.5 JPY`
  - `trades.db` 直近3h:
    - `MomentumBurst +661.8 JPY`
    - `MicroLevelReactor +200.8 JPY`
    - `M1Scalper-M1 -1395.5 JPY`
    - `scalp_ping_5s_flow_live -5593.4 JPY`
  - `ps` / `scripts/local_v2_stack.sh status` では
    - profile 外の `quant-scalp-ping-5s-c` / `quant-scalp-ping-5s-c-exit` が常駐していた
  - `quant-m1scalper.log` では
    - `signal_vshape_rebound action=OPEN_LONG conf=81`
    - `tag_filter_block tag=M1Scalper-vshape-rebound-long allow=['breakout-retest-long', 'nwave-long']`
    が出ており、件数を増やせる winner 候補が allowlist で落ちていた
  - `trades.db` の `M1Scalper` source tag 実績:
    - `M1Scalper-nwave-long: +98.4 JPY / 30`
    - `M1Scalper-breakout-retest-long: +81.2 JPY / 2`
    - `M1Scalper-vshape-rebound-long: +16.7 JPY / 2`
    - `M1Scalper-buy-dip: -2209.4 JPY / 425`
    - `M1Scalper-trend-long: -2277.3 JPY / 520`

- 対応:
  - `ops/env/local-v2-stack.env`
    - `STRATEGY_CONTROL_ENTRY_SCALP_PING_5S_C=0`
    - `STRATEGY_CONTROL_ENTRY_SCALP_PING_5S_FLOW=0`
    - `M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest-long,nwave-long,vshape-rebound-long`

- 判断:
  - loser 全体を広げるのではなく、`scalp_ping_5s_c/flow` の stray entry を止め、
    `M1` は直近 positive な long setup だけ追加するのが最も収益寄り

- 再検証条件:
  1. `quant-m1scalper.log` で `tag_filter` に `vshape-rebound-long` が含まれること
  2. `quant-scalp-ping-5s-c` の新規 entry が止まること
  3. 次回 1-3h 集計で `M1Scalper-M1` の net 勾配が改善し、`flow/c` の新規損失寄与が増えないこと

## 2026-03-06 15:35 UTC / 2026-03-07 00:35 JST - 収益改善: `MicroLevelReactor` の lot / 頻度を winner 側だけ増やす

- 市況確認（ローカルV2実測 + OANDA API）:
  - `USD/JPY mid=157.694 spread=0.8p`
  - `oanda_open_trades.py -> []`
  - `pdca_profitability_report_latest.md`
    - 24h: `2361 trades / win 49.2% / PF=0.69 / net=-8827.2 JPY`
    - 7d: `PF=0.59 / net=-11608.5 JPY`
  - loser は依然 `scalp_ping_5s_flow_live -6915.8 JPY`, `M1Scalper-M1 -4213.7 JPY`
  - winner は `MomentumBurst +2017.1 JPY`, `MicroLevelReactor +259.5 JPY`

- 事実:
  - `trades.db` 7d 集計:
    - `MicroLevelReactor: avg_units=119.6 / avg_intent=601.4 / avg_prob=0.440 / net=+259.5 JPY`
    - 勝っているのに intent 比で実約定サイズが薄すぎた
  - `orders.db` では `MicroLevelReactor` の recent rows が
    - `probability_scaled raw_units=94 -> scaled_units=42`
    - `entry_probability=0.445`
    を繰り返し、preserve-intent で 55% 以上削られていた
  - 同じ `request_json.entry_thesis.forecast_fusion` では
    - `units_before=454 -> units_after=211`
    - `entry_probability_before=0.73 -> 0.445`
    - `forecast_allowed=false reason=style_mismatch_range`
    で、worker 側 forecast fusion でも先に圧縮されていた
  - さらに `quant-v2-runtime.env` の strategy-specific forecast gate は
    - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_MICROLEVELREACTOR=0.22`
    - `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_MICROLEVELREACTOR=0.40`
    で、実際の winner payload にあった `target_reach_prob=0.103 / range_pressure=0.1599` でも upstream で厳しかった
  - `quant-micro-levelreactor.log` では 2026-03-05 13:27 UTC に
    - `OPEN_SKIP ... note=entry_probability:entry_probability_reject_threshold`
    が連続しており、frequency も `reject_under=0.52` に削られていた
  - 逆に `MAX_SIGNALS_PER_CYCLE=1` は dedicated 1-strategy worker では主要因ではなかった

- 対応:
  - `ops/env/quant-micro-levelreactor.env`
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MicroLevelReactor:1.35` を追加
    - `ORDER_MANAGER_PRESERVE_INTENT_REJECT_UNDER: 0.52 -> 0.40`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE=0.60` を追加
    - `STRATEGY_FORECAST_FUSION_DISALLOW_UNITS_MULT=0.80` を追加
    - `STRATEGY_FORECAST_FUSION_DISALLOW_PROB_MULT=0.82` を追加
    - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_MICROLEVELREACTOR: 0.22 -> 0.08`
    - `FORECAST_GATE_STYLE_RANGE_MIN_PRESSURE_STRATEGY_MICROLEVELREACTOR: 0.40 -> 0.15`
  - 再起動後の実プロセス env では `local-v2-stack.env` の `MICRO_MULTI_BASE_UNITS=48000` が後勝ちしていたため、
    dedicated lot 増加は `BASE_UNITS` 変更ではなく `MICRO_MULTI_STRATEGY_UNITS_MULT` で確実に入れた

- 判断:
  - 「全体ロット不足」ではなく、勝っている `MicroLevelReactor` だけが forecast/probability の二段圧縮で薄くなっていた
  - loser 側を reopen するより、winner dedicated worker の reject と scale を緩める方が収益寄りで安全

- 再検証条件:
  1. `orders.db` で `MicroLevelReactor` の `entry_probability_reject` が減ること
  2. `probability_scaled` の `scaled_units/raw_units` が `0.45` 近辺から改善すること
  3. 次の 1-3h で `MicroLevelReactor` の `filled` 件数と avg units が増えても net/PF が悪化しないこと
## 2026-03-07 01:26 JST / local-v2: `margin_snapshot_failed` 抑制のため bounded stale margin を 60s へ延長

- 市況確認（ローカルV2実測 + OANDA API, 2026-03-07 01:23-01:25 JST）:
  - `USD/JPY mid=157.602 spread=0.8p`
  - `ATR14(M1)=3.99p`, `range30m=19.0p`, `range60m=24.7p`
  - `open_trades=0`
  - `balance/nav=37914.75/37914.75 JPY`, `margin used=0`, `margin available=37914.75 JPY`
  - `data_lag_ms=1414`, `decision_latency_ms=17.6`

- 事実:
  - `logs/health_snapshot.json` と `pdca_profitability_report.py` では口座余力は十分で、
    spread も通常水準だった。
  - その一方 `orders.db` 24h では `margin_snapshot_failed=18`, `api_error=6`, `rejected=32`。
  - `logs/local_v2_stack/quant-order-manager.log` には
    `margin guard snapshot failed: 503 ... /summary` と
    `using stale margin snapshot ... reason=refresh_in_progress` が併存し、
    stale reuse が 15 秒を超える burst で途切れていた。
  - 24h 収益は `net_jpy=-7136.5 / PF=0.68`。主因は strategy expectancy 側だが、
    `/summary` flap による no-entry は機会損失として別で潰す価値がある。

- 対応:
  - `ops/env/local-v2-stack.env`
    - `ORDER_MARGIN_STALE_ALLOW_SEC=15 -> 60`

- 意図:
  - strategy local 判定や exit ロジックは変えず、OANDA `/summary` の minute-scale `503`
    による `margin_snapshot_failed` だけを減らす。
  - 既存の `ORDER_MARGIN_STALE_MIN_FREE_RATIO=0.30` /
    `ORDER_MARGIN_STALE_MIN_HEALTH_BUFFER=0.25` は維持し、
    stale fallback を無制限にはしない。

- 再検証条件:
  1. `quant-order-manager.log` で `using stale margin snapshot` が継続しつつ、
     `margin guard snapshot failed` が burst 後に減ること。
  2. 次の 1-3h で `orders.db` の `margin_snapshot_failed` 増分が現状ペース以下になること。
  3. `reject_rate` を悪化させず、`filled` の連続性が維持されること。

## 2026-03-07 01:36 JST / local-v2: M1Scalper `close_reject_profit_buffer` を 0.10p へ緩和

- 市況確認（ローカルV2実測 + OANDA API, 2026-03-07 01:23-01:25 JST）:
  - `USD/JPY mid=157.602 spread=0.8p`
  - `ATR14(M1)=3.99p`, `range30m=19.0p`, `open_trades=0`
  - `summary/pricing` は 5/5 `200 OK`、latency は概ね `220-310ms`

- 事実:
  - `orders.db` 24h の `M1Scalper-M1` は `close_reject_profit_buffer=521`。
  - reject 時の `min_profit_pips` は全件 `0.20`、`est_pips` は平均 `0.069p`。
  - 同 reject 分布は `est_pips<0.05: 279`, `<0.10: 300`, `<0.15: 401` で、
    `0.10-0.20p` に 221 件の tiny-profit exit が滞留していた。
  - reject を一度でも踏んだユニーク ticket は 141 本で、最終着地は
    `avg_pips=+0.119 / net_jpy=-46.3` と全体ではわずかに負け。
  - 内訳は mixed で、
    `candle_bearish_engulfing / candle_hanging_man / candle_bullish_engulfing`
    は reject 後の final がプラスだった一方、
    `candle_inverted_hammer / candle_hammer / candle_shooting_star`
    は final がマイナスへ反転していた。
  - `STOP_LOSS_ORDER` へ落ちた 8 本は `net_jpy=-228.0` と損失寄与が明確だった。
  - `trades.db` 24h の `M1Scalper-M1` positive close は `1340` 本で、
    `pl_pips<0.10` は 8 本、`pl_pips<0.20` は 49 本。

- 判断:
  - `0.20p -> 0.00p` まで緩めると upside を切り過ぎるリスクがある。
  - 一方 `0.20p` 維持では `0.10-0.20p` の tiny-profit exit を大量に拒否し、
    反転負けへつながる ticket が残る。
  - よって `M1Scalper` の `min_profit_pips` は保守的に `0.10p` へ半減し、
    近BE付近の tiny-profit exit だけを通しやすくする。

- 対応:
  - `config/strategy_exit_protections.yaml`
    - `M1Scalper.min_profit_pips=0.20 -> 0.10`

- 再検証条件:
  1. 次の 1-3h / 24h で `orders.db` の `M1Scalper-M1 close_reject_profit_buffer` が減少すること。
  2. `trades.db` の `M1Scalper-M1 MARKET_ORDER_TRADE_CLOSE` 平均損益が `-0.483p` から改善すること。
  3. `M1Scalper-M1` の `PF(pips)` が `0.68` 近辺から改善し、`STOP_LOSS_ORDER` の純損失が増えないこと。

## 2026-03-07 01:45 JST / local-v2: `MicroLevelReactor` の winner sizing を 1.60 へ増量

- 市況確認（ローカルV2実測 + OANDA API, 2026-03-07 01:37-01:45 JST）:
  - `USD/JPY mid=157.555-157.564 spread=0.8p`
  - `ATR14(M1)=3.73p`, `M1 60本レンジ=19.0p`, `open_trades=0`
  - `summary/pricing` は継続 `200 OK`、latency は概ね `230-325ms`

- 事実:
  - corrected 集計（`datetime(close_time)` 基準）では、直近 `30m` の active 群は
    - `MicroLevelReactor: 7 trades / +71.3 JPY / avg_pips=+1.229`
    - `scalp_ping_5s_b_live: 17 trades / -0.1 JPY / avg_pips=-0.582`
  - `60m` では `MicroLevelReactor: 16 trades / -39.3 JPY` だが、
    直近 `30m` は再びプラスへ戻していた。
  - `orders.db` 直近 `60m` の `MicroLevelReactor` は
    - `filled=16`
    - `avg filled units=1110.4`
    - `avg entry_units_intent=3452.3`
    - `fill/intent ratio=0.322`
  - `quant-micro-levelreactor.log` では dedicated env の
    `s_mult=1.35`, `dyn=1.40`, `cap=0.95` が実際に使われていた。
  - つまり、winner なのに probability/forecast 経路で実約定サイズがまだ薄い。

- 判断:
  - 現在の active 収益源は `MicroLevelReactor` で、`flow` と `M1` は直近 active 導線の主役ではない。
  - shared logic を緩めるより、winner dedicated env の `strategy_units_mult` を少し持ち上げる方が
    境界が小さく、収益速度を上げやすい。
  - 直近 `open_trades=0`、margin 余力も大きいため、この増量は許容範囲。

- 対応:
  - `ops/env/quant-micro-levelreactor.env`
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MicroLevelReactor:1.35 -> 1.60`

- 再検証条件:
  1. 次の `30-60m` で `MicroLevelReactor` の `avg filled units` が `1110` 近辺から増えること。
  2. `MicroLevelReactor` の `net_jpy / PF` が悪化せず、少なくとも `30m` プラス圏を維持すること。
  3. `orders.db` の `rejected` / `margin_snapshot_failed` が `MicroLevelReactor` で増えないこと。

## 2026-03-07 01:50 JST / local-v2: `MomentumBurst` を増量し、`scalp_ping_5s_b_live` の thin-edge 許容を停止

- 市況確認（ローカルV2実測 + OANDA API, 2026-03-07 01:49-01:50 JST）:
  - `USD/JPY mid=157.581 spread=0.8p`
  - `ATR14(M1)=3.61p`, `M1 60本レンジ=20.2p`, `open_trades=0`
  - `pricing/openTrades` は `200 OK`、一方 `account summary` は断続 `503` が残存
  - `health_snapshot`: `data_lag_ms=861.6`, `decision_latency_ms=15.1`

- 事実:
  - `trades.db` 7d では
    - `MomentumBurst: 31 trades / +1856.9 JPY / avg_pips=+3.8 / win=87.1%`
    - `MicroLevelReactor: 117 trades / +220.2 JPY / avg_pips=+3.207 / win=59.8%`
    - `M1Scalper-M1: -6172.5 JPY`, `scalp_ping_5s_flow_live: -7131.1 JPY`
  - corrected 集計の直近 `60m` では
    - `scalp_ping_5s_b_live: 42 fills / -0.4 JPY / avg_pips=-0.536`
    - `MicroLevelReactor: 16 fills / -39.3 JPY / avg_pips=-0.144`
    - `MomentumBurst` はこの1hでは未約定
  - `orders.db` 直近 `60m` の `scalp_ping_5s_b_live` は
    - `probability_scaled=21`, `filled=40`, `rejected(STOP_LOSS_ON_FILL_LOSS)=4`, `margin_snapshot_failed=4`
  - `local-v2-stack.env` は `SCALP_PING_5S_B_LOOKAHEAD_ALLOW_THIN_EDGE=1` を上書きしており、
    B_live の低エッジ許容を戻していた。
  - `quant-micro-momentumburst` は `local-v2-stack.env` の `MICRO_MULTI_BASE_UNITS=48000` が後勝ちするため、
    dedicated service 側の base units 変更だけでは live sizing を強めにくい。

- 判断:
  - 「今すぐ回っている負け筋」は `scalp_ping_5s_b_live` の薄利薄損ショートで、
    ここは停止ではなく thin-edge 許容を止めて entry 品質を戻す。
  - 一方、低頻度でも強い winner は `MomentumBurst` なので、
    shared layer を緩めず dedicated `strategy_units_mult` だけを追加して次の発火で厚く取る。

- 対応:
  - `ops/env/quant-micro-momentumburst.env`
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MomentumBurst:1.25` を追加
  - `ops/env/local-v2-stack.env`
    - `SCALP_PING_5S_B_LOOKAHEAD_ALLOW_THIN_EDGE=1 -> 0`

- 再検証条件:
  1. 次の `1-3h` で `scalp_ping_5s_b_live` の `filled / probability_scaled` 比率が下がりつつ、
     `avg_pips` と `PF` が改善すること。
  2. 次の `1-24h` で `MomentumBurst` の再発火時 `filled units` が従来より増え、
     7d winner 特性を崩さないこと。
  3. `orders.db` の `margin_snapshot_failed` と `api_error` が悪化せず、
     `open_trades=0` からの burst で margin cap に詰まらないこと。

## 2026-03-07 06:59 JST / local-v2: JST 7-8時メンテ帯のため追加チューニングを保留

- 市況確認（ローカルV2実測 + OANDA API, 2026-03-07 06:59 JST）:
  - `USD/JPY mid=157.8215`
  - `spread=6.3p`
  - `ATR14(M1)=1.92p`, `M1 60本レンジ=11.6p`
  - `summary/openTradeCount=0`, `marginAvailable=37839.6842`
  - `pricing` は 3 連続で同一 timestamp を返し、更新停止状態

- 事実:
  - 直近 `15m/30m/60m` の `trades.db` は新規 close なし。
  - `health_snapshot` は `git_rev=dcbe4482`, `data_lag_ms=1064.7`, `decision_latency_ms=11.7`。
  - `quant-micro-levelreactor` live env では
    `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE=0.70` と
    `MICRO_MULTI_STRATEGY_UNITS_MULT=MicroLevelReactor:1.60` を確認済み。

- 判断:
  - これは通常の流動性劣化ではなく、JST 7-8時のメンテ時間帯に一致する。
  - AGENTS の運用ルールどおり、この時間帯に追加の攻め設定を入れるのは保留する。
  - 既存の `dcbe4482` 反映確認だけ完了とし、次の評価はメンテ明け後に行う。

## 2026-03-06 16:56 UTC / 2026-03-07 01:56 JST - `MicroLevelReactor` の preserve-intent floor を `0.70` へ引き上げ

- 市況:
  - `tick_cache.json`: `mid=157.524`, `spread=0.8p`, tick age `0.3s`
  - `factor_cache.json` M1: `close=157.53`, `ATR14=3.30p`, `regime=Range`, `timestamp=2026-03-06T16:52:59.947137+00:00`
  - `pdca_profitability_report.py`: 24h `PF=0.67`, `net_jpy=-7294.6`, `open_trades=0`
  - `summary/openTrades` は断続 `503` が残るが、`pricing` / tick / local health は正常

- 事実:
  - `MicroLevelReactor` は 24h `117 trades / +220.2 JPY / avg_pips=+3.207 / win=59.8%` の winner。
  - 一方 `orders.db` の直近 `probability_scaled` では、同戦略の実オーダが
    - `entry_units_intent=2234 -> raw_units=1340 -> scaled_units=737`
    - `entry_units_intent=3403 -> raw_units=2042 -> scaled_units=1123`
    - `entry_units_intent=5454 -> raw_units=3272 -> scaled_units=1800`
    と一貫して `raw/intent=0.60`, `scaled/intent≈0.33` に張り付いていた。
  - `entry_probability` は `0.525-0.550` 帯で、まず preserve-intent floor が `0.60` で raw units を切り、
    その後 probability scaling が掛かる二段圧縮になっていた。
  - `MicroLevelReactor` の直近クラスター（`2026-03-06 16:37-16:38 UTC`）は `7 trades / +71.3 JPY` と利益化しており、
    losing worker ではなく winner worker のサイズ回復が最優先だった。

- 判断:
  - 前回の `MICRO_MULTI_STRATEGY_UNITS_MULT=1.60` 反映後はまだ post-change サンプルが薄く、
    さらに生の strategy intent を増やすより、floor に張り付いている preflight 圧縮を先に戻す方が筋。
  - shared order path は触らず、winner 専用 worker の `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE` だけを上げる。

- 対応:
  - `ops/env/quant-micro-levelreactor.env`
    - `ORDER_MANAGER_PRESERVE_INTENT_MIN_SCALE: 0.60 -> 0.70`

- 狙い:
  - 同じ `entry_probability≈0.55` 帯でも、`raw_units` を `intent` の 70% まで戻し、
    realized units を約 `+16.7%` 押し上げる。
  - loser 側の B / M1 / flow や共通 `order_manager` の選別ロジックは変更しない。

- 再検証条件:
  1. 次の `1-3h` で `orders.db` の `MicroLevelReactor` `probability_scaled` が `raw/intent=0.70` に寄ること。
  2. `scaled/intent` が従来の `0.315-0.330` より改善し、`filled units` が増えること。
  3. `MicroLevelReactor` の `avg_pips` / `net_jpy` が悪化せず、`margin_snapshot_failed` の増加を伴わないこと。

## 2026-03-07 01:53 UTC / 2026-03-07 10:53 JST - 週末クローズ帯で spread `6.3p` のため「勝ち筋追加」は保留、次回昇格候補は `TrendBreakout`

- 市況（local-v2 実測 + OANDA snapshot）:
  - `health_snapshot.json`:
    - `trades_last_close=2026-03-06T21:40:06Z`
    - `orders_status_1h=[]`
    - `data_lag_ms=1065`
    - `decision_latency_ms=11.7`
  - `orderbook_snapshot.json`:
    - `bid=157.790 / ask=157.853 / spread=6.3p`
    - `latency_ms=163`
  - `factor_cache.json`:
    - `ATR(M1)=1.94p / ATR(M5)=4.97p / ATR(H1)=18.82p`
  - `oanda_open_positions_live_USD_JPY.json`:
    - `long_units=871 / short_units=0`

- 判断:
  - spread `6.3p` は現行 `scalp` / `micro` 系の通常ガードレンジ外で、AGENTS の「市況悪化時は作業保留」に該当。
  - このため `trade_min` への新規 worker 追加、`local_v2_stack restart`、live 反映確認は行わない。
  - 今回は候補選定と監査ログ更新に留める。

- 候補比較（`logs/trades.db` 7d/24h + `config/dynamic_alloc.json`）:
  - `TrendBreakout`
    - 7d `3 trades / +264.4 JPY`
    - 24h `3 trades / +264.4 JPY`
    - dynamic alloc `score=0.575 / lot_multiplier=1.0`
    - 2026-03-05 JST の crash 修正と、2026-03-06 JST の M1 family open-trades guard / fail-closed guard 追加が既に main 側へ入っている。
  - `MicroTrendRetest-short`
    - 7d `15 trades / +8.7 JPY`
    - 24h `10 trades / -19.0 JPY`
    - dynamic alloc `score=0.552 / lot_multiplier=1.113`
    - `STOP_LOSS_ORDER=5`, `TAKE_PROFIT_ORDER=1` と直近の損失寄与がまだ重い。

- 次回通常流動性帯での第1候補:
  - `quant-scalp-trend-breakout(+exit)` を canary 追加候補に固定する。
  - 理由:
    - loser `scalp_ping_5s_flow_live` や `M1Scalper-M1` と違い、直近実現損益が正。
    - crash 修正と open-trades family guard が完了しており、live 安全性の前提が最も揃っている。
    - `MicroTrendRetest` よりも直近損失ドリフトが小さい。

- 次回再開条件:
  1. spread が通常帯（概ね `<=1.0p`）へ戻ること。
  2. `orders_status_1h` に live 注文が再開していること。
  3. その時点で `TrendBreakout` を `trade_min` に追加する実装と反映確認を再開すること。

## 2026-03-07 02:00 UTC / 2026-03-07 11:00 JST - `MicroTrendRetest-short` 専用フィルタを実装し、`trade_min` に常駐化

- 市況（local-v2 実測 + OANDA snapshot）:
  - `orderbook_snapshot.json`
    - `bid=157.790 / ask=157.853 / spread=6.3p`
    - `latency_ms=163`
  - `factor_cache.json`
    - `ATR(M1)=1.94p / ATR(M5)=4.97p / ATR(H1)=18.82p`
  - `oanda_account_snapshot_live.json` / `oanda_open_positions_live_USD_JPY.json`
    - `margin_used=5498 JPY`
    - `USD/JPY long_units=871 / short_units=0`

- 実測根拠:
  - `logs/trades.db` 7d
    - `MicroTrendRetest-long: 17 trades / -162.2 JPY / PF 0.21`
    - `MicroTrendRetest-short: 15 trades / +8.7 JPY / PF 1.06`
  - 既存 dedicated worker は `MICRO_STRATEGY_ALLOWLIST=MicroTrendRetest` のみで、
    `tag=MicroTrendRetest-long|short` を分離できなかった。

- 実装:
  - `workers/micro_runtime/config.py`
    - `MICRO_MULTI_SIGNAL_TAG_CONTAINS` を追加。
  - `workers/micro_runtime/worker.py`
    - candidate `signal_tag` に `contains` フィルタを追加。
    - 起動ログに `signal_tag_contains` を出すよう変更。
  - `ops/env/quant-micro-trendretest.env`
    - `MICRO_MULTI_SIGNAL_TAG_CONTAINS=short`
  - `scripts/local_v2_stack.sh`
    - `PROFILE_trade_min` に `quant-micro-trendretest(+exit)` を追加。

- 反映確認:
  - `python -m py_compile workers/micro_runtime/config.py workers/micro_runtime/worker.py workers/micro_trendretest/worker.py`
    - pass
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env`
    - `quant-micro-trendretest` / `quant-micro-trendretest-exit` が `running`
  - `logs/local_v2_stack/quant-micro-trendretest.log`
    - `worker start (interval=4.0s signal_tag_contains=short)` を確認

- 判断:
  - shared micro runtime の最小変更で、負けている long を dedicated worker から外し、
    `MicroTrendRetest-short` だけを `trade_min` の canary に載せる形にできた。
  - 週末クローズ帯のため、今回確認できたのは service 起動と env 読み込みまで。
    live fill / PF / reject の再評価は通常流動性帯で継続する。

## 2026-03-07 11:00 JST / local-v2: `M1Scalper-M1` の side filter を `long` に復帰

- 市況確認（OANDA 実測 / Saturday close）:
  - `pricing` は金曜クローズの同一 timestamp を返し続け、`USD/JPY mid=157.8215 / spread=6.3p`
  - `openTrades=0`, `marginUsed=0`, `orders_status_1h=[]`
  - 週末クローズ中のため、新規 live fill での再評価は不可

- 事実:
  - live `quant-m1scalper` env は `M1SCALP_SIDE_FILTER=none` で、intended operational value の `long` と不整合だった。
  - `trades.db` 24h の `M1Scalper-M1` を `source_signal_tag` 由来で side 集計すると
    - `short: 474 trades / -812.7 JPY / avg_pips=-0.432 / win=41.8%`
    - `long: 138 trades / -46.1 JPY / avg_pips=+0.062 / win=64.5%`
  - source tag 別でも
    - `M1Scalper-sell-rally: 475 trades / -614.5 JPY`
    - `M1Scalper-nwave-short: 13 trades / -352.1 JPY`
    - `M1Scalper-breakout-retest-long: +81.2 JPY`
  - 現状の主損失は short 側に集中していた。

- 判断:
  - `none -> long` は新規ロジック追加ではなく、intended worker policy への復帰。
  - 24h 実測でも short 側 cut の効果が最大だったため、週明け前の改善として最優先。

- 対応:
  - `ops/env/quant-m1scalper.env`
    - `M1SCALP_SIDE_FILTER=none -> long`

- 再検証条件:
  1. 週明け再開後の `trades.db` で `M1Scalper-M1` short 由来の新規 close が消えること。
  2. `M1Scalper-M1` 24h `net_jpy` と `avg_pips` が改善すること。
  3. `orders.db` で `M1Scalper-M1` が `breakout-retest-long` / `nwave-long` 中心に回ること。

## 2026-03-07 11:10 JST / local-v2: autorecover に market sanity guard を追加、次回 `trade_min` へ `TrendBreakout` を昇格

- 市況確認（ローカルV2実測）:
  - `logs/orderbook_snapshot.json`: `bid=157.790 / ask=157.853 / spread=6.3p`
  - `logs/tick_cache.json`: 最終 tick age は約 `14987s`
  - `logs/health_snapshot.json`: `orders_status_1h=[]`, `open_trades=0`
  - 土曜クローズ帯で、AGENTS の「市況悪化時は作業保留」に該当

- 事実:
  - `logs/local_v2_autorecover.log` では、クローズ帯でも `stack up succeeded profile=trade_min` が繰り返し出ていた。
  - 同時間帯の `logs/local_v2_stack/quant-position-manager.log` と各 exit worker では
    `127.0.0.1:8301` 接続拒否が断続し、`position-manager` の不要な再起動が収益導線を乱していた。
  - 直近勝ち筋の `TrendBreakout` は 7d `3 trades / +264.4 JPY / avg_pips +6.1 / win 100%`。

- 対応:
  - `scripts/local_v2_autorecover_once.sh`
    - `orderbook_snapshot.json` を見て、`spread>2.2p` / `tick_age>90s` / `JST 7-8時台` では non-core recovery を抑止する `market sanity guard` を追加
    - ただし `quant-market-data-feed` / `quant-strategy-control` / `quant-order-manager` / `quant-position-manager` が `stopped/stale` のときは、実障害復旧を塞がないよう guard を bypass する
  - `scripts/local_v2_stack.sh`
    - `PROFILE_trade_min` に `quant-scalp-trend-breakout` / `quant-scalp-trend-breakout-exit` を追加

- 狙い:
  - クローズ帯・メンテ帯での不要な autorecover による `position-manager` 再起動ループを止める。
  - core 4 サービスの自己復旧は維持し、market-data-feed 障害や core crash の復旧は止めない。
  - 通常流動性へ戻った次回 `trade_min` 起動で、winner の `TrendBreakout` を自動的に載せる。

- 再検証条件:
  1. 週明け前クローズ帯で `local_v2_autorecover.log` に不要な `stack up succeeded` が増えないこと。
  2. 次回通常流動性帯の `trade_min` 起動後、`TrendBreakout` worker が起動し `orders/trades` に新規実績が乗ること。
  3. `position-manager` の `connection refused` が減り、`orders.db` の `margin_snapshot_failed` / `api_error 503` が悪化しないこと。

## 2026-03-07 JST / local-v2 replay: M1 family (`TrendBreakout` / `PullbackContinuation` / `FailedBreakReverse`) を replay 対象へ追加

- 目的:
  - 週末のうちに、M1 family 3本を `scripts/replay_workers.py` / `scripts/replay_exit_workers_groups.py` から直接回せる状態にする。

- 実装:
  - `scripts/replay_workers.py`
    - `trend_breakout` / `pullback_continuation` / `failed_break_reverse` を追加
    - `factor_cache` + `M1Scalper.check` ベースの近似 replay helper を実装
    - 壊れた tick 行は skip するよう `load_ticks()` を fail-open 化
  - `scripts/replay_exit_workers_groups.py`
    - M1 family 3 worker を受理
    - class ベース exit worker を持たない worker 向けに simple exit adapter fallback を追加
  - `docs/REPLAY_STANDARD.md`
    - 上記 worker 名と近似条件を追記

- テスト:
  - `pytest tests/replay/test_m1_family_replay.py`
    - `2 passed`
  - `pytest tests/workers/test_m1scalper_split_workers.py`
    - `4 passed`
  - `python -m py_compile scripts/replay_workers.py scripts/replay_exit_workers_groups.py`
    - pass

- 週末 replay 実測:
  - `python scripts/replay_workers.py --worker trend_breakout --ticks tmp/vm_ticks/logs/replay/USD_JPY/USD_JPY_ticks_20260220.jsonl`
    - `trades=0`
    - `tmp/vm_ticks/logs/replay/.../20260220` は `6 ticks` しかなく、smoke 相当
  - `python scripts/replay_exit_workers_groups.py --ticks tmp/vm_ticks/logs/archive/replay.20260220-094854.dir/USD_JPY/USD_JPY_ticks_20260212.jsonl --workers trend_breakout,pullback_continuation,failed_break_reverse --no-hard-sl --exclude-end-of-replay --out-dir tmp/replay_exit_workers_groups_m1_family_20260212`
    - `trend_breakout / pullback_continuation / failed_break_reverse` の `base` はすべて `trades=0`
    - `summary_all.json` の `selection` も 3 worker 全て `requested=0 / applied=0 / excluded=0`
    - 対象 tick は `137,948` 行、spread percentile は `q20≈0.8p / q80≈0.8p`

- 判断:
  - replay 導線は通ったが、`2026-02-12` のフル日 tick でも M1 family 3本は entry が 1 本も立たなかった。
  - live 昇格本命は引き続き `TrendBreakout`。ただし週明け前の追加判断は `longer replay window` と通常流動性帯の live canary で継続する。

## 2026-03-07 02:52 UTC / 2026-03-07 11:52 JST - replay: `TrendBreakout` の `0 trades` は strategy no-signal ではなく tick coverage miss

- 市況確認（local-v2, weekend close）:
  - `logs/orderbook_snapshot.json`: `bid=157.790 / ask=157.853 / spread=6.3p`
  - `logs/health_snapshot.json`: `orders_status_1h=[]`, `last close=2026-03-06 21:40:06 UTC`
  - 週末クローズ帯のため、live 変更は行わず replay/ログ解析のみ実施

- 対象:
  - `logs/replay/USD_JPY/USD_JPY_ticks_20260306.jsonl`
  - `logs/trades.db`
  - `logs/local_v2_stack/quant-scalp-trend-breakout.log`
  - `tmp/replay_trend_breakout_20260306.json`
  - `tmp/replay_exit_workers_groups_trend_breakout_20260306/summary_all.json`

- 事実:
  - `TrendBreakout` の live close は `2026-03-06 09:06:51Z` / `09:06:58Z` open、`09:13:11Z` / `09:13:13Z` close の 2 件、各 `+89.6 JPY`。
  - `entry_thesis.source_signal_tag` は両方とも `M1Scalper-breakout-retest-long`、`entry_probability=0.86`、`entry_units_intent=1250`。
  - worker log でも `2026-03-06 18:06:51 JST` に同タグで `sent units=1250` を確認。
  - 一方、`logs/replay/USD_JPY/USD_JPY_ticks_20260306.jsonl` の tick 窓は `2026-03-06T11:17:48Z -> 21:59:05Z` で、実 live open (`09:06Z`) を含んでいない。
  - その状態で direct replay は `trades=0` のままだが、今回追加した `summary.coverage` は
    - `tick_count=70490`
    - `live_trade_overlap.overlap_count=0`
    - `live_trade_overlap.total_strategy_trades=2`
    を返し、「実 trade はあるが replay 窓に乗っていない」ことを明示した。
  - `replay_exit_workers_groups.py` 側の `summary_all.json` にも `entry_replay.summary.coverage` を保持し、
    `base_scenarios.all.selection.requested=0` の原因を同じファイル内で辿れるようにした。

- 対応:
  - `scripts/replay_workers.py`
    - `summary.coverage` を追加し、tick 窓 (`tick_start/end/count/span`) を常時出力。
    - M1 family は `logs/trades.db` と照合して `live_trade_overlap` を返す。
  - `scripts/replay_exit_workers_groups.py`
    - `summary_all.json` に `entry_replay.summary.coverage` を保持。
  - `docs/REPLAY_STANDARD.md`
    - replay 判定前に `summary.coverage` を見る運用ルールを追記。

- 判断:
  - `20260306` の `TrendBreakout replay=0 trades` は、現時点では strategy quality の根拠にならない。
  - 以後、M1 family の replay 結果は `coverage.live_trade_overlap.overlap_count > 0` を満たした窓だけを live RCA / 昇格判断の根拠に使う。
  - exact replay 優先窓は以下。
    - `TrendBreakout`: `2026-03-06 09:05-09:21 UTC`
    - `PullbackContinuation`: `2026-03-05 12:59-13:40 UTC`
    - `FailedBreakReverse`: `2026-03-05 14:48-15:58 UTC`
    ただし現ローカルには該当 `20260305/06` tick が未揃い。

## 2026-03-07 03:34 UTC / 2026-03-07 12:34 JST - replay: live trade から exact window を自動監査する wrapper を追加

- 対象:
  - `scripts/replay_live_window_audit.py`
  - `tests/scripts/test_replay_live_window_audit.py`
  - `tmp/replay_live_window_audit_m1_family_auto/report.json`

- 目的:
  - `trades.db` から live trade 時刻を引き、exact replay 窓を自動生成する。
  - 手元の tick assets に coverage がある窓だけ標準 replay を回し、missing 窓は必要日付付きで report 化する。

- 実装:
  - worker ごとの canonical `strategy_tag` を使って `trades.db` を参照。
  - `pre_minutes` / `post_minutes` で replay 窓を作成し、重なる窓は merge。
  - 既定 tick globs は `logs/replay`, `logs/archive/replay.*.dir`, `tmp`, `tmp/vm_ticks` を自動探索。
  - 各 window に `required_tick_basenames` を出し、covered 時だけ clipped tick JSONL を書く。
  - `--run-replay` で `scripts/replay_exit_workers_groups.py --no-hard-sl --exclude-end-of-replay` を窓ごとに起動可能。

- テスト:
  - `python -m py_compile scripts/replay_live_window_audit.py tests/scripts/test_replay_live_window_audit.py`
    - pass
  - `pytest -q tests/scripts/test_replay_live_window_audit.py`
    - `4 passed`

- 実行結果（default globs, local logs）:
  - `python scripts/replay_live_window_audit.py --workers trend_breakout,pullback_continuation,failed_break_reverse --trades-db logs/trades.db --pre-minutes 5 --post-minutes 15 --out-dir tmp/replay_live_window_audit_m1_family_auto`
    - `TrendBreakout`: `live_trade_count=2`, `window_count=1`
      - window=`2026-03-06T09:01:51Z -> 2026-03-06T09:28:13Z`
      - `coverage.status=missing`
      - `required_tick_basenames=['USD_JPY_ticks_20260306.jsonl']`
    - `PullbackContinuation`: `live_trade_count=0`, `window_count=0`
    - `FailedBreakReverse`: `live_trade_count=0`, `window_count=0`

- 判断:
  - 現ローカルで exact replay を前に進めるボトルネックは、依然として `USD_JPY_ticks_20260306.jsonl` の該当 UTC 窓不足。
  - ただし今後は、missing を人手で推定せず `report.json` から直ちに判定できる。

## 2026-03-07 11:45 JST / local-v2週末仕込み: dynamic alloc を pocket 協調化し、loader の 0.45 floor バグを修正

- 市況確認:
  - `logs/orderbook_snapshot.json`: `bid=157.790 / ask=157.853 / spread=6.3p`
  - tick age は `15000s` 超で土曜クローズ帯
  - 週明け前のため live restart は実施せず、offline 仕込みのみ実施

- 事実:
  - 7d pocket 集計では `micro=+1313.5 JPY / PF(pips)=1.209` が唯一プラス。
  - `scalp=-5908.1 JPY / PF(pips)=0.592`、`scalp_fast=-7311.7 JPY / PF(pips)=0.348` で、pocket 間の強弱差が大きい。
  - 既存 `scripts/dynamic_alloc_worker.py` は strategy ごとの `lot_multiplier` を出していたが、
    pocket 全体の勝ち負けは multiplier に入っていなかった。
  - さらに `workers/common/dynamic_alloc.py` が `allocation_policy.min_lot_multiplier=0.45` を item にも強制しており、
    `config/dynamic_alloc.json` 上で `M1Scalper-M1=0.10`, `scalp_ping_5s_flow_live=0.218` を出しても loader では `0.45` に丸められていた。

- 対応:
  - `scripts/dynamic_alloc_worker.py`
    - strategy tag 正規化に `trendbreakout -> TrendBreakout` を追加し、重複集計を解消
    - pocket ごとの weighted score / PF / realized JPY を計算し、`pocket_profiles` と `pocket_caps` を出力
    - 最終 `lot_multiplier` を `strategy_lot_multiplier x pocket_lot_multiplier` に変更
    - ただし pocket loser 内の winner（例: `TrendBreakout`）は pocket penalty を受けないよう保護
  - `workers/common/dynamic_alloc.py`
    - item が見つかったときは policy の `min_lot_multiplier` を強制せず、
      `effective_min_lot_multiplier` か item 固有 `min_lot_multiplier` のみを下限として使うよう修正

- 週明け向けにできた状態:
  - `config/dynamic_alloc.json`
    - `pocket_profiles.micro.lot_multiplier=1.14`
    - `pocket_profiles.scalp=0.68`
    - `pocket_profiles.scalp_fast=0.68`
    - `MomentumBurst lot_multiplier=0.969`
    - `MicroLevelReactor lot_multiplier=0.902`
    - `TrendBreakout lot_multiplier=1.000`（loser pocket penalty を neutralize）
    - `M1Scalper-M1 lot_multiplier=0.100`
    - `scalp_ping_5s_flow_live lot_multiplier=0.218`

- 検証:
  - `python3 -m py_compile scripts/dynamic_alloc_worker.py workers/common/dynamic_alloc.py`
  - `python3 scripts/dynamic_alloc_worker.py --lookback-days 7 --limit 0 --min-trades 12 --pf-cap 2.0 --target-use 0.88 --half-life-hours 36 --min-lot-multiplier 0.45 --max-lot-multiplier 1.65 --soft-participation 1 --allow-loser-block 0 --allow-winner-only 0`
  - `python3 - <<'PY' ... load_strategy_profile(...) ... PY`
    - `MomentumBurst=0.969`, `MicroLevelReactor=0.902`, `M1Scalper-M1=0.100`, `TrendBreakout=1.000`, `scalp_ping_5s_flow_live=0.218`

- 狙い:
  - 週明けに worker 群が同じ `dynamic_alloc.json` を読み、勝っている `micro` へ自動で厚く、負けている `scalp/scalp_fast` を自動で薄くする。
  - loser の縮小が loader で無効化されていた状態を解消し、`M1Scalper-M1` / `flow` の drag を本当に落とす。

## 2026-03-07 11:44 JST / local-v2週末仕込み: `scalp_ping_5s_flow_live` の thin-edge / tail-loss を圧縮

- 市況確認:
  - `logs/pdca_profitability_latest.md`（2026-03-07 11:44:50 JST）
    - `USD/JPY mid=157.8215 / spread=6.3p`
    - `open_trades=0`
  - `logs/oanda_account_snapshot_live.json`
    - `2026-03-07 11:08:15 JST` に更新
  - 土曜クローズ帯のため、live fill / PF の即時再評価は保留し、offline 実装とローカル反映まで進める。

- 事実:
  - `logs/pdca_profitability_latest.md`
    - 24h loser: `scalp_ping_5s_flow_live -1768.257 JPY / 97 trades / win 18.6% / PF 0.22`
    - 7d loser: `scalp_ping_5s_flow_live -7000.614 JPY / 414 trades / win 31.9% / PF 0.33`
  - `logs/trades.db` 直近72h:
    - `scalp_ping_5s_flow_live` は `avg tp_pips=1.416`, `avg sl_pips=1.260`
    - `TAKE_PROFIT_ORDER 125 / avg +1.462p`
    - `STOP_LOSS_ORDER 228 / avg -1.414p`
    - `MARKET_ORDER_TRADE_CLOSE 65 / avg -3.829p`
  - `logs/orders.db` では `close_request.exit_reason=max_adverse` の deep loser が残り、
    worst trade は `ticket_id=428528 / -17.6p / MARKET_ORDER_TRADE_CLOSE`。
  - 一方 `M1Scalper-M1` は現行 allowlist
    (`breakout-retest-long,nwave-long,vshape-rebound-long`) に限ると、
    直近72h `34 trades / avg +1.647p / avg hold 145.7s` でプラス。

- 対応:
  - `workers/scalp_ping_5s_flow/exit_worker.py`
    - `entry_thesis.force_exit_max_hold_sec`
      / `entry_thesis.force_exit_max_floating_loss_pips` を読み、
      flow trade では `reentry` / `direction_flip` より前に
      `time_stop` / `max_adverse` を強制できるよう修正。
  - `config/strategy_exit_protections.yaml`
    - `scalp_ping_5s_flow(_live)` を `B/C` 共用 profile から分離。
    - fallback の `loss_cut_hard_pips=1.8`, `loss_cut_max_hold_sec=120`,
      `non_range_max_hold_sec=90`, `direction_flip` 早期化へ更新。
  - `ops/env/scalp_ping_5s_flow.env`
    - `SCALP_PING_5S_FLOW_LOOKAHEAD_ALLOW_THIN_EDGE=0`
    - `SCALP_PING_5S_FLOW_LOOKAHEAD_EDGE_HARD_REJECT_PIPS=0.18`
    - `SCALP_PING_5S_FLOW_SIGNAL_WINDOW_ADAPTIVE_LIVE_SCORE_MIN_PIPS=0.08`
  - `ops/env/quant-scalp-ping-5s-flow-exit.env`
    - fallback loss-cut を `2.2p / 120s / cooldown 3s` へ圧縮。

- 検証:
  - `python3 -m py_compile workers/scalp_ping_5s_flow/exit_worker.py`
    - pass
  - `python3 - <<'PY' ... yaml.safe_load('config/strategy_exit_protections.yaml') ... PY`
    - pass

- 判断:
  - 主因は執行レイテンシではなく、`flow` の thin-edge entry と
    `MARKET_ORDER_TRADE_CLOSE` 側の tail-loss 放置だった。
  - `M1Scalper-M1` は現行 allowlist 部分がすでに正なので、
    今回は `flow` を優先し、M1 追加改修は週明けの実測再評価後に限定する。

- 週明け再検証条件:
  1. `logs/orders.db` で `scalp_ping_5s_flow_live close_request(exit_reason=max_adverse|time_stop)` の深い tail が減ること。
  2. `logs/trades.db` で `scalp_ping_5s_flow_live MARKET_ORDER_TRADE_CLOSE` 平均が `-3.829p` から改善すること。
  3. `logs/pdca_profitability_latest.md` で `scalp_ping_5s_flow_live` の 24h `PF / avg_pips / net_jpy` が改善すること。

## 2026-03-07 20:55 JST / local-v2週末検証: `TrendBreakout` live-window replay の主因は exact tick 欠損だけでなく warmup 欠損

- 市況確認:
  - `logs/orderbook_snapshot.json`
    - spread `6.3p`, latency stale
  - `logs/health_snapshot.json`
    - last close `2026-03-06T21:40:06Z`
    - `orders_status_1h=[]`
  - 土曜クローズ帯のため、live 変更は行わず replay/実装だけ進めた。

- 事実:
  - exact live 窓 `2026-03-06 09:01:51 -> 09:28:13 UTC` を
    `--allow-candle-sim-fallback` だけで replay すると、
    overlap は `2 trades` まで取れても `entry_replay.trades=0`。
  - 同じ窓に `--replay-warmup-minutes 120` を付けると、
    `tmp/replay_live_window_audit_trend_breakout_fallback_warm120/report.json`
    で `coverage.status=candle_simulated`、
    `summary_all.json` で `entry_replay.trades=2 / total_pnl_pips=14.041 / win_rate=1.0`。
  - 実 replay の first trade は
    `2026-03-06T09:06:10.721Z / tag=M1Scalper-breakout-retest-long` で、
    live の `09:06:51 / 09:06:58 UTC` 2本に十分近い。
  - `sim/pseudo_ticks.py` は OANDA S5 candle の nanosecond timestamp
    (`...000000000Z`) を受けられるよう修正済み。

- 判断:
  - `TrendBreakout` の false zero は「exact live 窓と重なっていない」ことではなく、
    replay が cold-start して `factor_cache` と breakout-retest 文脈を作れないことが主因だった。
  - M1 family の live-window audit は、
    exact coverage 監査と replay 用 warmup を分けて扱う必要がある。
  - 120分 warmup は今回の `TrendBreakout` では十分で、0 trades 問題を解消した。

- 運用ルール:
  - `TrendBreakout` / `PullbackContinuation` / `FailedBreakReverse` の replay では
    `scripts/replay_live_window_audit.py` の worker 既定 warmup `120m` を標準にする。
    明示 override が必要なときだけ `--replay-warmup-minutes` を渡す。
  - exact tick が欠けている場合も、まず `--allow-candle-sim-fallback` を併用して
    warmup 付き replay まで回し、0 trades を即「戦略不発」とは判断しない。
  - `summary_all.json` の `entry_replay.summary.factor_readiness` と
    `last_reject_sample` を先に見れば、
    warmup 不足なのか tag filter なのかを replay 出力だけで切り分けられる。

## 2026-03-07 21:10 JST / local-v2: 閉ループ断線の主因は analysis timer 未接続

- 市況確認:
  - 実行時点は 2026-03-07（土）JST。OANDA `/pricing` は `tradeable=false`。
  - `USD/JPY` last quote は `2026-03-06T21:59:05Z`、`closeout spread=7.7p`。
  - `logs/orderbook_snapshot.json` / `tick_cache` は約 50,531 秒 stale、`spread_state.spread_pips=6.3`。
  - live 市場は閉場なので、新規売買の live 検証は保留し、offline 実装/検証のみ進めた。

- 事実:
  - core V2 は稼働中:
    - `quant-market-data-feed`, `quant-strategy-control`,
      `quant-order-manager`, `quant-position-manager` は `running`。
  - 直近 24h 実績:
    - `trades_count_24h=344`
    - `data_lag_ms avg=1419.271`, `decision_latency_ms avg=27.446`,
      `reject_rate avg=0.022`
    - 赤字寄与上位は `scalp_ping_5s_flow_live -5593.35 JPY / PF 0.193`,
      `M1Scalper-M1 -1395.52 JPY / PF 0.629`
  - ただし local feedback artifacts は stale:
    - `logs/strategy_feedback.json` mtime = 2026-03-04
    - `logs/trade_counterfactual_latest.json` mtime = 2026-03-04
    - `config/pattern_book*.json` mtime = 2026-03-05
    - `config/dynamic_alloc.json` だけ 2026-03-07 更新
  - `scripts/install_local_v2_launchd.sh` / `local_v2_autorecover_once.sh` は
    売買 worker の復旧は行うが、
    `quant-dynamic-alloc.timer` / `quant-pattern-book.timer` /
    `quant-strategy-feedback.timer` / `quant-trade-counterfactual.timer`
    相当をローカル常駐導線で起動していなかった。

- 判断:
  - 「予測・分析・売買・事後反省・次回反映」の閉ループ不全は、
    strategy 実装単体よりも、local watchdog が analysis workers を定期実行していない
    運用導線の断線が主因。
  - `strategy_entry` が使う `analysis.strategy_feedback.current_advice()` は
    `logs/strategy_feedback.json` に依存し、
    `dynamic_alloc` / `pattern_gate` も file-based reload 前提なので、
    artifacts 更新が止まると学習結果が次回 entry に戻らない。

- 対応:
  - `scripts/run_local_feedback_cycle.py` を追加し、
    `dynamic_alloc / pattern_book / strategy_feedback / trade_counterfactual`
    を interval 管理付きで実行する local cycle を実装。
  - `scripts/local_v2_autorecover_once.sh` に
    `run_feedback_cycle_async()` を追加し、
    stack 健全時/復旧時に上記 cycle を非同期起動するよう修正。
  - 最新/履歴/個別ログ:
    - `logs/local_feedback_cycle_latest.json`
    - `logs/local_feedback_cycle_history.jsonl`
    - `logs/local_feedback_cycle/*.log`

- 再検証条件:
  1. `python3 scripts/run_local_feedback_cycle.py --force` 後に
     `logs/strategy_feedback.json`, `config/dynamic_alloc.json`,
     `config/pattern_book_deep.json`, `logs/trade_counterfactual_latest.json`
     の mtime が更新されること。
  2. 週明け market open 後、`strategy_entry` 経路で
     `analysis_feedback_*`, `dynamic_alloc_*`, `pattern_*` の
     reject/trim/scale が再び新しい artifact に追随していること。
  3. `logs/local_feedback_cycle_history.jsonl` に
     interval 実行が蓄積し、stale 再発が無いこと。

## 2026-03-07 21:20 JST / local-v2週末仕込み: `strategy_feedback` のローカル閉ループ欠損を修正

- 市況確認:
  - `logs/tick_cache.json`
    - 最終 tick は `2026-03-07 06:59 JST` 近辺で停止
  - `logs/factor_cache.json`
    - `timestamp=2026-03-06T21:59:00Z`, `USD/JPY close=157.822`, `spread ~= 0.8p`, `M1 atr_pips=1.82`
  - `logs/health_snapshot.json`
    - `trades_last_close=2026-03-06T21:40:06Z`, `orders_status_1h=[]`
  - 2026-03-07 21:20 JST は土曜クローズ帯のため、live fill 検証は保留し、ローカル実装と常駐分析導線の復旧に限定。

- 事実:
  - `logs/strategy_feedback.json` は `updated_at=2026-03-04T03:50:24Z` かつ `strategies={}` のまま stale。
  - `execution/strategy_entry.py` は `analysis/strategy_feedback.current_advice()` で同 JSON を読むだけで、
    ローカルV2側に timer 相当が無いと次回エントリーへ反映されない。
  - `python3 -m analysis.strategy_feedback_worker` を local clone で実行しても、
    systemd unit の `EnvironmentFile=/home/tossaki/QuantRabbit/...` をそのまま読み、
    mac の repo clone path と不一致のため env 解決に失敗し `0 strategies` を出力していた。

- 修正:
  - `analysis/strategy_feedback_worker.py`
    - Linux 固定 repo path を local clone path へ再解決。
    - `logs/local_v2_stack/pids/*.pid` を読んで non-systemd 環境でも running services を判定。
    - `STRATEGY_FEEDBACK_LOOP_SEC` を追加し、oneshot worker を local_v2 では常駐 loop に切替可能化。
  - `scripts/local_v2_stack.sh`
    - `quant-strategy-feedback` を local stack managed service として追加し、
      `trade_min` / `trade_all` で起動対象化。
  - `ops/env/local-v2-stack.env`
    - `STRATEGY_FEEDBACK_LOOP_SEC=600` を追加。

- オフライン検証:
  - `pytest -q tests/analysis/test_strategy_feedback_worker.py`
    - local-v2 pid + Linux 固定 EnvironmentFile を模擬し、
      `scalp_ping_5s_b_live` の feedback が生成されることを確認。
  - 実機ローカル:
    - `scripts/local_v2_stack.sh up --services quant-strategy-feedback --env ops/env/local-v2-stack.env`
    - `logs/strategy_feedback.json` が non-empty strategies を持って更新されることを確認する。

- 期待効果:
  - local-v2 でも `trades.db -> strategy_feedback.json -> strategy_entry` の直接ループが常時回る。
  - stale/empty feedback で entry quality が固定化する経路を解消し、勝ち/負けの反映を 10 分粒度で次回トレードへ戻せる。

## 2026-03-07 21:27 JST / local-v2週末仕込み: `counterfactual -> reentry` の未接続と launchd path ノイズを修正

- 市況確認:
  - `OANDA /pricing` は `tradeable=false`、最終 quote は `2026-03-06T21:59:05Z`
  - `USD/JPY` は `157.783/157.860`、spread は約 `7.7 pips`
  - `logs/factor_cache.json` の `M1 ATR ~= 1.82 pips`
  - 週末クローズ帯のため、改善対象は live 発注ではなく local PDCA 導線の閉ループ化に限定

- 観測:
  - `logs/local_feedback_cycle_latest.json` は `dynamic_alloc / pattern_book / trade_counterfactual` までは更新していたが、
    `replay_quality_gate` が cycle 対象外で `config/worker_reentry.yaml` へ戻る線が切れていた。
  - `logs/trade_counterfactual_latest.json` の `top_close_reasons` は live 実績に `TAKE_PROFIT_ORDER` /
    `STOP_LOSS_ORDER` / `MARKET_ORDER_TRADE_CLOSE` があるのに `unknown` 偏重だった。
  - `logs/local_v2_autorecover.launchd.err` には `~/Documents/...` 経由の
    `Operation not permitted` / `getcwd` ノイズが混在し、launchd 常駐の安定性を落としていた。

- 修正:
  - `scripts/run_local_feedback_cycle.py`
    - `replay_quality_gate` を既定ON job として追加。
    - `ops/env/quant-replay-quality-gate.env` を読ませ、
      closed 帯で `trade_counterfactual -> replay_quality_gate -> worker_reentry.yaml`
      の auto-improve を回すよう変更。
  - `analysis/trade_counterfactual_worker.py`
    - live trade 読み込みで `close_reason` を保持し、
      `reason="unknown"` 固定を解消。
    - `orders.db` 読み取りが一時失敗しても spread なしで継続し、
      counterfactual 全体を落とさない fail-open へ変更。
  - `scripts/install_local_v2_launchd.sh`
    - plist に書く stack/env path を canonical path 化。
    - `bash -lc` を `bash -c` へ変更し、`WorkingDirectory=/` を明示。

- 検証:
  - `bash -n scripts/install_local_v2_launchd.sh`
  - `pytest -q tests/scripts/test_run_local_feedback_cycle.py tests/analysis/test_trade_counterfactual_worker.py`
  - `scripts/status_local_v2_launchd.sh`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager,quant-strategy-feedback`

- 効果:
  - local でも replay/analysis の改善が `worker_reentry.yaml` 経由で次回トレードへ戻る。
  - counterfactual の失敗理由が `unknown` 汚染から改善し、reentry 改善の根拠が実測 close reason と整合する。
  - `orders.db` 読み取り揺れで `trade_counterfactual` が丸ごと停止する頻度を下げ、
    closed 帯の自動改善ループを維持しやすくなる。
  - launchd 自動復帰の path/cwd ノイズを下げ、週明け以降の常駐性を強化する。

## 2026-03-07 21:40 JST / local-v2 watchdog の symlink 起点を修正して feedback-cycle 断続停止を解消

- 市況確認:
  - 2026-03-07 土曜クローズ帯。`tradeable=false`、週末 spread 拡大のため live fill 検証は保留。
  - 修正対象はローカルV2の常駐/学習導線に限定。

- 事実:
  - `scripts/status_local_v2_launchd.sh` では LaunchAgent が `running` 表示でも、
    plist の `ProgramArguments` は `/Users/tossaki/Documents/App/QuantRabbit/...` を参照していた。
  - `logs/local_v2_autorecover.launchd.err` には
    `bash: /Users/tossaki/Documents/App/QuantRabbit/scripts/local_v2_autorecover_once.sh: Operation not permitted`
    が連続しており、watchdog one-shot が Documents symlink 経由で不安定化していた。
  - その結果、`run_local_feedback_cycle.py` を繋いでも
    `dynamic_alloc / pattern_book / trade_counterfactual` の定期実行が launchd 側で断続停止し、
    「分析結果を次回 entry へ返す」閉ループが再び途切れる余地が残っていた。

- 修正:
  - `scripts/local_v2_stack.sh`
  - `scripts/local_v2_autorecover_once.sh`
  - `scripts/install_local_v2_launchd.sh`
    - repo root を `pwd -P` で物理パスへ正規化するよう修正。
  - `scripts/status_local_v2_launchd.sh`
    - plist が `Documents/App/QuantRabbit` を参照していたら警告を返すよう追加。
  - `docs/OPS_LOCAL_RUNBOOK.md`
  - `docs/ARCHITECTURE.md`
    - local launchd/watchdog は物理パス固定が前提であることを明文化。

- 検証:
  - `bash -n scripts/local_v2_stack.sh scripts/local_v2_autorecover_once.sh scripts/install_local_v2_launchd.sh scripts/status_local_v2_launchd.sh`
  - 修正後に `scripts/install_local_v2_launchd.sh --profile trade_min --env ops/env/local-v2-stack.env` を再実行し、
    `scripts/status_local_v2_launchd.sh` で symlink 警告が消えること。
  - `logs/local_v2_autorecover.launchd.err` に新しい `Operation not permitted` が追記されないこと。

- 再検証条件:
  1. `launchctl print gui/${UID}/com.quantrabbit.local-v2-autorecover` の `ProgramArguments` が
     `/Users/tossaki/App/QuantRabbit/...` を指すこと。
  2. 週明け market open 中に `logs/local_feedback_cycle_history.jsonl` が継続更新されること。
  3. `logs/strategy_feedback.json`, `config/dynamic_alloc.json`,
     `config/pattern_book_deep.json`, `logs/trade_counterfactual_latest.json`
     の mtime が watchdog 周期に追随して stale 化しないこと。

## 2026-03-07 21:55 JST / `replay_quality_gate` の入力不足を soft-skip 化して local feedback cycle の誤警報を解消

- 市況確認:
  - 土曜クローズ帯で replay 用 tick file が 1 本しかなく、walk-forward の `train=2 test=1` 条件を満たさない。
  - closed帯の replay 学習は続けるが、入力不足そのものは常駐導線の障害とは分離して扱う。

- 事実:
  - `logs/replay_quality_gate_latest.json` は `returncode=2` かつ
    `stderr_tail=Insufficient tick files for walk-forward: files=1 train=2 test=1`。
  - 一方で worker wrapper は過去 run
    `tmp/replay_quality_gate/20260225_032552/quality_gate_report.json`
    を拾って `gate_status=pass` を書いており、
    `run_local_feedback_cycle.py` 側では job 全体が `error` 扱いになっていた。
  - つまり「入力不足なのに stale report を pass として見せつつ、終了コードだけ error」という
    観測上の齟齬があった。

- 修正:
  - `analysis/replay_quality_gate_worker.py`
    - `Insufficient tick files` / `No tick files matched` /
      `No tick files after min_tick_lines filter` を soft-skip として分類。
    - 非0終了時に fresh report が無ければ stale report を再利用しない。
    - worker の戻り値は 0 とし、state には `upstream_returncode=2`,
      `gate_status=skipped`, `soft_skip_reason=...` を残す。
  - `tests/analysis/test_replay_quality_gate_worker.py`
    - stale report が存在しても soft-skip になり、`report_json_path` が空になることを追加検証。
  - `docs/OPS_LOCAL_RUNBOOK.md`
  - `docs/ARCHITECTURE.md`
    - replay入力不足時の soft-skip 方針を追記。

- 検証:
  - `pytest -q tests/analysis/test_replay_quality_gate_worker.py`
  - `logs/local_feedback_cycle_latest.json` で `replay_quality_gate.status=skipped` かつ
    全体 `status` が不要に `error` へ落ちないこと。

- 再検証条件:
  1. tick file が不足している closed帯では `replay_quality_gate` が `skipped` になること。
  2. tick file が十分な closed帯では fresh report を生成し、`report_json_path` が当該 run を指すこと。
## 2026-03-07 23:06 JST / local read-only MCP は無駄ではないが、観測導線として使うには軽量化と fail-fast が必要

- 事実:
  - `scripts/mcp_sqlite_readonly.py --db logs/orders.db` に JSON-RPC で
    `initialize -> tools/call(query)` を送ると、
    `entry_intent_board / orders / sqlite_sequence` を正常返却できた。
  - 一方で修正前は `notifications/initialized` に対しても
    `Unknown method` を返しており、MCP クライアントとの相性が悪かった。
  - さらに SQLite 側は `max_rows` があっても内部で `fetchall()` 後に slice していたため、
    `logs/orders.db` / `logs/trades.db` の大きい table に LIMIT 無しで触れると無駄が大きかった。
  - `scripts/mcp_oanda_observer.py --readonly` は修正前、
    current shell に `OANDA_ACCOUNT_ID` 等が無いと
    `initialize` 後の `tools/call(summary)` が
    `startup config error: Environment variable oanda_account_id is not set`
    で止まり、repo 標準の `config/env.toml` 解決系に追従していなかった。
  - 2026-03-07 23:00-23:06 JST 時点の OANDA pricing 実測は
    `tradeable=false`, bid/ask=`157.790/157.853`, spread=`6.3 pips`,
    `M5 ATR14 ≒ 5.03 pips`。週末クローズ帯で通常執行条件ではない。

- 判断:
  - MCP 自体は「`logs/*.db` + OANDA 観測を read-only で同じ作法から触れる」点で有効。
  - ただし、観測専用ツールとして残すなら
    1) SQLite の返却上限を runtime でも厳守すること
    2) MCP notification に無応答で追従すること
    3) OANDA 引数を fail-fast 検証すること
    4) OANDA 資格情報を `utils.secrets.get_secret()` に寄せること
    が必須だった。

- 対応:
  - `mcp_sqlite_readonly.py` を `fetchmany()` + `truncated` 返却へ変更。
  - `mcp_sqlite_readonly.py` / `mcp_oanda_observer.py` とも、
    notification の無応答処理と invalid call の fail-fast を追加した。
  - `mcp_oanda_observer.py` は `utils.secrets.get_secret()` を使うよう修正し、
    current shell でも `initialize -> tools/call(summary)` が成功することを確認した。

## 2026-03-09 09:15 UTC / 2026-03-09 18:15 JST - 直近毀損の主因は trend 相場での逆張り再開。`MicroLevelReactor` gate を再度締め、`scalp_extrema_reversal_live` に trend 継続ガードを追加

- 市況確認（ローカルV2実測 + OANDA API）:
  - `2026-03-09 09:09 JST` OANDA snapshot:
    - `USD/JPY bid=158.444 ask=158.452 mid=158.448 spread=0.8p`
    - `M5 ATR14=8.7p`, 直近30本 change `+35.9p`
    - `openTrades=1`, `margin_avail=1,139.52 JPY`
  - `2026-03-09 09:14 JST` factor 実測:
    - `M1 adx=17.11`, `ma_gap=1.375p`, `range_score=0.264`, `range_mode=TREND`, `range_active=false`
  - 6h health:
    - `data_lag_ms avg=902.7`, `decision_latency_ms avg=15.7`

- 事実:
  - 24h 集計は `63 trades / win=42.9% / PF=0.253 / net_jpy=-757.27 / net_pips=-26.3`。
  - 損失は `00 UTC` に集中し、`6 trades / -513.88 JPY`。
  - `MicroLevelReactor`
    - `2026-03-09 09:06 JST` に `fade-upper` short を送信し、
      log では `range=0.25`, `pf=0.95`, `win=0.46`, `units=-3865`。
    - 直近6h 集計は `50 trades / -129.62 JPY`。
  - `scalp_extrema_reversal_live`
    - 直近 `6 trades / 0 wins / -7.42 JPY / avg_pips=-2.3`
    - `2026-03-09 09:05 JST` 前後に上昇継続中の short を連発。
  - `free_margin_ratio=1.0` 混入疑惑は false alarm:
    - `2026-03-09 09:13 JST` 再確認時点で OANDA summary / `utils.oanda_account` / `metrics.db`
      はすべて `marginUsed=0 / free_margin_ratio=1.0 / openTradeCount=0` で整合。

- 対応:
  - `ops/env/quant-micro-levelreactor.env`
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MicroLevelReactor:1.60 -> 1.35`
    - `MICRO_MULTI_MLR_MIN_RANGE_SCORE=0.05 -> 0.30`
    - `MICRO_MULTI_MLR_MAX_ADX=36.0 -> 24.0`
    - `MICRO_MULTI_MLR_MAX_MA_GAP_PIPS=6.5 -> 2.8`
  - `workers/scalp_extrema_reversal/worker.py`
    - `range_score / range_active / ma_gap / adx` を使う trend continuation guard を追加。
    - `_place_order()` が actual `free_margin_ratio` 取得前に `compute_cap()` していた順序も修正。

- 判断:
  - 現時点の悪化は「導線停止」ではなく、「trend 相場に range/reversal が再び広がった」ことが主因。
  - 2026-03-07 に winner 回復目的で緩めた `MicroLevelReactor` dedicated gate は、
    2026-03-09 の `range_mode=TREND` では過剰だった。
  - `scalp_extrema_reversal_live` は極値 + 短期 reversal だけで通っていたため、
    trend continuation の局所ガードが必要。

- 再検証条件:
  1. 反映後 30-90 分で `orders.db` の `MicroLevelReactor` が weak-range short に偏らないこと。
  2. `scalp_extrema_reversal_live` の trend continuation 中の `filled` が減り、
     no-entry または reject に吸収されること。
  3. free margin が薄い局面で `scalp_extrema_reversal_live` の cap が actual `free_margin_ratio` に追随すること。

## 2026-03-09 09:42 UTC / 2026-03-09 18:42 JST - 改善確認では未回復。`scalp_extrema_reversal_live` の range-active 通過穴と `MomentumBurst` の dyn_alloc 上振れを追加修正

- 市況確認（ローカルV2実測 + OANDA API）:
  - `2026-03-09 18:40 JST` OANDA snapshot:
    - `USD/JPY bid=158.481 ask=158.489 mid=158.485 spread=0.8p`
    - `M5 ATR14=7.14p`, 直近30本 range `28.8p`, change `+10.5p`
    - `pricing=249ms`, `summary=216ms`, `open_trades=246ms`
  - local health:
    - core 4 service は `running`
    - `snapshot_age_sec=0`
    - `data_lag_ms avg=842.2`, `decision_latency_ms avg=16.0`

- 事実:
  - 24h/6h 集計は `65 trades / net_jpy=-755.63 / PF=0.255 / win_rate=43.1%`。
  - 直近反映後でも closed trade は `1 trade / net_jpy=-0.02 / PF=0.0` で、改善を示すサンプルはまだ出ていない。
  - `scalp_extrema_reversal_live`
    - `2026-03-09 09:40-09:42 JST` に short を 4 本連発し、4 本とも `STOP_LOSS_ORDER`。
    - 4 本とも `range_active=true / range_mode=RANGE / range_score=0.359-0.368 / ADX=9.65-9.74 / ma_gap=1.16-1.25p` で、`range_active` が gate の通過条件になっていた。
  - `MomentumBurst`
    - 24h の損失最大は `-516.44 JPY / 2 trades / 0 wins`。
    - 負け約定では `dynamic_alloc.lot_multiplier=1.65`、一方で同じ `entry_thesis.history_perf` は `pf=0.928 / lot_multiplier=0.625 / n=8`。
    - つまり recent regime が負け側でも、dyn_alloc boost が上書きしてサイズを増やしていた。

- 対応:
  - `workers/scalp_extrema_reversal/worker.py`
    - `range_mode=RANGE` では `range_active=true` を単独通過条件にしない。
    - `range_score` floor と `against_gap_pips` upper bound を追加し、
      weak-range + continuation drift の short/long を block する。
  - `ops/env/quant-scalp-extrema-reversal.env`
    - `COOLDOWN_SEC=30 -> 120`
    - `MAX_OPEN_TRADES=3 -> 1`
    - `CAP_MAX=0.95 -> 0.70`
    - `TREND_GATE_RANGE_SCORE_MIN=0.40`
    - `TREND_GATE_RANGE_MAX_AGAINST_GAP_PIPS=1.00`
  - `workers/micro_runtime/worker.py`
    - recent history が `pf<1` または `lot_multiplier<1` なら、
      `dyn_alloc` boost を `1.0x` まで clamp する。
  - `ops/env/quant-micro-momentumburst.env`
    - `MICRO_MULTI_BASE_UNITS=62000 -> 52000`
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MomentumBurst:1.25 -> 1.00`
    - `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=180`

- 判断:
  - 前回の改善で `MicroLevelReactor` の weak-range fade は抑えられたが、
    収益未回復の時点では `scalp_extrema_reversal_live` の連打と
    `MomentumBurst` の過大サイズが新しい主因になっていた。
  - 이번対応は停止ではなく、strategy-local の range質判定と size上限を実測に合わせて詰め直したもの。

- 再検証条件:
  1. `scalp_extrema_reversal_live` が `range_score < 0.40` かつ `against_gap_pips > 1.0` のとき `filled` ではなく no-entry になること。
  2. 同 worker が同時多発の short/long を出さず、`MAX_OPEN_TRADES=1` と `COOLDOWN_SEC=120` で 1 本ずつ処理されること。
  3. `MomentumBurst` の `entry_thesis.dynamic_alloc_clamped_by_history` が recent loser regime で記録され、units が以前より縮小すること。

## 2026-03-09 18:58 JST - 利益最大化の観点では「止血だけ」では足りない。winner 集中のため local-v2 override を追加

- 直近実績（`logs/trades.db` 7d/30d 集計）:
  - winner は実質 `MomentumBurst` のみ:
    - 7d `33 trades / +1340.50 JPY / PF 2.53 / win_rate 81.8%`
  - loser が損益を強く食っている:
    - `scalp_ping_5s_flow_live`: 7d `-7131.11 JPY / PF 0.295`
    - `M1Scalper-M1`: 7d `-6172.49 JPY / PF 0.555`
    - `scalp_ping_5s_b_live`: 7d `-185.40 JPY / PF 0.416`
  - `MomentumBurst` は直近2敗があっても 7d/30d とも winner を維持しているため、
    利益最大化の次手は「loser を薄くし、winner にだけ厚く寄せる」こと。

- 事実:
  - `local-v2-stack.env` では `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_REJECT_BELOW` を上げていたが、
    `SCALP_PING_5S_B_ENTRY_LEADING_PROFILE_ENABLED=1` が無く、閾値強化が実質未適用だった。
  - `scalp_ping_5s_flow_live` は leading profile 有効でも threshold がまだ低く、
    `BASE_ENTRY_UNITS=80` が loser PF と釣り合っていなかった。
  - `M1Scalper-M1` は long-only 化後も 7d `-6172 JPY` で、
    `BASE_UNITS=1200` と `DYN_ALLOC_MULT_MAX=1.75` がまだ強すぎる。

- 対応:
  - `ops/env/local-v2-stack.env`
    - `scalp_ping_5s_b_live`
      - `ENTRY_LEADING_PROFILE_ENABLED=1`
      - `REJECT_BELOW=0.88`, `REJECT_BELOW_SHORT=0.92`
      - `ENTRY_LEADING_PROFILE_BOOST_MAX=0.00`, `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.90`
      - `MAX_ACTIVE_TRADES=1`, `MAX_PER_DIRECTION=1`
      - `BASE_ENTRY_UNITS=10`
      - `LOOKAHEAD_EDGE_MIN_PIPS=0.55`
      - `LOOKAHEAD_UNITS_MAX_MULT=1.00`
    - `scalp_ping_5s_flow_live`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.72`, `...SHORT=0.80`
      - `ENTRY_LEADING_PROFILE_BOOST_MAX=0.00`, `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.90`
      - `BASE_ENTRY_UNITS=36`
      - `LOOKAHEAD_EDGE_HARD_REJECT_PIPS=0.30`
      - `LOOKAHEAD_UNITS_MAX_MULT=1.00`
    - `M1Scalper-M1`
      - `BASE_UNITS=400`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.58`
      - `ENTRY_LEADING_PROFILE_BOOST_MAX=0.00`, `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.90`
      - `DYN_ALLOC_MULT_MAX=1.05`
    - `micro`
      - `MICRO_MULTI_STRATEGY_UNITS_MULT=MomentumBurst:1.35,MicroTrendRetest:1.15,...loser strategies downweighted...`
      - `MomentumBurst` と `MicroTrendRetest-short` を相対増量し、
        `MicroLevelReactor / MicroRangeBreak / MicroPullbackEMA / MicroCompressionRevert / MicroVWAPRevert` を明確に減衰

- 判断:
  - 「爆速で増やす」ための実務上の意味は、回転を上げることではなく、
    gross profit を作る winner へロットを寄せ、gross loss を作る loser の回転とサイズを下げること。
  - 今回は stop ではなく `override` で厚み配分を寄せ直し、
    loser 側の `leading_profile/lookahead` 増量も封じた。

- 再検証条件:
  1. 24h で `scalp_ping_5s_flow_live` と `M1Scalper-M1` の gross loss 増加ペースが鈍ること。
  2. `scalp_ping_5s_b_live` の 約定数が明確に減り、`ENTRY_LEADING_PROFILE` が reject/scale として効くこと。
  3. `MomentumBurst` の units が micro 内相対でやや厚くなりつつ、`PF>1` を維持すること。

## 2026-03-09 19:20 JST - さらに利益速度を上げるため、micro は winner-only に戻し loser pockets をもう一段削る

- 直近事実:
  - `metrics.db` 24h では `order_success_rate=98.57%`, `reject_rate=1.43%`,
    `decision_latency_ms=16.46` と、今の主因は拒否や遅延ではなかった。
  - `orders.db` 直近 3000 event でも `rejected=18 (0.6%)` に対し、
    `filled=461`。主なエラーは `STOP_LOSS_ON_FILL_LOSS` と `503`。
  - つまり「もっと稼ぐ」ための優先度は、さらに loser の厚みを落として
    winner の signal selection を強制すること。

- 対応:
  - `ops/env/local-v2-stack.env`
    - `micro`
      - `MICRO_MULTI_DYN_ALLOC_WINNER_ONLY=1`
      - `MICRO_MULTI_DYN_ALLOC_WINNER_SCORE=0.55`
      - `MICRO_MULTI_STRATEGY_UNITS_MULT=MomentumBurst:1.60,MicroTrendRetest:1.30,...`
      - `MicroLevelReactor / MicroRangeBreak / MicroPullbackEMA / MicroCompressionRevert / MicroVWAPRevert`
        はさらに減衰
    - `scalp_ping_5s_b_live`
      - `BASE_ENTRY_UNITS=6`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW_SHORT=0.94`
      - `LOOKAHEAD_EDGE_MIN_PIPS=0.65`
      - `DYN_ALLOC_MULT_MAX=0.80`
    - `scalp_ping_5s_flow_live`
      - `BASE_ENTRY_UNITS=18`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.80`, `...SHORT=0.86`
      - `LOOKAHEAD_EDGE_HARD_REJECT_PIPS=0.40`
      - `DYN_ALLOC_MULT_MAX=0.55`
    - `M1Scalper-M1`
      - `BASE_UNITS=200`
      - `ENTRY_LEADING_PROFILE_REJECT_BELOW=0.64`
      - `ENTRY_LEADING_PROFILE_UNITS_MAX_MULT=0.80`
      - `DYN_ALLOC_MULT_MAX=0.90`

- 判断:
  - 直近7dの明確 winner は `MomentumBurst`、次点が `MicroTrendRetest-short`。
  - `MICRO_MULTI_DYN_ALLOC_WINNER_ONLY=1` を `score=0.55` で戻すことで、
    micro では winner 候補が存在する局面で loser micro へ signal slot を渡さない。
  - scalp_fast / M1 は「停止」ではなく、利益毀損が出る速度だけをさらに落とす。

## 2026-03-09 19:28 JST - `MicroLevelReactor` は long 側だけ残して再度利益側へ寄せる

- 直近実測:
  - `MicroLevelReactor` 7d 集計では
    - `OPEN_LONG`: `238 trades / +277.13 JPY / PF 1.27 / avg_abs_units 450.9`
    - `OPEN_SHORT`: `72 trades / -342.60 JPY / PF 0.379 / avg_abs_units 591.2`
  - つまり戦略全体が負けているのではなく、`OPEN_SHORT` が
    `OPEN_LONG` の利益を食っていた。

- 対応:
  - `ops/env/quant-micro-levelreactor.env`
    - `MICRO_MULTI_SIGNAL_TAG_CONTAINS=breakout-long,bounce-lower`
      を追加し、`OPEN_LONG` 系だけ通す。
  - `ops/env/local-v2-stack.env`
    - `MICRO_MULTI_DYN_ALLOC_LOSER_SCORE=0.20`
      として、`MicroLevelReactor` の strategy score `0.222` を loser block から外す。
    - `MICRO_MULTI_STRATEGY_UNITS_MULT` の `MicroLevelReactor` は `0.80` へ戻す。

- 判断:
  - margin headroom の都合で global size は上げず、
    short 側だけ切って long winner 側へ枠を戻すほうが合理的。

## 2026-03-09 02:02 UTC / 2026-03-09 11:02 JST - local-v2 の無建玉主因は `TrendBreakout` の tag mismatch と `RangeFader` の min-units floor

- 市況確認:
  - `logs/tick_cache.json` の直近 USD/JPY は `bid=158.655 / ask=158.663 / spread=0.8p`。
  - `logs/replay/USD_JPY/USD_JPY_M1_20260309.jsonl` 集計では `ATR14=2.21p`、直近60分レンジは `18.5p`。
  - `logs/health_snapshot.json` は `data_lag_ms=638.9`, `decision_latency_ms=15.5` と平常圏。
  - `logs/oanda_account_snapshot_live.json` / `logs/oanda_open_positions_live_USD_JPY.json` では `margin_used=0`、USD/JPY 建玉は `long=0 / short=0`。

- 実測:
  - `logs/strategy_control.db` と `logs/local_v2_stack/quant-strategy-control.log` は
    `global(entry=True, exit=True, lock=False)` で、共通 lock は主因ではなかった。
  - `logs/orders.db` の直近24hは `preflight_start=172`, `submit_attempt=101`, `filled=100`,
    `entry_probability_reject=13`。システム全体停止ではなく、特定 worker の入口が詰まっていた。
  - `TrendBreakout` は `logs/local_v2_stack/quant-scalp-trend-breakout.log` で
    `tag_filter_block tag=M1Scalper-trend-long`, `...sell-rally`, `...buy-dip` が継続。
    一方で `ops/env/local-v2-stack.env` の共通
    `M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest-long,nwave-long,vshape-rebound-long`
    が dedicated env より後勝ちし、`TrendBreakout` 専用の aperture を潰していた。
  - `logs/trades.db` の直近72hでは `TrendBreakout` は `2 trades / +179.18 JPY / avg +6.2p` で、
    winner 候補をタグ不一致で塞いでいた。
  - `RangeFader` の reject は `entry_probability_below_min_units` に集中し、
    `entry_probability=0.34-0.36`, `entry_units_intent=576-711`, `scaled_units=202-254`
    のケースが `ORDER_MIN_UNITS_STRATEGY_RANGEFADER*=120` に引っ掛かっていた。
  - `scalp_ping_5s_b/d` は同時点でも `pred 0.12-0.49p < cost 1.15-1.21p` の
    `edge_negative_block` が継続しており、ここを緩めるのは negative expectancy を増やすだけなので今回は触らない。

- 対応:
  - `ops/env/quant-scalp-trend-breakout.env`
    - `M1SCALP_SIGNAL_TAG_CONTAINS=breakout-retest,trend-long,trend-short,nwave-long,nwave-short`
      に更新し、trend continuation 系 signal を受けられるようにした。
    - `M1SCALP_ALLOW_REVERSION=0` は維持し、逆張り再versionまでは開けない。
  - `ops/env/local-v2-stack.env`
    - 共通 `M1SCALP_SIGNAL_TAG_CONTAINS` override を削除し、
      M1 family 各 worker が dedicated env の tag filter をそのまま使うよう戻した。
  - `ops/env/quant-order-manager.env`
    - `ORDER_MIN_UNITS_STRATEGY_RANGEFADER*` と alias の
      `ORDER_MIN_UNITS_STRATEGY_SCALP_RANGEFAD` を `60` へ統一。
    - これで probability 縮小後の `69-91 units` 帯を hard reject せず、
      `submit_attempt` へ流せる状態に戻す。

- 再検証条件:
  1. restart 後に `TrendBreakout` で `preflight_start -> submit_attempt -> filled` が再発すること。
  2. `RangeFader` の `entry_probability_below_min_units` が減少し、
     `submit_attempt/filled` が出ること。
  3. `data_lag_ms` / `decision_latency_ms` が引き続き平常圏を維持すること。

## 2026-03-09 06:20 UTC / 2026-03-09 15:20 JST - local-v2 の Brain safe canary が restart で脱落していたため、default env 合成へ戻す

- 市況確認:
  - OANDA 直照会の USD/JPY は `bid=158.470 / ask=158.478 / spread=0.8p`。
  - `M5 x24` の平均レンジは `8.60p`、直近14本 ATR proxy は `9.58p`、直近30分レンジは `12.4p`。
  - OANDA API は `pricing=260ms`, `summary=205ms`, `openTrades=281ms`, `candles=215ms` で全て `200`。
  - `logs/health_snapshot.json` は `data_lag_ms=694.8`, `decision_latency_ms=20.1`, `trades_count_24h=275`。

- 実測:
  - `logs/local_v2_stack/quant-order-manager.log` の 2026-03-09 14:59 / 15:08 / 15:13 JST 再起動では、
    env chain が `quant-v2-runtime.env -> quant-order-manager.env -> local-v2-stack.env` までで止まり、
    safe profile が読まれていなかった。
  - 実効設定も `BRAIN_ENABLED=0`, `ORDER_MANAGER_BRAIN_GATE_ENABLED=0` で、
    `logs/brain_state.db` の `brain_decisions` 直近24h は `0件`。
  - つまり週末に手動で入れた safe profile は、月曜の restart/watchdog 導線では持続していなかった。

- 対応:
  - `ops/env/local-v2-stack.env`
    - `LOCAL_V2_EXTRA_ENV_FILES=ops/env/profiles/brain-ollama-safe.env` を設定。
    - これで `scripts/local_v2_stack.sh --env ops/env/local-v2-stack.env` の restart と
      watchdog / launchd 復旧が同じ safe Brain canary を自動で合成する。
  - `docs/OPS_LOCAL_RUNBOOK.md`
    - local-v2 の既定が safe canary 追随であることを明記し、
      反映コマンド例を `--env ops/env/local-v2-stack.env` のみへ更新。

- 反映:
  - `python3 scripts/prepare_local_brain_canary.py --warmup`
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env --services quant-order-manager,quant-strategy-control`
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager`

- 検証結果:
  - `prepare_local_brain_canary.py --warmup` の 2026-03-09 15:26 JST 出力は
    `market_ready=true`, `quality_gate_ok=true`, `ollama_ready=true`, `enable_recommended=true`。
  - `quant-order-manager.log` の 2026-03-09 15:27 JST 最新起動では、
    env chain に `extra=/Users/tossaki/App/QuantRabbit/ops/env/profiles/brain-ollama-safe.env` が出て、
    effective env も `BRAIN_ENABLED=1`, `ORDER_MANAGER_BRAIN_GATE_ENABLED=1`,
    `BRAIN_OLLAMA_MODEL=qwen2.5:7b` へ戻った。
  - `quant-strategy-control.log` も同時刻の env chain で `extra=/Users/tossaki/App/QuantRabbit/ops/env/profiles/brain-ollama-safe.env` を読んでいる。
  - `local_v2_stack status` では core 4 がすべて `running`。
  - 反映後 2 分（15:27-15:29 JST）は新規 order が入らず、
    `orders.db` の `brain_shadow` / `brain_block` と `brain_state.db` の当日 `brain_decisions` はまだ未発火。
  - その後 15:30:03 JST に `orders.db` へ `scalp_fast / brain_shadow` が 1 件見えたが、
    process env では `BRAIN_POCKET_ALLOWLIST=micro` が入っていた。
    調査すると `workers/common/brain.py` は allowlist 外 pocket に対し
    `reason=disabled` の ALLOW を返し、`execution/order_manager.py` 側がそれまで
    `brain_shadow` と metric に記録していた。
    実 LLM 呼び出しではなく、shadow 監査のノイズだった。

- 追加対応:
  - `execution/order_manager.py`
    - `ORDER_MANAGER_BRAIN_GATE_MODE=shadow` でも、`brain_decision.reason=disabled`
      の場合は `brain_shadow` / `order_brain_shadow` を記録しないよう修正。
  - `tests/execution/test_order_manager_log_retry.py`
    - allowlist 外 pocket の `disabled` 決定が shadow ログへ漏れない回帰テストを追加。

- 次の確認点:
  1. 反映後最初の micro preflight で `orders.db` に `brain_shadow` が出ること。
  2. `brain_state.db` に 2026-03-09 JST の `brain_decisions` 行が再開すること。

## 2026-03-09 06:40 UTC / 2026-03-09 15:40 JST - `MomentumBurst` は shared sizing rebalance 後に正転したため、reaccel 条件だけを少し緩めて entry 数を戻す

- 市況確認:
  - `logs/orderbook_snapshot.json` の最新 best bid/ask は `158.566 / 158.574`、spread は `0.8p`、stream latency は `352.8ms`。
  - `logs/health_snapshot.json` は `generated_at=2026-03-09T06:19:55Z`, `data_lag_ms=860.9`, `decision_latency_ms=21.6`。
  - `logs/oanda_open_positions_live_USD_JPY.json` は `long_units=0`, `short_units=0` で flat。

- 実測:
  - `logs/trades.db` 集計では、shared sizing rebalance 後の直近2時間で
    `MicroLevelReactor 53 trades / +177.8 JPY / win_rate 73.6%`,
    `RangeFader 20 trades / +28.94 JPY / win_rate 100%`,
    `MomentumBurst 3 trades / +32.23 JPY / win_rate 66.7%`。
  - `MomentumBurst` は 24h ではまだ `5 trades / -484.21 JPY` だが、7d では `36 trades / +1372.73 JPY / win_rate 80.6%`。
  - よって loser を増やす shared gate 緩和ではなく、`MomentumBurst` の strategy-local reentry だけを少し増やすのが筋。

- 対応:
  - `strategies/micro/momentum_burst.py`
    - `REACCEL_EMA_DIST_PIPS=2.0`
    - `REACCEL_DI_GAP=6.0`
    - `REACCEL_ROC5_MIN=0.02`
  - `ops/env/quant-micro-momentumburst.env`
    - `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=90`
  - `tests/strategies/test_momentum_burst.py`
    - pullback 後の modest breakdown でも `OPEN_SHORT` が出る回帰ケースを追加。

- 反映後の確認点:
  1. `MomentumBurst` の `filled` が増えても shared micro gate の reject 理由が増えないこと。
  2. `MomentumBurst` の 2h/24h PnL が再び負側へ大きく崩れないこと。
  3. `MicroLevelReactor` / `RangeFader` の winner flow を食い潰さないこと。

## 2026-03-09 15:18-16:10 JST - `close_reject_no_negative` は主に duplicate close loop であり、RangeFader と `scalp_ping_5s_flow` の failed close retry を抑制する

- 市況確認:
  - `python3 scripts/pdca_profitability_report.py --instrument USD_JPY ...` の 2026-03-09 15:18 JST 時点で、
    USD/JPY は `158.438 / 158.446`、spread `0.8 pips`、OANDA pricing/summary/openTrades はすべて `~225-245ms`。
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services ...` では
    core/runtime と主要 strategy service は `running`。

- 実測:
  - 24h の `orders.db` では `close_reject_no_negative=4,984`、`7,308` 行中 `68.2%`。
  - その `4,984` 行は `18` trade に集中し、`RangeFader-neutral-fade 2957`、`RangeFader-sell-fade 1509`、`RangeFader-buy-fade 518`。
  - `18/18` trade は最終的に `close_ok` となり、最終 `close_reason=MARKET_ORDER_TRADE_CLOSE`、合計 `+28.3 pips / +29.7 JPY`。
  - 7d では `close_reject_no_negative=5,509`。大半は同じ duplicate close だが、
    `scalp_ping_5s_flow_live` だけは `3` trade (`424165`, `424436`, `424185`) が reject 後に負けへ悪化した。

- 原因:
  - `execution/order_manager.py` は negative close を `strategy_exit_protections` と reason allowlist で拒否する。
  - 一方で `workers/scalp_rangefader/exit_worker.py` と
    `workers/scalp_ping_5s_flow/exit_worker.py` は close が失敗しても local state を即座に破棄していた。
  - そのため 0.7 秒 loop ごとに同じ `max_adverse` / `max_hold_loss` / `reentry_reset` を再送し、
    `close_reject_no_negative` をノイズ化していた。

- 対応:
  - `RangeFader` と `scalp_ping_5s_flow` の exit worker に
    `close` 成否の真偽値返却と、reason-scoped retry cooldown を追加。
  - close 成功時だけ `_states` と必要な `_direction_flip_states` を破棄し、
    failed close は worker-local state を保持したまま一定時間 backoff する。
  - 今回は `RangeFader.neg_exit.allow_reasons` を広げていない。
    実測で `RangeFader` の reject は全件がより良い価格で閉じており、
    allowlist 拡張は winner の premature cut を増やすリスクが高いため。

- 検証:
  - `pytest -q tests/workers/test_scalp_rangefader_exit_worker.py tests/workers/test_scalp_ping_5s_flow_exit_retry.py`
    は `4 passed`。
  - `python3 -m py_compile workers/scalp_rangefader/exit_worker.py workers/scalp_ping_5s_flow/exit_worker.py`
    も成功。

- 次の確認点:
  1. `orders.db` の `close_reject_no_negative` が、同一 trade への連打ではなく散発的な reject に縮小すること。
  2. `logs/local_v2_stack/quant-scalp-rangefader-exit.log` と
     `logs/local_v2_stack/quant-scalp-ping-5s-flow-exit.log` で同一 trade の failed close 連打が止まること。
  3. `scalp_ping_5s_flow_live` の near-break-even close reject が 24h で再発しないこと。

## 2026-03-09 15:45-16:05 JST - negative close は worker から `strategy_tag/pocket/instrument` を明示的に渡し、close policy の推測依存を外す

- 市況確認:
  - `python3 scripts/pdca_profitability_report.py --instrument USD_JPY ...` の 2026-03-09 15:48 JST 時点で、
    USD/JPY は `158.546 / 158.554`、spread `0.8 pips`、pricing/summary/openTrades は `228.7 / 232.0 / 271.5 ms`。
  - `local_v2_stack status` では core 4 + `quant-scalp-rangefader-exit` + `quant-scalp-ping-5s-flow-exit` は `running`。

- 実測:
  - 7d の実害 trade `424165 / 424436 / 424185` はいずれも `trades.db` 上では
    `strategy_tag=scalp_ping_5s_flow_live` だった。
  - しかし `orders.db` の `close_reject_no_negative` row では request payload の
    `strategy_tag/pocket` が空で、reject reason は主に `__de_risk__` と `reentry_reset` だった。
  - `execution.order_manager._neg_exit_decision()` を current config で直に叩くと、
    `scalp_ping_5s_flow_live` の policy では `__de_risk__` は許可される一方、
    strategy context なし default policy では拒否される。

- 原因:
  - close path は worker から `trade_id/client_order_id` だけを渡し、
    service 側で `strategy_tag/pocket/instrument` を推測していた。
  - この推測が崩れると、`scalp_ping_5s_flow_live` 専用の no-block neg-exit policy を使えず、
    default policy に落ちて `__de_risk__` が reject されうる。

- 対応:
  - `execution/order_manager.py` の `close_trade()` に
    `strategy_tag/pocket/instrument` を追加。
  - `workers/order_manager/worker.py` の `/order/close_trade` も同 context を転送するよう更新。
  - `workers/scalp_ping_5s_flow/exit_utils.py` と
    `workers/scalp_rangefader/exit_utils.py` は explicit context を forwarding。
  - `workers/scalp_ping_5s_flow/exit_worker.py` と
    `workers/scalp_rangefader/exit_worker.py` は close context を組み立てて
    `_attempt_close()` / partial close / de-risk close に常時渡す。
  - `workers/scalp_ping_5s_flow/pro_stop.py` と
    `workers/scalp_rangefader/pro_stop.py` の `maybe_close_pro_stop()` も
    explicit context をそのまま close request へ forwarding するように揃えた。

- 検証:
  - `pytest -q tests/execution/test_order_manager_exit_policy.py tests/workers/test_exit_utils_close_context.py tests/workers/test_scalp_rangefader_exit_worker.py tests/workers/test_scalp_ping_5s_flow_exit_retry.py`
    は `18 passed`。
  - `python3 -m py_compile execution/order_manager.py workers/order_manager/worker.py workers/scalp_ping_5s_flow/exit_utils.py workers/scalp_rangefader/exit_utils.py workers/scalp_ping_5s_flow/exit_worker.py workers/scalp_rangefader/exit_worker.py`
    も成功。

- 次の確認点:
  1. `scalp_ping_5s_flow_live` の negative close が発生した時に、`orders.db` の close row に `strategy_tag/pocket/instrument` が埋まること。
  2. `__de_risk__` / `risk_reduce` / `reentry_reset` が flow policy どおり通り、default policy へ落ちないこと。

## 2026-03-09 15:39 JST - `scalp_ping_5s_d_live` replay が `0 trades` だった主因は tick 欠損ではなく replay 既定選択のズレ

- 市況確認:
  - `logs/orderbook_snapshot.json` の最新 best bid/ask は `158.576 / 158.584`、spread は `0.8p`、latency は `101.5ms`。
  - `logs/health_snapshot.json` は `generated_at=2026-03-09T06:28:31Z`、`trades_last_close=2026-03-09T06:25:22Z`、`trades_count_24h=277`。
  - `local_v2_stack status` で core 4 は `running`。

- 実測:
  - 同一窓 `2026-03-09 05:19:37Z - 05:57:39Z` に対し、
    `SCALP_REPLAY_PING_VARIANT=D` だけで `scripts/replay_exit_workers.py --sp-only --sp-live-entry`
    を回すと `summary_overall.trades=0` だった。
  - 同じ窓で `SCALP_REPLAY_MODE=scalp_ping_5s_d` を明示すると
    `3 trades / -7.2 pips` まで戻った。
  - 原因は `replay_exit_workers.py` が allowlist 未指定時でも既定 `SCALP_REPLAY_MODE=spread_revert`
    を内部 allowlist に積み、`SCALP_REPLAY_PING_VARIANT=D` だけでは
    ping5s D を有効化していなかったこと。

- 対応:
  - `scripts/replay_exit_workers.py`
    - `SCALP_REPLAY_PING_VARIANT` が明示され、`SCALP_REPLAY_MODE/ALLOWLIST/POCKET` が未指定のときは、
      replay 選択を variant 側へ自動解決するよう修正。
    - effective `mode/allowlist/pocket` を replay JSON `meta` に追加。
    - ping5s replay の default pocket も worker config に合わせて `scalp_fast` へ揃えるよう修正。
  - `tests/scripts/test_replay_exit_workers.py`
    - variant D の implicit selection と、explicit replay mode を上書きしない回帰テストを追加。

- 検証:
  - `pytest -q tests/scripts/test_replay_exit_workers.py` は `15 passed`。
  - `python3 -m py_compile scripts/replay_exit_workers.py tests/scripts/test_replay_exit_workers.py` は pass。
  - 修正後、同じ replay コマンドを `SCALP_REPLAY_PING_VARIANT=D` のみで再実行すると
    `tmp/replay_exit_workers_ping5s_d_autofix.json` は
    `summary_overall.trades=3`, `meta.scalp_entry_mode_effective=scalp_ping_5s_d`,
    `meta.scalp_entry_pocket_effective=scalp_fast` を返した。

- 残課題:
  - replay が `0 trades` になる silent fail は潰れたが、
    live の `entry_probability` / side-metrics flip / lookahead 補正をどこまで replay に持ち込むかは別タスク。

## 2026-03-09 06:44 UTC / 2026-03-09 15:44 JST - Brain `no_llm` の主因は cold start だったため、safe canary に Ollama keep-alive を追加

- 市況確認:
  - OANDA 直照会の USD/JPY は `bid=158.497 / ask=158.505 / spread=0.8p`。
  - `M5 x24` の ATR proxy は `9.18p`、直近30分レンジは `22.0p`。
  - API は `pricing=286ms` で正常。

- 実測:
  - 2026-03-09 15:39 JST の Brain 発火は
    `MomentumBurst-open_short / micro / ALLOW / reason=no_llm / llm_ok=0 / latency_ms=4202` だった。
  - `brain_state.db` の同一行は `response_json` が `runtime_guard` のみで、
    `error=no_llm`、`response` 本体は空。parse failure ではなく、Ollama 応答が取れなかった経路だった。
  - 同じ `context_json` から live prompt を再現して `qwen2.5:7b` を実測すると、
    初回 `max_tokens=64` は `4920ms`、その後の連続呼び出しは
    `96-192 tokens` で `1386-1943ms` に収まった。
  - `ollama ps` でも qwen の unload 期限が短く、first-hit だけ遅い cold start と整合した。

- 対応:
  - `ops/env/profiles/brain-ollama-safe.env`
    - `BRAIN_OLLAMA_KEEP_ALIVE=-1` を追加し、safe canary では qwen を常駐化。
  - `utils/ollama_client.py`
    - `/api/chat` request に `keep_alive` を渡せるよう拡張。
  - `workers/common/brain.py`
    - live Brain / prompt autotune / runtime autotune の Ollama 呼び出しへ `BRAIN_OLLAMA_KEEP_ALIVE` を伝播。
  - `scripts/prepare_local_brain_canary.py`
    - warmup request にも `keep_alive` を載せ、readiness JSON へ `ollama_keep_alive` を出力。
  - `tests/utils/test_ollama_client.py`
    - `keep_alive` request field の回帰テストを追加。

- 意図:
  - `shadow` は動いていても `llm_ok=0 / no_llm` では比較不能なので、
    micro pocket の first-hit を常駐モデルで受け、`llm_ok=1` のサンプルを増やす。

- 検証:
  - `python3 -m py_compile utils/ollama_client.py workers/common/brain.py scripts/prepare_local_brain_canary.py tests/utils/test_ollama_client.py`
  - `pytest -q tests/utils/test_ollama_client.py`
  - 反映後に `ollama ps` の `UNTIL` が常駐化すること。
  - 次の micro Brain event で `brain_state.db.llm_ok=1` かつ `reason!=no_llm` を確認すること。

## 2026-03-09 07:02 UTC / 2026-03-09 16:02 JST - ping5s D replay は adaptive signal window と entry probability 補正を live helper へ寄せた

- 市況確認:
  - `logs/orderbook_snapshot.json` の最新 best bid/ask は `158.616 / 158.624`、spread は `0.8p`、latency は `298.7ms`。
  - `logs/health_snapshot.json` は `generated_at=2026-03-09T06:49:15Z`、`git_rev=fa5fdcd4`、`trades_count_24h=279`。
  - `local_v2_stack status` で core 4 は `running`。

- 実測:
  - 前段の replay fix 後も、`scalp_ping_5s_d_live` の replay entry は
    `confidence/100` ベースで、live worker が使う `adaptive signal window`,
    `side_metrics_direction_flip`, `entry_probability_alignment`,
    `entry_probability_band_allocation` を素通りしていた。
  - そのため zero-trade は解消しても、entry thesis と replay sizing が live からまだ乖離していた。

- 対応:
  - `scripts/replay_exit_workers.py`
    - ping5s replay signal に対して、live worker の
      `_maybe_adapt_signal_window`, `_adaptive_live_score_blocked`,
      `_maybe_side_metrics_direction_flip`,
      `_adjust_entry_probability_alignment`,
      `_entry_probability_band_units_multiplier`,
      `_resolve_final_signal_for_side_filter`,
      `_is_signal_mode_blocked`
      を再利用する補正層を追加。
    - replay signal に
      `entry_probability`, `entry_probability_raw`,
      `entry_probability_units_mult`, `entry_probability_band_units_mult`,
      `signal_window_adaptive_*`,
      `side_metrics_direction_flip_*`
      を埋めるようにした。
    - `ScalpReplayEntryEngine` は `confidence/100` ではなく、
      signal 側の `entry_probability` を優先して `entry_thesis` へ渡すようにした。
  - `tests/scripts/test_replay_exit_workers.py`
    - live probability 補正と side-metrics flip が replay signal へ反映される回帰テストを追加。
    - adaptive live-score block が replay でも entry を止める回帰テストを追加。

- 検証:
  - `pytest -q tests/scripts/test_replay_exit_workers.py` は `18 passed`。
  - `python3 -m py_compile scripts/replay_exit_workers.py tests/scripts/test_replay_exit_workers.py` は pass。
  - 実 replay でも `SCALP_REPLAY_MODE=scalp_ping_5s_d` のみで
    `tmp/replay_exit_workers_ping5s_d_liveadj2.json` は
    `summary_overall.trades=3`, `total_pnl_pips=-7.2` を維持し、
    trade count を崩さず補正層を追加できた。

- 残課題:
  - まだ replay は `lookahead_units_mult`, `dynamic_alloc`, `side_adverse_stack`, `allowed_lot`
    までは live と一致していない。
  - 次は lot/sizing の live parity をどこまで replay に持ち込むかを切る。

## 2026-03-09 16:05 JST - Brain shadow の文脈欠落と confidence スケール誤読を修正

- 市況:
  - `scripts/collect_local_health.sh` の最新 `logs/health_snapshot.json` は更新成功。
  - `scripts/prepare_local_brain_canary.py --warmup` 系の直近 readiness では
    `spread_pips=0.8`, `tick_age_sec=1.2`, `atr_proxy_pips=3.766`,
    `recent_range_pips_6m=6.4`, `ollama warmup=589.6ms`。
  - core 4 は running 維持で、safe canary は `micro` 限定 shadow のまま。

- 事象:
  - keep-alive 反映後、`brain_decisions` の `micro` は
    `2026-03-09T06:39:07Z no_llm / ALLOW / 4202ms`
    と
    `2026-03-09T06:48:04Z llm_ok=1 / REDUCE 0.5 / 1418ms`
    まで改善した。
  - ただし `llm_ok=1` の勝ちトレード
    `qr-1773038883320-micro-momentum788fab3aa`
    は実損益 `+4.3 pips / +120.228 JPY` だったのに、
    Brain は `REDUCE 0.5` を返していた。

- 原因:
  - `workers/common/brain.py` の `_compact_context()` は
    `entry_thesis` / `meta` を残していたが、
    `_json_text()` が再度 `_compact_jsonable()` を通し、
    top-level `max_items=8` 制限で `entry_thesis/meta` を落としていた。
  - その結果 `brain_state.db.context_json` は
    `ts/strategy_tag/pocket/side/units/sl/tp/confidence` だけになり、
    `entry_probability`, `entry_units_intent`, `tp_pips`, `sl_pips`,
    `dynamic_alloc`, `forecast_fusion` が LLM に渡っていなかった。
  - あわせて top-level `confidence` には `0.842...` の確率値が入り、
    LLM が `80` 点の confidence と `0.84` の probability を同一軸で読み違えていた。

- 対応:
  - `workers/common/brain.py`
    - compact 済み context を prompt / `brain_state.db` へ保存する際に再圧縮しないよう修正。
    - Brain 向け `confidence` は `entry_thesis.confidence` を優先し、
      なければ `entry_probability` を `0-100` に正規化して使うよう修正。
    - prompt rules に
      `confidence=0-100`,
      `entry_probability*=0.0-1.0`,
      両者を同一スケール比較しないこと、
      `confidence>=75 && entry_probability>=0.80` では
      spread/execution/regime が悪くない限り `ALLOW` を優先すること
      を明示した。
  - `tests/workers/test_brain_ollama_backend.py`
    - `sl/tp` があるケースでも prompt と `brain_state.db.context_json` の両方に
      `entry_thesis/meta` が残る回帰テストを追加。
    - normalized `confidence=80.0` と scale guidance が prompt に入ることを固定。

- 検証:
  - `pytest -q tests/workers/test_brain_ollama_backend.py` は `3 passed`。
  - `python3 -m py_compile workers/common/brain.py tests/workers/test_brain_ollama_backend.py` は pass。
  - 同じ winning sample を `qwen2.5:7b` / `timeout=4.0` / `max_tokens=128-192` で再評価すると、
    変更前は `REDUCE 0.6-0.8` だったのが、
    変更後は `ALLOW 1.0 / reason=\"High confidence and entry probability with stable spread\"`
    に変わった。

- 次の観測点:
  - live shadow で `brain_shadow` の `ALLOW/REDUCE` 構成比が変わるか。
  - `reason=no_llm` が再増加しないか。
  - `micro` の勝ちトレードで不要な `REDUCE` が減るか。

## 2026-03-09 16:18 JST - ping5s replay units mismatch は `dynamic_alloc` を replay sizing に戻して一段縮めた

- 市況確認:
  - `logs/orderbook_snapshot.json` の最新 best bid/ask は `158.616 / 158.624`、spread は `0.8p`、latency は `106.8ms`。
  - `logs/health_snapshot.json` は `generated_at=2026-03-09T07:11:04Z`、`git_rev=35432c3b`、`trades_count_24h=283`。
  - `local_v2_stack status` で core 4 は `running`。

- 実測:
  - 直前の replay parity では `adaptive signal window` と `entry_probability` は live に寄ったが、
    sizing 側はなお `dynamic_alloc` を落としていた。
  - 現行 `config/dynamic_alloc.json` では
    `scalp_ping_5s_d_live / scalp_fast` の `lot_multiplier=0.45`, `trades=43`, `score=0.089`。
  - つまり replay の `entry_units_intent` は、D worker で live が常時 `0.45x` している分だけ
    恒常的に大きく見えていた。

- 対応:
  - `scripts/replay_exit_workers.py`
    - ping5s replay signal で `workers.common.dynamic_alloc.load_strategy_profile()` を読み、
      found profile の `lot_multiplier` を `entry_units_intent` に掛けるようにした。
    - `DYN_ALLOC_MULT_MIN / DYN_ALLOC_MULT_MAX` clamp を live worker と同じ順で適用。
    - replay signal / `entry_thesis` に
      `dynamic_alloc.{strategy_key,score,trades,lot_multiplier}` を残すようにした。
  - `tests/scripts/test_replay_exit_workers.py`
    - profile ありで `entry_units_intent` が縮小される回帰テストを追加。
    - `lot_multiplier` clamp の回帰テストを追加。

- 検証:
  - `pytest -q tests/scripts/test_replay_exit_workers.py` は `20 passed`。
  - `python3 -m py_compile scripts/replay_exit_workers.py tests/scripts/test_replay_exit_workers.py` は pass。

- 残課題:
  - replay 未移植は `lookahead_units_mult`, `side_adverse_stack`, `allowed_lot`。
  - 特に live vs replay の最終 units 差をさらに詰めるには、
    `allowed_lot` と `lookahead_units_mult` の順を replay へ持ち込む必要がある。

## 2026-03-09 17:03 JST - `RangeFader` negative close reject は `neg_exit` override の defaults 上書きが原因

- 市況確認:
  - `logs/orders.db` / OANDA 観測では USD/JPY は spread `0.8p`、API 応答 `204-349ms`、local-v2 core 4 は稼働中。
  - エントリー不足の主因だった stale `dynamic_alloc` は解消後で、`RangeFader` の filled は再開していた。

- 実測:
  - `orders.db` では `2026-03-09 07:59-08:00 UTC` に
    `RangeFader-buy-fade` / `RangeFader-neutral-fade` の
    `close_reject_no_negative` が連発し、
    `exit_reason=max_adverse` と `max_hold_loss` が主因だった。
  - 実行時 `_strategy_neg_exit_policy("RangeFader-buy-fade")` は
    `allow_reasons=['reversion_*']` のみを返し、
    defaults の `max_adverse` 等が落ちていた。

- 原因:
  - `config/strategy_exit_protections.yaml` の `RangeFader.neg_exit.allow_reasons`
    が `reversion_*` だけを持ち、merge 時に defaults allowlist を丸ごと置換していた。
  - そのため worker 側が `allow_negative=True` を送っても、
    `order_manager` は strategy-local policy で negative close を拒否していた。

- 対応:
  - `RangeFader.neg_exit.allow_reasons` を defaults 相当の full list へ戻し、
    追加分として `reversion_*` と `max_hold_loss` を保持した。
  - derived tag (`RangeFader-buy-fade`, `RangeFader-neutral-fade`) でも
    `max_adverse` / `max_hold_loss` / `reversion_*` が通る回帰を追加した。

- 検証:
  - `pytest -q tests/execution/test_order_manager_exit_policy.py`
  - `python3 -m py_compile tests/execution/test_order_manager_exit_policy.py`

## 2026-03-09 08:55 UTC / 2026-03-09 17:55 JST - local-v2: 「待てば助かる EXIT」と「lane別の低頻度」を実測確認

- 市況確認（ローカルV2実測 + OANDA API）:
  - `2026-03-09T08:51:24Z` pricing:
    - `USD/JPY bid=158.586 ask=158.594 mid=158.590 spread=0.8p`
    - `M5 ATR14=8.11p`, `range_1h=33.0p`, `range_4h=44.8p`
  - OANDA API 応答:
    - `pricing=342-361ms(200/200/200)`
    - `summary=341-379ms(200/200)`
    - `openTrades=331-388ms(200/200)`
    - `candles(M5)=359-364ms(200/200)`
  - account/openTrades:
    - `openTradeCount=0`, `USD/JPY net_units=0`
  - local health:
    - `data_lag_ms=541.2`, `decision_latency_ms=21.45`

- 実測:
  - 直近 `308` closed trades は
    `win_rate=56.8%`, `PF=1.741`, `net_pips=+211.3` まで改善した一方、
    `net_jpy=-572.2` でまだ負側。
  - entry execution 自体は prior `308` trades 比で改善:
    - `slip_p95=0.4p -> 0.2p`
    - `latency_submit_p95=309.9ms -> 276.5ms`
    - つまり「入った価格」よりも、
      「EXITの形」と「どの lane が何回入るか」の問題が残っている。
  - `MomentumBurst` の recent `STOP_LOSS_ORDER` 7件を
    `tick_entry_validate + trade_sl_perspectives` で見ると、
    `TP_touch<=600s = 2/7`。
    - `ticket=454318`: `post_close_tp_touch_s=69`
    - `ticket=454360`: `post_close_tp_touch_s=343`
  - `scalp_extrema_reversal_live` の recent `STOP_LOSS_ORDER` 7件では
    `TP_touch<=600s = 3/7`。
    - `ticket=451906`: `post_close_tp_touch_s=159`
    - `ticket=451916`: `post_close_tp_touch_s=88`
    - `ticket=452470`: `post_close_tp_touch_s=538`
  - lane 別の filled cadence は大きく偏る:
    - `MicroLevelReactor-bounce-lower`: `159 fills`, `avg_gap_sec=146.2`
    - `MomentumBurst-open_long`: `10 fills`, `avg_gap_sec=3273.8`
    - `MomentumBurst-open_short`: `6 fills`, `avg_gap_sec=1843.2`
    - `scalp_extrema_reversal_live`: `8 fills`, `avg_gap_sec=1302.0`
  - つまり「頻度不足」は全体停止ではなく、
    `MomentumBurst` と `scalp_extrema_reversal_live` へ集中。

- 低頻度の主因:
  - `MomentumBurst`
    - `orders.db` では `filled=16`, `entry_probability_reject=0`, `perf_block=0` で、
      order-manager 側 reject は主因ではない。
    - 一方 `quant-micro-momentumburst.log` では
      `hist_block tag=MomentumBurst-open_long strategy=MomentumBurst n=12 score=0.196 reason=low_recent_score`
      が `17:21 JST` 以降に反復。
    - 同ログには
      `perf_block tag=MomentumBurst reason=hard:margin_closeout_n=4 rate=0.129 n=31`
      も断続。
    - `brain_state.db / metrics.db` では micro shadow の
      `brain_latency_ms=1.4s-4.2s` が記録され、
      momentum lane だけ preflight が相対的に重い。
  - `scalp_extrema_reversal_live`
    - `orders.db` 24h 集計で `filled=8` に対し
      `strategy_cooldown=39`, `perf_block=52`, `entry_probability_reject=1`。
    - `metrics.db` では
      `order_perf_block reason=hard:failfast:pf=0.06 win=0.08 n=12`
      と、`risk_mult_total=0.55` が継続。
    - つまりこの lane は「待てば戻る EXIT」も一部あるが、
      同時に perf/cooldown で entry もかなり抑えられている。
  - `scalp_ping_5s_b_live`
    - `quant-scalp-ping-5s-b.log` の最新 `entry-skip summary` は
      `lookahead_block`, `no_signal:revert_not_found`,
      `no_signal:momentum_tail_failed`, `no_signal:side_filter_block` が主。
    - `global_lock` は `strategy_control.db` で `0`、共通停止は主因ではない。

- 判断:
  - ユーザー指摘どおり、現状の主問題は
    「spread/slippage悪化」ではなく、
    `MomentumBurst` / `scalp_extrema_reversal_live` での
    `EXIT早すぎ` と `entry cadence不足`。
  - ただし頻度不足の中身は lane ごとに違う。
    - `MomentumBurst`: signal/history block + micro Brain latency
    - `scalp_extrema_reversal_live`: perf_block + cooldown + low win regime
    - `scalp_ping_5s_b_live`: strict setup gate
  - したがって、次の改善も共通レイヤ緩和ではなく、
    strategy-local の EXIT/reentry/cooldown/history gate を個別に詰めるべき。

- 次の観測点:
  1. `MomentumBurst` で `hist_block(low_recent_score)` が減り、
     `avg_gap_sec` が `30-60m` 窓で短縮すること。
  2. `MomentumBurst` の recent stop-loss 群で
     `post_close_tp_touch` が再発しないかを ticket 単位で継続監査すること。
  3. `scalp_extrema_reversal_live` は `perf_block` / `strategy_cooldown` が減っても
     `PF` が改善しないなら、頻度増より EXIT/entry quality 改善を優先すること。

## 2026-03-09 08:55 UTC / 2026-03-09 17:55 JST - local-v2: entry precision は改善、ただし early exit と reentry scarcity で取り切れていない

- 市況確認（ローカルV2実測 + OANDA API）:
  - `2026-03-09 17:51 JST` OANDA snapshot:
    - `USD/JPY bid=158.586 ask=158.594 mid=158.590 spread=0.8p`
    - `M5 ATR14=8.11p`, `range_1h=33.0p`, `range_4h=44.8p`
    - `openTrades=0`, `USD/JPY net_units=0`
  - OANDA API 応答品質:
    - `pricing=342-361ms (200/200/200)`
    - `summary=341-379ms (200/200)`
    - `openTrades=331-388ms (200/200)`
    - `candles(M5)=359-364ms (200/200)`
  - local health:
    - `data_lag_ms=541.2`, `decision_latency_ms=21.4`
    - core 4 +主要 worker は `running`

- 事実:
  - 直近 closed `308 trades` は
    `win_rate=56.8%`, `PF=1.741`, `expectancy=+0.686p`, `net_pips=+211.3` まで回復。
    一方で `net_jpy=-572.2` で、依然として資金寄与はマイナス。
  - ひとつ前の `308 trades` は
    `win_rate=28.6%`, `PF=0.486`, `expectancy=-0.687p`, `net_pips=-211.5`, `net_jpy=-148.3`。
    つまり「entry quality が前より良い」は事実で、直近は pips ベースでは明確に改善している。
  - 実行品質（filled entry recent 308 vs prior 308）:
    - `slip_p95=0.4p -> 0.2p`
    - `latency_submit_p95=309.9ms -> 276.5ms`
    - `spread_mean=0.800p -> 0.808p`
    - `latency_preflight_p95=413.5ms -> 534.8ms`
  - つまり悪化源は「約定滑り」ではなく、
    1) loser 側の oversized trade
    2) negative close 後の戻り取り逃し
    3) reentry 減少
    へ移っている。

- early exit / post-close recovery:
  - `logs/replay/USD_JPY/USD_JPY_ticks_20260309.jsonl` と直近 closed `120 trades` を照合すると、
    負け `39 trades` のうち
    - `5分以内に建値回復`: `25`
    - `15分以内に建値回復`: `36`
  - strategy 別の negative trade に対する `15分以内建値回復率`:
    - `MomentumBurst`: `6/8 = 75.0%`
    - `scalp_extrema_reversal_live`: `8/11 = 72.7%`
    - `scalp_ping_5s_d_live`: `5/5 = 100%`
    - `RangeFader`: `4/4 = 100%`
    - `MicroLevelReactor`: `96/96 = 100%`
  - 代表例:
    - `MomentumBurst ticket=454360`: `-4.2p` close 後、`15分 MFE=+11.9p`
    - `RangeFader ticket=454316`: `-3.1p` close 後、`15分 MFE=+10.7p`
    - `scalp_ping_5s_d_live ticket=454054`: `-1.4p` close 後、`15分 MFE=+17.9p`

- reentry / entry frequency:
  - filled recent `308` の pocket 構成は `micro=227 / scalp=60 / scalp_fast=21`。
    prior `308` は `micro=158 / scalp_fast=150` で、頻度低下は主に `scalp_fast` 側。
  - filled gap も `mean=1.54分 -> 1.96分`, `p90=2.67分 -> 3.90分` へ拡大。
  - negative trade 後の same-strategy `15分以内 reentry`:
    - `RangeFader`: `0/4`
    - `scalp_ping_5s_b_live`: `0/4`
    - `scalp_ping_5s_d_live`: `1/5 = 20.0%`
    - `MomentumBurst`: `3/8 = 37.5%`
    - `scalp_extrema_reversal_live`: `4/11 = 36.4%`
    - `MicroLevelReactor`: `86/96 = 89.6%`
  - つまり「戻るのに入らない」は主に `MomentumBurst` と `scalp_fast` / `RangeFader` に集中している。

- 詰まりの主因（直近 orders/logs 実測）:
  - `orders.db` の直近日中 block/reject:
    - `perf_block=52`（全件 `scalp_fast / scalp_extrema_reversal_live`）
    - `strategy_cooldown=39`（同上）
    - `entry_probability_reject=46`
    - `rejected=1`
  - `quant-scalp-ping-5s-b.log`:
    - `lookahead_block` と `side_filter_block` が数十件単位で継続
    - 一時的に `spread_blocked=148/148` も発生
  - `quant-scalp-ping-5s-flow.log`:
    - `adaptive_live_score_block` と `revert_not_found` が継続
  - `MomentumBurst`:
    - `perf_block tag=MomentumBurst reason=hard:margin_closeout_n=4 rate=0.129 n=31` が
      `07:37/07:39/07:51/08:34/08:52/08:58 UTC` に継続
    - 一方で recent `16 trades / avg_abs_units=4329 / net_jpy=-616.2` と、
      依然として oversized loser でもある
  - `RangeFader`:
    - `58 trades / win_rate=93.1% / net_jpy=+54.3` の winner だが、
      負け4本は全て `15分以内建値回復` かつ `15分以内再エントリーなし`

- 判断:
  - 現時点の主問題は「entry precision の悪化」ではない。
  - 収益を取り切れていない主因は、
    1. exit が早すぎて negative close 後の回復を逃していること
    2. `MomentumBurst` / `scalp_fast` / `RangeFader` で reentry frequency が不足していること
    3. その一方で `MomentumBurst` は loser 時の absolute units がまだ大きいこと
  - よって次の改善は
    `滑り対策` ではなく、strategy-local の `exit patience / reentry cadence / cooldown or gate` を優先すべき。

- 次の観測点:
  1. `MomentumBurst` と `scalp_extrema_reversal_live` の negative close 後 `5-15分` の MFE を継続監視し、早すぎる exit 条件を特定する。
  2. `RangeFader` / `MomentumBurst` / `scalp_fast` の negative close 後 `15分` 以内 reentry 率を改善指標に置く。
  3. `scalp_fast` の `perf_block / cooldown / lookahead_block / adaptive_live_score_block` を loosening するなら、同時に `net_pips` が再悪化しないことを確認する。

## 2026-03-09 09:25 UTC / 2026-03-09 18:25 JST - local-v2: `MomentumBurst` / `scalp_extrema_reversal_live` の strategy-local patience/cadence 修正を実装

- 実装:
  - `MomentumBurst` は signal 時の hard SL を `max(2.4, atr_pips * 1.25)` へ拡張。
    直近 stop-out 群（`sl=3.38-4.56p`）の手前刈りを減らし、同時に allowed-lot 側で units を自然縮小させる。
  - `scalp_extrema_reversal_live` は
    `COOLDOWN_SEC=120 -> 60`,
    `SL_ATR_MULT=0.95`, `TP_ATR_MULT=1.25`,
    `SL_MIN/MAX=1.2/2.6`, `TP_MIN/MAX=1.4/3.2`
    を dedicated env + worker local cap で反映。
  - `scalp_extrema_reversal_live` の `perf_guard` は
    `SCALP_EXTREMA_REVERSAL_PERF_GUARD_HARD_FAILFAST_ENABLED=0`
    を追加し、hard reject ではなく `reduce` を優先させる。

- ねらい:
  - `MomentumBurst` は `STOP_LOSS_ORDER` 連発が主因で、
    `scalp_extrema_reversal_live` は `strategy_cooldown=71` / `perf_block=52` / `filled=12`
    と cadence 枯渇が主因だった。
  - 共通 gate や時間帯封鎖ではなく、strategy-local の stop/cooldown/perf profile だけを動かす。

- 直後の再検証条件:
  1. `orders.db` で `scalp_extrema_reversal_live` の `perf_block:hard:failfast` が消え、`filled` が増えること。
  2. `trades.db` で `MomentumBurst` / `scalp_extrema_reversal_live` の `STOP_LOSS_ORDER` 比率が下がること。
  3. `strategy_cooldown` が支配的に残る場合のみ、次段で shared `stage_tracker` の strategy-local override 要否を再評価すること。

## 2026-03-09 09:31 UTC / 2026-03-09 18:31 JST - local-v2 post-deploy check: 直後の実トレードはまだ悪化、改善は未確認

- 市況確認:
  - `2026-03-09T09:29:50Z` pricing:
    - `USD/JPY bid=158.425 ask=158.433 spread=0.8p`
  - local health:
    - `data_lag_ms=246.4`, `decision_latency_ms=13.1`
  - core 4 + `quant-micro-momentumburst` + `quant-scalp-extrema-reversal` は `running`
  - OANDA `openTrades=0`, `USD/JPY net_units=0`

- 変更後の即時成績:
  - `2026-03-09 18:18 JST` 再起動以降の closed trades:
    - `MomentumBurst`: `2 trades / net_jpy=-76.8 / net_pips=-1.4 / win_rate=50.0%`
    - `scalp_ping_5s_d_live`: `2 trades / net_jpy=-12.5 / net_pips=-2.9 / win_rate=0.0%`
    - 合計 `4 trades / net_jpy=-89.3 / net_pips=-4.3`
  - 直近 `1h` は `6 trades / net_jpy=-123.6 / net_pips=-1.6 / PF=0.725`
  - 直近 `3h` は `36 trades / net_jpy=-221.8 / net_pips=+9.8 / PF=0.821`
  - 直近 `24h` でも `314 trades / net_jpy=-695.7 / net_pips=+209.7 / PF=0.734`

- 変更後の lane 別挙動:
  - `MomentumBurst` は post-deploy 直後に `OPEN_SHORT` を 2 本約定。
    - `ticket=454416`: `+53.6 JPY / +3.0p`
    - `ticket=454428`: `-238.9 JPY / -4.4p`
    - net でまだ負け。
  - `quant-order-manager.log` では post-deploy の `MomentumBurst-open_short` が
    `probability_scale:1.000` で `-5404/-5429 units` まで通っており、
    loser 1本の絶対損失がまだ重い。
  - `quant-micro-momentumburst.log` は `18:32 JST` に
    `hist_block tag=MomentumBurst-open_short strategy=MomentumBurst n=12 score=0.196 reason=low_recent_score`
    を再度記録。つまり直後に2本撃った後、history gate でまた止まり始めている。
  - `scalp_extrema_reversal_live` は post-deploy 後 `orders.db` に
    `preflight/filled/perf_block` が 1件も増えておらず、今回の loosen がまだ評価できるほど signal が出ていない。
    pre-restart に見えていた `perf_block:hard:failfast` は旧ログ側。
  - `scalp_ping_5s_d_live` は `18:30 JST` に
    `STOP_LOSS_ON_FILL_LOSS` reject -> protection fallback -> filled の後、
    `ticket=454440` が `-9.0 JPY / -1.5p` で終了。

- 判断:
  - ユーザー指摘どおり、現時点では「うまくいっていない」で正しい。
  - ただし post-deploy の失敗主因は `scalp_extrema_reversal_live` ではなく、
    1. `MomentumBurst` がまだ大きい short clip を打って loser 1本で赤化していること
    2. `scalp_ping_5s_d_live` が `STOP_LOSS_ON_FILL_LOSS` fallback 経由で薄く負けていること
    3. `extrema` はまだ post-change の検証サンプルが出ていないこと
  - つまり、今回の loosen の成否判定前に、
    `MomentumBurst` の per-trade downside をもう一段落とす必要がある。

- 次の観測点:
  1. `MomentumBurst` の post-deploy `OPEN_SHORT` で `units ~5400` が続くかを最優先で監視する。
  2. `scalp_extrema_reversal_live` は post-restart 後に `filled` が出るまで、改善/悪化の判定を保留する。
  3. `scalp_ping_5s_d_live` の `STOP_LOSS_ON_FILL_LOSS` fallback fill が続くなら、entry hard-stop gap の再点検が必要。

## 2026-03-09 09:31 UTC / 2026-03-09 18:31 JST - pattern_book 未コミット差分の棚卸しと反映要否

- 市況確認:
  - `USD/JPY` は `158.445 / 158.453`、直近 tick spread 平均は約 `0.8 pips`。
  - `logs/factor_cache.json` の `M1 ATR` は `3.03 pips`、直近 tick range は約 `3.0 pips`。
  - `logs/metrics.db` の `data_lag_ms` は直近で `54.7-930.7ms`、`decision_latency_ms` は `13-18ms`。
  - `logs/orders.db` 直近 24h は `filled=313`, `rejected=2` で、流動性/応答品質の悪化による作業保留条件には該当しなかった。

- Git / 差分の実測:
  - cleanup 着手時の dirty file は `config/pattern_book.json`, `config/pattern_book_deep.json` のみ。
  - `main == origin/main` で未 push commit はなく、未反映の code/config deploy は確認されなかった。
  - 差分は `quant-pattern-book` の live snapshot 更新で、`as_of`, `patterns_total`, `deep_analysis` の集計前進が中心だった。

- 反映判断:
  - local-v2 runtime は既に現行 snapshot を参照しており、`workers/common/pattern_gate.py` は `logs/patterns.db` を優先し、JSON は fallback のみ。
  - 従って本件は repo cleanup 対象ではあるが、追加の `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager` は不要と判断した。

## 2026-03-09 18:44 JST - side 別対症療法をやめ、indicator-quality guard を両方向へ導入

- 市況確認:
  - `USD/JPY` は `158.322 / 158.330`、spread `0.8 pips`。
  - `logs/health_snapshot.json` は `data_lag_ms=1025.5`, `decision_latency_ms=14.06`。
  - core 4 と `quant-micro-momentumburst`, `quant-scalp-extrema-reversal` は稼働中で、問題は infra 停止ではなく trade economics 側。

- 実測:
  - 直近 `1h` は `9 trades / net_jpy=-667.7 / net_pips=-13.6 / win_rate=33.3% / PF=0.328`。
  - 直近 `6h` の悪化主因は `MicroTrendRetest-short -540.6 JPY`、`MomentumBurst -214.3 JPY`、`scalp_ping_5s_d_live -35.1 JPY`。
  - 直近の大きい micro 負けは
    - `MicroTrendRetest-short` 2本: `-5100 units`, `-5.8p / -4.8p`
    - `MomentumBurst` 2本: `-5429 / -5433 units`, `-4.4p / -3.7p`
  - `orders.db` の `entry_thesis` では、両戦略とも負けた short が `pattern_tag ... rsi:os ... d:short` かつ `trend_snapshot = {tf:H4, direction:long, gap_pips:33.692, adx:31.52}` を伴っていた。

- 判断:
  - 問題を `short が悪い / long が悪い` で切るのは不正確。
  - 本質は `RSI 過熱/売られ過ぎ`, `ADX+MA gap に対する伸び切り`, `higher-TF 逆行`, `連続バー偏り` といった indicator state の質が悪いまま entry していること。
  - 改善は side 固有の対症療法ではなく、両方向に効く `indicator-quality guard` を strategy-local に入れる方針へ切り替える。

- 実装:
  - `workers/micro_runtime/worker.py` は strategy 実行前に `mtf` と `trend_snapshot` を factor view へ注入し、戦略側が live でも higher-TF context を見て判定できるようにした。
  - `MomentumBurst` は `trend_snapshot` 逆行ブロックと、`long/short` 両方向の overextension guard を導入し、`RSI 54-70 / 30-46` の quality band に収めた。
  - `MicroTrendRetest` は既存の symmetric `RSI/ADX/gap` quality check に `trend_snapshot` 逆行ブロックを追加した。

- 検証:
  - `pytest -q tests/strategies/test_momentum_burst.py tests/strategies/test_trend_retest.py tests/workers/test_micro_multistrat_trend_flip.py`
  - `39 passed`
  - `python3 -m py_compile strategies/micro/momentum_burst.py strategies/micro/trend_retest.py workers/micro_runtime/worker.py tests/strategies/test_momentum_burst.py tests/strategies/test_trend_retest.py tests/workers/test_micro_multistrat_trend_flip.py`

## 2026-03-09 20:03 JST - e132c325 反映後の初回 live 確認

- 市況確認:
  - `USD/JPY` 直近 tick は `158.424 / 158.432`、spread `0.8 pips`。
  - `factor_cache M1 close=158.453`, `atr_pips=3.74`。
  - `logs/health_snapshot.json` 更新時点で `data_lag_ms=423.5`, `decision_latency_ms=21.08`。
  - core 4 と `quant-micro-momentumburst`, `quant-micro-trendretest`, `quant-scalp-rangefader` は `running`。

- post-deploy 実測:
  - 対象窓: `2026-03-09 19:13 JST` restart 後。
  - closed trade は `4 trades / net_jpy=+259.0 / net_pips=+9.1 / win_rate=100%`。
  - 内訳は `MomentumBurst 1本 +244.4 JPY / +4.6p`、`scalp_ping_5s_d_live 1本 +12.0 JPY / +2.1p`、`scalp_extrema_reversal_live 2本 +2.6 JPY / +2.4p`。
  - `MomentumBurst` の post-deploy filled は `OPEN_LONG 5313 units` が 1本のみで、
    `pattern_tag=c:spin_up|w:upper|tr:up_strong|rsi:mid_high|vol:tight|atr:low|d:long`,
    `trend_snapshot={tf:H4, direction:long, gap_pips:33.692, adx:31.52}` の aligned long だった。
  - 同窓で `MomentumBurst` / `MicroTrendRetest` の `rsi:os` + `H4 long` 逆行 short fill は確認されなかった。
  - `MicroTrendRetest` は post-deploy でまだ order/trade サンプルが出ていない。

- 残課題:
  - 初回サンプルとしては改善方向だが、`MicroTrendRetest` は約定ゼロのため有効性未判定。
  - `quant-micro-trendretest.log` には `20:00:59 JST` に `stale factors age=445.0s` 警告が出ており、entry quality とは別に factor freshness の監視は継続要。

## 2026-03-09 20:42 JST - `MicroTrendRetest` が short 固定になっていて entry 頻度を削っていた

- 市況確認:
  - `USD/JPY` 直近 tick は `158.472 / 158.480`、spread `0.8 pips`。
  - `factor_cache M1 close=158.466`, `atr_pips=2.69`。
  - `logs/health_snapshot.json` は `data_lag_ms=205.9`, `decision_latency_ms=11.03`。

- 実測:
  - 直近 `30m` は `0 trades`、直近 `1h` は `1 trade` のみ。
  - `orders.db` でも直近 `1h` は `preflight_start=1 / filled=1` で、`entry_probability_reject` や `perf_block` が主因ではなかった。
  - `quant-micro-trendretest.log` は worker start 時に `signal_tag_contains=short` を出しており、
    dedicated env `ops/env/quant-micro-trendretest.env` に `MICRO_MULTI_SIGNAL_TAG_CONTAINS=short` が残っていた。
  - その結果、`MicroTrendRetest` は strategy 自体が `OPEN_LONG` を返しても worker 層で long tag を拾えず、entry 頻度が構造的に落ちていた。

- 判断:
  - これは `long/short を指標状態で対称に扱う` という現行方針と矛盾する。
  - 頻度低下の明確な構成要因なので、`signal_tag_contains=short` は撤去して両方向通過へ戻す。

## 2026-03-09 20:17 JST - local Brain を観測専用から profit-oriented PDCA へ切り替え

- 市況確認:
  - `python3 scripts/prepare_local_brain_canary.py --warmup` の最新 readiness は
    `USD/JPY 158.414 / 158.422`, spread `0.8 pips`,
    `recent_range_6m=3.8 pips`, `atr_proxy=3.171 pips`,
    `tick_age=-0.4s`, `market_ready=true`, `enable_recommended=true`。
  - `ollama qwen2.5:7b` warmup は `414.3ms`、`quality_gate_ok=true`。

- 直前の問題認識:
  - post-fix shadow は `llm_ok=1` が出ている一方、
    `ALLOW` が loser を通し、`REDUCE` が winner を削るケースが混在していた。
  - 問題は「LLM が未接続」ではなく、
    `MomentumBurst-open_short` / `MicroTrendRetest-short` を中心に
    recent loser pattern を prompt / runtime guard が十分に使えていないことだった。

- 実装判断:
  - 観測待ちをやめ、Brain を `profit mode` の PDCA レーンへ引き上げた。
  - `micro-only shadow` は維持したまま、`BRAIN_SAMPLE_RATE=1.0` に上げ、
    prompt/runtime autotune を safe profile 内で常時有効化した。
  - Brain context に strategy/pocket ごとの `recent_outcome`
    (`trades`, `wins`, `win_rate`, `avg_pips`, `profit_factor`) を注入し、
    prompt に「`profit_factor < 1` かつ `avg_pips < 0` なら
    exceptional setup 以外は `ALLOW` を `REDUCE` 側へ寄せる」規則を追加した。
  - runtime guard でも同じ recent loser signal を使い、
    `ALLOW -> REDUCE` への deterministic な loss-bias fallback を追加した。
  - autotune input には `strategy_filled_trade_outcome` 集計を追加し、
    per-strategy/per-action の realized outcome を次回の prompt/runtime 調整へ渡すようにした。

- 検証:
  - `pytest -q tests/workers/test_brain_ollama_backend.py tests/workers/test_brain_history_prompt_autotune.py tests/scripts/test_prepare_local_brain_canary.py`
  - `19 passed`
  - `python3 -m py_compile workers/common/brain.py scripts/prepare_local_brain_canary.py tests/workers/test_brain_ollama_backend.py tests/workers/test_brain_history_prompt_autotune.py tests/scripts/test_prepare_local_brain_canary.py`
  - 新規テストでは、recent loser を持つ `MomentumBurst-open_short / micro` に対して
    fake LLM の `ALLOW` が最終的に `REDUCE 0.62` へ下がることを固定した。

- 運用メモ:
  - `prepare_local_brain_canary.py` は `BRAIN_PROFILE_MODE=profit` を安全プロファイルとして扱うよう更新した。
  - これにより `profit mode + sample_rate=1.0 + autotune on` でも
    readiness は `profile_safe=true` のまま live restart 可能になった。

## 2026-03-09 20:27 JST - 旧 generic profile が profit guard を鈍らせていたため、micro profit lane を専用 profile へ分離

- 問題:
  - 反映後の process env は `profit mode` だったが、
    実際に読み込む既定 `config/brain_runtime_param_profile.json` は
    `2026-03-05` の generic throughput profile のままで、
    `outcome_min_trades=12` と reaction が遅かった。
  - これでは `micro-only` shadow で loser cluster を見つけても、
    `4-6本` の現実的な sample 数では loss-bias guard が発火しにくい。

- 対応:
  - `config/brain_prompt_profile_profit_micro.json`
    と `config/brain_runtime_param_profile_profit_micro.json`
    を新設し、safe Brain lane から明示参照するようにした。
  - runtime 初期値は `outcome_min_trades=4`,
    `min_guard_samples=18`,
    `outcome_negative_reduce_scale=0.62`
    へ下げ、`micro` の recent loser に faster response する形へ寄せた。
  - prompt 初期ルールにも
    `MomentumBurst-open_short` / `MicroTrendRetest-short`
    の loser cluster を full `ALLOW` しにくくする方針を明示した。

## 2026-03-09 21:09 JST - MicroLevelReactor の shadow 成績を根拠に Brain を限定 apply へ昇格

- 市況確認:
  - `prepare_local_brain_canary.py --warmup` の最新 readiness は
    `USD/JPY 158.445 / 158.453`, spread `0.8 pips`,
    `atr_proxy=2.812 pips`, `recent_range_6m=1.4 pips`,
    `tick_age=-0.1s`, `enable_recommended=true`。

- 実測:
  - `2026-03-09 21:03 JST` 時点で
    `logs/brain_prompt_autotune_profit_latest.json` /
    `logs/brain_runtime_param_autotune_profit_latest.json`
    が更新され、dedicated profit lane の autotune report 出力を確認。
  - 直近 13 decision の filled trade outcome は
    `ALLOW: trades=7, PF=0.1586, win_rate=28.6%, avg_pips=-2.3857`
    に対し、
    `REDUCE: trades=5, PF=1.0682, win_rate=60.0%, avg_pips=+0.04`
    だった。
  - `MicroLevelReactor-bounce-lower` の直近 join でも
    `ALLOW 4本 = realized_pl -17.837 / avg_pips -0.725`
    に対し、
    `REDUCE 4本 = realized_pl -0.733 / avg_pips -0.075`
    まで損失が圧縮されていた。

- 判断:
  - ここでは `shadow 継続` より `MicroLevelReactor だけ apply`
    の方が期待値改善に寄ると判断した。
  - 一方で `MomentumBurst` / `MicroTrendRetest` はまだ post-profit-profile の
    sample が薄いため、Brain 適用対象から一旦外す。

## 2026-03-09 21:22 JST - local LLM が market/execution 文脈を読めておらず cache も粗すぎたため、safe canary の入力と再利用粒度を修正

- 市況確認:
  - `logs/tick_cache.json` 直近300 tick: USD/JPY `158.414` 付近、平均 spread `0.8 pips`、3分レンジ `3.6 pips`。
  - `logs/factor_cache.json` の M1: `atr_pips=2.53`, `range_score=0.331`, `regime=Range`。
  - `logs/orders.db` 直近1h: `filled=272`, `rejected=2`, `brain_shadow/brain_apply` 系以外の API 崩れは見えず続行可能。

- 問題:
  - `logs/brain_state.db` の直近 `context_json` では `spread_pips` / `atr_pips` / `ticks` が欠落し、
    Brain は実質 `entry_probability + recent_outcome` だけで判断していた。
  - `brain_runtime_param_autotune_profit_latest.json` も
    `market_summary.spread_pips.count=0`, `atr_pips.count=0` で、
    local LLM の runtime tuning が市場条件を学習できていなかった。
  - `workers/common/brain.py` の cache key は `(strategy_tag, pocket)` 固定だったため、
    同一 strategy 内の side 違い・確率帯違いでも 15 秒は同じ判断を再利用し得た。
  - 同 report は `error=no_response` を返しており、safe profile の async autotune timeout も短かった。

- 対応:
  - Brain context へローカル tick (`bid/ask/mid/spread_pips/age_sec`) と
    `factor_cache` の M1 snapshot (`atr_pips`, `range_score`, `adx`, `rsi`, `regime`) を
    fallback 注入する。
  - Brain cache を `strategy+pocket` 固定から、
    `side + probability/confidence/spread/ATR/recent_outcome bucket` を含む
    setup fingerprint 単位へ変更する。
  - shared Ollama runtime では background prompt/runtime autotune が
    live preflight 実行中/直後を `live_preflight_inflight` /
    `live_preflight_cooldown` として退避し、safe canary の `no_llm` を
    自己衝突で増やさないようにする。
  - safe profile に `BRAIN_PROMPT_AUTO_TUNE_TIMEOUT_SEC=12`,
    `BRAIN_RUNTIME_PARAM_AUTO_TUNE_TIMEOUT_SEC=12`,
    `BRAIN_AUTOTUNE_LIVE_PRIORITY_COOLDOWN_SEC=30` を追加し、
    preflight latency を増やさずに background autotune の `no_response` を減らす。

- 検証方針:
  - `tests/workers/test_brain_ollama_backend.py` に
    `side 別 cache 分離` と `live tick/M1 factor fallback` の unit test を追加する。
  - 反映後は `brain_state.db.context_json` の `spread_pips` / `atr_pips` / `ticks` 充足率と、
    `brain_*_autotune_profit_latest.json` の `market_summary` 非ゼロ化を監査する。

## 2026-03-09 21:45 JST - Brain timeout 連発時は fail-open cooldown に退避し、entry cadence を落とさない

- 市況確認:
  - `prepare_local_brain_canary.py --warmup`: USD/JPY `158.424/158.416`、spread `0.8 pips`、ATR proxy `2.834 pips`、recent 6m range `3.0 pips`、Ollama warmup `1059 ms` で続行可能。
  - `logs/orders.db` 直近1h: `preflight_start=24`, `filled=24`, `brain_shadow=8` で、Brain block は出ていない。

- 問題:
  - `logs/brain_state.db` 直近6hでは `MicroLevelReactor-bounce-lower` が `llm_fail` を 5 回出しており、21:18-21:19 JST に `4001-4009 ms` timeout が連発していた。
  - safe canary は fail-open なので entry 数自体は落としていないが、同一 strategy/pocket の近接 setup ごとに 4 秒待ちを繰り返し、timing 劣化で cadence を毀損していた。

- 対応:
  - `workers/common/brain.py` に strategy/pocket 単位の failfast state を追加し、timeout / no-response が連続した場合は live LLM 呼び出しを短時間 skip して `llm_fail_fast` として fail-open へ切り替える。
  - safe canary env に
    `BRAIN_FAILFAST_CONSECUTIVE_FAILURES=2`,
    `BRAIN_FAILFAST_COOLDOWN_SEC=30`,
    `BRAIN_FAILFAST_WINDOW_SEC=60`
    を追加し、micro entry の cadence を守る。

- 意図:
  - local LLM が不安定な瞬間に「さらに待つ」より、safe canary の fail-open 原則で timing を優先する。
  - entry 頻度は落とさず、LLM が読める局面では判断を使い、読めない瞬間だけ遅延を切り離す。

## 2026-03-09 22:45 JST - local-v2: `TickImbalance` / `WickReversalBlend` 停止原因は `stage_tracker` の naive/aware UTC 混在

- 市況確認:
  - OANDA `pricing(USD_JPY)` は `2026-03-09 13:37:59Z` 時点で
    `bid=158.236 / ask=158.244`、spread 約 `0.8 pips`、`elapsed_ms=227`。
  - OANDA `candles(USD_JPY, M5, count=20)` からの直近 complete bar 基準では
    `ATR14 ≒ 7.879 pips`、直近 1h range `41.6 pips`。
  - `logs/health_snapshot.json` では直近 24h `trades_count_24h=370`、
    `data_lag_ms≈315`, `decision_latency_ms≈29` で、相場/導線とも作業継続可能。

- 実測:
  - `scripts/local_v2_stack.sh status --profile trade_all --env ops/env/local-v2-stack.env` で、
    明確に止まっていた entry worker は
    `quant-scalp-tick-imbalance` と `quant-scalp-wick-reversal-blend` の 2 本だけ。
  - 両ログ末尾は
    `TypeError: can't subtract offset-naive and offset-aware datetimes`
    で、`execution/stage_tracker.py:is_blocked()` の
    `cooldown_until - current` が crash 点だった。
  - 24h 損益寄与は
    `MicroLevelReactor: 237 trades / PF 1.338 / +97.4 pips`,
    `RangeFader: 79 / PF 1.182 / +14.7 pips`,
    `MomentumBurst: 22 / PF 0.807 / -7.3 pips`,
    `TickImbalance: 1 / -8.0 pips`。
    停止 2 本をここで増量する根拠は無く、まず「停止解除」が優先。
  - 7d では `MomentumBurst` は `53 trades / PF 2.797 / +110.5 pips` と戻り余地があり、
    一方 `M1Scalper-M1` と `scalp_ping_5s_b_live` は依然 loser。
    「全部総動員」は、loser widening ではなく crash recovery + winner 維持で進めるべき。

- 対応:
  - `execution/stage_tracker.py`
    の public naive UTC contract は維持しつつ、
    `ensure_cooldown()` / `is_blocked()` では `_coerce_utc()` で再正規化してから比較/減算するよう修正。
  - `tests/test_stage_tracker.py` に
    naive cooldown public value でも `is_blocked()` / `ensure_cooldown()` が動く回帰を固定。

- 検証:
  - `pytest -q tests/test_stage_tracker.py`
  - 反映後に
    `scripts/local_v2_stack.sh restart --profile trade_all --env ops/env/local-v2-stack.env`
    を実行し、
    `quant-scalp-tick-imbalance` / `quant-scalp-wick-reversal-blend` が
    `running` に戻ることを確認する。

## 2026-03-09 22:25 JST - 損失拡大RCAを踏まえ、MomentumBurst short と MicroLevelReactor long を strategy-local に再調整

- 市況確認:
  - `scripts/pdca_profitability_report.py --instrument USD_JPY --top-n 5`
    時点の OANDA snapshot は `USD/JPY 158.300`、spread `0.8 pips`、
    open trades `0` で、常時異常 spread ではなく局所スパイク中心だった。
  - `logs/metrics.db` 直近24h は `decision_latency_ms avg=17.6 / max=1401.8`、
    `data_lag_ms avg=1369 / max=279272`、
    `order_success_rate avg=0.992`、
    `reject_rate avg=0.008` で、導線停止より
    `close_reject_no_negative` ノイズと戦略期待値劣化が主因。

- 実測:
  - `trades.db` 24h は `369 trades / win_rate 52.3% / PF(pips) 1.12 / net_jpy -1360.6`。
  - `MomentumBurst` は `long +112.1` に対し `short -598.5` で、
    loser short の平均は `RSI 35.3`, `trend_gap_pips -3.04`, `atr_pips 3.39` と
    既に伸び切った down move の chase が目立った。
  - `MicroLevelReactor` long は `235 trades / net_jpy -180.6` で、
    `pattern_meta.shape=trend_dn` かつ `wick=upper` が
    `22 trades / net_jpy -384.3` と最大の負け cluster だった。

- 対応:
  - `strategies/micro/momentum_burst.py`
    で short 専用の RSI/drift/reaccel threshold を env 化し、
    `RSI low + trend_gap large negative + EMA stretch` の exhausted short を skip する guard を追加。
  - `strategies/micro/level_reactor.py`
    で long breakout の RSI floor を引き上げ、
    `bounce-lower` は lower wick 優位かつ bearish continuation でない candle だけ通すようにした。
  - `ops/env/quant-micro-levelreactor.env`
    で shared range-score floor は維持したまま、
    long 向け RSI / candle confirmation 閾値を明示。
  - `ops/env/quant-micro-momentumburst.env`
    で short exhaustion / short reaccel の閾値を dedicated env として固定。

- 意図:
  - 時間帯封鎖や shared gate 強化ではなく、
    「MomentumBurst short の late chase」と
    「MicroLevelReactor long の反発確認前エントリー」
    という敗因 cluster を strategy-local に潰しつつ、
    shared sizing や broad participation は落とさない。

## 2026-03-09 23:15 JST - local-v2 の loser size leak を止め、M1Scalper flipped continuation を cluster guard で抑制

- 市況確認:
  - OANDA pricing は `USD/JPY 158.322 / 158.330`、spread `0.8 pips`、
    pricing 応答 `206ms`、`openTradeCount=4`、`openPositionCount=1`、
    `pendingOrderCount=8` だった。
  - `logs/health_snapshot.json` は `generated_at=2026-03-09T14:01:45Z`、
    `data_lag_ms=249.3`, `decision_latency_ms=16.6` で、
    導線停止ではなく strategy / sizing 側の期待値悪化が主因だった。

- 実測:
  - `trades.db` 直近 24h は
    `MomentumBurst -486.4 JPY`, `MicroLevelReactor -175.3 JPY`,
    `MicroTrendRetest-long/-short -616.4 JPY`, `RangeFader -50.9 JPY`。
  - `config/dynamic_alloc.json` 再生成前提では
    `MicroTrendRetest-long 0.365`, `MicroTrendRetest-short 0.45`,
    `MicroLevelReactor 0.452`, `MomentumBurst 0.785`, `RangeFader 0.213`
    まで縮んでいるのに、micro runtime の shared `strategy_units_mult`
    が後段で loser の size を戻し得る構造だった。
  - `M1Scalper-M1` は live で tag filter 済みでも、
    `sell-rally -> trend-long` / `buy-dip -> trend-short` の flipped continuation が
    `ultra_low ATR × tight BBW × stretched gap × RSI extreme` で
    7d loser cluster を作っていた。

- 対応:
  - `workers/micro_runtime/worker.py`
    で、`dynamic_alloc` が `lot_multiplier < 1.0` もしくは
    history-underperforming を返した戦略には
    `MICRO_MULTI_STRATEGY_UNITS_MULT` の正方向 boost を上乗せしないようにした。
    `entry_thesis` には `strategy_units_mult_applied` / `strategy_units_mult_guard`
    を残し、監査可能にした。
  - `systemd/quant-dynamic-alloc.service`
    を `--half-life-hours 24` へ変更し、
    直近 24-48h の悪化を 7d 勝ち残りより早く sizing に反映するようにした。
  - `strategies/scalping/m1_scalper.py` と
    `ops/env/quant-m1scalper.env`
    に flipped continuation 専用の extreme guard を追加し、
    `ATR<=3.2`, `BBW<=0.0014`, `trend_gap>=2.5 pips`,
    `long RSI>=58` / `short RSI<=42` の反転追随を strategy-local に reject するようにした。

- 意図:
  - loser lane の shared re-inflation を止めつつ、
    winner lane (`MomentumBurst`) は相対優位を保ったまま残す。
  - `M1Scalper` は broad tag filter に頼らず、
    直近 loser cluster だけを strategy-local に切り落とす。

## 2026-03-10 00:35 JST - `MomentumBurst short` と `MicroLevelReactor long` の live loser cluster を strategy-local に圧縮

- 市況確認:
  - OANDA pricing は `USD/JPY 158.210 / 158.226`、mid `158.218`、spread `1.6 pips`、
    pricing 応答 `avg 225.5ms`、M1 ATR14 `3.821 pips`、M5 ATR14 `10.143 pips`、
    12-bar M5 range `23.8 pips` で、作業保留が必要な異常市況ではなかった。
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    では `quant-market-data-feed / quant-strategy-control / quant-order-manager /
    quant-position-manager` を含む local-v2 主要サービスが稼働中だった。
  - `logs/metrics.db` 24h は `decision_latency_ms avg=17.613 / max=1401.776`、
    `data_lag_ms avg=1279.118 / max=279272.689`、`order_success_rate avg=0.99`、
    `reject_rate avg=0.01` で、導線停止や reject 多発は主因ではなかった。

- 実測:
  - `trades.db` 直近24h は `384 trades / win_rate 51.3% / PF 0.596 /
    expectancy -4.0 JPY / net -1550.1 JPY`。
  - `MomentumBurst` は `24 trades / -617.2 JPY / avg_units 4171.5` で、
    `short 11 trades / -598.5 JPY / avg_units 4458.0` が主犯だった。
    loser short の平均 `pattern_meta.rsi=35.3`、最小 `23.06` で、
    `trend_snapshot.direction=long, gap_pips>=31.8, adx>=30.5`
    に対して oversold short chase が走っていた。
  - `MicroLevelReactor` は `237 trades / -175.3 JPY` で、
    `long 235 trades / -180.6 JPY / avg_units 264.8`。
    loser cluster は
    `c:trend_dn|w:upper|tr:flat|rsi:mid_low|vol:tight|atr:low|d:long`
    が `7 trades / -220.7 JPY`、
    `c:trend_dn|w:upper|tr:dn_strong|rsi:neutral|vol:tight|atr:ultra_low|d:long`
    が `6 trades / -167.5 JPY` で、
    bearish continuation candle の bounce-lower long が中心だった。
  - `orders.db` 24h の entry reject は `STOP_LOSS_ON_FILL_LOSS` 4件のみで、
    `filled 384 / rejected 4`。open trades も 0 で、
    期待値悪化を strategy-local に止血する局面だった。

## 2026-03-10 01:50 JST - micro loser cluster の緊急止血

- 実測:
  - 調査窓 `2026-03-09 01:50 JST -> 2026-03-10 01:50 JST` は
    `396 trades / -1839.3 JPY / PF 0.554 / win_rate 49.75%`。
  - 市況は通常帯で、`USD/JPY 157.976 / spread 0.8 pips / M1 ATR 3.115 pips`。
    `quant-market-data-feed` / `quant-order-manager` は稼働、口座建玉は 0。
  - 赤字主因は `MomentumBurst -617.2 JPY (24 trades)`、
    `MicroTrendRetest-long -290.2 JPY (17 trades)`、
    `MicroLevelReactor -175.3 JPY (237 trades)`。
  - `MomentumBurst` は `STOP_LOSS_ORDER 11件 / -1820.6 JPY` が重く、
    short 側だけで `-598.5 JPY`。`MicroTrendRetest-long` は
    `2026-03-10 01:23 JST` 台に同系セットアップを連打して `-214.4 JPY`。

- 変更:
  - `ops/env/quant-micro-momentumburst.env`
    で `MomentumBurst` の dedicated sizing を `0.90` へ下げ、
    short 側は `MOMENTUMBURST_SHORT_DRIFT_CEIL=-0.05`,
    `MOMENTUMBURST_SHORT_EXHAUSTION_RSI_MAX=40`,
    `MOMENTUMBURST_REACCEL_DI_GAP_SHORT=8.0`
    として low-ATR / tight 文脈の late short を削る。
  - `ops/env/quant-micro-trendretest.env`
    で `MicroTrendRetest` の dedicated sizing を `0.85` にし、
    `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=24`
    を追加して同一分足での burst stacking を止める。

- 意図:
  - 共通 gate や時間帯 block は追加せず、
    直近24hの loser cluster を strategy-local / dedicated env のみで止血する。

## 2026-03-10 08:25 JST - `MomentumBurst` / `MicroTrendRetest` の strategy-local quality guard を追加

- 実測:
  - 直近24hは `455 trades / -1885.5 JPY / PF 0.551` まで悪化。
  - `MomentumBurst -617.2 JPY`、`MicroTrendRetest-long -290.2 JPY` が継続して主因。
  - 市況は `USD/JPY 157.807 / spread 0.8 pips / M1 ATR 2.457 pips` と通常帯で、
    「低ATR / tight 文脈での strategy-local quality 不足」が継続していた。

- 変更:
  - `strategies/micro/momentum_burst.py`
    に `tight short context` 判定を追加し、
    low-ATR / low-vol / chop-range 文脈では
    `drift_pips`, `DI gap`, `ROC5`, `ema_slope_10`
    が十分に downside impulse を示す short だけを通すようにした。
  - `strategies/micro/trend_retest.py`
    に `retest close recovery` 判定を追加し、
    低ATR時に retest 極値へ貼り付いたままの long/short を reject するようにした。
  - `tests/strategies/test_momentum_burst.py`
    と `tests/strategies/test_trend_retest.py`
    へ loser cluster 向けの境界テストを追加した。

- 検証:
  - `pytest tests/strategies/test_momentum_burst.py tests/strategies/test_trend_retest.py tests/workers/test_micro_multistrat_trend_flip.py`
    は `50 passed`。

- 意図:
  - shared gate 追加ではなく、
    `MomentumBurst short` と `MicroTrendRetest` の
    low-ATR / tight loser cluster を strategy 本体で落とす。

- 変更:
  - `strategies/micro/momentum_burst.py`
    で short 側の stretched quality 判定を
    `REACCEL_DI_GAP_SHORT / REACCEL_ROC5_MIN_SHORT` に揃えた。
    さらに oversold short exhaustion 判定を
    「stretch していれば generic early-return より先に落とす」順序へ変更した。
  - `ops/env/quant-micro-momentumburst.env`
    で `MOMENTUMBURST_RSI_SHORT_MIN=36`,
    `MOMENTUMBURST_RSI_SHORT_MAX=42`,
    `MOMENTUMBURST_SHORT_DRIFT_CEIL=0.10`,
    `MOMENTUMBURST_REACCEL_COOLDOWN_SEC=45`
    とし、oversold short と連続 reaccel long を絞った。
  - `strategies/micro/level_reactor.py`
    で breakout-long に supportive candle 条件を追加し、
    `bounce-lower` は bearish continuation candle を confidence 減点ではなく
    hard reject に変更した。
  - `ops/env/quant-micro-levelreactor.env`
    で `MLR_LONG_RSI_MIN=54`,
    `MLR_LONG_BOUNCE_RSI_MAX=52`,
    `MLR_BOUNCE_BODY_BEAR_MAX_PIPS=0.8`,
    `MLR_BOUNCE_MIN_LOWER_WICK_PIPS=1.0`
    とし、long 側の entry quality を引き上げた。

- 検証:
  - `pytest tests/strategies/test_momentum_burst.py tests/strategies/test_level_reactor.py`
    は `23 passed`。

- 意図:
  - shared gate や時間帯 block は増やさず、
    `MomentumBurst short` の late chase と
    `MicroLevelReactor long` の bearish bounce を
    strategy-local に圧縮する。

## 2026-03-10 08:45 JST - forecast を local-v2 live 導線へ接続（narrow canary）

- 実測:
  - 市況は通常帯。`USD/JPY 157.83` 近辺、recent spread は `0.8 pips`、
    `factor_cache` の `M1 ATR` は `1.67 pips`、`decision_latency_ms` は約 `14.5ms`、
    `data_lag_ms` は約 `1869ms` で live entry 導線の健全性は維持されていた。
  - 直近24hでは `MomentumBurst`, `MicroTrendRetest`, `MicroLevelReactor` が
    主な毀損源だが、forecast は未接続ではなく strategy-side の
    forecast context / fusion と worker-side gate では既に利用されていた。
  - 一方で dedicated `quant-forecast` は local-v2 stack 管理外で、
    order-manager 側も `ORDER_MANAGER_FORECAST_GATE_ENABLED=0` に加え
    `preserve_strategy_intent` 下では forecast 分岐へ到達しないため、
    service を live 発注直前へ活かせていなかった。

- 変更:
  - `local_v2_stack` の trade profile に `quant-forecast` を追加し、
    restart / watchdog / autorecover の対象へ組み込んだ。
  - `order_manager` へ
    `ORDER_MANAGER_FORECAST_GATE_APPLY_WITH_PRESERVE_INTENT`
    を追加し、`preserve_strategy_intent` を維持したまま
    dedicated forecast gate を opt-in で使えるようにした。
  - `ops/env/quant-order-manager.env` で
    `ORDER_MANAGER_FORECAST_GATE_ENABLED=1` /
    `FORECAST_GATE_ENABLED=1` /
    `ORDER_MANAGER_FORECAST_GATE_APPLY_WITH_PRESERVE_INTENT=1`
    を有効化し、allowlist を
    `MicroLevelReactor, MomentumBurst, MicroTrendRetest, M1Scalper-M1, RangeFader, scalp_ping_5s_b_live, scalp_ping_5s_c_live, scalp_ping_5s_flow_live`
    に限定して narrow canary 化した。

- 意図:
  - `forecast_improvement_latest.json` の verdict が `mixed` のため、
    全戦略一括ではなく毀損寄与戦略と既存 tuned strategy に限定して
    order-manager forecast gate を live 接続する。
  - strategy-local forecast/fusion を壊さず、
    dedicated service の `allow/reduce/block` を restart 耐性つきで
    live 経路へ反映する。

## 2026-03-10 09:15 JST / local-v2 profitability hotfix: current loser cluster と stale 14d feedback の窓ズレ

- 市況確認:
  - `logs/tick_cache.json`
    - 最新 tick は `mid 157.932`, spread は約 `0.8 pips`
  - `logs/factor_cache.json`
    - `M1 close=157.947`, `atr_pips=3.31`, `timestamp=2026-03-10T00:09:59Z`
  - `logs/health_snapshot.json`
    - `generated_at=2026-03-10T00:05:20Z`, `trades_last_close=2026-03-09T23:03:40Z`,
      `orders_status_1h` は `entry_probability_reject=18` が主で、
      異常停止を示す snapshot ではなかった

- 実測:
  - 直近24hの loser cluster は
    `MomentumBurst 22 trades / -7.9 pips`,
    `MicroTrendRetest-long 17 trades / -45.7 pips`,
    `MicroTrendRetest-short 2 trades / -10.6 pips`
    に集中した。
  - `MicroLevelReactor` は 24h 合算では `220 trades / +51.1 pips` だが、
    勝敗数は `95W / 120L` で、直近6hは `34 trades / -36.1 pips` の loser burst へ反転している。
  - `logs/strategy_feedback.json` は `updated_at=2026-03-10T00:04:35Z`,
    `version=2026-02-24`, `7 strategies` を保持し、
    `MomentumBurst` に `entry_probability_multiplier=1.0732`,
    `entry_units_multiplier=1.3107` を、
    `MicroLevelReactor` に `entry_units_multiplier=1.141`,
    `tp_distance_multiplier=1.0407` をまだ返していた。
  - 一方で `MicroTrendRetest-long/short` は current loser cluster に入っているのに
    `strategy_feedback.json` には現れず、split tag 側の current loser を
    live へ十分速く戻せていない。

- 判断:
  - `STRATEGY_FEEDBACK_LOOKBACK_DAYS=14` は local-v2 の cadence に対して長すぎ、
    直近24h-6hの悪化より 14d aggregate を優先して stale boost を残している。
  - broad に shared gate を増やすより、
    dedicated env 側で loser cluster の参加率と cadence を先に落とし、
    `strategy_feedback` の窓だけを `3d` へ縮めるのが最小リスク。

- 即時 hotfix 方針:
  - `MomentumBurst`, `MicroTrendRetest`, `MicroLevelReactor` は
    dedicated env 側の units / cooldown / cadence を先に de-risk する。
  - `quant-strategy-feedback` は止めず、
    `STRATEGY_FEEDBACK_LOOKBACK_DAYS` を `14 -> 3` へ寄せて
    current loser turn で stale boost を早く中立化する。
  - `MicroLevelReactor` のように 24h aggregate がまだ残る戦略は、
    全停止ではなく `entry_units_multiplier <= 1.0` を先に狙い、
    loser burst 収束後に再評価する。

## 2026-03-10 09:28 JST / local-v2 profitability hotfix: current loser 3戦略へ forecast sub-guards を追加

- 実測:
  - `orders.db` 直近24hの mechanism status は `brain_shadow=21` のみで、
    current loser 3戦略に対する `forecast_*` / `pattern_*` の reject は観測されなかった。
  - `ops/env/profiles/brain-ollama-safe.env` の live canary は
    `BRAIN_STRATEGY_ALLOWLIST=MicroLevelReactor` で、Brain は現行 safe canary のまま。
  - `config/pattern_book.json` に `MomentumBurst / MicroTrendRetest / MicroLevelReactor`
    の top-level strategy entry はなく、`logs/patterns.db` の `pattern_actions` も
    current loser 3戦略では大半が `learn_only / insufficient_samples` だった。

- 判断:
  - 「全部の仕組み」を広げる候補のうち、今すぐ live で意味を持つ追加レバーは
    pattern/brain の broad opt-in ではなく、既に live な forecast gate の
    strategy-local sub-guards を loser 3戦略に足すことだった。

- 反映:
  - `MomentumBurst`
    - `FORECAST_GATE_STYLE_TREND_MIN_STRENGTH_STRATEGY_MOMENTUMBURST=0.56`
    - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_MOMENTUMBURST=0.54`
    - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_MOMENTUMBURST=0.14`
    - `FORECAST_GATE_EXPECTED_PIPS_CONTRA_MAX_STRATEGY_MOMENTUMBURST=-0.02`
    - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_MOMENTUMBURST=0.18`
  - `MicroTrendRetest`
    - `FORECAST_GATE_STYLE_TREND_MIN_STRENGTH_STRATEGY_MICROTRENDRETEST=0.58`
    - `FORECAST_GATE_EDGE_BLOCK_STRATEGY_MICROTRENDRETEST=0.58`
    - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_MICROTRENDRETEST=0.18`
    - `FORECAST_GATE_EXPECTED_PIPS_CONTRA_MAX_STRATEGY_MICROTRENDRETEST=-0.02`
    - `FORECAST_GATE_TARGET_REACH_MIN_STRATEGY_MICROTRENDRETEST=0.22`
  - `MicroLevelReactor`
    - `FORECAST_GATE_EXPECTED_PIPS_MIN_STRATEGY_MICROLEVELREACTOR=0.16`
    - `FORECAST_GATE_EXPECTED_PIPS_CONTRA_MAX_STRATEGY_MICROLEVELREACTOR=-0.02`

- 意図:
  - shared gate を増やさず、既存 forecast runtime override と
    dedicated service env だけで loser lane の low-edge / contra / weak-target-reach を
    先に落とす。
## 2026-03-10 11:00 JST / local-v2 regime coverage: thin transition/chop 向け precision wrapper を追加

- 実測:
  - local-v2 の `2026-03-10 10:59-11:00 JST` は
    `USD/JPY 157.622/157.630`, `spread=0.8p`, 直近レンジ `5分 2.0p / 15分 3.9p / 60分 11.0p`。
  - `factor_cache.json` は
    `M1 RSI 42.3 / ADX 16.5 / -DI優位`,
    `M5 RSI 43.8 / ADX 20.7 / -DI優位`,
    `H1 RSI 39.0 / -DI優位`,
    `H4 RSI 52.8 / +DI優位`
    で、強い順張りではなく `transition/chop`。
  - `local_v2_stack/pids/*.pid` では既に trade_all 相当の多数 worker が稼働していた一方、
    `orders.db` / `trades.db` の直近60分は
    `scalp_extrema_reversal_live` の `3 trades / +0.56 JPY` 以外は寄与が薄く、
    `LevelReject`, `WickReversalBlend`, `MicroVWAPRevert`, `MicroVWAPBound`, `FalseBreakFade`, `TickImbalance`
    では新規約定が見えなかった。

- 判断:
  - 問題は「worker 数不足」ではなく、
    既存多数 worker の中でも
    `低ボラ / 薄い range / entry drought`
    を専用に埋める scalp lane が足りないことだった。
  - broad gate 緩和ではなく、既存 precision base を再利用した
    wrapper worker を足す方が速く、shared layer を汚さない。

- 反映:
  - 新規 ENTRY/EXIT ペア:
    - `quant-scalp-precision-lowvol`
    - `quant-scalp-vwap-revert`
    - `quant-scalp-drought-revert`
  - いずれも `workers.scalp_wick_reversal_blend` の
    precision base を env projection wrapper で再利用し、
    `SCALP_PRECISION_*` を strategy-local prefix へ写して動かす。
  - `precision_lowvol` は
    `ADX/BBW/ATR` が細い帯での BB touch + tick reversal を拾う。
  - `vwap_revert` は
    VWAP gap を使った内側回帰を拾う。
  - `drought_revert` は
    `DROUGHT_MINUTES=6` の entry drought 時だけ gap-filler として起動する。
  - `scripts/local_v2_stack.sh` の `PROFILE_trade_min` / `PROFILE_trade_all`
    へ 3 ペアを追加し、通常 restart / watchdog 後も維持する。

- 意図:
  - 「どんな局面でも入る」を shared gate の無差別緩和ではなく、
    `transition/chop` と `entry drought` を埋める strategy-local lane 増設で実現する。
  - 既存 momentum / breakout lane を壊さず、
    薄い相場での participation だけを厚くする。

## 2026-03-10 12:27 JST / `RangeFader` profitable buy cluster を `perf_block` 履歴から分離

- 実測:
  - `2026-03-10 12:27 JST` 時点の USD/JPY は
    `157.846/157.854`, spread `0.8p`, 直近レンジ `300 ticks=2.0p / 900 ticks=4.3p`。
    `M1 RSI 54.76 / ADX 29.33 / +DI 23.39 / -DI 17.03`,
    `M5 RSI 63.6 / ADX 20.4` で、pricing stream は `200 OK` 継続だった。
  - `RangeFader-buy-fade` の直近72h注文は
    `105 preflight_start / 25 filled / 86 entry_probability_reject / 80 perf_block`。
  - 一方で `RangeFader` buy side の直近72hトレードは
    `33 trades / +13.17 JPY / avg +0.67 pips / win_rate 84.8%` で、
    buy side 自体は profitable を維持していた。
  - 直近30-90分の live no-fill 主因は `RangeFader-buy-fade` で、
    `MomentumBurst` は signal formation 自体が 0 件だったため、
    今回の participation 改善対象から外した。

- 判断:
  - broad な `confidence` 引き上げや shared gate 緩和ではなく、
    profitable な buy cluster だけを別 tag に切り出して
    directional `perf_block` 履歴から分離するのが最も狭い改善だった。
  - `workers.common.perf_guard` の `split_directional=true` を前提に、
    `RangeFader-buy-supportive` を新設すれば
    既存 `RangeFader-buy-fade` / `sell-fade` / `neutral-fade` の failfast 履歴を継承しない。

- 反映:
  - `strategies/scalping/range_fader.py`
    - `plus_di/minus_di`, `ema_slope_10`, spread, ADX, mean 乖離で
      `buy_supportive` 文脈を判定。
    - 元の `long_gate` を超えた近傍RSIでも、
      supportive buy 文脈だけは `RangeFader-buy-supportive` を返す。
    - `RangeFader-buy-supportive` にだけ
      `RANGE_FADER_BUY_SUPPORT_CONF_BONUS=6` を加え、
      `confidence 45 -> 51` 相当まで引き上げて
      `entry_probability_reject` 側の目減りを減らす。
    - sell / neutral の判定は変更しない。
  - `workers/scalp_rangefader/config.py`
    - `BUY_COOLDOWN_SEC` を追加し、既定を `COOLDOWN_SEC * 0.7` とした。
  - `workers/scalp_rangefader/worker.py`
    - `RangeFader-buy-*` tag だけ短い cooldown を使い、
      sell / neutral は従来 cooldown を維持する。

- 意図:
  - shared `order_manager` / `perf_guard` / dynamic alloc を触らず、
    profitable buy cluster の participation だけを strategy-local に戻す。
  - loser history を引きずる既存 `RangeFader-*` tag を壊さず、
    `supportive buy` のみ別 lane として監査可能にする。

## 2026-03-10 13:11-13:20 JST / `scalp_extrema_reversal_live` long を M5-supportive shallow pullback へ拡張

- 実測:
  - `2026-03-10 13:11 JST` の USD/JPY は
    `157.888/157.896`, spread `0.8p`,
    `M1 RSI 49.6 / ADX 16.8`,
    `M5 close > ema20 / RSI 61.2 / ADX 21.2` で、
    強トレンドではなく `M5 bullish shallow pullback` だった。
  - `2026-03-10 13:20 JST` でも spread は `0.8p` のまま、
    feed の `data_lag_ms` は `~212ms` で local-v2 は正常帯を維持した。
  - `scalp_extrema_reversal_live` の直近90分 closed trade は
    `long 4 trades / +1.78 JPY / +2.8 pips / win_rate 75%`,
    `short 12 trades / -8.716 JPY / -15.8 pips / win_rate 16.7%` だった。
  - 同じ90分の注文は
    `buy filled=4` に対して `sell filled=12` で、
    shared preflight 側の reject ではなく long signal 生成が薄かった。

- 判断:
  - 今の live ボトルネックは shared gate ではなく、
    `EXTREMA_RSI_LONG_MAX=46.0` と `LOW_BAND_PIPS=0.9`
    が `M5 上向きの浅い押し目 long` を取り逃がすことだった。
  - broad な threshold 緩和ではなく、
    `M5 close >= ema20`, `M5 RSI`, `DI gap`, `ema_slope_10`,
    `M1 ADX`, `M1-ema20 gap`
    を満たすときだけ long の `RSI cap / low band / confidence`
    を少し広げるのが最も狭い改善だった。

- 反映:
  - `workers/scalp_extrema_reversal/worker.py`
    - `M5 supportive` 判定を追加し、
      条件を満たす long にだけ
      `RSI cap 46 -> 50`,
      `low band 0.9 -> 1.2 pips`,
      `confidence +4`
      を適用する。
    - short ロジック、shared preflight、order_manager は変更しない。
    - `entry_thesis.extrema` に
      `supportive_long`, `supportive_long_context`,
      `long_rsi_cap`, `long_low_band_pips`
      を残し、監査可能にした。
  - `ops/env/quant-scalp-extrema-reversal.env`
    - current 運用値として
      `LONG_SUPPORT_*` を dedicated env に明示した。

- 意図:
  - `short loser cluster` を shared 層で止めるのではなく、
    `M5 support 付き shallow long` だけを strategy-local に増やす。
  - `scalp_extrema_reversal_live` の既存 tag / exit 契約を維持し、
    反応速度だけを上げる。

### 2026-03-10 local-v2 `MicroLevelReactor` bounce-lower の no-wick countertrend probe を遮断

- 市況:
  - `2026-03-10 14:56-14:57 JST` の USD/JPY は
    `157.664/157.672`, spread `0.8p`,
    直近レンジが `60/120/300/900 ticks = 1.6/2.1/2.5/5.6 pips`。
  - `M1 close 157.667 < ema20 157.684`,
    `M5 close 157.685 < ema20 157.729`,
    `H1 close 157.778 < ema20 157.976` で、
    短期は still soft-down だった。
  - `health_snapshot.json` は fresh、`git_rev=98295a99`、
    `data_lag_ms ~= 685` で local-v2/OANDA 応答は通常帯だった。

- 実測:
  - 直近90分の closed trade は
    `MicroLevelReactor 6 trades / -13.056 JPY / -11.9 pips / win_rate 0%`
    で最大の active loser だった。
  - 同 cluster は `2026-03-10 04:17 UTC` に 6 連続 long fill され、
    すべて `entry_probability=0.4901`, `confidence=76`,
    `pattern_tag=c:maru_up|w:none|tr:dn_strong|rsi:os|vol:tight|atr:ultra_low|d:long`
    だった。
  - つまり負け筋は `bounce-lower` 自体ではなく、
    `local MA gap が down-strong` なのに
    `下ヒゲなしの陽線 probe` を反発と誤認した部分だった。

- 対応:
  - `strategies/micro/level_reactor.py`
    - `bounce-lower` で `ma10-ma20 <= -0.6 pips` の countertrend probe を検知したときだけ、
      `body >= 0.2 pips`,
      `lower wick >= 1.0 pips`,
      `lower wick > upper wick`
      を追加要件にした。
    - local MA gap が down-strong でないときは、
      従来どおり `body-only reclaim` も通す。
  - `ops/env/quant-micro-levelreactor.env`
    - `MLR_BOUNCE_COUNTERTREND_MIN_GAP_PIPS=0.6`
    - `MLR_BOUNCE_COUNTERTREND_MIN_BODY_PIPS=0.2`
    - `MLR_BOUNCE_COUNTERTREND_MIN_LOWER_WICK_PIPS=1.0`
    - `MLR_BOUNCE_CONTINUATION_ATR_MAX=1.8`
    - `MLR_BOUNCE_CONTINUATION_ADX_MIN=22.0`
    - `MLR_BOUNCE_CONTINUATION_DI_GAP_MIN=18.0`
    - `MLR_BOUNCE_CONTINUATION_MIN_LOWER_WICK_PIPS=0.4`
      を dedicated env に明示した。
  - 同じ `bounce-lower` 内で
    `ATR <= 1.8`, `ADX >= 22`, `minus_di - plus_di >= 18`
    の `continuation probe` も別扱いし、
    `tiny lower wick` のままでは long しないようにした。

- 意図:
  - broad に `MicroLevelReactor` を止めず、
    `dn_strong + no-wick` の loser cluster だけを strategy-local に削る。
  - `ultra-low ATR + strong -DI` の continuation probe も同じ branch で削る。
  - shared preflight / sizing / exit worker は変更しない。

- 検証:
  - `python3 -m pytest -q tests/strategies/test_level_reactor.py`
  - `python3 -m py_compile strategies/micro/level_reactor.py tests/strategies/test_level_reactor.py`

### 2026-03-10 local-v2 `scalp_extrema_reversal_live` の non-supportive long countertrend を遮断

- 市況:
  - `2026-03-10 15:58 JST` の USD/JPY は
    `157.536/157.544`, spread `0.8p`。
  - `M5 close 157.569 < ema20 157.611`,
    `H1 close 157.641 < ema20 157.944` で、
    短中期は still soft-down だった。

- 実測:
  - 直近90分の `scalp_extrema_reversal_live` は
    `28 trades / -12.929 JPY / -20.6 pips`。
  - 24h の long だけでも
    `19 trades / -7.568 JPY / win_rate 31.6%`。
  - このうち `supportive_long=false` かつ
    `trend_gate.ma_gap_pips <= -0.5`
    の long cluster は
    `7 trades / -5.267 JPY / -7.3 pips / win_rate 14.3%`
    だった。

- 対応:
  - `workers/scalp_extrema_reversal/worker.py`
    - `non-supportive long` で
      `ma10-ma20 <= -0.5 pips`
      の local countertrend gap を検知したら、
      `OPEN_LONG` を生成しないようにした。
    - `supportive_long=true` の long は、
      同じ ma gap でも従来どおり通す。
  - `ops/env/quant-scalp-extrema-reversal.env`
    - `SCALP_EXTREMA_REVERSAL_LONG_COUNTERTREND_GAP_BLOCK_PIPS=0.50`
      を dedicated env に明示した。

- 意図:
  - `scalp_extrema_reversal_live` 全体を止めず、
    `soft-down M1 gap に逆らう non-supportive long`
    だけを strategy-local に削る。
  - short 側と shared gate / order_manager / exit worker は変更しない。

- 検証:
  - `python3 -m pytest -q tests/workers/test_scalp_extrema_reversal_worker.py` -> `10 passed`
  - `python3 -m py_compile workers/scalp_extrema_reversal/worker.py tests/workers/test_scalp_extrema_reversal_worker.py`

## 2026-03-10 16:44-16:47 JST / local-v2 `precision_lowvol` と `drought_revert` crash 復旧 + `MomentumBurst` dedicated env を現行 cadence へ補正

- 市況:
  - `2026-03-10 16:44 JST` 時点の USD/JPY は
    local `tick_cache` で `157.326/157.334`, spread median/p95 `0.8p/0.8p`。
  - `factor_cache` は `M1 close 157.334 / ATR 3.29p / ADX 32.19 / RSI 34.16`,
    OANDA `pricing` は `200 OK / 200ms`, `candles` は `200 OK / 275ms`,
    `M1 ATR proxy 3.6p`, `20-candle range 17.9p` だった。
  - `health_snapshot.json` は fresh で
    `data_lag_ms ~= 161`, `decision_latency_ms ~= 17.5`,
    `orders_status_1h = entry_probability_reject 84 / perf_block 59 / filled 2`。
  - よって今回は spread 異常・API 劣化・グローバル lock ではなく、
    strategy-local / worker-side の entry cadence 不全として扱った。

- 実測:
  - `scripts/local_v2_stack.sh status` では
    `quant-scalp-precision-lowvol` と `quant-scalp-drought-revert`
    が `stale_pid_file` で停止していた。
  - 両 worker のログは
    `TypeError: _signal_precision_lowvol() missing 1 required positional argument: 'range_ctx'`
    と
    `TypeError: _signal_drought_revert() missing 1 required positional argument: 'range_ctx'`
    で即落ちしていた。
  - 同時に `quant-micro-momentumburst` の実効 env は
    `MOMENTUMBURST_REACCEL_COOLDOWN_SEC=45`,
    `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=150`
    のままで、`docs/RISK_AND_EXECUTION.md` /
    `docs/WORKER_REFACTOR_LOG.md` に残した現行値
    `35 / 120` より重かった。

- 対応:
  - `workers/scalp_wick_reversal_blend/worker.py`
    - strategy dispatch を `_dispatch_strategy_signal(...)` へ切り出し、
      `DroughtRevert` / `PrecisionLowVol` を
      `range_ctx` 必須 signal として明示した。
    - これで dedicated wrapper 経由でも
      `_signal_drought_revert` / `_signal_precision_lowvol`
      に `range_ctx` が渡るようにした。
  - `ops/env/quant-micro-momentumburst.env`
    - `MICRO_MULTI_STRATEGY_UNITS_MULT=MomentumBurst:1.05`
    - `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=120`
    - `MOMENTUMBURST_REACCEL_COOLDOWN_SEC=35`
      へ更新し、現行運用台帳と dedicated env のズレを解消した。

- 意図:
  - 「候補を作る lane が process crash で死んでいる」状態を先に戻し、
    shared gate を緩めずに entry source を復元する。
  - `MomentumBurst` は shared override を維持したまま、
    dedicated env の stale cooldown だけを current 値へ揃えて
    reaccel cadence を戻す。

- 検証:
  - `python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_dispatch.py`
  - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py`

## 2026-03-10 16:59-17:05 JST / local-v2 `trade_min` に winner-cover lane を昇格

- 市況:
  - `2026-03-10 16:59-17:00 JST` の USD/JPY は
    local `tick_cache` で `157.298/157.306`, spread median/p95 `0.8p/0.8p`。
  - `factor_cache` は `M1 close 157.345 / ATR 2.80p / ADX 30.06 / RSI 49.34`。
  - OANDA live は `pricing 200 OK / 328ms`, `candles 200 OK / 218ms`,
    closeout spread `1.7p`, `ATR20 proxy 2.78p`, `20-candle range 8.8p`。
  - よって市況悪化や API 劣化ではなく、participation 配置の問題として扱った。

- 実測:
  - `trade_min` の定義には
    `RangeFader` / `MicroRangeBreak` / `session_open`
    が含まれておらず、watchdog / restart で常設維持されない状態だった。
  - 一方で local 実行中には
    `quant-scalp-rangefader`, `quant-micro-rangebreak`, `quant-session-open`
    が別起動で `running` だった。
  - `trades.db` 180日集計では
    `MicroRangeBreak = 281 trades / +7752.2 JPY`,
    `RangeFader = 248 trades / +253.4 JPY`,
    `session_open_breakout = 3 trades / +11.7 JPY`。
  - 直近窓では `MomentumBurst` 以外の winner が薄く、
    active winner lane を watchdog 管理下へ戻すことを優先した。

- 対応:
  - `scripts/local_v2_stack.sh`
    - `PROFILE_trade_min` に
      `quant-scalp-rangefader(+exit)`,
      `quant-micro-rangebreak(+exit)`,
      `quant-session-open(+exit)`
      を追加した。

- 意図:
  - shared gate / sizing を緩めず、
    既に local で稼働実績のある winner-cover lane を
    `trade_min` の標準構成へ昇格させる。
  - これで `restart --profile trade_min` と watchdog 復旧後も、
    `RangeFader` / `MicroRangeBreak` / `session_open`
    が常設 worker として残る。

- 検証:
  - `bash -n scripts/local_v2_stack.sh`
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env`
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
  - `scripts/collect_local_health.sh`

## 2026-03-10 17:08-17:16 JST / `RangeFader` no-entry の主因は perf-guard metric と scaled min-units のズレ

- 市況:
  - `2026-03-10 17:09-17:10 JST` の USD/JPY は
    local `tick_cache` で `157.454/157.462`, spread median/p95 `0.8p/0.8p`。
  - `factor_cache` は `M1 close 157.451 / ATR 3.32p / ADX 29.06 / RSI 65.28`。
  - OANDA `pricing` は `200 OK / 304ms`、closeout `157.448/157.466`。
  - 市況停止ではなく order-manager 側の entry path が主因。

- 実測:
  - `orders.db` 直近60分は `filled=40` あるが、
    `RangeFader-buy/sell/neutral` だけで
    `preflight_start 1580 / perf_block 1580 / entry_probability_reject 689` と詰まっていた。
  - `quant-order-manager.log` の reject note は
    `perf_block:failfast_soft:pf=0.70 win=0.78 n=79`。
  - ただし `trades.db` の同じ `RangeFader` 7日集計は
    `pf_jpy=0.70` に対し `pf_pips=1.18`, `avg_pips=+0.186`, `win_rate=78.5%`。
    低単位・高頻度ゆえに `realized_pl` 評価でだけ failfast していた。
  - さらに probability-scaled intents の分布は
    `buy p50=38`, `sell p50=34`, `neutral p50=36` で、
    現行 `ORDER_MIN_UNITS_STRATEGY_RANGEFADER*=60` では大半が
    `entry_probability_below_min_units` になっていた。
  - 24h の `orders.db` 実分布では
    `RangeFader-buy-fade >=35` が `577/1070 (53.9%)`,
    `RangeFader-neutral-fade >=35` が `486/806 (60.3%)`,
    `RangeFader-sell-fade >=35` が `464/1108 (41.9%)`。
    `sell-fade` だけは `>=30` に下げると `748/1108 (67.5%)` まで回復するため,
    追加緩和は sell lane のみに限定する。

- 対応:
  - `ops/env/quant-order-manager.env`
    - `RANGEFADER_PERF_GUARD_VALUE_COLUMN=pl_pips`
    - `ORDER_MIN_UNITS_STRATEGY_SCALP_RANGEFAD=35`
    - `ORDER_MIN_UNITS_STRATEGY_RANGEFADER=35`
    - `ORDER_MIN_UNITS_STRATEGY_RANGEFADER_BUY_FADE=35`
    - `ORDER_MIN_UNITS_STRATEGY_RANGEFADER_SELL_FADE=30`
    - `ORDER_MIN_UNITS_STRATEGY_RANGEFADER_NEUTRAL_FADE=35`

- 意図:
  - shared perf-guard 全体は変えず,
    `RangeFader` だけ評価軸を `pl_pips` に切り替えて
    cost-heavy な JPY failfast を外す。
  - 同時に scaled `min_units` を現在の実分布へ合わせ,
    near-miss intents を注文まで通す。

- 検証:
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-order-manager`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-order-manager,quant-scalp-rangefader,quant-scalp-rangefader-exit`
  - `ps eww -p <quant-order-manager pid>`
  - `logs/local_v2_stack/quant-order-manager.log`

## 2026-03-10 17:34-17:39 JST / `scalp_extrema_reversal_live` の shallow countertrend long を追加遮断

- 市況:
  - `2026-03-10 17:34 JST` の USD/JPY は local `tick_cache` で
    `157.492/157.500`, spread `p50/p95=0.8p/0.8p`, tick age `0.605s`。
  - `factor_cache` は `M1 close 157.498 / ATR 3.51p / ADX 23.76 / RSI 64.09`。
  - `health_snapshot` は `data_lag_ms=341.9`, `decision_latency_ms=18.4`,
    `mechanism_integrity=yes`。
  - 市況停止ではなく strategy-local な loser 再発として処理。

- 実測:
  - fresh loss は直近90分 `2026-03-10 16:04-17:34 JST` に
    `-5.994 JPY / -7.0p`。
  - 内訳は
    `scalp_extrema_reversal_live` 2本
    (`2026-03-10 16:19:08 JST -1.596 JPY / -2.8p`,
    `2026-03-10 16:20:24 JST -2.278 JPY / -3.4p`)
    と `session_open_breakout` 1本 (`17:28:10 JST -2.12 JPY / -0.8p`)。
  - `scalp_extrema_reversal_live` の2本はどちらも
    `supportive_long=false`, `ma_gap_pips=-0.32/-0.40`,
    `ADX=12.0/12.6`, `range_score=0.306/0.315` の
    shallow countertrend long だった。
  - 現行 `SCALP_EXTREMA_REVERSAL_LONG_COUNTERTREND_GAP_BLOCK_PIPS=0.50`
    ではこの2本が通っていたが、24h closed trade に対する再判定では
    `0.30` へ下げると該当2本だけが block 対象になり、
    勝ち玉は追加で削らなかった。

- 対応:
  - `ops/env/quant-scalp-extrema-reversal.env`
    - `SCALP_EXTREMA_REVERSAL_LONG_COUNTERTREND_GAP_BLOCK_PIPS=0.30`

- 意図:
  - `supportive_long=true` は維持しつつ、
    弱い trend / weak range 帯の `non-supportive` long だけを
    strategy-local に追加遮断する。
  - shared gate / order_manager / exit worker には触れない。

- 検証:
  - `scripts/local_v2_stack.sh restart --env ops/env/local-v2-stack.env --services quant-scalp-extrema-reversal`
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-scalp-extrema-reversal,quant-order-manager,quant-position-manager`
  - `ps eww -p <quant-scalp-extrema-reversal pid>`
  - env 読み込み下で `EXTREMA_LONG_COUNTERTREND_GAP_BLOCK_PIPS=0.30`,
    `ma_gap=-0.32/-0.40 -> blocked`, `ma_gap=-0.29 -> pass` を確認

## 2026-03-10 17:50-17:58 JST / `scalp_extrema_reversal_live` を停止ではなく shallow-probe quality guard へ置換

- 市況:
  - `2026-03-10 17:50 JST` 時点の USD/JPY は local 実測で通常帯。
  - local tick は `157.562/157.570`、spread `p50/p95=0.8p/0.8p`、
    `M1 ATR 3.03p`、OANDA `pricing/candles` は `200 OK`。
  - 市況停止ではなく、strategy-local な loser cluster 改善として扱う。

- 直前 hotfix の見直し:
  - `LONG_COUNTERTREND_GAP_BLOCK_PIPS=0.30` は
    fresh loss 2 本を止めるには効いたが、
    user intent の「止めるのではなく改善する」には blunt すぎる。
  - そのため gap hardening は `0.50` に戻し、
    loser pattern にだけ当たる quality guard へ置換する。

- 実測と cluster:
  - fresh loss 2 本はどちらも
    `supportive_long=false`,
    `dist_low_pips=0.296/0.300`,
    `long_bounce_pips=0.2/0.3`,
    `tick_strength=0.2`,
    `ADX=12.0/12.6`,
    `range_score=0.315/0.306`
    の shallow probe long だった。
  - 7d の closed `scalp_extrema_reversal_live` / non-supportive long 21 本に
    同条件
    `dist_low<=0.30 && long_bounce<=0.30 && tick_strength<=0.20 && adx<=13 && range_score<=0.32`
    を当てると、
    `2 trades blocked / 0 winners blocked / net -3.874 JPY`
    で、fresh loser 2 本だけに一致した。

- 対応:
  - `workers/scalp_extrema_reversal/worker.py`
    - non-supportive long に対して
      `dist_low / long_bounce / tick_strength / adx / range_score`
      を束ねた `long_shallow_probe_block` を追加。
    - `entry_thesis.extrema.long_shallow_probe_block` を監査用に露出。
  - `ops/env/quant-scalp-extrema-reversal.env`
    - `SCALP_EXTREMA_REVERSAL_LONG_COUNTERTREND_GAP_BLOCK_PIPS=0.50`
    - `SCALP_EXTREMA_REVERSAL_LONG_SHALLOW_PROBE_*`
      (`DIST_LOW_MAX=0.30`, `BOUNCE_MAX=0.30`,
      `TICK_STRENGTH_MAX=0.20`, `ADX_MAX=13.0`,
      `RANGE_SCORE_MAX=0.32`)

- 意図:
  - `supportive_long=true` の lane は維持する。
  - broad な long kill は避け、
    shallow countertrend long の loser micro-pattern だけを
    strategy-local に削る。
  - shared gate / order_manager / exit worker 契約は変えない。

- 検証:
  - `python3 -m py_compile workers/scalp_extrema_reversal/worker.py tests/workers/test_scalp_extrema_reversal_worker.py`
  - `python3 -m pytest -q tests/workers/test_scalp_extrema_reversal_worker.py`
  - explorer review でも
    「7d で blocked 2 本は loser のみ、winner 巻き込み 0」
    を確認。

## 2026-03-10 18:05-18:12 JST / `scalp_ping_5s_c_live` を strategy-control reopen

- 市況:
  - `2026-03-10 18:05 JST` の USD/JPY は
    `157.712/157.720`, spread `p50/p95=0.8p/0.8p`,
    `M1 ATR 2.963p`, `ADX 55.75`, `RSI 79.34`。
  - OANDA `summary/pricing/candles` はすべて `200 OK`。
  - 相場停止ではなく participation 配置の問題として扱う。

- 実測:
  - 直近90分の order-manager 実数は
    `session_open_breakout` のみ `OPEN_SCALE/REQ/FILLED=1`。
  - `scalp_ping_5s_c_live` は
    `OPEN_REJECT=2` で、両方とも
    `strategy_control_entry_disabled`。
  - 2本の blocked signal は
    `entry_probability=0.712 / 0.922`,
    `confidence=92`,
    `entry_units_intent=25 / 21` で、
    低品質だから落ちたのではなく strategy-control で閉じていた。
  - `logs/strategy_control.db` では
    `scalp_ping_5s_c` が `entry_enabled=0`。

- 追加根拠:
  - `scalp_ping_5s_c_live` は
    7d `21 trades / -4.659 JPY`,
    14d `31 trades / +3.624 JPY` で、
    直近 stray loser の恒久停止を続けるほどの壊れ方ではない。
  - 既存の forecast / perf / probability / min-units guard は
    そのまま残っているため、
    reopen しても shared gate を緩める変更ではない。

- 対応:
  - `ops/env/local-v2-stack.env`
    - `STRATEGY_CONTROL_ENTRY_SCALP_PING_5S_C=1`

- 意図:
  - `scalp_ping_5s_c_live` をフルノーガードで増やすのではなく、
    既存 guard 群を維持したまま
    strategy-control だけ reopen して、
    現在捨てている strong setup を再び通す。

## 2026-03-10 19:45-20:10 JST / perf scale を boost-only から dynamic reduce へ拡張

- 市況:
  - `2026-03-10 19:47 JST` の USD/JPY は
    `157.822/157.830`, spread `0.8p`,
    `ATR(M1/M5/H1)=2.85/6.83/22.77p`,
    `data_lag_ms=640`, `decision_latency_ms=15.5ms`。
  - `health_snapshot` では直近1h `filled=14`、
    `orders_recent` は `10:41-10:44 UTC` まで更新。
  - `quant-market-data-feed.log` の OANDA pricing stream は
    直近 restart 後も `HTTP 200` 継続。
  - 過去には `summary/openTrades` の `502/503/read timeout` があるが、
    直近末尾では再発していないため今回は通常帯として扱った。

- 実測:
  - 直近3h `orders.db` は
    `scalp perf_block=1580`, `scalp probability_scaled=1586`, `scalp filled=11`。
  - `perf_block` の大半は `RangeFader-*` で、
    entry path は
    `order_manager_perf_guard_strategy -> block`
    `reason=failfast_soft:pf=0.70 win=0.78 n=79`。
  - 同時に `config/dynamic_alloc.json` は
    `RangeFader score=0.086 lot_multiplier=0.231` まで縮小済みで、
    既に dynamic alloc は participation を絞っていた。
  - 一方 `perf_scale('RangeFader','scalp')` は
    `pf=0.698 / win_rate=0.785 / avg_pips=0.186` で、
    旧実装だと `boost 1.05` を返していた。
    block 側と scale 側で評価軸が噛み合っていなかった。
  - `scalp_extrema_reversal_live` は 7d
    `pf=0.217 / win=0.242 / avg_pips=-1.061`、
    `MomentumBurst` は 7d
    `pf=1.548 / win=0.709 / avg_pips=1.831` と、
    戦略ごとに「縮小すべき lane」と
    「継続 participation すべき lane」が明確に分かれていた。

- 対応:
  - `workers/common/perf_guard.py`
    - `perf_scale` を boost-only から
      `boost / flat / reduce` の対称スコアへ拡張。
    - `pf`, `win_rate`, `avg_pips` の miss を penalty として扱い、
      `pf<1.0 && avg_pips<0.0` の lane は追加 penalty を入れる。
  - `workers/common/dyn_size.py`
    - `perf_scale` の `multiplier<1.0` を実 units に反映。
    - `spread/adx/signal/perf` のどれかが悪化しているときは
      base floor への強制ブレンドをやめ、
      downscale をそのまま残す。
  - `ops/env/quant-order-manager.env`
    - `RANGEFADER_PERF_GUARD_MODE=reduce`
      を追加し、
      soft failfast を即 block ではなく
      warn + dynamic sizing に寄せる。

- 意図:
  - hard failfast / margin closeout / SL-loss-rate の
    hard block は残す。
  - soft 悪化は
    `block` ではなく `size down` へ寄せ、
    市況に合わせて participation を滑らかに調整する。
  - shared order-manager に新しい global gate は追加せず、
    既存 perf-guard / dyn-size の役割分担だけを整える。

- 検証:
  - `pytest tests/test_perf_guard_failfast.py tests/workers/common/test_dyn_size.py -q`
    -> `19 passed`
  - ローカル確認では
    `perf_scale('RangeFader','scalp') -> reduce 0.95`,
    `perf_scale('scalp_extrema_reversal_live','scalp_fast') -> reduce 0.8`
    を確認。

## 2026-03-10 20:05 JST / strategy_entry participation alloc の boost path 復旧（local-v2）
- 目的:
  - 「全部動的にして、市況でちゃんとトレードできるようにする」方針に合わせ、
    共通レイヤが underused winner の participation 回復を潰していないかを local-v2 実測で確認する。
- 市況確認:
  - `logs/orderbook_snapshot.json` 直近値は `bid=157.830 / ask=157.838 / spread=0.8p`、
    provider `oanda-stream`, latency `202.5ms`。
  - `logs/factor_cache.json` は `ATR(M1)=2.81p / ATR(M5)=6.83p / ATR(H1)=22.77p`。
  - `logs/metrics.db` 直近2h平均は `data_lag_ms=546.26`, `decision_latency_ms=16.5`。
  - `bash scripts/collect_local_health.sh` は `health_snapshot.json updated=yes` を返し、
    OANDA pricing/account snapshot も直近更新だったため通常帯として作業継続。
- 実測:
  - 直近6h `orders.db` は `scalp perf_block=1580 / entry_probability_reject=689 / filled=11`。
  - `logs/entry_path_summary_latest.json` では
    `RangeFader attempts=1710 fills=46 share_gap=0.696389 hard_block_rate=1.9228` と過剰試行が継続。
  - 一方で `WickReversalBlend 4/4`, `PrecisionLowVol 6/6`, `MomentumBurst 22/22` のように
    filled quality が高い lane もあり、participation を戻す余地があった。
  - しかし `execution/strategy_entry.py` の `_apply_participation_alloc()` は
    `lot_multiplier` を常に `<=1.0` に clamp しており、
    `scripts/participation_allocator.py` が将来 `boost_participation` を出しても
    units boost は execution 側で無効化される状態だった。
- 変更:
  - `execution/strategy_entry.py`
    - `participation_alloc.action == boost_participation` のときだけ
      modest な units boost を許可。
    - 上限は `STRATEGY_PARTICIPATION_ALLOC_MULT_MAX` と
      artifact `allocation_policy.max_units_boost` の両方で clamp。
    - trim lane / stale or missing payload / non-explicit action は従来どおり trim-only。
    - `sell` 側でも reason が逆転しないよう、`abs(units)` 基準で
      `boost_participation / overused_trim` を監査記録。
  - `workers/common/participation_alloc.py`
    - `allocation_policy.max_units_boost` / `max_probability_boost` など
      artifact の policy 上限を loader から返すよう更新。
- 意図:
  - 新しい global gate は足さず、既存 `strategy_entry` の soft participation layer を
    artifact 設計どおりに動かす。
  - overused loser の trim は維持しつつ、
    underused winner の modest scale-up だけを explicit signal 時に解放する。
- 検証:
  - `pytest -q tests/execution/test_strategy_entry_adaptive_layers.py tests/scripts/test_participation_allocator.py`
    -> `6 passed`
  - `python3 -m compileall execution/strategy_entry.py workers/common/participation_alloc.py`
    -> 成功

## 2026-03-10 20:xx JST / RangeFader の cadence_floor を strategy-local cooldown へ接続
- 目的:
  - マイナスが出すぎる loser lane を、shared gate 追加ではなく strategy-local frequency 制御で動的に絞る。
- 市況確認:
  - 追加確認時も `logs/orderbook_snapshot.json` は `spread=0.8p`、
    `logs/factor_cache.json` は `ATR(M1)=3.54p / M5=7.33p / H1=22.82p` で通常帯。
- 実測:
  - `config/participation_alloc.json` は `RangeFader action=trim_units cadence_floor=0.9` を出していた。
  - しかし runtime では `cadence_floor` が未使用で、
    `logs/entry_path_summary_latest.json` は `RangeFader attempts=1710 / fills=46 / perf_block=1664`。
  - つまり「動的に頻度を落とせ」という artifact が出ていても、
    worker 側 cooldown は静的のままだった。
- 変更:
  - `workers/scalp_rangefader/worker.py`
    - `participation_alloc` を read-only で読み、
      fresh `trim_units` + `protect_frequency=true` + `cadence_floor<1.0`
      のときだけ cooldown を `base / cadence_floor` へ延長。
    - 例: `20s / 0.9 = 22.2s`。
    - stale / missing / hold / boost は no-op のまま。
- 意図:
  - 新しい global gate を足さず、
    RangeFader の strategy-local cooldown だけを artifact 駆動へ寄せる。
  - サイズだけではなく頻度側も動的にすることで、
    loser lane の「出すぎ」を抑える。
- 検証:
  - `pytest -q tests/workers/test_scalp_rangefader_worker.py`
    -> `5 passed`
  - `pytest -q tests/execution/test_strategy_entry_adaptive_layers.py tests/scripts/test_participation_allocator.py`
    -> `6 passed`
  - `python3 -m compileall workers/scalp_rangefader/worker.py tests/workers/test_scalp_rangefader_worker.py`
    -> 成功

## 2026-03-10 20:5x JST / micro runtime の loser cadence を dynamic_alloc へ接続
- 目的:
  - 「マイナスが出すぎる / 稼ぐのが遅い」状態に対し、
    micro loser lane の頻度を strategy-local に落としつつ、
    side 別の成績を runtime がちゃんと参照できる状態へ戻す。
- 市況確認:
  - `logs/orderbook_snapshot.json` 直近値は `bid=157.801 / ask=157.809 / spread=0.8p`、
    provider `oanda-stream`, latency `212.9ms`。
  - `logs/factor_cache.json` は `ATR(M1)=2.21p / ATR(M5)=6.70p / ATR(H1)=22.82p`。
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    は core と micro worker を含め running。
  - `logs/local_v2_stack/quant-order-manager.log` と
    `quant-micro-{momentumburst,levelreactor,trendretest}.log` の更新時刻は
    `2026-03-10 20:51-20:53 JST` で fresh。
- 実測:
  - `logs/trades.db` 直近24h:
    - `MomentumBurst`: `24 trades / realized=-617.2 JPY / -17.1 pips`
    - `MicroTrendRetest-short`: `2 trades / realized=-540.6 JPY / -10.6 pips`
    - `MicroTrendRetest-long`: `17 trades / realized=-290.2 JPY / -45.7 pips`
    - `MicroLevelReactor`: `228 trades / realized=-79.7 JPY / +40.2 pips`
  - `config/dynamic_alloc.json` (`as_of=2026-03-10T11:47:03Z`) は
    `MicroLevelReactor=0.14`, `MomentumBurst=0.301`,
    `MicroTrendRetest-long=0.14`, `MicroTrendRetest-short=0.14`
    を出していた。
  - しかし `workers/micro_runtime/worker.py` は
    `dynamic_alloc` を `strategy_name` だけで引いており、
    `MicroTrendRetest-long/-short` の side 別 profile を live sizing / loser cadence に使えていなかった。
  - さらに cooldown 延長は `participation_alloc` と `dynamic_alloc` を
    積算で掛けてしまう形になり得たため、
    loser 抑制を二重に強める設計になりやすかった。
- 変更:
  - `workers/micro_runtime/worker.py`
    - `MicroTrendRetest-long-trendflip` のような tag から
      `MicroTrendRetest-long` を優先解決し、未解決時だけ base strategy へ戻す
      micro-local profile loader を追加。
    - `dynamic_alloc` の live load を signal tag 解決後へ移動し、
      loser block / sizing / cooldown が同じ resolved key を参照するよう整理。
    - `participation_alloc` cadence と `dynamic_alloc` loser multiplier を
      `max(base, base/cadence_floor, ref/dyn_mult)` で合成し、
      二重積算ではなく「強い方だけ採用」に変更。
    - base cooldown が 0 の戦略は `LOOP_INTERVAL_SEC` を参照基準にして、
      loser lane でも frequency 制御が効くようにした。
- 現行 env での実効イメージ:
  - local-v2 shared `MICRO_MULTI_DYN_ALLOC_MULT_MIN=0.68`
    のため `0.14/0.301` は live では `0.68` まで clamp。
  - `MicroLevelReactor`: `8s / 0.68 ≒ 11.8s`
  - `MicroTrendRetest`: `45s / 0.68 ≒ 66.2s`
  - `MomentumBurst`: `120s / 0.68 ≒ 176.5s`
    (`reaccel` は `35s / 0.68 ≒ 51.5s`)
  - `participation_alloc` が今後 `trim_units + cadence_floor<1.0` を出した場合も、
    その延長値と dynamic_alloc 延長値の大きい方だけを採用する。
- 意図:
  - shared order-manager / common gate に新しい一律判定は足さず、
    micro worker の strategy-local cadence だけを
    live artifact 駆動へ寄せる。
  - 負けている lane の size と cadence を同じ profile key で整合させ、
    `MicroTrendRetest` の side 別 loser を runtime が見落とさないようにする。
- 検証:
  - `pytest -q tests/workers/test_micro_multistrat_trend_flip.py`
    -> `25 passed`
  - `python3 -m compileall workers/micro_runtime/worker.py tests/workers/test_micro_multistrat_trend_flip.py`
    -> 成功

## 2026-03-10 21:xx JST / micro loser cluster を strategy-local quality 改善へ変換
- 目的:
  - cadence や lot をさらに削るのではなく、
    `MomentumBurst` と `MicroTrendRetest` の負けパターン自体を
    strategy-local entry quality guard に変換する。
- 市況確認:
  - 再確認時点でも `logs/orderbook_snapshot.json` は
    `bid=157.826 / ask=157.834 / spread=0.8p / latency=156.2ms`。
  - `logs/factor_cache.json` は
    `ATR(M1)=2.14p / ATR(M5)=5.79p / ATR(H1)=22.82p`。
  - 主要 service は local-v2 running 継続。
- 実測:
  - `logs/trades.db` 直近24h:
    - `MomentumBurst-open_short`: `11 trades / -598.5 JPY / -13.8p`
      で、`7` 負けのうち `6` が `STOP_LOSS_ORDER`。
      負け側は `M1 RSI≈32.7`, `range_score≈0.158`,
      `abs(MA10-MA20)≈3.57p` に偏り、`maru_dn/flat` の continuation candle が多い。
    - `MicroTrendRetest-long`: `17 trades / -290.2 JPY / -45.7p`,
      `14` 負けで全件 `STOP_LOSS_ORDER`。
      そのうち `9/14` が
      `spin_dn|lower|tr:up_strong|rsi:ob|...|d:long` で、勝ち側には 0 件。
    - `MicroTrendRetest-short`: `2 trades / -540.6 JPY / -10.6p`,
      `2/2` 負けで `spin_up|...|rsi:os|d:short` に揃っていた。
  - `projection` や `micro_chop` より、
    今回は `pattern shape + RSI + DI/ADX + low-range clean trend` の方が
    敗因の説明力が高かった。
- 変更:
  - `strategies/micro/momentum_burst.py`
    - low-range clean trend (`range_score<=0.22`) で
      `rsi<=35`, 強い `DI gap`, 大きい bearish breakdown candle の
      late short chase を reject する guard を追加。
    - これは cadence trim ではなく、
      clean trend の末端で売り遅れる cluster を strategy-local に落とす変更。
  - `strategies/micro/trend_retest.py`
    - small MA-gap 条件でも
      `long + rsi>=62 + bearish small-body reclaim`,
      `short + rsi<=38 + bullish small-body reclaim`
      を reject する対称 guard を追加。
    - 既存の `trend_snapshot` / `retest_close_recovery` / `max_retest_dist`
      の上に、reclaim candle 自体の exhaustion を追加で見る形。
- 意図:
  - loser lane の frequency を一律に削るのではなく、
    `entry` の質を上げて `STOP_LOSS_ORDER` cluster を先に潰す。
  - 共有 gate や `order_manager` は触らず、
    `MomentumBurst` と `MicroTrendRetest` の strategy-local contract の中で完結させる。
- 検証:
  - `pytest -q tests/strategies/test_trend_retest.py tests/strategies/test_momentum_burst.py`
    -> `37 passed`
  - `pytest -q tests/strategies/test_trend_retest.py tests/strategies/test_momentum_burst.py tests/workers/test_micro_multistrat_trend_flip.py tests/execution/test_strategy_entry_adaptive_layers.py tests/scripts/test_participation_allocator.py`
    -> `68 passed`
  - `python3 -m compileall strategies/micro/trend_retest.py strategies/micro/momentum_burst.py tests/strategies/test_trend_retest.py tests/strategies/test_momentum_burst.py`
    -> 成功

## 2026-03-10 21:14 JST / current 状況確認 + micro quality の追加改善
- current 状況:
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    は core / scalp / micro 全サービス running。
  - `bash scripts/collect_local_health.sh` は
    `health_snapshot.json updated=yes`, `snapshot_age_sec=0`, `mechanism_integrity=yes`。
  - `logs/health_snapshot.json` は
    `generated_at=2026-03-10T12:14:28Z`, `git_rev=6084a691`, `trades_count_24h=223`。
  - 市況は `bid=157.899 / ask=157.907 / spread=0.8p / latency=313ms`、
    `ATR(M1)=2.81p / M5=6.09p / H1=21.74p` で通常帯。
  - 直近約定は `RangeFader` と `WickReversalBlend` が継続し、
    `MomentumBurst` / `MicroTrendRetest` は再起動後まだ新規約定なし。
- 追加で見えた負け cluster:
  - `MomentumBurst` の reaccel long は直近7日でも `2/2` 負け、
    どちらも `pattern_tag=trend_up|w:upper|tr:flat|...|d:long`,
    `trend_snapshot(H4 long, gap 37.683p, adx 32.44)` で
    weak follow-through の breakout だった。
  - `MicroTrendRetest-short` の `2/2` 負けは
    `rsi<=30` の oversold かつ bullish reclaim に揃っていた。
- 変更:
  - `strategies/micro/momentum_burst.py`
    - long `reaccel` でだけ、
      flat / upper-wick / weak body の breakout を reject する
      follow-through guard を追加。
    - open を持たない fixture では no-op にして、既存 test 契約は維持。
  - `strategies/micro/trend_retest.py`
    - short 側の reclaim exhaustion を強め、
      `rsi<=38` の oversold short で bullish reclaim が high-close なら reject。
- 意図:
  - entry 数を一律に減らさず、
    weak breakout / oversold reclaim だけを strategy-local に外す。
- 検証:
  - `pytest -q tests/strategies/test_momentum_burst.py tests/strategies/test_trend_retest.py`
    -> `40 passed`
  - `pytest -q tests/strategies/test_momentum_burst.py tests/strategies/test_trend_retest.py tests/workers/test_micro_multistrat_trend_flip.py tests/execution/test_strategy_entry_adaptive_layers.py tests/scripts/test_participation_allocator.py`
    -> `71 passed`
  - `python3 -m compileall strategies/micro/momentum_burst.py strategies/micro/trend_retest.py tests/strategies/test_momentum_burst.py tests/strategies/test_trend_retest.py`
    -> 成功

## 2026-03-10 21:22 JST / current 状況確認 + RangeFader headwind guard 追加
- current 状況:
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    は core / scalp / micro 全サービス running。
  - `bash scripts/collect_local_health.sh` は
    `health_snapshot.json updated=yes`, `snapshot_age_sec=0`, `mechanism_integrity=yes`。
  - `logs/health_snapshot.json` は
    `generated_at=2026-03-10T12:14:28Z`, `git_rev=6084a691`, `trades_count_24h=223`。
  - 市況は `bid=157.920 / ask=157.928 / spread=0.8p / latency=127ms`、
    `ATR(M1)=2.92p / M5=6.01p / H1=21.74p` で通常帯。
  - 直近 closed trade は `2026-03-10 21:14-21:16 JST` の `RangeFader` short が
    `-2.9p` から `-4.9p` の連続 loss。
  - 最新 orders は `2026-03-10 21:20-21:21 JST` の `RangeFader sell`
    `entry_probability_reject` が連打。
- 追加で見えた敗因:
  - 24h 累積損失の上位は引き続き
    `MomentumBurst -617.2 JPY`,
    `MicroTrendRetest-short -540.6 JPY`,
    `MicroTrendRetest-long -290.2 JPY`。
  - ただし live の新しい負け lane は `RangeFader` 側で、
    直近 short loss は `range_score=0.225-0.252`,
    `entry_probability=0.36-0.37` の weak fade に偏っていた。
  - 同時点の `M1` は `RSI=64.35`, `ADX=37.13`, `+DI 28.25 > -DI 14.91`,
    `ema_slope_10=0.00597` で bullish headwind が明確だった。
- 変更:
  - `strategies/scalping/range_fader.py`
    - `range_score` が低く、`ADX` と `DI gap` と `ema_slope_10` が
      trend continuation を示すときは、
      weak `sell-fade` / `neutral-fade` short を strategy-local に抑止。
    - 同じ条件を long 側にも対称に入れ、
      bearish headwind 下の weak long fade も抑止。
    - ただし極端な overextension は残すため、
      `momentum_pips` が十分に伸びたケースまでは block しない。
  - `strategies/micro/trend_retest.py`
    - `short` に入れていた reclaim close-position guard を `long` 側にも対称追加し、
      `rsi>=62` の bearish low-close reclaim も reject。
- 意図:
  - cadence や shared gate ではなく、
    `RangeFader` / `MicroTrendRetest` の strategy-local entry quality 改善で
    current loser cluster を潰す。
- 検証:
  - `pytest -q tests/strategies/test_trend_retest.py tests/strategies/test_scalp_thresholds.py tests/workers/test_scalp_rangefader_worker.py`
    -> `25 passed`
  - `pytest -q tests/strategies/test_scalp_thresholds.py tests/workers/test_scalp_rangefader_worker.py tests/strategies/test_momentum_burst.py tests/strategies/test_trend_retest.py tests/workers/test_micro_multistrat_trend_flip.py tests/execution/test_strategy_entry_adaptive_layers.py tests/scripts/test_participation_allocator.py`
    -> `84 passed`
  - `python3 -m compileall strategies/scalping/range_fader.py tests/strategies/test_scalp_thresholds.py strategies/micro/trend_retest.py tests/strategies/test_trend_retest.py`
    -> 成功

## 2026-03-10 21:40 JST / 「市況適応できているか」実測診断
- current 状況:
  - `scripts/local_v2_stack.sh status --env ops/env/local-v2-stack.env --services quant-market-data-feed,quant-strategy-control,quant-order-manager,quant-position-manager`
    は4サービスとも `running`。
  - `scripts/collect_local_health.sh` は
    `health_snapshot.json updated=yes`, `mechanism_integrity=yes`。
  - `python3 scripts/pdca_profitability_report.py --instrument USD_JPY`
    は `generated_at_jst=2026-03-10T21:35:23+09:00`,
    `USD_JPY mid=157.904`, `spread=0.8p`,
    `24h trades=218`, `PF(pips)=0.25`, `net_jpy=-1050.433`,
    `7d PF(pips)=0.62`, `net_jpy=-13820.152`。
  - `logs/metrics.db` 直近500件平均は
    `decision_latency_ms=21.433`, `data_lag_ms=567.321`,
    latest は `reject_rate=0.0`, `order_success_rate=1.0`。
- 観測できた「適応している部分」:
  - `entry_thesis` 契約は生きており、直近2000 orders で
    `entry_probability=1845`, `entry_units_intent=1845`,
    `forecast=1845`, `dynamic_alloc=1796`。
  - micro runtime は `range_active/range_score` と `micro_chop_*` を factor に注入し、
    `dynamic_alloc`, `strategy_units_mult`, `chop_units_mult`,
    `entry_probability`, `entry_units_intent` を合成して sizing している。
  - current logs でも `RangeFader` は
    `range_reason=volatility_compression`、
    `MicroLevelReactor` は `mlr_range_gate_block ... chop=0.55` を出しており、
    regime-aware の strategy-local gate 自体は動いている。
- ただし「適応が弱い / 範囲が狭い」点:
  - Brain は safe canary のままで、
    `LOCAL_V2_EXTRA_ENV_FILES=ops/env/profiles/brain-ollama-safe.env`,
    `BRAIN_POCKET_ALLOWLIST=micro`,
    `BRAIN_STRATEGY_ALLOWLIST=MicroLevelReactor`。
    `logs/brain_state.db` の直近24h decision は
    `53件`, すべて `MicroLevelReactor-bounce-lower / REDUCE / avg_scale=0.549`。
    つまり LLM は live 全体の主判定ではなく、micro の一部だけ。
  - forecast gate は order-manager で live だが allowlist 制で、
    `FORECAST_GATE_STRATEGY_ALLOWLIST=MicroLevelReactor,MicroTrendRetest,MomentumBurst,M1Scalper-M1,RangeFader,scalp_ping_5s_b_live,scalp_ping_5s_c_live,scalp_ping_5s_flow_live`。
    直近24h order の `entry_thesis.forecast.reason` は
    `not_applicable=7758`, `__missing__=6243` が大半で、
    prediction が付いていても「使える判定」まで至らないケースが多い。
  - pattern gate は global opt-in ではなく、
    直近24h orders で `pattern_gate_opt_in=1` は `228件` のみ。
  - fallback candle artifact は stale で、
    `logs/oanda/candles_M1_latest.json` は `2026-03-04T03:04:00Z`,
    `H1/H4` は `2025-11-04` のまま。
    `workers/common/forecast_gate.py` の fallback path はこの M1 file を参照するため、
    live service / factor で埋まらない経路は stale context を掴むリスクがある。
- 今回の主因整理:
  - 「市況に合わせる仕組み」はあるが、現行 live は
    `LLM=一部 canary`, `forecast=allowlist`, `pattern=opt-in` のため、
    ユーザーが期待する “全体が動的に考える” 状態ではない。
  - 24h では normal spread / low reject / low latency なのに負けているため、
    主因はインフラではなく strategy quality / calibration。
  - 高めの `entry_probability` でも負けている例があり、
    `MomentumBurst avg_prob=0.848 / exp_pips=-0.713`,
    `MicroTrendRetest-long avg_prob=0.657 / exp_pips=-2.688`,
    `WickReversalBlend avg_prob=0.807 / exp_pips=-0.578`。
    現行の probability / forecast / local guard が current regime に十分再較正されていない。
- 優先改善:
  - Brain を broad 化する前に、
    `MomentumBurst / MicroTrendRetest / WickReversalBlend / RangeFader`
    の loser cluster を strategy-local で先に潰す。
  - stale な `logs/oanda/candles_*.json` を更新し、
    forecast fallback が古い文脈へ落ちない状態を回復する。
  - `entry_probability` が高いのに負ける lane を
    `pattern_tag / RSI / ADX / MA gap / trend_snapshot / divergence`
    で再クラスタし、probability と size の較正を戦略別に分離する。

## 2026-03-10 21:59 JST / lane-aware feedback 復旧 + current loser 即応
- 目的:
  - `RangeFader-buy/sell/neutral-fade` と `MicroLevelReactor-bounce/fade` を
    canonical `strategy_tag` へ潰さず、動的配分・loser cluster・pattern gate を
    lane 単位で再較正できる状態へ戻す。
  - 同時に current loser である `RangeFader` の under-min-units reject と
    `MicroLevelReactor-bounce-lower` の weak reclaim を即応で改善する。
- 仮説:
  - live の `trades.entry_thesis.strategy_tag_raw` には raw lane が残っている一方、
    `dynamic_alloc` / `participation_alloc` / `loser_cluster` / `pattern_id`
    が canonical `strategy_tag` を優先しており、
    `RangeFader-neutral-fade` のような winner/loser lane を同じ `RangeFader`
    として扱っていた。
  - `RangeFader` は probability-scale 後の units が `sell/neutral` lane で
    min-units を割りやすく、`MicroLevelReactor` は
    low ATR + strong `-DI` の continuation で weak reclaim long をまだ拾っていた。
- 変更:
  - `utils/strategy_tags.py` に `extract_strategy_tags()` を追加し、
    `entry_thesis.strategy_tag_raw` を最優先、
    canonical tag は fallback として返す形へ整理。
  - `scripts/entry_path_aggregator.py`
    `scripts/participation_allocator.py`
    `scripts/dynamic_alloc_worker.py`
    `scripts/loser_cluster_worker.py`
    は raw lane 優先へ更新。
    `trades.strategy_tag` の schema/意味は変更せず、
    既存 `entry_thesis.strategy_tag_raw` だけを根拠に lane-aware 化した。
  - `analysis/pattern_book.py` は `build_pattern_id()` で
    `strategy_tag_raw` を優先するよう変更し、
    pattern gate も同じ lane 粒度で学習/照合できるようにした。
  - `strategies/micro/level_reactor.py` は
    strong continuation (`low ATR + strong ADX/-DI`) を別閾値に切り出し、
    weak body / weak lower-wick の `bounce-lower` を reject。
  - `ops/env/quant-order-manager.env` は
    `ORDER_MIN_UNITS_STRATEGY_RANGEFADER_SELL_FADE=25`,
    `ORDER_MIN_UNITS_STRATEGY_RANGEFADER_NEUTRAL_FADE=30`
    へ下げ、shared loosen なしで winner lane の生存率を戻した。
- 影響範囲:
  - shared feedback artifact の粒度が `strategy family` から `lane` へ細かくなる。
    対象は `entry_path_summary_latest.json`, `participation_alloc.json`,
    `dynamic_alloc.json`, `loser_cluster_latest.json`, `pattern_book*.json`。
  - `order_manager` / `position_manager` の schema や共通 reject 条件は変更していない。
- 生成物確認:
  - `logs/entry_path_summary_latest.json` は
    `RangeFader-buy-fade / sell-fade / neutral-fade` を個別 key として出力。
  - `config/participation_alloc.json` は
    `RangeFader-buy-fade` `trim_units lot_multiplier=0.8296`,
    `RangeFader-sell-fade` `trim_units lot_multiplier=0.8296`,
    `RangeFader-neutral-fade` `hold lot_multiplier=1.0`
    を確認。
  - `config/dynamic_alloc.json` は
    `RangeFader-buy/sell/neutral-fade`,
    `MicroLevelReactor-bounce-lower/fade-upper`
    を別 key として出力。
- 検証:
  - `pytest -q tests/strategies/test_level_reactor.py`
    -> `10 passed`
  - `pytest -q tests/workers/test_scalp_rangefader_worker.py`
    -> `5 passed`
  - `pytest -q tests/test_dynamic_alloc_worker.py tests/scripts/test_participation_allocator.py tests/scripts/test_entry_path_aggregator.py tests/scripts/test_loser_cluster_worker.py tests/analysis/test_pattern_book.py`
    -> `19 passed`
  - `pytest -q tests/workers/test_pattern_gate.py`
    -> `6 passed`
  - `python3 scripts/entry_path_aggregator.py ...`
    `python3 scripts/participation_allocator.py ...`
    `PYTHONPATH=/Users/tossaki/Documents/App/QuantRabbit python3 scripts/dynamic_alloc_worker.py ...`
    `python3 scripts/loser_cluster_worker.py ...`
    `PYTHONPATH=/Users/tossaki/Documents/App/QuantRabbit python3 scripts/pattern_book_worker.py ...`
    を実行し、live artifact を更新済み。

## 2026-03-10 13:17 UTC / 2026-03-10 22:17 JST - 「逆に張っている」仮説と restart persistence 監査
- 対象（実測）:
  - `logs/trades.db`, `logs/orders.db`, `logs/metrics.db`, `logs/health_snapshot.json`
  - `logs/tick_cache.json`, `logs/factor_cache.json`
  - `logs/local_v2_stack/quant-micro-levelreactor.log`, `logs/local_v2_stack/quant-micro-trendretest.log`
  - `config/dynamic_alloc.json`, `config/pattern_book.json`,
    `config/participation_alloc.json`, `config/auto_canary_overrides.json`
  - `logs/strategy_feedback.json`, `logs/brain_state.db`
  - `scripts/local_v2_stack.sh`, `analysis/strategy_feedback.py`,
    `analysis/auto_canary.py`, `workers/common/dynamic_alloc.py`,
    `workers/common/participation_alloc.py`, `workers/common/brain.py`,
    `workers/common/pattern_gate.py`
- 市況・稼働:
  - USD/JPY は `157.768`、spread `0.8 pips`
    （`logs/tick_cache.json` 末尾）。
  - `logs/factor_cache.json` は `M1 ATR=2.84 pips`, `M5 ATR=6.29 pips`,
    `H1 ATR=21.73 pips`。
  - `metrics.db` 直近6hは `data_lag_ms avg=548.4`,
    `decision_latency_ms avg=17.0`。
  - `health_snapshot.json` は fresh、`trade_min` の主要サービスは running。
  - `orders.db` 直近24hは
    `preflight_start=1818`, `probability_scaled=1803`,
    `perf_block=1583`, `entry_probability_reject=1467`,
    `submit_attempt=207`, `filled=205`, `rejected=2`。
    現時点の悪化は OANDA/API 不調より strategy quality 側が主因。
- 24h 収益:
  - `214 trades`, `win_rate=30.8%`, `PF(pips)=0.27`,
    `net_pips=-315.3`, `net_jpy=-963.0`,
    `expectancy=-1.47 pips/trade`。
  - 同一 exit 時刻のまま side だけ反転した仮説では
    `+315.3 pips / +963.0 JPY` 相当なので、
    「方向/タイミングの問題」は確かにある。
  - ただし全戦略が単純に long/short 反転しているわけではない。
- 主損失クラスター:
  - `MicroTrendRetest-long`
    - `17 trades`, `all long`, `trend_snapshot.direction=long`,
      `pattern_meta.trend_bucket=up_strong`,
      `rsi_bucket=ob(9) / mid_high(8)`,
      `net_pips=-45.7`, `net_jpy=-290.2`
    - これは「逆方向」より
      `up_strong + ob/mid_high` で long を追いかけ過ぎている lane。
  - `MicroLevelReactor-bounce-lower`
    - `44 trades`, `all long`, `trend_snapshot.direction=long`,
      `pattern_meta.trend_bucket=dn_strong`,
      `net_pips=-75.8`, `net_jpy=-75.2`
    - 大局 long のまま、局所の `dn_strong` 継続に対して
      bounce long を早く入れている。
      「逆に張っているように見える」主因はここ。
    - worker log でも restart 後に
      `mlr_range_gate_block active=False score=0.150 adx=37.21 ma_gap=4.05`
      や
      `... score=0.161 adx=39.62 ma_gap=3.92`
      が出ており、継続局面の抑止が弱い。
  - `WickReversalBlend`
    - `9 trades`, `win_rate=77.8%`,
      `net_pips=-5.2`, `net_jpy=-156.5`
    - これは reverse side ではなく、
      一部 loser の損失幅が勝ちを食っている payoff/exit 問題。
- 判定:
  - 「ほとんど逆にトレードしている」は **部分的に真**。
  - より正確には、
    - `MicroLevelReactor-bounce-lower` は
      局所 continuation に対する早すぎる long
    - `MicroTrendRetest-long` は
      過熱帯 long の chase
    - `WickReversalBlend` は
      payoff/exit の非対称
    の混合。
  - したがって修正方針は
    「全体 side を反転」ではなく、
    strategy-local quality guard と exit/payoff の分離改善が正。
- restart persistence 監査:
  - `scripts/local_v2_stack.sh` は service 起動時に
    base/service/override/extra env を毎回読込み、
    `ops/env/local-v2-stack.env` は
    `LOCAL_V2_EXTRA_ENV_FILES=ops/env/profiles/brain-ollama-safe.env`
    を固定している。
    manual restart / watchdog / launchd 復旧で同じ Brain 導線を維持する。
  - live runtime は disk artifact を都度読む設計:
    - `analysis/strategy_feedback.py`
      -> `logs/strategy_feedback.json`
    - `analysis/auto_canary.py`
      -> `config/auto_canary_overrides.json`
    - `workers/common/dynamic_alloc.py`
      -> `config/dynamic_alloc.json`
    - `workers/common/participation_alloc.py`
      -> `config/participation_alloc.json`
    - `workers/common/pattern_gate.py`
      -> `logs/patterns.db`, `config/pattern_book_deep.json`
    - `workers/common/brain.py`
      -> `logs/brain_state.db` に memory/decisions を永続化
  - 実測でも、主要サービスは `2026-03-10 22:01 JST` 前後に起動し直していたが、
    その後
    `auto_canary_overrides.json=22:02:57`,
    `strategy_feedback.json=22:09:55`,
    `dynamic_alloc.json=22:10:03`,
    `pattern_book.json=22:10:43`,
    `participation_alloc.json=22:11:01`
    と fresh 更新されている。
  - `brain_state.db` も
    `brain_memory rows=14`,
    `last_memory_update=2026-03-10T12:50:00+00:00`,
    `brain_decisions 24h=44`
    を保持しており、restart で Brain memory が消えた形跡はない。
  - 揮発するのは process 内 cache / cooldown 類
    （`strategy_feedback._CACHE`,
    `auto_canary._CACHE`,
    `dynamic_alloc._CACHE`,
    `participation_alloc._CACHE`,
    `brain._CACHE`,
    `brain._FAILFAST_STATE`,
    live Ollama priority state など）で、
    restart 直後に短い warm-up はある。
  - ただし current loser は restart 後の数分だけではなく
    直近24h全体で継続しており、
    「restart のたびに thinking が初期化して current losses を作っている」
    ことは現時点では主因と断定できない。
- 次アクション:
  - `MicroLevelReactor-bounce-lower` に対して
    `dn_strong + high ADX + ma_gap expansion` の continuation 抑止を
    strategy-local で強める。
  - `MicroTrendRetest-long` は
    `up_strong + ob/mid_high` の chase を減衰/skip する。
  - `WickReversalBlend` は reverse 判定と切り離して、
    exit/payoff の非対称だけを別件で詰める。

## 2026-03-10 22:41 JST / reverse-entry RCA を strategy-local dynamic quality / exit へ反映
- 対象:
  - `MicroLevelReactor-bounce-lower`
  - `MicroTrendRetest-long`
  - `WickReversalBlend`
- 実装:
  - `MicroLevelReactor` は
    `recent M1 continuation + wide negative ma_gap + DI continuation`
    を合算して pressure を出し、
    pressure が強い局面では
    `bounce-lower` long に
    `strong body reclaim` と `strong lower-wick reclaim`
    を要求するよう更新。
  - `MicroTrendRetest` は
    `gap/ATR`, `ADX`, `trend_snapshot` の same-direction 圧力と
    retest depth / overshoot / close position を使って
    `up_strong + mid_high/ob` の shallow chase long を
    strategy-local に reject するよう更新。
  - `WickReversalBlend` は
    signal 生成時に `wick_ratio`, `tick_strength`, `follow`,
    `retrace_from_extreme`, `projection.score` から entry quality を計算し、
    `entry_thesis` へ保存。
    exit worker はその quality と trade ごとの `sl_pips/tp_pips/current ATR`
    を使って `profit_take`, `trail_start`, `trail_backoff`,
    `loss_cut_hard_pips`, `loss_cut_max_hold_sec` を動的化する。
- 方針:
  - dedicated env の固定 tightening は今回の主解にしない。
  - shared gate / time block / order-manager の後付け一律判定も増やさない。
  - entry と exit の両方を strategy-local contract の中で閉じる。
- 検証:
  - `pytest -q tests/strategies/test_level_reactor.py tests/strategies/test_trend_retest.py tests/workers/test_scalp_wick_reversal_blend_policy.py tests/workers/test_scalp_wick_reversal_blend_exit_worker.py`
    -> `33 passed`
  - `pytest -q tests/workers/test_micro_multistrat_trend_flip.py tests/workers/test_scalp_wick_reversal_blend_dispatch.py`
    -> `27 passed`
- 未反映事項:
  - まだ `main` の local-v2 service restart / live post-check は未実施。
    commit/push 後に `scripts/local_v2_stack.sh restart` と `status` /
    主要 worker log 確認まで続ける。

## 2026-03-11 04:36 JST / 改善レイヤの snapshot 依存と current flow 非追随の切り分け
- Why/Hypothesis:
  - ユーザー指摘どおり、「entry 実行は live でも、改善・配分・共通 penalty の一部が snapshot artifact 依存なら、相場の流れが変わった直後に効かなくなる」可能性がある。
  - 特に `RangeFader` 系の repeated low-probability entry は、strategy-local の flow 判定が弱く、shared 側の artifact trim/reject が後追いになっている疑いがある。
- Expected Good:
  - live tick/candle/factor に主導権を戻し、artifact は slow baseline / audit に降格すれば、同じ低品質 setup の連発を減らせる。
  - strategy 全体ではなく `setup_fingerprint / trend_snapshot / microstructure bucket` 単位で補正できれば、regime shift 後の立ち上がりが速くなる。
- Expected Bad:
  - snapshot 系を急に弱めすぎると、低 sample 戦略で loser lane の抑制が外れ、参加率だけ先に回復して損失が増える。
  - freshness gate を厳格化しすぎると artifact 欠損時に保守的 skip が増え、entry 数が落ちる。
- Observed/Fact:
  - 2026-03-11 04:32 JST 前後の OANDA 観測は `USD/JPY 157.998` 近辺、spread `0.8 pips`、ATR14 は `M1 2.59 pips / M5 7.43 pips`、pricing/openTrades/candles 応答は `236-355 ms` で、異常流動性ではなかった。
  - local-v2 主要サービスは active。`logs/health_snapshot.json` では `data_lag_ms=628`, `decision_latency_ms=21.5` と execution path は健全。
  - 直近15分は `orders.db` で `entry_probability_reject=2216`, `perf_block=1583`, `filled=202`。`trades.db` は `200 trades / -197.8 pips / avg -0.989 pips`。
  - `RangeFader-sell-fade` は 2026-03-11 04:34-04:35 JST に `entry_probability 0.371-0.378`、`confidence 33`、`range_reason=volatility_compression` のほぼ同一 thesis を数秒おきに再送しており、current flow に対する strategy-local の再評価よりも shared reject が先に働いている。
  - code 上も freshness 方針が不統一。
    - stale skip あり: `participation_alloc`, `market_context`, `order_manager` の tick/factor stale guards。
    - stale age gate 弱い/未実装: `dynamic_alloc`, `strategy_feedback`, `auto_canary`, `pattern_gate`。
  - artifact 実測:
    - `config/dynamic_alloc.json as_of=2026-03-10T19:33:58Z`（約17秒齢）
    - `config/participation_alloc.json as_of=2026-03-10T19:31:03Z`（約192秒齢）
    - `config/pattern_book_deep.json as_of=2026-03-10T19:30:03Z`（約252秒齢）
    - `analysis/macro_snapshot_builder.py` は 10分 refresh、`analysis/market_context.py` は 30秒 refresh / 最大1800秒 stale 許容。
- Verdict:
  - `pending`
  - 「全部が静止画」は誤りだが、「改善レイヤの一部が窓集計 artifact 依存で、しかも freshness 制御が揃っていない」は事実。
  - 現症の主因は「shared gate が厳しすぎること単体」ではなく、「strategy-local が current flow を十分に折り込まず、artifact ベース trim/reject が後追いで効いていること」。
- Next Action:
  - `RangeFader` を優先対象にし、strategy-local で `recent 3-5 candles pressure / ma_gap-ATR ratio / setup_fingerprint` を使って low-quality fade の repeated emission を止める。
  - `dynamic_alloc` と `strategy_feedback` に payload age / drift overlay の扱いを追加し、slow baseline と fast overlay を分離する。
  - shared layer には新しい一律 gate を足さず、`entry_thesis` に `setup_fingerprint`, `flow_regime`, `microstructure_bucket` を標準化して、strategy-local と exit worker に引き回す。

## 2026-03-11 04:45 JST / `MicroTrendRetest` の shallow chase retest を動的ガードで対称補強
- Why/Hypothesis:
  - ユーザー指摘どおり、単発の価格スナップショットに合わせた fixed tuning は
    1分後に腐るため採らない。
  - 代わりに local-v2 実測の multi-window RCA を実施した。
    `OANDA snapshot: USD_JPY mid=157.962 spread=0.8p`,
    `M1 ATR14=2.63p / ATR60=3.11p / 60m range=20.4p`,
    `M5 ATR14=8.84p` で市況は通常帯。
  - そのうえで `MicroTrendRetest-long` は
    `24h net_jpy=-290.2, PF=0.115`,
    `7d net_jpy=-452.4, PF=0.152` と継続劣化。
    `MicroTrendRetest-short` も `7d net_jpy=-531.9, PF=0.220`。
  - loser cluster は
    `up_strong/down_strong + vol:tight + atr:low/ultra_low + shallow retest`
    に寄っており、
    現行 `_chase_reset_ok()` が
    same-direction pressure 下の weak reclaim candle をまだ通していると判断。
- Expected Good:
  - `STOP_LOSS_ORDER` へ落ちる shallow chase retest を
    `ATR / breakout stretch / retest depth / candle body-wick / recovery`
    で動的に reject し、
    時間帯 block や固定価格帯 tuning なしに
    `MicroTrendRetest` の expectancy を改善する。
- Expected Bad:
  - low ATR の clean retest まで過剰に落とすと
    participation が下がるリスクがある。
  - 特に strong continuation 後の深め reset を許したいので、
    `breakout stretch` と `retest depth` を ATR 比で併用し、
    shallow case だけを狙う。
- Observed/Fact:
  - `strategies/micro/trend_retest.py` の `_chase_reset_ok()` を更新し、
    same-direction chase pressure 下の
    `low ATR + shallow retest + weak reclaim candle`
    を long/short 対称に reject する guard を追加。
  - 実装は price snapshot 固定ではなく、
    `ATR / breakout stretch / retest depth / body / wick / close recovery`
    の runtime feature だけを使用。
  - テスト:
    - `./.venv/bin/pytest -q tests/strategies/test_trend_retest.py` -> `16 passed`
    - `./.venv/bin/pytest -q tests/workers/test_micro_multistrat_trend_flip.py` -> `27 passed`
- Verdict: pending
- Next Action:
  - commit/push 後に `quant-micro-trendretest` 系を restart し、
    `MicroTrendRetest-long/-short` の `STOP_LOSS_ORDER` 件数、
    `filled` / `strategy_cooldown` / `perf_block` 比率、
    24h `net_jpy` / `PF` を再観測する。

## 2026-03-11 04:55 JST / stale artifact を no-op 化し、`RangeFader` を current flow へ寄せる
- Why/Hypothesis:
  - 2026-03-11 04:36 JST の RCA どおり、問題は「全部が静止 snapshot」ではなく、
    live entry path の上に stale artifact と strategy-local current-flow 不足が重なっていたことだった。
  - shared artifact が stale のまま trim/block すると、
    相場が切り替わっても loser setup の後追い reject か、古い winner/loser bias が残る。
  - `RangeFader` は repeated low-probability fade を shared reject で落とすのではなく、
    recent continuation と `ma_gap/ATR` で strategy-local に止めるほうが正しい。
- Expected Good:
  - stale `strategy_feedback` / `auto_canary` / `pattern_gate` が live entry に効き続ける状態を減らせる。
  - `entry_thesis` に `flow_regime`, `microstructure_bucket`, `setup_fingerprint`,
    `continuation_pressure` を残すことで、entry/exit/監査が同じ live context を参照できる。
  - `RangeFader` の same-thesis repeated fade が減り、
    shared `entry_probability_reject` / `perf_block` へ行く前に loser fade を間引ける。
- Expected Bad:
  - freshness gate を厳しくしすぎると、artifact 欠損窓で補助 trim が薄くなり entry が荒れるリスクがある。
  - `RangeFader` の flow guard が強すぎると、極端な逆張り winner まで取り逃がす可能性がある。
- Observed/Fact:
  - 実装:
    - `analysis/strategy_feedback.py`
      - feedback / counterfactual payload に最大 age 判定を追加し、stale main payload は空扱い、fresh counterfactual のみ overlay 可能にした。
    - `analysis/auto_canary.py`
      - stale override を no-op にした。
    - `workers/common/pattern_gate.py`
      - stale DB/JSON source を `db_stale/json_stale` として読み飛ばすようにした。
    - `execution/strategy_entry.py`
      - stale `dynamic_alloc` profile は trim しない。
      - `technical_context` から `live_setup_context` を生成し、
        `flow_regime`, `microstructure_bucket`, `setup_fingerprint` を `entry_thesis` に注入するようにした。
    - `strategies/scalping/range_fader.py`
      - recent 4-bar continuation + `ma_gap/ATR` + `ADX/DI` による `flow_headwind` pressure で
        continuation 強めの shallow fade を reject するようにした。
      - allowed signal に `continuation_pressure`, `flow_regime`, `ma_gap_pips`, `gap_ratio`, `setup_fingerprint` を露出するようにした。
    - `workers/scalp_rangefader/worker.py`
      - 上記 signal metadata を `entry_thesis` へコピーするようにした。
  - テスト:
    - `./.venv/bin/pytest -q tests/analysis/test_strategy_feedback.py tests/analysis/test_auto_canary.py` -> `6 passed`
    - `./.venv/bin/pytest -q tests/workers/test_pattern_gate.py tests/workers/test_scalp_rangefader_worker.py` -> `15 passed`
    - `./.venv/bin/pytest -q tests/execution/test_strategy_entry_adaptive_layers.py tests/strategies/test_scalp_thresholds.py` -> `16 passed`
- Verdict: pending
- Next Action:
  - `main` へ push 後に local-v2 を restart し、
    `RangeFader-sell-fade` / `RangeFader-buy-fade` の repeated reject 本数、
    `entry_probability_reject` / `perf_block` / `filled` 比率、
    `signal_flow_context` と `live_setup_context` の監査 payload を live で確認する。

## 2026-03-11 05:11 JST / `strategy_feedback` を strategy-wide から setup-scoped override へ分解
- Why/Hypothesis:
  - stale artifact を切っても、fresh な `strategy_feedback` が strategy-wide 一律補正のままだと、
    current setup が変わった直後も「最近の losers/winners」の bias が broad に残る。
  - 本当に動的にするなら、shared feedback 自体を
    `setup_fingerprint / flow_regime / microstructure_bucket` 単位で current setup match に限定する必要がある。
- Expected Good:
  - `RangeFader` や `MicroTrendRetest` のような同一 strategy 内の mixed regime で、
    loser cluster の trim が clean setup へ波及しにくくなる。
  - shared feedback が「戦略単位の古い平均」ではなく、
    current live setup に一致した recent cluster にだけ効く。
- Expected Bad:
  - setup 粒度を細かくしすぎると sample 不足で override が sparse になる。
  - `entry_thesis` の setup metadata 欠損時は従来の strategy-wide advice へ fallback するため、
    新旧 path の混在期間は効き方が uneven になり得る。
- Observed/Fact:
  - `analysis/strategy_feedback_worker.py`
    - `trades.entry_thesis` から `setup_fingerprint`, `flow_regime`, `microstructure_bucket` を抽出し、
      `setup_fingerprint` / `flow_micro` / `flow_regime` / `microstructure_bucket`
      の specificity で `setup_overrides` を生成するようにした。
    - local `logs/trades.db` の closed trade `17211` 件では explicit `setup_*` field が 0 件だったが、
      `technical_context` は `4865` 件に残っていた。
      そのため `technical_context` / `spread_pips` / `range_score` / `units` から
      setup identity を再構成する fallback を追加し、
      新規約定待ちをせず historical cluster から `setup_overrides` を作れるようにした。
  - `analysis/strategy_feedback.py`
    - `current_advice(..., entry_thesis=...)` を追加し、
      current setup に一致した override を base strategy advice の上に適用するようにした。
    - 一致した override は `_meta.setup_override` に残す。
  - `execution/strategy_entry.py`
    - feedback 適用時に live `entry_thesis` を `current_advice()` へ渡すようにした。
  - テスト:
    - `./.venv/bin/pytest -q tests/analysis/test_strategy_feedback.py` -> `6 passed`
    - `./.venv/bin/pytest -q tests/analysis/test_strategy_feedback_worker.py` -> `10 passed`
    - `./.venv/bin/pytest -q tests/execution/test_strategy_entry_adaptive_layers.py` -> `9 passed`
- Verdict: pending
- Next Action:
  - `main` へ push 後に local-v2 を restart し、
    `logs/strategy_feedback.json` に `setup_overrides` が載る strategy を確認する。
    特に historical `technical_context` 由来の override が即時に生成されるかを見る。
  - live order/trade 監査で `_meta.setup_override` と
    `live_setup_context` / `setup_fingerprint` の一致率、
    blanket trim の減少、
    `entry_probability_reject` と `filled` の比率変化を追う。

## 2026-03-11 05:40 JST / `participation_alloc` を strategy-wide から setup-scoped override へ分解
- Why/Hypothesis:
  - `strategy_feedback` だけ setup-scoped にしても、
    shared participation が strategy-wide のままだと
    同一 strategy 内の mixed regime を still broad に trim/boost してしまう。
  - `entry_path_summary` から setup identity を集計し、
    `participation_alloc` でも current setup match の override を使えば、
    shared participation も「今の型」にだけ効かせられる。
- Expected Good:
  - `RangeFader` のように同一 strategy 内で buy/sell/neutral や regime が混在していても、
    loser setup の trim が winner setup を巻き添えで削りにくくなる。
  - shared participation が current setup 単位の overuse / underuse を反映する。
- Expected Bad:
  - setup 粒度が細かいと sample が割れ、override が sparse になる。
  - setup realized P/L の backfill が弱い戦略では、
    share / hard-block 中心の override になり、効き方が uneven になり得る。
- Observed/Fact:
  - `scripts/entry_path_aggregator.py`
    - `orders.request_json.entry_thesis` から `setup_fingerprint`,
      `flow_regime`, `microstructure_bucket` を抽出し、
      strategy ごとに `setups` 集計を出すようにした。
  - `scripts/participation_allocator.py`
    - strategy-level allocation に加えて setup 別 `setup_overrides` を生成するようにした。
    - setup realized P/L を `trades.entry_thesis` から backfill する path を追加した。
  - `workers/common/participation_alloc.py`
    - live `entry_thesis` の current setup と一致する `setup_overrides` を選ぶようにした。
  - `execution/strategy_entry.py`
    - participation loader に live `entry_thesis` を渡し、
      一致した `setup_override` を監査 payload に残すようにした。
  - テスト:
    - `./.venv/bin/pytest -q tests/scripts/test_participation_allocator.py tests/workers/common/test_participation_alloc.py tests/execution/test_strategy_entry_adaptive_layers.py` -> `23 passed`
  - live dry-run:
    - `entry_path_aggregator.build_report(...)` + `participation_allocator.build_participation_alloc(...)`
      で `21` strategy 中 `4` strategy に `setup_overrides` が即時生成された。
    - sample:
      - `RangeFader-buy-fade: 1`
      - `RangeFader-neutral-fade: 4`
      - `RangeFader-sell-fade: 2`
      - `scalp_extrema_reversal_live: 2`
- Verdict: pending
- Next Action:
  - `main` へ push 後に local-v2 へ反映し、
    `config/participation_alloc.json` と `logs/entry_path_summary_latest.json` を最新生成する。
  - live order 監査で `entry_thesis["participation_alloc"].setup_override` が
    current setup と一致すること、
    RangeFader 系の blanket trim が減ることを確認する。

## 2026-03-11 09:07 JST / local-v2: `M1Scalper` breakout/vshape を live setup payload 化
- Why/Hypothesis:
  - `M1Scalper` は live factor を十分に持っているのに、
    breakout/vshape の final threshold と worker 側 sizing/thesis が固定値寄りで、
    current setup の quality が shared layer へ十分に露出していなかった。
  - breakout/vshape の signal で `flow_regime / continuation_pressure / setup_quality / setup_fingerprint`
    を live 算出し、worker がそのまま `entry_thesis` へ保存すれば、
    `M1Scalper-M1` を strategy-wide に見る static bias を薄くできる。
- Expected Good:
  - breakout/vshape の entry が current continuation / regime / volatility に応じて
    `entry_probability` と size を変える。
  - shared `strategy_feedback` / `participation_alloc` が
    `M1Scalper` を broad loser lane ではなく current setup cluster で学習できる。
- Expected Bad:
  - setup payload が細かすぎると sample が割れ、
    setup-scoped shared overlay の効き方が sparse になる。
  - signal quality が弱い局面では `setup_size_mult` が downscale し、
    participation が一時的に落ちる可能性がある。
- Observed/Fact:
  - 市況:
    - `USD_JPY bid/ask=158.082/158.090`、spread 約 `0.8 pips`。
    - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env` で
      core/strategy service は `running`。
    - `scripts/collect_local_health.sh` は `snapshot_age_sec=0`, `stale_warn=no`。
  - `strategies/scalping/m1_scalper.py`
    - breakout-retest / vshape-rebound signal に
      `flow_regime`, `continuation_pressure`, `microstructure_bucket`,
      `setup_quality`, `setup_fingerprint`, `entry_probability`, `setup_size_mult`
      を live factor ベースで付与するようにした。
    - `BODY/RETEST/MOMENTUM/RSI/TTL` は `atr_pips`, `adx`, `range_score`,
      `ema gap`, recent continuation から effective threshold を再計算するようにした。
  - `workers/scalp_m1scalper/worker.py`
    - signal setup payload を `entry_thesis` の
      `m1_setup`, `flow_regime`, `microstructure_bucket`, `setup_fingerprint`,
      `setup_quality`, `signal_entry_probability` へ保存するようにした。
    - `setup_size_mult` は worker local の size scale に反映するようにした。
  - テスト:
    - `./.venv/bin/pytest -q tests/workers/test_m1scalper_nwave_tolerance_override.py tests/workers/test_m1scalper_setup_context.py tests/workers/test_m1scalper_quickshot.py` -> `7 passed`
    - `./.venv/bin/pytest -q tests/workers/test_m1scalper_nwave_tolerance_override.py tests/workers/test_m1scalper_setup_context.py tests/workers/test_m1scalper_quickshot.py tests/workers/test_m1scalper_open_trades_guard.py tests/workers/test_m1scalper_config.py tests/replay/test_m1_family_replay.py` -> `20 passed`
- Verdict: pending
- Next Action:
  - local-v2 反映後、`orders.db` の `M1Scalper-breakout-retest-*` / `M1Scalper-vshape-rebound-*`
    で `entry_thesis.m1_setup.setup_fingerprint` と `entry_probability` が入っていることを確認する。
  - `strategy_feedback` / `participation_alloc` の setup-scoped overlay が
    `M1Scalper` の broad loser trim ではなく current setup match で効くかを live order/trade で追う。

## 2026-03-11 09:52 JST / local-v2: `dynamic_alloc` / `auto_canary` / `M1Scalper EXIT` を setup/trade-local 化
- Why/Hypothesis:
  - `participation_alloc` / `strategy_feedback` を setup-scoped にしても、
    `dynamic_alloc` と `auto_canary` が strategy-wide のままだと
    current setup と無関係な blanket trim が残る。
  - `M1Scalper` も entry だけ setup payload を持っていて、
    exit が fixed threshold 寄りだと winner setup を伸ばせず loser setup の cut も遅れる。
  - slow adaptive layer と exit worker まで同じ setup identity でつなげば、
    「静止した市況の snapshot」ではなく current setup と trade thesis に沿った補正へ寄せられる。
- Expected Good:
  - `dynamic_alloc` と `auto_canary` が current setup match に限定され、
    同一 strategy 内の別 lane を巻き添えで trim しにくくなる。
  - `M1Scalper` は aligned `breakout_retest` を伸ばしやすく、
    headwind `vshape_rebound` は hold / adverse / soft-exit を tighter に寄せられる。
  - `loser_cluster -> auto_canary` も setup context を持つので、
    canary line が strategy-wide ではなく setup-scoped で回り始める。
- Expected Bad:
  - setup 粒度を上げる分、sample が割れて override が sparse になる lane が出る。
  - `setup_fingerprint` が細かすぎる strategy は、
    `flow_regime / microstructure_bucket` fallback より exact fingerprint が先に選ばれ、
    current data の sparse/noisy override を拾うリスクがある。
  - `M1Scalper` exit の dynamic threshold が強すぎると、
    弱い setup で早利食いしすぎて cadence が落ちる可能性がある。
- Observed/Fact:
  - 市況:
    - `USD/JPY bid/ask=158.094/158.102`, spread 約 `0.8 pips`。
    - `logs/health_snapshot.json` は `generated_at=2026-03-11T00:52:04Z`,
      `data_lag_ms=846.0`, `decision_latency_ms=16.37`, `missing_mechanisms=null`。
  - `scripts/dynamic_alloc_worker.py`
    - recent `trades.entry_thesis` から setup identity を再構成し、
      `config/dynamic_alloc.json` に `11` strategy / `55` setup override を生成した。
  - `scripts/loser_cluster_worker.py` / `scripts/auto_canary_improver.py`
    - loser cluster に `setup_context` を保持し、
      `config/auto_canary_overrides.json` に `13` strategy / `46` setup override を生成した。
  - `workers/common/dynamic_alloc.py` / `analysis/auto_canary.py`
    - live `entry_thesis` の current setup と一致する override だけを返すようにした。
  - `execution/strategy_entry.py`
    - `dynamic_alloc` / `auto_canary` 適用時に live `entry_thesis` を loader へ渡し、
      一致した override を監査 payload に残すようにした。
  - `workers/scalp_m1scalper/exit_worker.py`
    - `entry_thesis.m1_setup` を優先して
      `profit_take_pips`, `lock_trigger_pips`, `max_hold_sec`, `max_adverse_pips`,
      `trail_*`, `rsi_fade_*`, `vwap_gap_pips`, `structure_*`, `atr_spike_pips`
      を trade-local に再計算するようにした。
  - テスト:
    - `./.venv/bin/pytest -q tests/workers/common/test_dynamic_alloc.py tests/analysis/test_auto_canary.py tests/execution/test_strategy_entry_adaptive_layers.py tests/scripts/test_auto_canary_improver.py tests/scripts/test_loser_cluster_worker.py tests/test_dynamic_alloc_worker.py tests/workers/test_m1scalper_nwave_tolerance_override.py tests/workers/test_m1scalper_setup_context.py tests/workers/test_m1scalper_exit_worker.py tests/workers/test_m1scalper_quickshot.py tests/workers/test_m1scalper_open_trades_guard.py tests/workers/test_m1scalper_config.py tests/replay/test_m1_family_replay.py`
      -> `62 passed`
- Verdict: pending
- Next Action:
  - local-v2 反映後、
    `orders.db` / `trades.db` で `entry_thesis.dynamic_alloc.setup_override`,
    `entry_thesis.auto_canary.setup_override`,
    `entry_thesis.m1_setup` と exit 監査の整合を spot check する。
  - `M1Scalper-M1`, `MicroLevelReactor-*`, `MicroTrendRetest-long`,
    `RangeFader-*` の live order/trade で
    strategy-wide blanket trim が減って current setup match の補正へ寄ったかを追う。

## 2026-03-11 10:25 JST / local-v2: `WickReversalBlend` short fade の bullish continuation headwind を worker local で遮断
- Why/Hypothesis:
  - live loser は `DroughtRevert` / `PrecisionLowVol` / `VwapRevertS` の short fade に寄っていた。
    2026-03-11 10:25 JST の local-v2 実測では `USD/JPY bid/ask=158.210/158.218`、spread 約 `0.8 pips`、
    `scalp` pocket は `-1659 units / 3 trades / unrealized -70.8 pips` の short 偏りだった。
  - `WickReversalBlend` には short 用 `flow_guard` が既にあったが、
    `DI gap` と `vwap stretch` の continuation 圧力が弱く、
    `PrecisionLowVol` では marginal headwind でも `vgap_bias_ok` が
    confidence/size boost を残していた。
  - `flow_guard` を live continuation ベースへ寄せ、
    `DroughtRevert` は projection deny も worker local で止めれば、
    shared post-hoc reject に頼らず wrong-way short を前段で削れる。
- Expected Good:
  - bullish continuation が残る short fade を signal 時点で reject し、
    `entry_probability_reject` の後段 reject より前に低品質 attempt を減らせる。
  - `PrecisionLowVol` の positive `vwap_gap` は
    clean fade のときだけ boost され、marginal headwind short の size/conf boost が消える。
  - `flow_guard` が `entry_thesis` に残るため、
    downstream の setup-scoped shared layer と exit 監査で
    `continuation_pressure / setup_quality` を読める。
- Expected Bad:
  - short fade の blocked rate が一時的に上がり、
    range revert 系の cadence が落ちる可能性がある。
  - `trend_stack` や `stretch_pressure` の重みが強すぎると、
    clean overextension まで block して winner fade を落とすリスクがある。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/worker.py`
    - `_reversion_short_flow_guard()` に
      `plus_di-minus_di`, `trend_stack`, `stretch_pressure` を追加した。
    - `setup_quality` が低く `trend_stack` が高い marginal short も reject するようにした。
    - `DroughtRevert` は `projection_decision(..., mode="range")` を必須化した。
    - `PrecisionLowVol` の `vgap_bias_ok` は
      `continuation_pressure + 0.05 <= max_pressure` かつ `setup_quality >= 0.66`
      のときだけ boost するようにした。
    - `flow_guard` を `entry_thesis` に露出し、
      `continuation_pressure / reversion_support / setup_quality / flow_regime`
      を記録するようにした。
  - テスト:
    - `./.venv/bin/pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py tests/workers/test_scalp_wick_reversal_blend_exit_worker.py tests/workers/test_scalp_wick_reversal_blend_dispatch.py`
      -> `13 passed`
- Verdict: pending
- Next Action:
  - local-v2 反映後、
    `orders.db` で `DroughtRevert` / `PrecisionLowVol` / `VwapRevertS` の
    short reject/filled mix と `entry_thesis.flow_guard` を spot check する。
  - `scalp` pocket の short 偏りと `RangeFader` 以外の loser lane が
    どこまで減るかを current live trades で追う。

## 2026-03-11 10:55 JST / local-v2: order-manager service 経由の二重 probability gate を解消
- Why/Hypothesis:
  - 2026-03-11 10:49 JST 時点の local-v2 実測では
    `USD/JPY 158.223`、tick spread 約 `0.8 pips`、
    `M1 ATR 2.61 pips`、`M5 ATR 5.88 pips`、
    `data_lag_ms 666`、`decision_latency_ms 17` で、
    OANDA pricing stream は `200 OK` 継続だった。
  - `logs/oanda_account_snapshot_live.json` では
    `nav=35,151.17`, `margin_used=10,501.80`, `margin_available=24,655.85`,
    `free_margin_ratio=0.7012` で、margin 使用率は約 `29.9%` に留まっていた。
  - `orders.db` / `quant-order-manager.log` では
    直近 6 時間の `entry_probability_reject=2640`、`perf_block=1583` が支配的で、
    `RangeFader-buy/sell/neutral-fade` が reject の大半を占めていた。
  - 実際の reject payload では
    `1628 -> 1067 -> 972 -> 136` と worker/strategy_entry 側で縮小した後、
    `order_manager_probability_gate` が `136 -> 60` に縮小し、
    さらに同じ gate がもう一度走って `60 -> block` になっていた。
  - 原因は `ORDER_MANAGER_SERVICE_ENABLED=1` の strategy runtime から
    `quant-order-manager` service へ委譲する前に client 側 `execution/order_manager.py`
    でも probability gate を実行し、service worker 側で同じ gate を再実行していたこと。
- Expected Good:
  - service 経由の market/limit order で `order_manager_probability_gate` が 1 回だけになり、
    `entry_probability_below_min_units` の過剰 reject を減らせる。
  - `dynamic_alloc` / `participation_alloc` / `blackboard_coordination` 後に残った
    intent size が二重に削られず、margin 利用率と filled cadence の回復が見込める。
  - service timeout / unhandled 時は local fallback 側が従来どおり
    probability gate を 1 回実行する。
- Expected Bad:
  - client 側の pre-service `probability_scaled` ログが減るため、
    監査の見え方が一時的に変わる。
  - loser lane の実発注が少し増える可能性があるため、
    live では `filled / reject_rate / margin_used` を同時監視する必要がある。
- Observed/Fact:
  - `execution/order_manager.py`
    - `market_order()` / `limit_order()` は
      `entry_intent_guard` pass 後に service へ委譲し、
      service が `handled` を返した場合は client 側 `probability_gate` を実行しないようにした。
    - service が `unhandled` のときだけ local fallback 側で
      `probability_gate` を実行するようにした。
  - `tests/execution/test_order_manager_log_retry.py`
    - service 成功時に client 側 `_probability_scaled_units()` が呼ばれないこと、
      service unhandled 時に local fallback で gate が走ることを追加検証した。
  - テスト:
    - `pytest tests/execution/test_order_manager_log_retry.py`
      -> `22 passed`
- Verdict: pending
- Next Action:
  - local-v2 反映後に `orders.db` で
    `RangeFader-*` の `entry_probability_reject` と `probability_scaled` の比率が
    どう変わるかを確認する。
  - `logs/oanda_account_snapshot_live.json` の
    `margin_used / margin_available / free_margin_ratio` と、
    `health_snapshot.json` の `order_success_rate / reject_rate` を追って、
    margin 使用率が `~30%` からどこまで戻るかを監査する。

## 2026-03-11 12:xx JST / local-v2: `WickReversalBlend` / `VwapRevertS` の sparse-thesis adverse hold を worker local で前倒し
- Why/Hypothesis:
  - 2026-03-11 現在の local-v2 実測では
    `health_snapshot.git_rev=3abc9423`, `mechanism_integrity.ok=true`,
    `data_lag_ms=456.9`, `decision_latency_ms=13.1` と導線自体は正常で、
    open trades も `0` だった。
  - 一方で直近 6 時間は
    `WickReversalBlend short -154.42 JPY`, `VwapRevertS short -24.94 JPY`,
    `RangeFader long/short -151.53 JPY` と loser が続いていた。
  - `WickReversalBlend` の大きい負け trade を見ると、
    `wick_blend_quality / wick / projection` を欠いた sparse `entry_thesis` のまま
    `time_stop` まで保有されていた。
  - `VwapRevertS` short も `exit_worker` の
    `wick_blend_exit_adjustments()` 適用対象から漏れており、
    `orders.db` / `trades.db` 上で `continuation_pressure / setup_quality / flow_guard`
    が欠落していた。
  - sparse thesis でも quality を rebuild し、
    `projection_headwind` と `continuation_pressure` を使って
    `hard cut / max hold` を前倒しすれば、
    trade-local の adverse hold を減らせる。
- Expected Good:
  - `WickReversalBlend` / `VwapRevertS` の wrong-way fade が
    `time_stop` まで残るケースを減らせる。
  - `entry_thesis` の nested `flow_guard` が落ちても、
    top-level dynamic fields から exit 側が同じ current setup を復元できる。
  - high-quality / no-headwind winner は必要以上に早く利食わず、
    headwind lane だけ tighter にできる。
- Expected Bad:
  - headwind 判定が強すぎると `VwapRevertS` の clean revert winner まで
    早利食いへ寄る可能性がある。
  - signal/top-level field の冗長化で order payload がやや膨らむ。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/policy.py`
    - `projection_headwind` を entry quality に追加した。
    - sparse thesis 向け `_fallback_wick_blend_trade_quality()` を追加した。
    - `wick_blend_exit_adjustments()` は
      `projection_headwind` / `continuation_pressure` / sparse thesis を見て
      `trail_*`, `loss_cut_hard_pips`, `loss_cut_max_hold_sec` を再計算するようにした。
    - high-quality / no-headwind lane では
      `trail_start` を base より不用意に下げない floor を戻した。
  - `workers/scalp_wick_reversal_blend/worker.py`
    - short `flow_guard` を
      `continuation_pressure / reversion_support / setup_quality / flow_regime`
      の top-level fields としても signal に保持するようにした。
    - `_build_entry_thesis()` は nested `flow_guard` が無くても
      top-level dynamic fields を `entry_thesis` に昇格するようにした。
  - `workers/scalp_wick_reversal_blend/exit_worker.py`
    - `wick_blend_exit_adjustments()` の live adjust を
      `WickReversalBlend` だけでなく `VwapRevertS` にも適用した。
  - テスト:
    - `pytest -q tests/workers/test_scalp_wick_reversal_blend_* tests/workers/test_scalp_precision_wrapper_env.py`
      -> `29 passed`
- Verdict: pending
- Next Action:
  - 反映後に `orders.db` / `trades.db` で
    `WickReversalBlend` / `VwapRevertS` の新規 trade に
    `continuation_pressure / setup_quality / flow_regime` が実際に残るかを spot check する。
  - `time_stop` close が `loss_cut_*` や earlier protective close へ置き換わるか、
    直近 6 時間の same-tag loser cluster で追う。

## 2026-03-11 14:xx JST / local-v2: `RangeFader` の post-commit loser lane を strategy-local にさらに削る
- Why/Hypothesis:
  - `0cc16fd8` 反映後の UTC 窓を正しく `julianday(...)` で切ると、
    `RangeFader` は `21 trades / -3.514 JPY / -5.6 pips / avg -0.27 pips`
    まで改善していた。
  - それでも loser は
    `RangeFader|short|neutral-fade|range_fade|p0 = 7 trades / -7.7 pips` と
    `RangeFader|long|buy-fade|range_fade|p0 = 3 trades / -3.6 pips`
    に集約していた。
  - いずれも huge loser ではなく、
    `~180 sec` 前後で小さく切られる low-edge lane の積み上がりだった。
  - `neutral-fade short p0` は entry quality が薄すぎるので block 寄りに、
    `buy-fade long p0` は entry を全停止せず trade-local exit を早めれば、
    current cadence を落としすぎずに expectancy を改善できる。
- Expected Good:
  - `neutral-fade short p0` の 0W-7L lane を entry 前に減らせる。
  - `buy-fade long p0` は勝ちを小さく拾い、負けをより短くする方向へ寄る。
  - `sell-fade short p0` の winner lane は維持しやすい。
- Expected Bad:
  - `neutral-fade short` を切りすぎると、薄いが有効な revert winner も減る可能性がある。
  - `buy-fade long` の exit を早めすぎると、遅れて戻る winner を取り逃がす可能性がある。
- Observed/Fact:
  - `strategies/scalping/range_fader.py`
    - `fragile_neutral_short_range_guard()` を追加した。
    - `flow_regime=range_fade`, `continuation_pressure=0`, `range_score>=0.45`,
      `gap_ratio>=0.35`, low momentum の `neutral-fade short` に対して、
      lane-aware quality floor を満たせない setup を skip するようにした。
  - `workers/scalp_rangefader/exit_worker.py`
    - `buy-fade long` の `range_fade/p0` で
      `setup_quality>=0.70` なら
      `profit_take`, `soft_adverse`, `max_hold`, `trail_*`, `lock_buffer`
      を tighter に再計算するようにした。
  - テスト:
    - `pytest -q tests/strategies/test_scalp_thresholds.py tests/workers/test_scalp_rangefader_exit_worker.py tests/workers/test_scalp_rangefader_worker.py`
      -> `26 passed`
- Verdict: pending
- Next Action:
  - 反映後に `RangeFader|short|neutral-fade|range_fade|p0` と
    `RangeFader|long|buy-fade|range_fade|p0` の trades / avg_pips を
    post-commit UTC 窓で再集計する。
  - `sell-fade short p0` の winner lane が維持されているかも同時に監査する。

## 2026-03-11 15:xx JST / local-v2: `VwapRevertS` の `gap:up_strong` hostile short を entry 前に止める
- Why/Hypothesis:
  - 最新の正しい UTC 窓では、fresh sample で `RangeFader` はほぼ止まり、
    post-restart では `scalp_extrema_reversal_live` の 1 本しか新規 closed trade が無かった。
  - それでも 6 時間 loser では
    `VwapRevertS short` が `-26.643 JPY / -19.6 pips` で最悪だった。
  - setup cluster で切ると
    `VwapRevertS|short|range_fade|unknown|rsi:overbought|atr:mid|gap:up_strong|volatility_compression`
    の 2 本が `-30.739 JPY / -26.1 pips / avg hold 1082 sec` を作っていた。
  - この lane は `projection.score` が negative でも、`vgap` extension と shallow reversal だけで short が通っていた。
  - hostile projection / strong extension / weak reversal / weak setup_quality を組み合わせて
    entry 前に止めれば、後段の exit に頼る前に最悪 lane を消せる。
- Expected Good:
  - `VwapRevertS short` の `gap:up_strong` loser lane を fresh sample で減らせる。
  - strong extension でも supportive projection や明確な reversal を持つ winner lane は残せる。
- Expected Bad:
  - `vgap/ATR` が大きい clean revert winner まで削ると、VwapRevertS の参加率が落ちる可能性がある。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/worker.py`
    - `_signal_vwap_revert()` に short lane block を追加した。
    - 条件は `projection.score <= -0.10`, `vgap/ATR >= 6.5`,
      `range_score >= 0.30`, shallow overbought RSI, `rev_strength < 0.90`,
      `flow_guard.setup_quality < 0.58` の組み合わせ。
  - `tests/workers/test_scalp_wick_reversal_blend_dispatch.py`
    - `gap:up_strong` hostile projection lane が block されることを追加検証した。
  - テスト:
    - `pytest -q tests/workers/test_scalp_wick_reversal_blend_dispatch.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py tests/workers/test_scalp_wick_reversal_blend_policy.py tests/workers/test_scalp_wick_reversal_blend_exit_worker.py`
      -> `22 passed`
- Verdict: pending
- Next Action:
  - 反映後に `VwapRevertS short` の new fills が出た場合、
    `setup_fingerprint`, `projection.score`, `setup_quality` を
    `orders.db` / `trades.db` で spot check する。
  - `gap:up_lean` や `gap:up_flat` の winner lane が維持されるかも同時に確認する。

## 2026-03-11 20:xx JST / local-v2: shared env blanket stop rollback を docs 正本へ反映
- Why/Hypothesis:
  - 2026-03-11 の RCA では loser lane が `PrecisionLowVol` / `WickReversalBlend` /
    `VwapRevertS` / `MicroCompressionRevert` / `MicroVWAPRevert` に集中したが、
    current architecture は shared env の blanket `STRATEGY_CONTROL_ENTRY_* = 0`
    ではなく strategy-local guard と setup-scoped shared trim を正としている。
  - あわせて shared `RISK_PERF_MIN_MULT` の broad raise は
    winner 以外まで一律に risk floor を押し上げるため、
    safer baseline を正本へ戻しておかないと次回 RCA で再び全体 raise に寄りやすい。
  - `ORDER_MANAGER_FORECAST_GATE_APPLY_WITH_PRESERVE_INTENT=1` の運用は
    preserve-intent が min-RR を bypass する誤解を生みやすく、
    `ORDER_MIN_RR_*` が risk/execution guard であることを明記する必要があった。
- Expected Good:
  - 次回の RCA / PDCA で blanket stop や broad multiplier raise へ戻らず、
    strategy-local guard と shared trim の責務分離を維持できる。
  - preserve-intent を使う order path でも
    min-RR floor が有効である前提を運用ログ/仕様で揃えられる。
- Expected Bad:
  - docs だけ先行して stale になると、runtime env と運用記録の間にズレが残る可能性がある。
- Observed/Fact:
  - `AGENTS.md` / `docs/WORKER_REFACTOR_LOG.md` / `docs/RISK_AND_EXECUTION.md`
    を更新し、shared env の blanket `STRATEGY_CONTROL_ENTRY_* = 0` を
    current 運用として扱わないこと、
    shared risk multiplier は safer baseline を維持すること、
    `ORDER_MIN_RR_*` は preserve-intent 下でも risk/execution guard として適用することを追記した。
- Verdict: good
- Next Action:
  - 次回の env / code change では
    `ops/env/local-v2-stack.env` と `ops/env/quant-order-manager.env` の実値が
    この方針と一致しているかを先に spot check してから RCA を進める。

## 2026-03-11 20:42 JST / local-v2: setup-scoped trim へ戻しつつ precision winner lane を攻める
- Why/Hypothesis:
  - 直近 60 分の `orders.db` では `scalp` pocket が `RangeFader short sell-fade` の churn に支配されており、
    `RangeFader|short|sell-fade|range_fade|p1` と `transition` lane が大量 preflight を消費していた。
  - 一方で `PrecisionLowVol` と `VwapRevertS` は current winner lane があるのに、
    `config/dynamic_alloc.json` の top-level `lot_multiplier` が
    setup override 無しで strategy-wide に効き、`0.14-0.24x` まで blanket trim されていた。
  - `DroughtRevert` は recent winner だが、専用 env の `*_UNIT_BASE_UNITS` / `*_UNIT_CAP_MAX` が
    runtime key と一致しておらず sizing が実質 default のまま、さらに long side は falling-knife guard が弱かった。
  - shared で広く止めるより、`dynamic_alloc` を setup-scoped に戻しつつ、
    `RangeFader` / `WickReversalBlend` の loser lane を strategy-local に締め、
    `DroughtRevert` / `PrecisionLowVol` / `VwapRevertS` の winner lane だけを厚くする方が current 方針に合う。
- Expected Good:
  - `PrecisionLowVol` / `VwapRevertS` は live setup に一致する override が無い限り
    blanket trim されず、winner lane の participation が戻る。
  - `RangeFader` の `transition` と `range_fade/p1` short fade、
    および `neutral-fade long range_fade/p0` の churn を entry 前に削れる。
  - `DroughtRevert` の strong reclaim long は
    stronger TP / size floor と正しい dedicated base units が効き、
    30 分窓の realized JPY を押し上げやすくなる。
- Expected Bad:
  - `dynamic_alloc` の trim を戻し過ぎると、
    `PrecisionLowVol` / `VwapRevertS` の non-winner lane まで participation が戻るリスクがある。
  - dedicated base units を 1.7-2.4x へ上げたため、
    short-term drawdown と margin pressure は悪化しうる。
- Observed/Fact:
  - `workers/common/dynamic_alloc.py`
    - live setup identity があるのに matching `setup_override` が無い場合、
      top-level negative `lot_multiplier` を runtime no-op に戻すようにした。
  - `execution/strategy_entry.py`
    - unmatched setup では `dynamic_alloc` trim が適用されず、
      `entry_thesis["dynamic_alloc"]` も汚さないことをテストで固定した。
  - `strategies/scalping/range_fader.py`
    - `sell-fade short` の fragile `transition` / `range_fade|p1` lane と
      `neutral-fade long range_fade|p0` lane を strategy-local guard で block するようにした。
  - `workers/scalp_wick_reversal_blend/worker.py`
    - `DroughtRevert` の long side に falling-knife guard を追加し、
      strong reclaim lane は TP / confidence / size floor を引き上げた。
    - `PrecisionLowVol` / `VwapRevertS` は `ma10-ma20` 由来の `gap:up_lean` winner lane を優遇し、
      `gap:down_flat` の weak short lane は penalty/block するようにした。
  - `ops/env/quant-scalp-drought-revert.env`
    - `SCALP_PRECISION_DROUGHT_REVERT_BASE_UNITS=22000`
    - `SCALP_PRECISION_DROUGHT_REVERT_CAP_MAX=1.08`
  - `ops/env/quant-scalp-precision-lowvol.env`
    - `SCALP_PRECISION_LOWVOL_BASE_UNITS=15000`
    - `SCALP_PRECISION_LOWVOL_CAP_MAX=0.98`
  - `ops/env/quant-scalp-vwap-revert.env`
    - `SCALP_PRECISION_VWAP_REVERT_BASE_UNITS=18000`
    - `SCALP_PRECISION_VWAP_REVERT_CAP_MAX=1.00`
  - テスト:
    - `pytest -q tests/workers/common/test_dynamic_alloc.py tests/execution/test_strategy_entry_adaptive_layers.py tests/strategies/test_scalp_thresholds.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py tests/workers/test_scalp_wick_reversal_blend_dispatch.py`
      -> `55 passed`
- Verdict: pending
- Next Action:
  - restart 後 30-60 分で
    `DroughtRevert`, `PrecisionLowVol`, `VwapRevertS`, `RangeFader`
    の `submit_attempt / filled / realized_jpy / avg_pips / setup_fingerprint`
    を再集計し、winner lane の厚みと loser lane の遮断が両立しているか確認する。
  - `PrecisionLowVol` / `VwapRevertS` の recent `request_json.entry_thesis.dynamic_alloc`
    が blank になり、shared top-level trim が消えていることを spot check する。

## 2026-03-11 21:40 JST / local-v2: shared setup override の低サンプル反応を 30 分窓向けに前倒し
- Why/Hypothesis:
  - 2026-03-11 21:39 JST 時点の local-v2 実測では
    `USD/JPY price=158.5035`、recent filled spread `0.8 pips`、
    `nav=35417.7654`、open trade `0`、`order_success_rate=1.0`、
    `reject_rate=0.0` で、ボトルネックは execution ではなく lane 配分だった。
  - `participation_alloc` は setup override を持てても
    strategy-level `min_attempts=20` と同じ閾値で setup を評価しており、
    current 4+ attempt / 3-5 fill の setup を shared boost/trim に上げにくかった。
  - さらに setup override は
    `max_units_boost / max_probability_boost`
    を loader へ渡しておらず、
    winner setup の `boost_participation` が
    probability-only になりやすかった。
  - `dynamic_alloc` も `setup_min_trades>=6` のため、
    `DroughtRevert|long|range_fade|unknown|rsi:oversold|atr:mid|gap:up_flat|volatility_compression`
    のような single-trade severe loser setup を current window で trim できていなかった。
- Expected Good:
  - current 30 分 loser setup を strategy-wide stop なしで細くし、
    share を winner setup へ戻しやすくなる。
  - `participation_alloc` の winner setup が
    probability だけでなく unit boost も runtime へ届く。
  - `dynamic_alloc` が acute loser setup を 2 分周期の artifact で捕まえられる。
- Expected Bad:
  - setup override の sample 閾値を下げるため、
    一時的なノイズ setup へ反応しやすくなる。
  - single-trade severe loser trim は conservative でも、
    一過性 stop を 1 本で細くしすぎるリスクがある。
- Observed/Fact:
  - `scripts/participation_allocator.py`
    - `setup_min_attempts` を独立化し、default `4` で setup override を評価するようにした。
    - setup override payload に
      `max_units_cut / max_units_boost / max_probability_boost`
      を含めるようにした。
  - `workers/common/participation_alloc.py`
    - setup override loader が
      `max_units_boost / max_probability_boost`
      を runtime profile へ保持するようにした。
  - `scripts/dynamic_alloc_worker.py`
    - `setup_min_trades` を独立化し、default `4` にした。
    - `sum_realized_jpy<=-8` かつ `weighted_win_rate<=0.25` /
      `jpy_pf<=0.25` の single-trade severe loser setup は、
      `setup_min_trades` 未満でも trim override を出すようにした。
  - テスト:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/scripts/test_participation_allocator.py tests/workers/common/test_participation_alloc.py`
      -> `15 passed`
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_dynamic_alloc_worker.py`
      -> `13 passed`
  - artifact 再生成:
    - `python3 scripts/entry_path_aggregator.py --lookback-hours 24 --limit 12000 --top-k 8`
    - `python3 scripts/participation_allocator.py --entry-path-summary logs/entry_path_summary_latest.json --trades-db logs/trades.db --output config/participation_alloc.json --lookback-hours 24 --min-attempts 20 --setup-min-attempts 4`
    - `PYTHONPATH=/Users/tossaki/Documents/App/QuantRabbit python3 scripts/dynamic_alloc_worker.py --limit 2400 --lookback-days 7 --min-trades 16 --setup-min-trades 4 --pf-cap 2.0 --target-use 0.88 --half-life-hours 24 --min-lot-multiplier 0.45 --max-lot-multiplier 1.65 --soft-participation 1 --allow-loser-block 0 --allow-winner-only 0`
    - 再生成後の `config/dynamic_alloc.json` では
      `DroughtRevert|long|range_fade|unknown|rsi:oversold|atr:mid|gap:up_flat|volatility_compression`
      に `lot_multiplier=0.45` の setup override が出た。
    - 同じく `PrecisionLowVol|short|...|gap:down_flat|volatility_compression`
      に `lot_multiplier=0.45` が出ており、
      `VwapRevertS|short|...|gap:up_lean|volatility_compression`
      の positive setup override も維持された。
- Verdict: pending
- Next Action:
  - `main` へ push 後に local-v2 を restart し、
    `DroughtRevert`, `PrecisionLowVol`, `VwapRevertS`, `RangeFader-sell-fade`
    の next 30-60 分 `gross_win / gross_loss / net_jpy`
    を setup ごとに再評価する。

## 2026-03-11 JST - 30分 speed-to-profit 向け participation trim/boost 加速
- Why/Hypothesis:
  - local-v2 実測では service / execution は正常で、
    `decision_latency_ms~17`, `order_success_rate~0.992`, `reject_rate~0.008`
    とボトルネックは runtime ではなかった。
  - 直近 30-120 分の loser は
    `PrecisionLowVol|short|range_fade|...|gap:down_flat`,
    `RangeFader|long|neutral-fade|range_fade|p0`,
    `RangeFader|short|sell-fade|range_fade|p1`,
    `DroughtRevert` の current lane に集中しており、
    既存 `participation_alloc` は
    「underused だが負けている lane」を trim できていなかった。
  - winner 側も `MomentumBurst-open_long` の current boost が
    `+2.6%` 前後と浅く、30 分窓の立ち上がりには弱かった。
- Expected Good:
  - `underused/high-fill` の loser lane を
    strategy-wide stop なしで current setup 単位に前捌きできる。
  - `MomentumBurst-open_long` など short-window winner の
    units/probability boost を少し厚くして、
    30 分 net JPY の立ち上がりを速くできる。
  - `run_local_feedback_cycle` の再計算でも同じ cap が継続し、
    manual regeneration と自動 cycle が乖離しない。
- Expected Bad:
  - recent loser への反応が速くなる分、
    一過性の drawdown lane に敏感になり、
    勝ち返し直前の setup を一時的に絞る可能性がある。
  - winner boost cap を `1.18 / 0.08` へ広げるため、
    artifact の誤学習があると短時間で size がやや乗りやすい。
- Observed/Fact:
  - `scripts/participation_allocator.py`
    - `underused でも realized_jpy が十分に悪い lane` を
      `trim_units + bounded probability_offset` に落とす
      `loss_drag` 分岐を追加した。
    - explicit `boost_participation` と small-sample winner boost を強め、
      default cap も `max_units_boost=0.18`,
      `max_probability_boost=0.08` へ更新した。
  - `execution/strategy_entry.py`
    - runtime の `STRATEGY_PARTICIPATION_ALLOC_MULT_MAX` default を `1.18`,
      `STRATEGY_PARTICIPATION_ALLOC_PROB_BOOST_MAX` default を `0.10`
      へ更新し、artifact 側の explicit boost を clip しにくくした。
  - `scripts/run_local_feedback_cycle.py`
    - `participation_allocator` 既定引数を
      `--max-units-boost 0.18 --max-probability-boost 0.08`
      に合わせた。
  - テスト:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/scripts/test_participation_allocator.py`
      -> `14 passed`
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/execution/test_strategy_entry_adaptive_layers.py`
      -> `14 passed`
    - `python3 -m py_compile scripts/participation_allocator.py execution/strategy_entry.py scripts/run_local_feedback_cycle.py`
  - artifact 再生成:
    - `python3 scripts/entry_path_aggregator.py --lookback-hours 24 --limit 12000 --top-k 8`
    - `python3 scripts/participation_allocator.py --entry-path-summary logs/entry_path_summary_latest.json --trades-db logs/trades.db --output config/participation_alloc.json --lookback-hours 24 --min-attempts 20 --setup-min-attempts 4 --max-units-cut 0.18 --max-units-boost 0.18 --max-probability-boost 0.08`
    - 再生成後:
      - `MomentumBurst-open_long -> lot_multiplier=1.0731, probability_boost=0.0224`
      - `PrecisionLowVol -> lot_multiplier=0.856, probability_offset=-0.056`
      - `PrecisionLowVol|short|range_fade|...|gap:down_flat -> lot_multiplier=0.856, probability_offset=-0.056`
      - `RangeFader-neutral-fade -> base hold` だが
        `RangeFader|long|neutral-fade|range_fade|p0 -> lot_multiplier=0.856, probability_offset=-0.056`
      - `DroughtRevert -> lot_multiplier=0.856, probability_offset=-0.056`
- Verdict: pending
- Next Action:
  - `main` へ push 後に local-v2 restart を実施し、
    next 30-60 分で
    `MomentumBurst-open_long`,
    `PrecisionLowVol short gap:down_flat`,
    `RangeFader long neutral-fade p0`,
    `DroughtRevert`
    の `fills / realized_jpy / avg units / entry_probability_after`
    を再確認する。

## 2026-03-11 DroughtRevert flat-gap long guard
- Why/Hypothesis:
  - 直近 closed trades では `DroughtRevert|long|range_fade|...|gap:up_flat/down_flat`
    が `-10.57 JPY`, `-10.28 JPY` と current drag で、
    いずれも `rsi<=45`, `abs(ma_gap_pips)<=0.59`, `price_gap_pips>=3.17`,
    `vwap_gap>=37.7` の「flat-gap なのに mean stretch が深い」lane だった。
  - `gap:up_lean` / `gap:down_strong` の勝ち lane まで殺さずに、
    flat-gap oversold long だけを strategy-local で落とす。
- Expected Good:
  - `DroughtRevert` の current long loser fill を減らし、
    30 分窓の giveback を抑える。
- Expected Bad:
  - reclaim が非常に強い flat-gap long を取り逃がす可能性がある。
  - そのため `rev_strength>=0.92`, `touch_ratio>=1.60`, `setup_quality>=0.62`
    の exceptional reclaim は通す。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/worker.py` の `_signal_drought_revert`
    に `oversold flat-gap + deep mean/VWAP stretch` guard を追加した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に current loser cluster 相当の regression test を追加した。
- Verdict: pending
- Next Action:
  - push/restart 後の next 30-60 分で
    `DroughtRevert|long|range_fade|...|gap:up_flat/down_flat`
    の `fills / realized_jpy / entry_probability_after` を再確認する。

## 2026-03-12 scalp_extrema_reversal short shallow-probe guard
- Why/Hypothesis:
  - 2026-03-12 02:01 JST 時点の local-v2 実測では、
    `30m=+0.300 JPY`, `120m=-1.151 JPY` で execution は正常だった一方、
    drag は `scalp_extrema_reversal_live` の short lane に集中していた。
  - worst fingerprints は
    `short|range_fade|...|volatility_compression = -3.706 JPY`
    と `short|range_compression|...|volatility_compression = -3.400 JPY`
    で、filled order の `entry_thesis` は
    `range_mode=RANGE`, `ma_gap_pips>0`, `dist_high<=0.4`,
    `short_bounce_pips<=0.4`, `tick_strength<=0.4`
    の shallow short probe だった。
  - shared `participation / dynamic_alloc / strategy_feedback` は既に trim 済みだったため、
    これ以上 shared を締めるより worker local で low-edge short を reject する方が妥当。
- Expected Good:
  - bullish continuation 気味の shallow short を減らし、
    `scalp_extrema_reversal_live` の current giveback を止める。
  - bearish `M5` support が揃う short と long lane は残す。
- Expected Bad:
  - shallow short の rare winner を一部取り逃がす。
  - ただし current lane は short 合計 `62 trades / -35.604 JPY` と負債が大きく、
    先に drag を切る方が期待値が高い。
- Observed/Fact:
  - `workers/scalp_extrema_reversal/worker.py` に
    `short_support_context`, `short_countertrend_block`,
    `short_shallow_probe_block` を追加した。
  - `ops/env/quant-scalp-extrema-reversal.env` に current 運用値
    `SHORT_COUNTERTREND_GAP_BLOCK_PIPS=0.45`,
    `SHORT_SHALLOW_PROBE_*`
    を明記した。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    は `15 passed`。
- Verdict: pending
- Next Action:
  - push/restart 後の next 30-60 分で
    `scalp_extrema_reversal_live` の
    `short|range_fade|...|volatility_compression`,
    `short|range_compression|...|volatility_compression`
    の `fills / realized_jpy / close_reason`
    を再確認する。

## 2026-03-12 PrecisionLowVol / DroughtRevert RR-floor relax and wider SL
- Why/Hypothesis:
  - ユーザー指摘どおり、current 問題は「entry が全部悪い」ではなく、
    hard SL が current ボラに対して浅すぎる lane が混ざっていた。
  - local tick 照合では `PrecisionLowVol` の current stop loss `7件` 中
    `2件` が `120s` 以内に `TP` 側へ到達し、
    `avg_sl=1.60`, `avg_tp=2.00`, `avg_mae_120=4.24`, `avg_mfe_120=1.21`
    だった。
  - `DroughtRevert` も current stop loss `5件` 中 `2件` が
    `300s` 以内に `TP` 側へ到達し、
    `avg_sl=1.42`, `avg_tp=1.84`, `avg_mae_120=4.20`
    だった。
  - さらに `PrecisionLowVol` 実約定では
    thesis `1.6 / 2.0` が actual `1.4 / 2.2` に寄る例があり、
    scalp pocket の global `ORDER_MIN_RR=1.50` が
    strategy-local 想定より `SL` を浅くしていた。
- Expected Good:
  - `PrecisionLowVol` / `DroughtRevert` の「数秒-十数秒で broker SL → その後戻る」
    lane を減らす。
  - `SL` 拡大で `units` は自動縮小し、1 trade あたりの JPY リスクを暴れさせずに
    生存時間だけを伸ばす。
- Expected Bad:
  - loser lane がそのまま走り続けた場合、1 trade の保持時間は伸びる。
  - current drag を止めるには十分でも、即座に gross profit cadence が
    大きく跳ねるとは限らない。
- Observed/Fact:
  - `execution/order_manager.py`
    に strategy-scoped `ORDER_MIN_RR_STRATEGY_*` override を追加した。
  - `ops/env/quant-order-manager.env`
    で `ORDER_MIN_RR_STRATEGY_PRECISIONLOWVOL=1.10`,
    `ORDER_MIN_RR_STRATEGY_DROUGHTREVERT=1.10`
    を設定した。
  - `workers/scalp_wick_reversal_blend/worker.py`
    で `PrecisionLowVol` / `DroughtRevert` の `sl_pips` を広げ、
    `tp_pips` は current ボラ帯で届く範囲に留めた。
  - `ops/env/quant-scalp-precision-lowvol.env` と
    `ops/env/quant-scalp-drought-revert.env`
    では shared `SCALP_PRECISION_PERF_GUARD_MODE=reduce` を明示し、
    current log 上の `perf guard blocked` で entry 自体が止まる状態を避けた。
  - `pytest -q tests/execution/test_order_manager_preflight.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    は `44 passed`。
- Verdict: pending
- Next Action:
  - push/restart 後の next 30-60 分で
    `PrecisionLowVol` / `DroughtRevert` の
    `STOP_LOSS_ORDER<=30s 件数`,
    `post_close_tp_touch`,
    `avg_hold_sec`,
    `realized_jpy`
    を再確認する。

## 2026-03-12 scalp_extrema_reversal short setup-pressure guard
- Why/Hypothesis:
  - 最新 180 分の local-v2 実測では、`scalp_extrema_reversal_live` の最速 drag は
    short `volatility_compression` に集中していた。
  - current fast SL sample は
    `dist_high<=0.893`, `short_bounce<=0.5`, `tick_strength<=0.5`
    の shallow short が連続し、`2.88-18.46s` で broker SL に刺さっていた。
  - 一方で RR 自体は current actual `1:1.3-1.4` 帯で極端には崩れておらず、
    問題は current lane の repeated entry quality と判断した。
- Expected Good:
  - current loser になっている short `volatility_compression` shallow lane を、
    連続負け中だけ strategy-local に抑える。
  - 強い reversal short や long side は残し、short 全停止にはしない。
- Expected Bad:
  - recent lane pressure が強い間、rare winner の shallow short も一部取り逃がす。
  - ただし setup-pressure は recent outcome 連動なので、固定 stop より復帰しやすい。
- Observed/Fact:
  - `workers/scalp_extrema_reversal/worker.py`
    に recent setup-pressure 判定を追加し、
    short `volatility_compression` の
    `sl_rate / fast_sl_rate / net_jpy`
    が悪化している間だけ shallow short を reject するようにした。
  - `ops/env/quant-scalp-extrema-reversal.env`
    に `SETUP_PRESSURE_*` と
    `SHORT_SETUP_PRESSURE_*`
    の current live 値を明記した。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    は `18 passed`。
- Verdict: pending
- Next Action:
  - push/restart 後の next 30-60 分で
    `scalp_extrema_reversal_live`
    の short `volatility_compression`
    について `fills / STOP_LOSS_ORDER<=30s / realized_jpy`
    を再確認する。

## 2026-03-12 ping_d on-fill protection realign
- Why/Hypothesis:
  - `scalp_ping_5s_d_live` の current drag には、
    TP disabled だった旧 fill に加えて、
    current TP-enabled fill でも actual RR が崩れる問題が残っていた。
  - local `orders.db` では latest fill が
    thesis `sl/tp = 1.0 / 1.4`
    なのに actual は `3.0 / 1.4` となっており、
    protection retry で使った basis と actual executed price のズレで
    intended RR を失っていた。
- Expected Good:
  - `ping_d` を含む small-target scalp が
    retry / fill drift で設計 RR を失わず、
    fill 後の実保護が thesis gap に近づく。
  - current `scalp_fast` lane の small winner capture を残したまま、
    broker protection 側の歪みだけを潰せる。
- Expected Bad:
  - fill ごとに protection update が追加で 1 回走る可能性がある。
  - ただし差分が無い fill では no-op とし、
    drift があるケースだけを realign する。
- Observed/Fact:
  - `execution/order_manager.py`
    に fill 後の executed price 基準で
    `SL/TP` を引き直す `_realign_protections_to_fill()`
    を追加した。
  - market order fill 後の `on_fill_protection`
    は submit 時の `sl_price / tp_price` をそのまま再設定せず、
    thesis gap と actual fill price から再計算した protection を使う。
  - `tests/execution/test_order_manager_preflight.py`
    は `38 passed`。
- Verdict: pending
- Next Action:
  - push/restart 後の next 30-60 分で
    `scalp_ping_5s_d_live`
    の `actual avg_sl / avg_tp / avg_rr / missing_tp`
    を再確認する。

## 2026-03-12 scalp_precision soft perf-guard disable for entry recovery
- Why/Hypothesis:
  - current local-v2 では `orders.db` / `trades.db` の最新時刻が
    `2026-03-12 02:55 JST` 付近で止まる一方、
    `metrics.db` と market data は更新を続けていた。
  - `quant-scalp-precision-lowvol.log`,
    `quant-scalp-drought-revert.log`,
    `quant-scalp-wick-reversal-blend.log`
    には current signal / loop が出ているのに、
    `perf guard blocked` が 03:38-03:41 JST まで継続していた。
  - `PrecisionLowVol` / `DroughtRevert` / `WickReversalBlend` は
    すでに worker-local の `setup_pressure`, `flow_guard`, `RR` 修正を入れており、
    その上で shared prefix `SCALP_PRECISION` の soft perf guard が
    entry を全落ちさせる状態は過剰と判断した。
- Expected Good:
  - soft block で止まっていた scalp precision 系の entry を live へ戻す。
  - guard は消さず、strategy-local の quality / RR / setup-pressure を主系に戻す。
  - `VwapRevertS` の hard loser lane は reopen しないため、
    worst lane の再悪化は避ける。
- Expected Bad:
  - `PrecisionLowVol` / `DroughtRevert` / `WickReversalBlend` が
    想定より早く再開し、直後に数件の small loser を出す可能性がある。
  - ただし `VwapRevertS` hard-failfast は残し、
    shared blanket reopen にはしない。
- Observed/Fact:
  - dedicated env
    `ops/env/quant-scalp-precision-lowvol.env`,
    `ops/env/quant-scalp-drought-revert.env`,
    `ops/env/quant-scalp-wick-reversal-blend.env`
    に perf guard bypass を追加した。
  - `PrecisionLowVol` / `DroughtRevert` は wrapper が
    `SCALP_PRECISION_*` を source prefix から再投影するため、
    bypass key は
    `SCALP_PRECISION_LOWVOL_PERF_GUARD_ENABLED=0` /
    `SCALP_PRECISION_DROUGHT_REVERT_PERF_GUARD_ENABLED=0`
    を正とする。
  - worker でも source-prefix key を直接読み、
    wrapper 投影の成否に依存せず `perf_guard.is_allowed()` を bypass する。
  - `execution/order_manager.py` は worker key を直接流用せず、
    `ORDER_MANAGER_PERF_GUARD_BYPASS_STRATEGY_*`
    で `PrecisionLowVol` / `DroughtRevert` / `WickReversalBlend`
    の strategy perf guard だけを skip する。
  - `WickReversalBlend` は wrapper を噛まないため、
    `SCALP_PRECISION_PERF_GUARD_ENABLED=0` をそのまま使う。
  - `SCALP_PRECISION_PERF_GUARD_MODE=reduce` は残すが、
    current live では worker-local guard を優先して
    soft perf block を明示的に無効化する。
  - `quant-scalp-vwap-revert.log` は
    `hard:failfast:pf=0.24 win=0.57 n=14`
    なので reopen 対象から除外した。
- Verdict: pending
- Next Action:
  - push/restart 後の next 15-30 分で
    `PrecisionLowVol` / `DroughtRevert` / `WickReversalBlend`
    の `new submit_attempt / filled / realized_jpy`
    と `perf guard blocked` 消失を確認する。

## 2026-03-12 `strategy_entry` live setup context regression audit
- Why/Hypothesis:
  - `workers/scalp_wick_reversal_blend`
    で coarse headwind label を
    `flow_headwind_regime`
    へ分離しても、
    `execution/strategy_entry.py`
    の dispatch / reject path test が
    `live_setup_context` stage を見ていないと
    richer setup context の欠落を再発防止できない。
  - local-v2 の active worker を見る限り、
    `RangeFader` と `M1Scalper` は
    worker 側の `flow_regime` 自体が
    `trend_* / range_fade / transition`
    などの richer label で、
    `ExtremaReversal` は同 label を持っていない。
    current 実害は `WickReversalBlend` 系だった。
- Expected Good:
  - `market_order` / `limit_order`
    の両入口で
    `live_setup_context`
    が `entry_path_attribution`
    に残ることを CI で固定できる。
  - coarse headwind label の再導入や
    stage 欠落が起きたときに
    reject path を含めて即座に検知できる。
- Expected Bad:
  - runtime 挙動は変わらないため、
    直近収益の改善はこの change 単体では起こらない。
  - 市況適応そのものの改善は
    worker local の signal / exit 側で
    継続して詰める必要がある。
- Observed/Fact:
  - `tests/execution/test_strategy_entry_forecast_fusion.py`
    の stale だった stage 期待値を
    `live_setup_context`
    付きへ更新した。
  - `flow_headwind_regime`
    を保持したまま
    `flow_regime / microstructure_bucket / setup_fingerprint`
    が richer live setup context で残ることを
    `market_order` / `limit_order`
    の両方で regression test 化した。
  - active worker audit では
    `strategies/scalping/range_fader.py`
    は `trend_* / range_fade / transition`
    を signal に載せており、
    `strategies/scalping/m1_scalper.py`
    も worker local で richer setup payload を生成する。
    `workers/scalp_extrema_reversal/worker.py`
    には同系統 label の overwrite は無かった。
  - `PYTHONPATH=. pytest -q tests/execution/test_strategy_entry_forecast_fusion.py`
    は `20 passed`。
- Verdict: good
- Next Action:
  - current active worker で coarse label を
    `flow_regime`
    に再投入する change を入れる場合は、
    `flow_headwind_regime`
    のような別 key に分離する。

## 2026-03-12 shared setup identity repair for pre-fix coarse flow labels
- Why/Hypothesis:
  - `WickReversalBlend` 系の coarse
    `flow_regime=range_fade`
    が fix 前 trade に残っている間、
    `dynamic_alloc` / `participation_alloc` /
    `strategy_feedback`
    が recent trade を current setup ごとに
    学習するとき、
    `setup_fingerprint`
    の richer phase
    (`range_compression`)
    と top-level `flow_regime`
    が食い違って miscluster する。
  - current live では worker 側を
    `flow_headwind_regime`
    へ分離済みでも、
    直近 6-24h の pre-fix trade が
    shared artifact を汚すなら
    改善の反映が遅れる。
- Expected Good:
  - common 形式の
    `setup_fingerprint`
    を持つ trade は、
    coarse top-level label が残っていても
    shared setup identity を
    `flow_regime / microstructure_bucket / setup_fingerprint`
    の整合した組で復元できる。
  - `RangeFader`
    のような custom fingerprint は壊さず、
    common fingerprint を使う strategy だけ
    repair できる。
- Expected Bad:
  - custom fingerprint の parse 条件が広すぎると
    別 strategy の独自 fingerprint を
    誤って common 形式として扱うリスクがある。
  - runtime の entry/exit ではなく
    shared feedback の分類修正なので、
    即時の P/L 改善は限定的。
- Observed/Fact:
  - `workers/common/setup_context.py`
    に common fingerprint parser を追加し、
    `derive_live_setup_context` /
    `extract_setup_identity`
    が parse 可能な
    `setup_fingerprint`
    を setup identity の正として扱うようにした。
  - pre-fix の recent trade で
    `flow_regime=range_fade`
    かつ
    `setup_fingerprint` が
    `...|range_compression|unknown|...`
    だった
    `DroughtRevert` /
    `PrecisionLowVol`
    3 件を確認し、
    修正後の
    `extract_setup_identity()`
    は
    `flow_regime=range_compression`
    を復元した。
  - custom fingerprint
    `RangeFader|short|sell-fade|trend_long|p0`
    は explicit
    `flow_regime / microstructure_bucket`
    をそのまま保持する unit test を追加した。
  - `PYTHONPATH=. pytest -q tests/workers/common/test_setup_context.py tests/workers/common/test_dynamic_alloc.py tests/workers/common/test_participation_alloc.py tests/execution/test_strategy_entry_forecast_fusion.py`
    は `34 passed`。
- Verdict: good
- Next Action:
  - next feedback cycle で
    `dynamic_alloc` /
    `participation_alloc`
    artifact の
    `PrecisionLowVol` /
    `DroughtRevert`
    setup override が
    `range_compression`
    側へ寄るかを確認する。

## 2026-03-12 PrecisionLowVol marginal short continuation-headwind guard
- Why/Hypothesis:
  - 2026-03-12 10:48 JST 時点の local-v2 実測では、
    USD/JPY は `158.824`、spread `0.8 pips`、M1平均レンジ `2.5 pips` で
    市況・execution は通常帯だった。
  - 直近24hの `trades.db` では
    `PrecisionLowVol` short `volatility_compression` が
    `24 trades / -81.7 JPY` と最大 drag だった。
  - 既存 weak overbought short guard
    (`rsi>=60`, `projection.score<=0`, `setup_quality<0.46`)
    の少し外側を集計すると、
    `continuation_pressure>=0.33`, `rsi>=59`,
    `projection.score<=0.08`, `setup_quality<0.44`
    が `8 trades / 0 wins / -107.6 JPY` に集中していた。
  - env 閾値を単純に広げるだけだと
    `continuation_pressure=0` の戻り勝ち short まで巻き込むため、
    headwind 条件つきの marginal guard を別建てした方が安全と判断した。
- Expected Good:
  - continuation headwind を背負った marginal reclaim short を前倒しで落とし、
    `PrecisionLowVol` の current drag を減らす。
  - headwind が薄い short と stronger projection short は残し、
    strategy 自体の参加は維持する。
- Expected Bad:
  - `continuation_pressure` が一時的に高い局面で、
    rare winner の short を少数取り逃がす可能性がある。
  - ただし current cluster は `0 wins` で集中しており、
    blanket stop より worker local の lane 切りの方が副作用は小さい。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/config.py`
    に `PREC_LOWVOL_MARGINAL_SHORT_*` を追加した。
  - `workers/scalp_wick_reversal_blend/worker.py`
    は `range_reason=volatility_compression` short で
    `continuation_pressure>=0.33`, `rsi>=59`,
    `projection.score<=0.08`, `setup_quality<0.44`
    の marginal short を reject するようにした。
  - `ops/env/quant-scalp-precision-lowvol.env`
    に current live 値
    `RSI_MIN=59.0`, `PROJECTION_SCORE_MAX=0.08`,
    `SETUP_QUALITY_MAX=0.44`,
    `CONTINUATION_PRESSURE_MIN=0.33`
    を明記した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    へ headwind あり/なしの regression を追加した。
  - `PYTHONPATH=. pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    は `19 passed`。
- Verdict: pending
- Next Action:
  - `git commit -> git push -> scripts/local_v2_stack.sh restart ...`
    で反映し、
    next 30-60 分の `PrecisionLowVol short fills / STOP_LOSS_ORDER / realized_jpy`
    を再確認する。

## 2026-03-12 scalp_extrema_reversal long drift-probe guard
- Why/Hypothesis:
  - 2026-03-12 11:10 JST 前後の local-v2 実測では、
    `PrecisionLowVol` への fix 後も
    `scalp_extrema_reversal_live` buy `volatility_compression|range_fade`
    が current active loser だった。
  - 直近24hの buy side では
    `range_fade` だけで `5 trades / -25.7 JPY`。
    current worst 2 trades は
    `supportive_long=0`, `long_bounce<=0.3`, `dist_low<=0.3`,
    `tick_strength<=0.2`, `ADX>=29`, `ma_gap_pips>=0.2`,
    `range_score≈0.40` に集中していた。
  - 既存 long guard は
    shallow probe (`ADX<=13`, `range_score<=0.32`) と
    mid-RSI probe (`rsi>=40`, `range_score>=0.55`) の間に隙間があり、
    current drift lane はそこで通っていた。
- Expected Good:
  - `scalp_extrema_reversal_live` の active loser だった
    浅い buy drift probe を worker local に削れる。
  - `supportive_long` と deeper probe は残し、
    long strategy 自体の参加は維持する。
- Expected Bad:
  - shallow long の rare winner を少数取り逃がす可能性がある。
  - ただし current cluster は `2 trades / 0 wins / -17.7 JPY`
    で集中しており、まずここを切る方が副作用は小さい。
- Observed/Fact:
  - `workers/scalp_extrema_reversal/worker.py`
    に `LONG_DRIFT_PROBE_*` を追加し、
    `volatility_compression` の non-supportive long で
    `dist_low<=0.35`, `bounce<=0.35`, `tick_strength<=0.25`,
    `ADX>=24`, `range_score>=0.38`, `ma_gap_pips>=0.15`, `rsi>=36`
    の drift probe を reject するようにした。
  - `ops/env/quant-scalp-extrema-reversal.env`
    に current live 値を明記した。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    へ drift probe block / keep regression を追加し、
    `PYTHONPATH=. pytest -q tests/workers/test_scalp_extrema_reversal_worker.py`
    は `23 passed`。
- Verdict: pending
- Next Action:
  - `git commit -> git push -> scripts/local_v2_stack.sh restart ...`
    で反映し、
    next 30-60 分の `scalp_extrema_reversal_live` buy
    `volatility_compression|range_fade` の
    `fills / realized_jpy / STOP_LOSS_ORDER`
    を再確認する。

## 2026-03-12 19:20 JST / local-v2: repeated improvements still fail because negative-expectancy lanes dominate participation
- Why/Hypothesis:
  - current local-v2 が稼げない主因は、インフラや OANDA 異常ではなく、
    `volatility_compression / range_fade` 系の低エッジ reversion が
    まだ実弾で多く通っており、
    winner lane より loser lane に participation が偏っているため。
  - shared trim / probability offset は効いているが、
    chronic loser を止め切るには弱く、
    repeated local fixes のたびに別の loser lane が残る構造になっている。
- Expected Good:
  - 次の改善優先度を
    `infra` ではなく
    `loser lane の遮断 / RR 再設計 / participation 再配分`
    へ固定できる。
  - 時間帯封鎖ではなく、
    setup fingerprint 単位の worker-local 改善へ戻せる。
- Expected Bad:
  - report-only のため、
    この記録自体では収益は改善しない。
  - loser lane を deeper に切ると、
    rare winner を少数取り逃がす副作用は残る。
- Observed/Fact:
  - 市況 / API / health:
    - `OANDA summary/pricing/openTrades = 200 OK`
    - latency `235-301ms`
    - `USD/JPY 158.775 / 158.783`, spread `0.8 pips`
    - `open_trades=0`
    - `data_lag_ms=141.4`, `decision_latency_ms=20.2`
    - factor cache: `M1 ATR14=2.18 pips`, range `5m=4.7p / 15m=8.5p / 60m=22.6p`
  - 収益:
    - 24h: `136 trades / win_rate 28.7% / PF 0.38 / net -78.5 JPY / -105.7 pips`
    - 7d: `4444 trades / win_rate 46.8% / PF 0.63 / net -13687.9 JPY / -2516.2 pips`
  - 7d の赤字寄与上位:
    - `scalp_ping_5s_flow_live`: `420 trades / -7131.1 JPY`
    - `M1Scalper-M1`: `2290 trades / -6172.5 JPY`
    - `RangeFader`: `295 trades / -206.9 JPY`
    - `PrecisionLowVol`: `50 trades / -141.2 JPY`
    - `scalp_extrema_reversal_live`: `160 trades / -125.9 JPY`
  - payoff / expectancy の歪み:
    - `M1Scalper-M1`: win rate `58.5%` だが
      avg win `1.667p` / avg loss `4.142p` / RR `0.40`
      で payoff が壊れている。
    - `scalp_ping_5s_flow_live`: win rate `31.4%`,
      avg win `1.398p` / avg loss `2.010p` / RR `0.70`
      で signal quality も payoff も足りない。
    - `PrecisionLowVol`: win rate `40.0%`,
      avg win `1.575p` / avg loss `1.627p`
      でコスト込み期待値が負。
    - `scalp_extrema_reversal_live`: win rate `23.1%`,
      avg win `1.524p` / avg loss `1.715p`
      で participation 維持に値しない。
  - 24h の active loser:
    - `PrecisionLowVol`: `40 trades / -174.8 JPY`
      loser cluster は
      `short range_fade/range_compression volatility_compression`
      と
      `long trend_short volatility_compression`
      に集中。
      sample では negative / weak projection でも
      `entry_probability 0.64-0.76`
      で通っていた。
    - `scalp_extrema_reversal_live`: `75 trades / -60.2 JPY`
      loser cluster は
      `long range_fade volatility_compression`,
      `short range_compression volatility_compression`,
      `short range_fade volatility_compression`
      に集中。
      sample では
      `supportive_long/short = false`,
      `range_score ≈ 0.44`
      の shallow probe が残っていた。
  - close reason:
    - `PrecisionLowVol`: `STOP_LOSS_ORDER 24 trades / -312.7 JPY`
    - `scalp_extrema_reversal_live`: `STOP_LOSS_ORDER 48 trades / -81.1 JPY`
    - `RangeFader`: `MARKET_ORDER_TRADE_CLOSE 67 trades / -33.7 JPY`
  - order funnel (24h):
    - unique candidate `2377`
    - `preflight_start 1470`
    - `perf_block 1150`
    - `entry_probability_reject 907`
    - `filled 250`
    filled 比率は低いのに、filled subset 自体がまだ負。
  - shared trim の弱さ:
    - current `participation_alloc` では
      `PrecisionLowVol` と `scalp_extrema_reversal_live` が loser 判定でも
      `units_multiplier=0.824`, `probability_offset=-0.07`
      程度で、
      current drag に対して cut が浅い。
  - winner under-allocation:
    - `MomentumBurst` は 24h `2 trades / +185.3 JPY`、
      7d `60 trades / +1306.2 JPY`。
      それでも
      `scalp_ping_5s_flow_live + M1Scalper-M1`
      の `2710 trades`
      に比べて participation が薄すぎる。
- Verdict: bad
- Next Action:
  - `M1Scalper-M1` は entry 増ではなく、
    `avg loss >> avg win` の payoff asymmetry を先に修正する。
  - `scalp_ping_5s_flow_live` は
    signal quality と RR が同時に負けているため、
    active participation 対象から外すか deeper trim が必要。
  - current loser の `PrecisionLowVol` と `scalp_extrema_reversal_live` は、
    mild trim ではなく
    `volatility_compression` reclaim / shallow probe の
    worker-local hard reject を追加で深くする。
  - winner lane (`MomentumBurst` など) の participation を
    loser trim と同じ速度で上げられるかを再設計する。

## 2026-03-12 19:38 JST / local-v2: feedback loops should only deepen proven edges and cut fresh losers earlier
- Why/Hypothesis:
  - current loss loop を悪化させていた主因は、
    `participation_alloc` が
    「薄い利益でも winner boost を出す」,
    `dynamic_alloc` が
    「fresh setup なら strategy trim を素通しする」,
    `strategy_feedback` が
    「前回より改善していなくても正の multiplier を残す」
    という 3 つの緩さを同時に持っていたこと。
  - このままだと feedback cycle を回すたびに、
    unproven lane へ size/probability が戻り、
    chronic loser は浅い trim のまま残る。
- Expected Good:
  - low-sample winner でも `profit_per_fill` が薄い lane は
    boost されず、
    「回すたびに期待値を下げる」復元ループを止められる。
  - explicit setup identity があっても、
    setup override 未学習の lane は
    strategy-level trim を維持できる。
  - positive feedback は
    `PF / avg_pips / loss_asymmetry`
    の改善が確認できた setup だけへ残る。
- Expected Bad:
  - 立ち上がり直後の winner lane は、
    以前より participation 回復が遅くなる。
  - deeper negative probability cut により、
    recovering lane が一時的に baseline 近辺へ留まる。
- Observed/Fact:
  - `scripts/participation_allocator.py`
    で `profit_per_fill / loss_per_fill` を導入し、
    loser trim の深さを per-fill 損失でも決めるようにした。
    winner boost は `profit_per_fill` 下限を満たす lane に限定し、
    負の確率 trim は `max_probability_cut` を別 cap で扱うようにした。
  - 同ファイルの最終 payload clamp は
    旧 `max_probability_boost` で負の trim を潰していたため、
    `max(max_probability_boost, max_probability_cut)` を使うよう修正した。
  - `workers/common/participation_alloc.py`
    と `execution/strategy_entry.py`
    で `max_probability_cut` を runtime へ通し、
    deep loser trim を live pre-order に反映できるようにした。
  - `workers/common/dynamic_alloc.py`
    は `explicit_setup_without_override`
    で `lot_multiplier=1.0` へ戻す処理を削除し、
    `setup_trim_fallback=strategy_level_trim`
    で blanket trim を維持するようにした。
  - `scripts/dynamic_alloc_worker.py`
    は `2 trades` でも
    `negative realized / negative avg realized / bad avg_pips or PF`
    の fast-reactive loser setup に対して
    setup override を emit するようにした。
  - `analysis/strategy_feedback_worker.py`
    は previous feedback を読み、
    `profitable_now && payoff_ok && improved_vs_prev`
    を満たすときだけ正の
    `entry_probability_multiplier / entry_units_multiplier / sl/tp multiplier`
    を残すようにした。
  - `scripts/run_local_feedback_cycle.py`
    は repo root を `PYTHONPATH` の先頭へ自動注入するようにし、
    `dynamic_alloc` job が
    `ModuleNotFoundError: utils`
    で stale 化する経路を潰した。
  - targeted test:
    `pytest -q tests/scripts/test_run_local_feedback_cycle.py tests/workers/common/test_dynamic_alloc.py tests/execution/test_strategy_entry_adaptive_layers.py tests/workers/common/test_participation_alloc.py tests/test_dynamic_alloc_worker.py tests/analysis/test_strategy_feedback_worker.py tests/scripts/test_participation_allocator.py`
    は `82 passed`。
  - `python3 scripts/run_local_feedback_cycle.py --force --job dynamic_alloc --job participation_allocator`
    は `dynamic_alloc=ok`, `participation_allocator=ok`
    で両 artifact 更新を確認した。
- Verdict: pending
- Next Action:
  - 自動 feedback cycle と local-v2 runtime へ反映し、
    next 30-60 分で
    `trim_units` lane の `probability_offset / realized_jpy_per_fill`
    と
    winner lane の `boost_participation`
    が thin-profit で再発しないかを確認する。
  - `M1Scalper-M1` と `scalp_ping_5s_flow_live`
    のような chronic loser で、
    shared trim だけで足りるか、
    まだ worker-local payoff 修正が必要かを再判定する。

## 2026-03-12 20:18 JST / local-v2: `MomentumBurst-open_long` の directional profile が micro runtime に届かず、winner/loser cadence が live signal に反映されていなかった
- Why/Hypothesis:
  - 「エントリーが少ない」の正体は
    global trade count 不足ではなく、
    winner lane の participation/cadence が
    実際の `signal.tag` に十分つながっていないこと。
  - fresh `config/dynamic_alloc.json` では
    `MomentumBurst-open_long` が direction split key として出ている一方、
    `workers/micro_runtime/worker.py`
    の `_strategy_profile_lookup_keys()`
    は
    `MicroTrendRetest-long/-short`
    と
    `MicroCompressionRevert-long/-short`
    しか見ておらず、
    `MomentumBurst-open_long`
    を live load できていなかった。
  - この穴があると、
    future winner lane の `boost_participation`
    や direction-specific loser trim が
    micro runtime の cooldown へ乗らず、
    「良い lane を増やす / 悪い lane を落とす」
    自動化が incomplete のまま残る。
- Expected Good:
  - `MomentumBurst-open_long/open_short`
    のような directional split key を
    live cooldown / dynamic_alloc / participation cadence にそのまま接続できる。
  - future winner lane が
    `boost_participation + cadence_floor>1.0`
    を出したとき、
    base strategy fallback に潰されず
    cooldown 短縮まで反映される。
  - `MicroLevelReactor-bounce-lower`
    のような setup tag は
    誤って `MicroLevelReactor-bounce`
    へ broad match せず、
    non-directional setup を勝手に再配線しない。
- Expected Bad:
  - direction split の recent loser profile が出ている lane は、
    これまでより明確に cooldown 延長が効く。
    current `MomentumBurst-open_long`
    は 3d `dynamic_alloc score=0.283 / lot_multiplier=0.592 / sum_realized_jpy=-64.41`
    のため、
    「ただ数を増やす」動きにはならない。
  - 今の M1 factor は
    `adx=13.82`, `rsi=45.37`, `ma10<ma20`, `close<ema20`
    で
    `MomentumBurst` long edge 自体が成立しておらず、
    lookup 修正だけでは即時 entry 増には直結しない。
- Observed/Fact:
  - `logs/pdca_profitability_latest.md` 時点で
    24h は `134 trades / PF 0.39 / net -76.3 JPY`。
    「全体が建っていない」より
    loser lane 偏重の問題が大きい。
  - fresh `config/dynamic_alloc.json`
    には
    `MomentumBurst-open_long`
    が存在したが、
    micro runtime の live lookup は base `MomentumBurst`
    にしか落ちていなかった。
  - `_strategy_profile_lookup_keys()`
    を directional token
    `long/short/open_long/open_short`
    の generic 解決へ更新し、
    `MomentumBurst-open_long-reaccel`
    などから
    `MomentumBurst-open_long`
    を優先解決するよう修正した。
  - 同時に、
    non-directional setup tag
    `MicroLevelReactor-bounce-lower`
    は base `MicroLevelReactor`
    のまま扱うガードを追加した。
  - `pytest -q tests/workers/test_micro_multistrat_trend_flip.py`
    は `29 passed`。
    `MomentumBurst-open_long` lookup と
    non-directional tag guard の回帰テストを追加した。
- Verdict: pending
- Next Action:
  - local-v2 runtime へ反映後、
    `MomentumBurst-open_long`
    の `OPEN_SCALE` / `OPEN_REQ`
    が direction-specific cooldown と整合するか確認する。
  - その上で still no-attempt が続くなら、
    共有 layer ではなく
    `strategies/micro/momentum_burst.py`
    の transition/reaccel long 条件を、
    current profitable setup fingerprint を根拠に
    strategy-local で見直す。

## 2026-03-12 20:24 JST / local-v2: `MomentumBurst` の transition long だけ mid-RSI を拾い、winner lane の entry scarcity を埋める
- Why/Hypothesis:
  - current `logs/pdca_profitability_latest.md`
    では
    24h winner は
    `MomentumBurst +185.32 JPY / 2 trades`
    だけで、
    loser の大半は
    `PrecisionLowVol`
    と
    `scalp_extrema_reversal_live`
    に偏っていた。
  - `logs/trades.db`
    の recent `MomentumBurst` winner は
    `flow_regime=transition`,
    `gap:up_strong`,
    `tr:up_strong`
    に集中していた一方、
    過去 RCA では
    `MomentumBurst` long の near-miss に
    `long_rsi`
    が残っていた。
  - したがって
    broad gate loosening ではなく、
    strong higher-TF uptrend + low range/chop の
    transition long にだけ
    RSI floor を少し下げるのが最短。
- Expected Good:
  - `MomentumBurst-open_long`
    の transition winner lane で、
    `RSI 52-54`
    の early continuation を拾える。
  - `range_fade` / high-chop / weak DI gap の long は
    従来どおり通さず、
    loser lane の増加を避けられる。
- Expected Bad:
  - higher-TF uptrend がある transition 局面で、
    これまでより少し早い long が増える。
    impulse が十分でない窓では
    shallow continuation を掴むリスクがある。
- Observed/Fact:
  - `strategies/micro/momentum_burst.py`
    に
    `_long_rsi_min()`
    を追加し、
    non-reaccel long かつ
    `range_active=false`,
    `range_score<=0.30`,
    `micro_chop_score<=0.58`,
    `plus_di-minus_di>=6`,
    `roc5>=0.022`,
    `ema_slope_10>=0.001`,
    `trend_snapshot.direction=long`,
    `trend_snapshot gap/adx` が十分なときだけ、
    long RSI floor を
    `54 -> 52`
    へ緩和した。
  - `reaccel` long,
    range/chop headwind,
    no higher-TF support,
    weak impulse では
    従来の `RSI_LONG_MIN=54`
    を維持する。
  - `tests/strategies/test_momentum_burst.py`
    に
    strong transition long の `mid-RSI` 通過ケースと、
    range/chop では同じ `mid-RSI` を通さない回帰テストを追加した。
  - `pytest -q tests/strategies/test_momentum_burst.py tests/workers/test_micro_multistrat_trend_flip.py`
    は `59 passed`。
  - synthetic fixture でも
    `rsi=52.6`,
    `trend_snapshot long`,
    `range_score=0.24`,
    `micro_chop_score=0.54`
    の transition long が
    `OPEN_LONG`
    になることを確認した。
- Verdict: pending
- Next Action:
  - `quant-micro-momentumburst`
    反映後、
    next 30-60 分で
    `MomentumBurst-open_long`
    の `OPEN_REQ` 件数と
    `entry_probability / confidence / close_reason`
    を確認し、
    `transition` lane が実際に増えるかを見る。
  - もし still scarce なら、
    次は `RSI` ではなく
    `long_bull_run`
    側の near-miss を strategy-local に点検する。

## 2026-03-12 22:01 JST / local-v2: `MomentumBurst` の transition long は softly-contra higher-TF snapshot でも positive projection があれば mid-RSI early continuation を拾う
- Why/Hypothesis:
  - 市況確認（`logs/market_context_latest.json`, `logs/health_snapshot.json`）では
    2026-03-12 22:01 JST 時点で
    `USD/JPY 158.8845`,
    `spread 0.8p`,
    `data_lag_ms 651.9`,
    `decision_latency_ms 12.9`
    で、
    execution failure より
    strategy-local cadence が論点だった。
  - `logs/orders.db`
    の直近48h
    `MomentumBurst-open_long`
    は
    `5 fills / 5 preflight / 0 rejected`
    で、
    shared gate では詰まっていない。
  - `logs/trades.db`
    の直近48h long は
    `transition 2 trades / +91.8 JPY`
    に対して
    `reaccel 3 trades / -25.4 JPY`
    だったため、
    増やすべきは
    `transition`
    だけ。
  - recent winner の一部は
    `projection.score≈0.195-0.245`
    を持ちながら
    `trend_snapshot.direction=short`
    で、
    current `_long_rsi_min()`
    の
    `trend_snapshot.direction=long`
    必須条件だと
    softly-contra snapshot 下の
    `RSI 52-54`
    early continuation を拾えない余地が残っていた。
- Expected Good:
  - `gap:up_strong / tr:up_strong / low range-chop / positive projection`
    の
    `transition long`
    だけを少し早く通せる。
  - `reaccel`
    や
    shared gate
    を緩めずに cadence を増やせる。
- Expected Bad:
  - positive projection が false positive のとき、
    softly-contra snapshot 下の shallow long が増える可能性。
  - ただし
    `gap/DI/ROC/EMA slope`,
    `range/chop`,
    `trend_snapshot_supports`,
    `price_action_direction`
    は維持する。
- Observed/Fact:
  - `strategies/micro/momentum_burst.py`
    に
    `MOMENTUMBURST_TRANSITION_LONG_PROJECTION_SCORE_MIN`
    と
    `_projection_score()`
    を追加し、
    `_long_rsi_min()`
    で
    `trend_snapshot`
    が non-long / weak でも
    `projection.score`
    が閾値以上なら
    `RSI 54 -> 52`
    緩和を許すようにした。
  - `ops/env/quant-micro-momentumburst.env`
    では
    `MOMENTUMBURST_TRANSITION_LONG_PROJECTION_SCORE_MIN=0.18`
    を current live 値とし、
    soft-contra snapshot 下の positive projection だけを拾う。
  - `tests/strategies/test_momentum_burst.py`
    に
    default block /
    strong projection allow /
    weak projection keep-block
    を追加した。
  - `pytest -q tests/strategies/test_momentum_burst.py`
    は
    `35 passed`。
- Verdict: pending
- Next Action:
  - `quant-micro-momentumburst`
    反映後、
    next 30-60 分で
    `MomentumBurst-open_long`
    の
    `OPEN_REQ / OPEN_SCALE / fills`
    を確認し、
    `transition long`
    の件数が増えるかを見る。
  - それでも still scarce なら、
    次は forecast gate ではなく
    `long_bull_run`
    か
    `transition`
    の
    `gap/DI/roc`
    近辺の near-miss を点検する。

### 2026-03-12 `MomentumBurst` の H4 tie-break で weak H1 逆風だけ neutralize
- Why/Hypothesis:
  - current local-v2 では
    24h closed trade が
    `MomentumBurst: 2 trades / +185.3 JPY`
    と winner 側なのに、
    直近 6h の order path では
    `MomentumBurst`
    の `preflight / submit / fill`
    が 0 件で、
    cadence 不足が続いていた。
  - runtime は
    `candles_m5 / candles_h1 / candles_h4`
    を strategy factor に渡している一方で、
    `strategies/micro/momentum_burst.py`
    の `_mtf_supports()`
    は実質
    `M5/H1`
    の 2 票だけで binary block していた。
    `M5 + H4`
    が同方向でも
    `H1`
    の弱い逆風だけで
    `transition long`
    を丸ごと落とす余地があった。
- Expected Good:
  - `M5 + H4`
    major trend が揃い、
    `H1 gap<=4 pips && H1 ADX<18`
    の shallow countertrend だけを neutralize して、
    `MomentumBurst`
    の cadence を strategy-local に少し戻せる。
  - shared gate / shared sizing / cooldown はそのまま維持できる。
- Expected Bad:
  - weak `H1`
    を広く取りすぎると、
    broad loosening になって continuation loser を増やす可能性。
  - そのため
    `M5 + H4`
    の同方向 2 票を必須とし、
    `H1`
    は
    `gap/adx`
    が shallow なときだけ neutralize する。
- Observed/Fact:
  - `workers/micro_runtime/worker.py`
    に
    `mtf_context`
    を追加し、
    `m5/h1/h4`
    の
    `gap_pips / adx / direction`
    を strategy factor へ引き回すよう更新した。
  - `strategies/micro/momentum_burst.py`
    の `_mtf_supports()`
    は
    `candles_h4`
    を投票へ加え、
    `M5 + H4`
    同方向かつ
    `H1 gap<=4.0`,
    `H1 adx<18.0`
    のときだけ
    `H1`
    の逆風を neutralize する。
  - `ops/env/quant-micro-momentumburst.env`
    に
    `MOMENTUMBURST_MTF_H1_WEAK_OPPOSE_GAP_PIPS_MAX=4.0`,
    `MOMENTUMBURST_MTF_H1_WEAK_OPPOSE_ADX_MAX=18.0`
    を dedicated 値として追加した。
  - `tests/strategies/test_momentum_burst.py`
    へ
    `H4 tie-break allow`,
    `strong H1 headwind keep-block`,
    `H4 disagreement keep-block`
    を追加し、
    `pytest -q tests/strategies/test_momentum_burst.py`
    は
    `38 passed`
    だった。
  - 軽い replay check（`2026-03-12` UTC day）では
    extra signal は 0 件で、
    broad cadence loosening ではなく
    rare setup 向けの narrow change に留まることを確認した。
- Verdict: pending
- Next Action:
  - `quant-micro-momentumburst`
    反映後、
    next `30-90m`
    の
    `MomentumBurst`
    で
    `M5/H4 aligned + weak H1 oppose`
    局面の
    `OPEN_REQ / fills`
    が増えるかを見る。
  - 追加 fills が出ても
    `fast SL`
    が増えるなら、
    次は
    `_long_rsi_min()`
    の projection-backed override を
    `trend_snapshot`
    依存ではなく
    `MTF support`
    依存へ寄せる。

### 2026-03-12 `MomentumBurst` H4 tie-break 回帰修正
- Why/Hypothesis:
  - 直前の
    `H4 tie-break`
    実装は
    `H4`
    を third vote として扱っており、
    `M5/H1 long + H4 short`
    の既存 winner lane まで block する回帰になっていた。
  - 実測では
    `2026-03-11 21:51 JST`
    close の
    `MomentumBurst` winner
    (`+78.1 JPY`)
    が
    `M5 long / H1 long / H4 short`
    だったのに、
    新実装の `_mtf_supports()`
    だと
    `False`
    になっていた。
- Expected Good:
  - 既存の
    `M5/H1`
    winner path を温存したまま、
    `M5/H1`
    disagreement 時だけ
    `H4`
    を tie-break に使える。
  - cadence 改善のための narrow override が、
    legacy winner lane を壊さない。
- Expected Bad:
  - `H4`
    の影響を弱めすぎると、
    tie-break としての追加価値が薄くなる可能性。
  - そのため
    `M5/H1`
    が同方向のときは legacy pass を優先し、
    `1 vs 1`
    の split にだけ
    `H4 + weak H1`
    条件を使う。
- Observed/Fact:
  - `strategies/micro/momentum_burst.py`
    の `_mtf_supports()`
    を修正し、
    `M5/H1`
    同方向はそのまま pass、
    `M5/H1`
    split のときだけ
    `H4`
    を tie-break に使う形へ戻した。
  - regression test として
    `M5/H1 long + H4 short`
    の legacy support keep-pass を
    `tests/strategies/test_momentum_burst.py`
    に追加した。
  - historical winner 再現で
    `2026-03-11T12:51:12Z`
    相当の
    `mtf_supports`
    は
    `False -> True`
    に戻った。
  - `pytest -q tests/strategies/test_momentum_burst.py`
    は
    `39 passed`
    だった。
- Verdict: good
- Next Action:
  - この修正を反映した上で、
    `MomentumBurst`
    の post-restart
    `OPEN_REQ / fills`
    を再観測する。
  - 次の cadence 改善は、
    legacy winner lane を壊さないことを
    historical winner replay で確認してから入れる。

### 2026-03-12 `live_setup_context` に tick pace と MTF macro suffix を追加
- Why/Hypothesis:
  - 2026-03-12 JST の local 実測では
    `USD/JPY 158.945/158.953`,
    `spread 0.8 pips`,
    `M5 ATR14 6.914 pips`,
    `H1 ATR14 21.036 pips`,
    `OANDA 279-301ms`
    で市況は通常帯だった。
  - 一方で直近24hは
    `259 trades / -211.4 JPY / win_rate 28.6%`
    かつ
    `range_fade|unknown`
    が
    `178 trades / -237.9 JPY`
    と支配的で、
    shared setup identity が `spread_pips/tick_rate` 欠損と M1-only 圧縮で粗すぎた。
  - `PrecisionLowVol` / `DroughtRevert` の loser lane と
    `scalp_extrema_reversal_live`
    の current short probe を、
    `MTF disagreement`
    を残した setup fingerprint に分ければ、
    strategy-local 改善と shared trim の両方が current setup 単位で効きやすくなる。
- Expected Good:
  - `tick_window` 由来の `microstructure_bucket=unknown` が減り、
    `tight/normal/wide + thin/normal/fast`
    へ復元される。
  - H1/H4/D1 が強い trend のときだけ
    `macro:trend_*` / `align:countertrend|mixed`
    が setup fingerprint へ残り、
    countertrend fade loser を別 lane として学習できる。
  - `scalp_extrema_reversal_live`
    でも worker 側 explicit contract により
    `technical_context`
    が live thesis に常時注入される。
- Expected Bad:
  - setup fingerprint の suffix 追加により、
    既存 shared artifact と exact match しない current lane が一時的に増える可能性。
  - そのため suffix は
    `macro_flow_regime != local flow_regime`
    または
    `align in {countertrend,mixed}`
    のケースに限定した。
- Observed/Fact:
  - `market_data/tick_window.py`
    の `summarize()`
    は
    `spread_pips`
    と
    `tick_rate`
    を返すよう更新した。
  - `workers/common/setup_context.py`
    は
    `H1/H4/D1`
    から
    `h1_flow_regime / h4_flow_regime / d1_flow_regime / macro_flow_regime / mtf_alignment`
    を導出し、
    必要時だけ
    `setup_fingerprint`
    へ
    `macro:*`
    と
    `align:*`
    を付与するよう更新した。
  - `workers/scalp_extrema_reversal/worker.py`
    は
    `technical_context_tfs/fields/ticks/candle_counts`
    を explicit に持つよう更新し、
    `tick_rate`
    も request するようにした。
  - テスト:
    - `python3 -m pytest tests/test_tick_window_reload.py -q`
      -> `3 passed`
    - `python3 -m pytest tests/workers/common/test_setup_context.py -q`
      -> `4 passed`
    - `python3 -m pytest tests/workers/test_scalp_extrema_reversal_worker.py -q`
      -> `28 passed`
    - `python3 -m pytest tests/execution/test_strategy_entry_adaptive_layers.py -k "inject_live_setup_context_records_flow_regime_and_fingerprint" -q`
      -> `1 passed`
    - `python3 -m pytest tests/execution/test_strategy_entry_forecast_fusion.py -k "preserves_richer_live_setup_context" -q`
      -> `2 passed`
- Verdict: pending
- Next Action:
  - local-v2 反映後の次 `30-120m` で、
    `orders.db / trades.db`
    から
    `microstructure_bucket=unknown`
    の比率と
    `macro:trend_*`
    suffix の出現を確認する。
  - `PrecisionLowVol` / `DroughtRevert` / `scalp_extrema_reversal_live`
    の loser lane が
    `countertrend`
    として分離されたら、
    次は worker local quality guard をその lane にだけ寄せる。

### 2026-03-12 `PrecisionLowVol / DroughtRevert` に strategy-local MTF flow guard を追加
- Why/Hypothesis:
  - 前段の `live_setup_context` 改善だけでは、
    `PrecisionLowVol` / `DroughtRevert`
    本体の signal 判定が still `M1 + range_ctx`
    主体で、
    higher timeframe continuation を entry 前に十分反映できない。
  - current loser は
    bullish higher timeframe continuation 下の
    weak short fade、
    および bearish continuation 下の weak long reclaim を含むため、
    `M5/H1/H4`
    を worker local の `continuation_pressure`
    へ直接戻す必要がある。
- Expected Good:
  - weak countertrend probe は
    `macro_flow_regime / mtf_alignment / mtf_countertrend_pressure`
    で worker local に落ち、
    same strategy の stronger reclaim / aligned fade は残せる。
  - `entry_thesis`
    にも
    `macro_flow_regime / mtf_alignment / m5/h1/h4_flow_regime`
    が残り、
    shared RCA / feedback が current setup をより細かく学習できる。
  - `technical_context_ticks`
    に
    `tick_rate`
    を追加したので、
    `PrecisionLowVol / DroughtRevert`
    でも `microstructure_bucket`
    が `unknown`
    へ落ちにくくなる。
- Expected Bad:
  - MTF pressure を強く掛けすぎると、
    revert winner lane まで broad に削る可能性。
  - そのため guard は blanket block ではなく、
    既存 `flow_guard`
    の `continuation_pressure / reversion_support / max_pressure`
    に only additive で寄せ、
    `strong_reclaim_probe`
    は維持した。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/worker.py`
    に
    `M5/H1/H4`
    の trend snapshot helper を追加し、
    `DroughtRevert` / `PrecisionLowVol`
    へだけ `fac_m5/fac_h1/fac_h4`
    を dispatch するよう更新した。
  - short/long の `reversion_*_flow_guard`
    は
    `macro_flow_regime / mtf_alignment / mtf_countertrend_pressure / mtf_aligned_support`
    を織り込んで
    `continuation_pressure`
    を動的化するよう更新した。
  - `DroughtRevert` long 側の duplicated local pressure 計算は
    `_reversion_long_flow_guard()`
    へ寄せ、
    `PrecisionLowVol`
    と同じ MTF-aware path を使うようにした。
  - `entry_thesis`
    は nested `flow_guard`
    から
    `macro_flow_regime / mtf_alignment / mtf_countertrend_pressure / m5/h1/h4_flow_regime`
    を top-level へ昇格し、
    `technical_context_ticks`
    には
    `tick_rate`
    を追加した。
  - テスト:
    - `python3 -m pytest tests/workers/test_scalp_wick_reversal_blend_dispatch.py -q`
      -> `17 passed`
    - `python3 -m pytest tests/workers/test_scalp_wick_reversal_blend_signal_flow.py -q`
      -> `24 passed`
    - `python3 -m pytest tests/workers/test_scalp_extrema_reversal_worker.py -q`
      -> `30 passed`
    - `python3 -m pytest tests/workers/common/test_setup_context.py -q`
      -> `4 passed`
    - `python3 -m compileall workers/scalp_wick_reversal_blend/worker.py workers/scalp_wick_reversal_blend/config.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> 成功
- Verdict: pending
- Next Action:
  - local-v2 反映後の次 `30-120m`
    で
    `PrecisionLowVol / DroughtRevert`
    の
    `mtf_alignment=countertrend`
    lane の
    `OPEN_REQ -> fills`
    減少と、
    stronger reclaim lane の残存を確認する。
  - それでも loser が残るなら、
    次は `RangeFader`
    側の
    `flow_headwind`
    と同じ粒度で
    `pattern_tag / projection.score / mtf_alignment`
    を組み合わせた lane split を追加する。
  - 今回の `tick_rate` request は `scalp_wick_reversal_blend` worker explicit contract で入れており、
    shared strategy contract 側の blanket 追加はまだ行っていない。

## 2026-03-13 03:48 JST / local-v2: `scalp_extrema_reversal_live` short の setup-pressure 中 positive-gap weak lane を追加で落とす
- Why/Hypothesis:
  - 市況確認では
    2026-03-13 03:47 JST 時点で
    `USD/JPY 159.394`,
    `spread 0.8p`,
    `M1 ATR 0.818p`,
    `15m range 2.7p`,
    `60m range 12.0p`,
    `OANDA pricing 234.7ms`,
    `openTrades=[]`,
    `decision_latency_ms 10-32`,
    `data_lag_ms 74-1415`
    で、
    market / execution は通常帯だった。
  - `logs/trades.db`
    の直近24hでは
    `scalp_extrema_reversal_live`
    が
    `93 trades / -86.098 JPY`
    と loser で、
    short `volatility_compression`
    だけでも
    `42 trades / -31.066 JPY`
    を削っていた。
  - そのうち
    `short_setup_pressure.active=1`
    かつ
    `ma_gap_pips>=0.15`
    の positive-gap lane は
    `3 trades / -5.390 JPY`
    で、
    `RSI 69.06`
    の winner `+0.79`
    は残る一方、
    `RSI 65.37 / 67.19`
    の weak short が
    `-1.236 / -4.944 JPY`
    と current drag になっていた。
  - 既存の
    `short_setup_pressure_block`
    は
    `short_bounce<=0.50`
    までしか見ておらず、
    `positive ma_gap + weak tick + RSI 未伸び切り`
    の中間 short lane が still 通っていた。
- Expected Good:
  - setup-pressure active 中の
    short `volatility_compression`
    で、
    `ma_gap>0`
    の weak countertrend short だけを
    strategy-local に落とせる。
  - `RSI>=69`
    まで伸びた stronger short や、
    supportive short は維持できる。
- Expected Bad:
  - `rsi<=68`
    の閾値が低すぎると、
    shallow positive-gap short の winner まで削る可能性。
  - そのため条件は
    `setup-pressure active`
    かつ
    `range_mode=RANGE`
    かつ
    `volatility_compression`
    かつ
    `ma_gap>=0.15`
    かつ
    `dist_high<=0.90`
    かつ
    `short_bounce<=0.75`
    かつ
    `tick_strength<=0.40`
    かつ
    `rsi<=68`
    の narrow lane に限定した。
- Observed/Fact:
  - `workers/scalp_extrema_reversal/worker.py`
    に
    `short_positive_gap_probe_block`
    を追加し、
    recent setup-pressure が active なときだけ
    positive-gap の weak short probe を reject するよう更新した。
  - 既存の
    `short_setup_pressure_block`
    や
    `short_shallow_probe_block`
    は維持し、
    新 block は
    `short_supportive=false`
    の narrow lane に only additive で入れた。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    に
    `positive-gap weak short block`
    と
    `RSI 69 の stronger short keep`
    を追加した。
  - 検証:
    - `python3 -m pytest tests/workers/test_scalp_extrema_reversal_worker.py -q`
      -> `32 passed`
    - `python3 -m py_compile workers/scalp_extrema_reversal/worker.py tests/workers/test_scalp_extrema_reversal_worker.py`
      -> 成功
- Verdict: pending
- Next Action:
  - local-v2 反映後の次
    `30-90m`
    で
    `scalp_extrema_reversal_live`
    short
    `volatility_compression`
    の
    `OPEN_REQ -> fills`
    と
    `STOP_LOSS_ORDER`
    を確認し、
    `short_positive_gap_probe_block`
    相当 lane の fill が消えるかを見る。
  - それでも current loser が残るなら、
    次は
    `range_score / supportive_short_context / M5 bearish support`
    を軸に
    short positive-gap lane をさらに split する。

## 2026-03-13 07:32 JST / local-v2: `PrecisionLowVol` short の mid-RSI continuation-headwind lane を追加で落とす
- Why/Hypothesis:
  - 市況確認では
    `logs/health_snapshot.json`
    の
    `2026-03-13 07:02 JST`
    時点で
    `openTrades=[]`,
    `decision_latency_ms 17.1`,
    `data_lag_ms 2291.6`
    で stack / OANDA 応答自体は生きていた。
    ただし
    JST 7-8 時の maintenance 帯で
    spread / stream は不安定だったため、
    live entry の評価ではなく
    直近 24h の closed trades cluster を根拠に RCA を進めた。
  - `logs/trades.db`
    の直近24hでは
    `PrecisionLowVol`
    が
    `42 trades / -181.678 JPY / win rate 31.0% / PF 0.468`
    と current loser だった。
  - 既存の
    `weak overbought short guard`
    と
    `marginal short continuation-headwind guard`
    の少し外側に、
    short `volatility_compression`
    の
    `rsi>=58`,
    `projection.score<=0.05`,
    `setup_quality<0.48`,
    `continuation_pressure>=0.33`
    という mid-RSI lane が残っており、
    この subset だけで
    `9 trades / 0 wins / -110.881 JPY`
    すべて
    `STOP_LOSS_ORDER`
    だった。
  - current marginal guard は
    `rsi>=59`
    と
    `setup_quality<0.44`
    を見ているため、
    `rsi 58.6-62.6`
    /
    `setup_quality 0.416-0.435`
    の loser lane が still 通っていた。
- Expected Good:
  - `PrecisionLowVol` short を blanket stop せず、
    continuation headwind を背負った
    mid-RSI の弱い reclaim short だけを
    worker local に前倒しで落とせる。
  - 既存
    `weak overbought short`
    /
    `marginal short`
    /
    `setup-pressure`
    より強い reclaim short は残せる。
- Expected Bad:
  - threshold を広げすぎると
    reclaim short の winner まで削る可能性がある。
  - そのため条件は
    `range_reason=volatility_compression`
    /
    `projection.score<=0.05`
    /
    `continuation_pressure>=0.33`
    /
    `rsi>=58`
    /
    `setup_quality<0.48`
    の narrow lane に限定し、
    env flag で切り戻せる形にした。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/config.py`
    に
    `PREC_LOWVOL_HEADWIND_SHORT_*`
    を追加し、
    new guard を dedicated config として明示した。
  - `workers/scalp_wick_reversal_blend/worker.py`
    の
    `_signal_precision_lowvol()`
    で
    short `volatility_compression`
    かつ
    `continuation_pressure>=0.33`
    /
    `rsi>=58`
    /
    `projection.score<=0.05`
    /
    `setup_quality<0.48`
    の lane を
    `headwind_short_lane`
    として reject するよう更新した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に
    `rsi=58.6`
    の current loser lane block と、
    `setup_quality` 回復時の keep を追加した。
  - 検証:
    - `python3 -m pytest tests/workers/test_scalp_wick_reversal_blend_signal_flow.py -q`
      -> `26 passed`
    - `python3 -m py_compile workers/scalp_wick_reversal_blend/config.py workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> 成功
- Verdict: pending
- Next Action:
  - 08:00 JST 以降の通常帯で
    `PrecisionLowVol` short `volatility_compression`
    の
    `OPEN_REQ -> fills`
    /
    `STOP_LOSS_ORDER`
    /
    `realized_jpy`
    を見て、
    `rsi 58-59`
    帯の loser short が消えるか確認する。
  - それでも short loser が残るなら、
    次は
    `projection.score / setup_quality / mtf_alignment`
    で
    reclaim short を finer split する。

## 2026-03-13 07:40 JST / local-v2: `scalp_extrema_reversal_live` long setup-pressure の neutral-gap / higher-range 窓を広げる
- Why/Hypothesis:
  - 同じ
    `2026-03-13 07:02 JST`
    の local-v2 snapshot では、
    maintenance 帯で spread は荒いものの
    stack / API 自体は生存していたため、
    RCA は直近 24h closed trades を基準に行った。
  - `logs/trades.db`
    の直近24hでは
    `scalp_extrema_reversal_live`
    が
    `100 trades / -103.25 JPY / win rate 25.0% / PF 0.295`
    と strong loser だった。
  - long `volatility_compression`
    かつ
    `long_setup_pressure.active=1`
    の current loser lane を見ると、
    既存
    `dist_low<=0.90`
    /
    `bounce<=0.35`
    /
    `tick_strength<=0.30`
    /
    `adx<=23`
    に加えて、
    `ma_gap_pips<=0.10`
    と
    `range_score<=0.60`
    まで広げると
    `3 trades / 0 wins / -8.04 JPY`
    を捉えられた。
  - 内訳は
    `ma_gap_pips=0.065, range_score=0.458`
    の neutral-to-positive gap loser 1 本と、
    `range_score=0.561 / 0.586`
    の slightly higher-range loser 2 本で、
    既存
    `ma_gap<=0.00`
    /
    `range_score<=0.55`
    の setup-pressure 窓の少し外側に残っていた。
- Expected Good:
  - `scalp_extrema_reversal_live` long を停止せず、
    recent outcome が悪化している間だけ
    neutral-gap / higher-range の weak probe を
    setup-pressure guard で落とせる。
  - stronger reclaim long や
    setup-pressure 非 active 時の long は維持できる。
- Expected Bad:
  - setup-pressure 窓を広げすぎると
    回復局面の long まで block する可能性がある。
  - そのため変更は
    `LONG_SETUP_PRESSURE_MA_GAP_MAX_PIPS 0.00 -> 0.10`
    と
    `LONG_SETUP_PRESSURE_RANGE_SCORE_MAX 0.55 -> 0.60`
    のみとし、
    recent setup-pressure active 時にしか効かない narrow widening に留めた。
- Observed/Fact:
  - `ops/env/quant-scalp-extrema-reversal.env`
    で
    `SCALP_EXTREMA_REVERSAL_LONG_SETUP_PRESSURE_MA_GAP_MAX_PIPS=0.10`
    と
    `SCALP_EXTREMA_REVERSAL_LONG_SETUP_PRESSURE_RANGE_SCORE_MAX=0.60`
    へ更新した。
  - `tests/workers/test_scalp_extrema_reversal_worker.py`
    に
    neutral-gap long が setup-pressure 下で block されることを確認する
    regression test
    を追加した。
  - 既存の
    `long_drift_probe`
    /
    `long_positive_gap_probe`
    /
    `supportive_long`
    の分岐はそのまま残し、
    shared gate や time block は増やしていない。
  - 検証:
    - `python3 -m pytest tests/workers/test_scalp_extrema_reversal_worker.py -q`
      -> `33 passed`
    - `python3 -m py_compile workers/scalp_extrema_reversal/worker.py tests/workers/test_scalp_extrema_reversal_worker.py`
      -> 成功
- Verdict: pending
- Next Action:
  - 08:00 JST 以降の通常帯で
    `scalp_extrema_reversal_live`
    long
    `volatility_compression`
    /
    `long_setup_pressure.active=1`
    の
    fills,
    `STOP_LOSS_ORDER`,
    `net_jpy`
    を確認する。
  - それでも loser lane が残るなら、
    次は
    `supportive_long_context / M5 support / range_score`
    を軸に
    setup-pressure 内 long probe をさらに split する。

## 2026-03-13 07:55 JST / local-v2: `participation_alloc` で fresh loser lane を fast trim する
- Why/Hypothesis:
  - 2026-03-13 07:49 JST の local-v2 実測では
    USD/JPY `159.310 / 159.318`,
    spread `0.8 pips`,
    ATR14 `M1=1.429 pips / M5=3.929 pips`,
    30-candle range `M1=5.6 pips / M5=15.6 pips`,
    OANDA `pricing=246.6ms / account_summary=319.7ms / openTrades=233.6ms / candles(M1,M5)=318-362ms`
    で市況・API は通常帯だった。
  - 一方で `logs/trades.db` の直近30分は
    `104 trades / -312.2 JPY / avg -3.0 JPY`
    と短期収益がマイナスで、
    `PrecisionLowVol=-126.9 JPY`,
    `WickReversalBlend=-69.2 JPY`,
    `scalp_extrema_reversal_live=-68.3 JPY`
    が current drag だった。
  - 既存 shared `participation_alloc` は
    `2 attempts / 2 fills / positive realized_jpy`
    の winner lane を早く `boost_participation`
    できる一方で、
    `2 fills + negative realized_jpy`
    の fresh loser lane は
    `min_attempts`
    未満だと `hold`
    に残りやすく、
    30分KPIに対する下押しを止め切れていなかった。
- Expected Good:
  - `DroughtRevert` / `WickReversalBlend`
    のような filled loser lane を
    strategy stop ではなく shared trim で早く減速できる。
  - `2 fills + positive realized`
    の winner boost ロジックは維持しつつ、
    current loser lane だけを対称に前倒しで落とせる。
- Expected Bad:
  - 2連敗直後の回復 lane まで shared trim すると
    rebound participation を削る恐れがある。
  - そのため negative `probability_offset`
    は
    `realized_jpy<=-12`
    か
    `loss_per_fill>=6`
    か
    `attempts>=3`
    のときだけ付与し、
    それ未満は units trim 優先に留める。
- Observed/Fact:
  - `scripts/participation_allocator.py`
    に small-sample loser branch を追加し、
    `attempts>=2`,
    `fills>=2`,
    `negative realized_jpy`,
    `loss_per_fill>=4`
    かつ正常 fill quality の lane を
    fast `trim_units`
    へ落とすようにした。
  - `tests/scripts/test_participation_allocator.py`
    に
    `WickReversalBlend`
    の
    `2 trades / 2 fills / -18.958 JPY`
    fast loser と、
    `PrecisionLowVol`
    の
    setup-scoped fast loser override
    の回帰を追加した。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/scripts/test_participation_allocator.py`
      -> `22 passed`
    - `python3 -m py_compile scripts/participation_allocator.py tests/scripts/test_participation_allocator.py`
      -> 成功
    - `python3 scripts/participation_allocator.py --entry-path-summary logs/entry_path_summary_latest.json --trades-db logs/trades.db --output config/participation_alloc.json --lookback-hours 6 --min-attempts 12 --setup-min-attempts 2 --max-units-cut 0.22 --max-units-boost 0.24 --max-probability-boost 0.10`
      -> `config/participation_alloc.json`
      で
      `DroughtRevert: lot_multiplier=0.8528 / probability_offset=-0.0731`,
      `WickReversalBlend: lot_multiplier=0.8302 / probability_offset=-0.0977`,
      `scalp_extrema_reversal_live: lot_multiplier=0.824 / probability_offset=-0.112`
      を確認した。
- Verdict: pending
- Next Action:
  - 08:00 JST 以降の通常帯で
    `orders.db` / `entry_thesis.participation_alloc`
    を見て、
    `DroughtRevert` /
    `WickReversalBlend`
    の current loser lane が
    `trim_units`
    のまま維持されるか確認する。
  - 直近30分の
    `net_jpy`
    が依然強くマイナスなら、
    次は
    `PrecisionLowVol`
    の single-fill loser を
    worker local guard で先に落とす。

## 2026-03-13 08:14 JST / local-v2: `PrecisionLowVol` short `gap:down_flat` の low-score loser lane を落とす
- Why/Hypothesis:
  - 2026-03-13 08:12 JST の local-v2 実測では
    USD/JPY `159.294 / 159.302`,
    spread `0.8 pips`,
    ATR14 `M1=1.629 pips / M5=4.107 pips`,
    OANDA `pricing=354ms / summary=223ms / openTrades=260ms / candles(M1,M5)=210-294ms`,
    `openTrades=0`
    で市況・API は通常帯だった。
  - 同時点の直近24h closed trades では
    `PrecisionLowVol`
    が
    `42 trades / -181.678 JPY / win rate 31.0%`
    と current loser で、
    short `volatility_compression`
    の
    `gap:down_flat`
    cluster だけで
    `11 trades / -89.30 JPY / win rate 18.2%`
    を削っていた。
  - その内側でも
    `range_score<=0.44`
    /
    `projection.score<=0.30`
    /
    `setup_quality<0.40`
    /
    `continuation_pressure>=0.24`
    /
    `rsi>=54`
    の lane は
    `8 trades / 0 wins / -88.24 JPY`
    かつ
    全て
    `STOP_LOSS_ORDER`
    で、
    ほぼこの low-score lane が
    `gap:down_flat`
    loser を作っていた。
  - 一方で
    `gap:down_flat`
    の winner 2 本は
    `range_score=0.469 / 0.610`
    と higher-score 側に寄っており、
    range score を軸に loser lane だけを薄く切り分けられる余地があった。
- Expected Good:
  - `PrecisionLowVol` short を blanket stop せず、
    `gap:down_flat`
    の low-score loser lane だけを
    worker local に前倒しで落とせる。
  - 既存の
    `gap:up_flat` shallow guard,
    mid-RSI continuation-headwind guard
    と合わせて、
    `PrecisionLowVol`
    short
    `volatility_compression`
    の低期待値 reclaim を段階的に削れる。
- Expected Bad:
  - `range_score<=0.44`
    を広く使いすぎると
    回復初動の reclaim short まで削る恐れがある。
  - そのため条件は
    `gap:down_flat`
    /
    `volatility_compression`
    /
    `continuation_pressure>=0.24`
    /
    `rsi>=54`
    /
    `projection.score<=0.30`
    /
    `setup_quality<0.40`
    に限定し、
    `range_score`
    が戻った lane は残す形にした。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/config.py`
    に
    `PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_GUARD_ENABLED`,
    `PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_RANGE_SCORE_MAX`,
    `PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_PROJECTION_SCORE_MAX`,
    `PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_SETUP_QUALITY_MAX`,
    `PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_CONTINUATION_PRESSURE_MIN`,
    `PREC_LOWVOL_DOWN_FLAT_LOW_SCORE_SHORT_RSI_MIN`
    を追加した。
  - `workers/scalp_wick_reversal_blend/worker.py`
    の
    `_signal_precision_lowvol()`
    で、
    short `volatility_compression`
    かつ
    `gap:down_flat`
    の
    `range_score<=0.44`
    /
    `continuation_pressure>=0.24`
    /
    `rsi>=54`
    /
    `projection.score<=0.30`
    /
    `setup_quality<0.40`
    を
    `down_flat_low_score_short_lane`
    として reject するよう更新した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に
    low-score loser lane block と、
    `range_score=0.61`
    の recover case で short を keep する回帰を追加した。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> `28 passed`
    - `python3 -m py_compile workers/scalp_wick_reversal_blend/config.py workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> 成功
- Verdict: pending
- Next Action:
  - 再起動後の通常帯で
    `PrecisionLowVol`
    short
    `volatility_compression`
    /
    `gap:down_flat`
    の
    `OPEN_REQ -> fills`
    /
    `STOP_LOSS_ORDER`
    /
    `realized_jpy`
    を確認する。
  - それでも short loser が残るなら、
    次は
    `gap:down_flat`
    の
    `range_fade`
    と
    `range_compression`
    を分けて、
    winner が残る score 帯だけを finer split する。

## 2026-03-13 08:27 JST / local-v2: all-strategy tick audit を基に stop band を strategy-local に広げる
- Why/Hypothesis:
  - ユーザ指摘どおり
    `SLで刈られてすぐ戻る`
    trade が本当に多いのかを、
    strategy ごとに切り分ける必要があった。
  - `tick_entry_validate`
    を
    `logs/trades.db`
    と
    `logs/replay/USD_JPY/USD_JPY_ticks_20260311.jsonl`,
    `logs/replay/USD_JPY/USD_JPY_ticks_20260312.jsonl`
    に対して実行し、
    `2026-03-11 00:00 UTC -> 2026-03-13 00:00 UTC`
    の
    `300 trades`
    を照合した。
  - 結果は一律ではなく、
    `SL後300秒以内にTP帯へ戻る`
    比率は
    `scalp_ping_5s_d_live=5/9`,
    `WickReversalBlend=3/6`
    で強い一方、
    `PrecisionLowVol=5/26`,
    `DroughtRevert=3/16`,
    `scalp_extrema_reversal_live=10/66`
    は
    `entry負け`
    の寄与も無視できなかった。
  - したがって
    `全戦略を一律に widen`
    ではなく、
    clear stop-hunt sample は stop band widening、
    mixed sample は
    entry guard + modest widening
    に分けるのが妥当と判断した。
- Expected Good:
  - `WickReversalBlend` /
    `scalp_ping_5s_d_live`
    の
    `即 SL -> 数十秒-数分で戻る`
    取りこぼしを減らせる。
  - `PrecisionLowVol` /
    `DroughtRevert` /
    `scalp_extrema_reversal_live`
    は
    bad entry を残したまま大きく widen しないため、
    負け trade の延命を避けられる。
- Expected Bad:
  - stop band を広げると
    1-trade あたりの nominal loss は増える。
  - そのため
    `PrecisionLowVol` /
    `DroughtRevert` /
    `WickReversalBlend`
    は strategy-local の modest widening に留め、
    `ping_d` /
    `extrema`
    は dedicated env で narrow に管理して切り戻し可能にした。
- Observed/Fact:
  - tick audit 集計:
    - `scalp_extrema_reversal_live`: `66 stop`, `36` が `30s以内`, `10` が `tp_after_sl<=300s`
    - `PrecisionLowVol`: `26 stop`, `17` が `30s以内`, `5` が `tp_after_sl<=300s`
    - `DroughtRevert`: `16 stop`, `9` が `30s以内`, `3` が `tp_after_sl<=300s`
    - `WickReversalBlend`: `6 stop`, `3` が `tp_after_sl<=300s`
    - `scalp_ping_5s_d_live`: `9 stop`, `8` が `30s以内`, `5` が `tp_after_sl<=300s`
  - `workers/scalp_wick_reversal_blend/worker.py`
    で
    `DroughtRevert`,
    `PrecisionLowVol`,
    `WickReversalBlend`
    の
    `sl_pips / tp_pips`
    band を広げた。
  - `ops/env/quant-scalp-extrema-reversal.env`
    で
    `SL_ATR_MULT=1.05`,
    `TP_ATR_MULT=1.35`,
    `SL_MIN/MAX=1.30/2.90`,
    `TP_MIN/MAX=1.60/3.40`
    へ更新した。
  - `ops/env/quant-scalp-ping-5s-d.env`
    で
    `TP_ENABLED=1`
    と
    D 専用の
    `SL_* / TP_*`
    を追加した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に
    widened stop band の回帰を追加した。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> `29 passed`
    - `python3 -m py_compile workers/scalp_wick_reversal_blend/config.py workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> 成功
- Verdict: pending
- Next Action:
  - 再起動後に
    `orders.db` /
    `trades.db`
    で
    `STOP_LOSS_ORDER`
    の
    `hold_sec`,
    `sl_pips`,
    `post_close_tp_touch`
    を strategy 別に再集計する。
  - 特に
    `scalp_ping_5s_d_live`,
    `WickReversalBlend`,
    `scalp_extrema_reversal_live`
    で
    `tp_after_sl<=300s`
    が減るかを確認し、
    まだ高ければ
    stop band ではなく
    entry timing / partial exit
    側へ移る。

## 2026-03-13 08:35 JST / local-v2: dedicated exit worker で post-entry の broker protection move を有効化
- Why/Hypothesis:
  - ユーザ指摘どおり
    「entry 後に
    `SL/TP`
    が市況追随できていない」
    仮説を、
    current live と tick 照合で切り分けた。
  - OANDA live
    `2026-03-13 08:34 JST`
    は
    `USD/JPY bid=159.308 ask=159.316 spread=0.8p`,
    `ATR14(M1)=1.257p`,
    `ATR14(M5)=3.221p`,
    `pricing=298ms`,
    `summary=170ms`,
    `openTrades=0`
    で通常帯だった。
  - `scripts/pdca_profitability_report.py`
    では
    24h
    `106 trades / win=33.0% / PF=0.52 / net_jpy=-307.2`
    で、
    top losers は
    `PrecisionLowVol=-117.25`,
    `WickReversalBlend=-69.224`,
    `scalp_extrema_reversal_live=-68.265`
    だった。
  - `tick_entry_validate`
    を
    `scalp_extrema_reversal_live`
    の
    `2026-03-12 22:45-23:20 JST`
    に当てると、
    `MARKET_ORDER_TRADE_CLOSE`
    2本で
    `post_close_tp_touch_s=320,121`
    が出ており、
    `TP_touch<=600s`
    も
    `5/7`
    だった。
  - 一方
    `PrecisionLowVol`
    の
    `2026-03-12 15:50-18:45 JST`
    は
    `TP_touch<=120s=0/8`,
    `<=600s=2/8`
    で、
    post-entry 管理だけでなく
    entry quality
    の問題も大きかった。
  - したがって
    shared exit manager
    を足さず、
    loser lane を持つ dedicated exit worker で
    `be_profile / tp_move`
    を broker-side protection へ反映するのが妥当と判断した。
- Expected Good:
  - `scalp_extrema_reversal_live`
    と
    `WickReversalBlend / PrecisionLowVol / DroughtRevert`
    が含み益化した後、
    broker `SL`
    を建値超えへ寄せ、
    `TP`
    も current price + buffer へ引き直せる。
  - `MARKET_ORDER_TRADE_CLOSE`
    後の
    `post_close_tp_touch`
    や、
    小さい winner の give-back を減らせる。
- Expected Bad:
  - `TP`
    を近づける分、
    一部の大きい runner は伸び切る前に刈られる。
  - `PrecisionLowVol`
    は
    current loser の主因が entry 側にもあるため、
    この変更だけでは PF 改善が限定的な可能性がある。
- Observed/Fact:
  - `workers/scalp_level_reject/exit_worker.py`
    と
    `workers/scalp_wick_reversal_blend/exit_worker.py`
    に、
    strategy-local
    `be_profile / tp_move`
    を使って
    broker `SL/TP`
    を live 更新する経路を追加した。
  - `config/strategy_exit_protections.yaml`
    に
    `scalp_extrema_reversal_live`,
    `WickReversalBlend`,
    `PrecisionLowVol`,
    `DroughtRevert`
    の
    `be_profile / tp_move`
    override を追加した。
  - 既存の
    `loss_cut / take_profit / lock_floor`
    などの exit 判断自体は変更せず、
    dedicated exit worker 内の protection move だけを増やした。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_exit_worker.py tests/workers/test_scalp_level_reject_exit_worker.py`
      -> `6 passed`
    - `python3 -m py_compile workers/scalp_level_reject/exit_worker.py workers/scalp_wick_reversal_blend/exit_worker.py tests/workers/test_scalp_wick_reversal_blend_exit_worker.py tests/workers/test_scalp_level_reject_exit_worker.py`
      -> 成功
- Verdict: pending
- Next Action:
  - 再起動後に
    `logs/local_v2_stack/quant-*-exit.log`
    と
    `quant-order-manager.log`
    で
    `protection_move`
    /
    `set_trade_protections`
    /
    `STOP_LOSS_ORDER`
    を追う。
  - 特に
    `scalp_extrema_reversal_live`
    の
    `MARKET_ORDER_TRADE_CLOSE -> post_close_tp_touch`
    が減るか、
    `PrecisionLowVol`
    の
    loser burst が残るかを分けて確認し、
    後者が残るなら
    entry guard
    側を優先して詰める。

## 2026-03-13 09:14 JST / local-v2: post-entry protection move を ATR/spread/setup-aware に更新
- Why/Hypothesis:
  - 08:35 JST の変更で
    dedicated exit worker に
    broker
    `SL/TP`
    の live move 自体は入ったが、
    trigger は strategy ごとの固定
    `be_profile / tp_move`
    依存が残っていた。
  - current live
    `2026-03-13 09:13 JST`
    は
    `USD/JPY bid=159.052 ask=159.060 spread=0.8p`,
    `ATR14(M1)=3.108p`,
    `ATR14(M5)=5.002p`,
    `fills_60m=105`,
    `rejects_60m=5`,
    `open_trades=0`
    で通常帯だった。
  - よって
    shared exit manager は増やさず、
    dedicated exit worker 内で
    `ATR / spread / setup_quality / continuation_pressure / reversion_support / extrema setup pressure`
    から
    `trigger / lock / buffer`
    を補正するのが妥当と判断した。
- Expected Good:
  - `PrecisionLowVol / WickReversalBlend`
    は
    headwind + wide-spread
    のとき
    `trigger`
    を早め、
    `lock_ratio`
    を上げ、
    `TP buffer`
    を狭めて give-back を減らせる。
  - `scalp_extrema_reversal_live`
    は
    supportive extrema setup
    のときだけ
    `trigger`
    を少し遅らせ、
    `TP buffer`
    を広げて、
    反転 winner を刈り過ぎにくくできる。
- Expected Bad:
  - stress 判定が過敏だと、
    runner を早く刈る。
  - supportive 側の補正が強すぎると、
    反転 fail で
    含み益の取りこぼしが増える。
- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/exit_worker.py`
    に
    `_wick_live_protection_adjustments`
    を追加し、
    headwind / setup-quality / spread
    に応じて
    `trigger_mult / lock_ratio_mult / buffer_mult`
    を返すようにした。
  - `workers/scalp_level_reject/exit_worker.py`
    に
    `_level_reject_live_protection_adjustments`
    を追加し、
    supportive extrema / setup-pressure / spread
    に応じて
    同種の multiplier を返すようにした。
  - `tests/workers/test_scalp_wick_reversal_blend_exit_worker.py`
    と
    `tests/workers/test_scalp_level_reject_exit_worker.py`
    に、
    `stressed -> tighten`,
    `supportive -> loosen`
    の比較テストを追加した。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_exit_worker.py tests/workers/test_scalp_level_reject_exit_worker.py`
      -> `8 passed`
    - `python3 -m py_compile workers/scalp_wick_reversal_blend/exit_worker.py workers/scalp_level_reject/exit_worker.py tests/workers/test_scalp_wick_reversal_blend_exit_worker.py tests/workers/test_scalp_level_reject_exit_worker.py`
      -> 成功
- Verdict: good
- Next Action:
  - live の
    `protection_move`
    ログに出す
    `stress / support`
    と
    `post_close_tp_touch`
    の改善有無を
    6h 窓で照合する。

## 2026-03-13 09:45 JST / local-v2: 低稼働の主因確認と fast PDCA 運用ルール化
- Why/Hypothesis:
  - user 指摘どおり、
    current live は
    「entry が少ない」
    状態で、
    low-activity 時の確認順序を運用側へ固定しておく必要があった。
- Expected Good:
  - 市況が通常帯なのに low-entry になったとき、
    15 分以内に dominant block family を切り分けられる。
  - 広域緩和ではなく、
    strategy-local
    の 1 変更ずつで PDCA を回せる。
- Expected Bad:
  - 低稼働と market pause を混同すると、
    不要な tuning を急いで入れるリスクがある。
- Observed/Fact:
  - `2026-03-13 09:42 JST`
    時点の live は
    `USD/JPY 159.161/159.169 spread=0.8p`,
    `ATR14(M1)=2.664p`,
    `ATR14(M5)=5.415p`,
    `open_trades=0`
    で通常帯だった。
  - exact time 差で取り直すと、
    recent は
    `fills_15m=0`,
    `fills_30m=1`,
    `fills_60m=2`
    だった。
    最終 fill は
    `2026-03-13 09:27 JST`
    の
    `scalp_ping_5s_c_live`
    だった。
  - recent active lane は
    `scalp_ping_5s_c_live`
    と
    `TickImbalance`
    だけで、
    `scalp_ping_5s_c_live`
    は
    `entry_probability_reject`
    が 2 件、
    worker log では
    `lookahead block`
    と
    `no_signal:revert_not_found`
    が主因だった。
  - `PrecisionLowVol`,
    `DroughtRevert`,
    `WickReversalBlend`,
    `TickImbalance`
    は worker log が
    `cluster cooldown`
    に張り付いていた。
  - `scalp_extrema_reversal_live`
    は
    `risk multiplier=0.55`
    かつ
    `strategy_cooldown:loss_streak`
    により recent entry が抑制されていた。
  - 上記を踏まえ、
    `docs/AGENT_COLLAB_HUB.md`
    と
    `docs/OPS_LOCAL_RUNBOOK.md`
    に
    low-entry fast PDCA の運用ルールを追記した。
- Verdict: good
- Next Action:
  - 次の調整は
    1. `scalp_ping_5s_c_live`
       の
       `lookahead_block / revert_not_found`
       のどちらを先に削るかを決める。
  - その後に
    2. scalp pocket の
       `cluster cooldown`
       が長すぎないかを点検する。

## 2026-03-13 10:09 JST / local-v2: `StageTracker` の stale cluster cooldown を current trades window 基準へ修正
- Why/Hypothesis:
  - user が貼った
    `2026-03-13 09:05 JST`
    の
    `TickImbalance`
    負け以降、
    `PrecisionLowVol / DroughtRevert / WickReversalBlend`
    が
    `cluster cooldown`
    に張り付いていた。
  - `logs/stage_state.db`
    を見ると
    `pocket_loss_window`
    に
    `trade_id=59370`
    の stale row と
    `59372 TickImbalance`
    が残り、
    実際は single-strategy の small loss なのに
    scalp pocket 全体へ cooldown が掛かっていた。
- Expected Good:
  - `StageTracker`
    が stale row に引っ張られず、
    current trades の close time と strategy breadth で
    cluster cooldown を再判定できる。
  - one loser lane の contained loss で
    `scalp`
    pocket 全体の entry を止めない。
- Expected Bad:
  - pocket-wide cooldown の発火が減るぶん、
    truly broad な loser burst まで見逃すと
    loser lane が増えるリスクがある。
- Observed/Fact:
  - `2026-03-13 10:09 JST`
    に
    updated
    `StageTracker`
    を local 実DBへ当てると、
    `pocket_loss_window`
    は
    `59372 TickImbalance`
    だけへ再同期され、
    stale row は削除された。
  - 同時点の
    `scalp`
    pocket は
    `loss_jpy=69.012`,
    `loss_pips=1.8`,
    `strategy_count=1`
    となり、
    `stage_cooldown`
    の
    `scalp loss_cluster_*`
    は消えた。
  - unit test は
    `tests/test_stage_tracker.py`
    で
    `8 passed`
    を確認した。
- Verdict: good
- Next Action:
  - next live では
    `cluster cooldown`
    の巻き添えが減る前提で
    `scalp_ping_5s_c_live`
    の
    `lookahead_block / revert_not_found`
    を個別に詰める。

## 2026-03-13 10:30 JST / local-v2: feedback cycle を current loser へ速く反応する配分に更新
- Why/Hypothesis:
  - `2026-03-13 10:30 JST`
    時点の live review では、
    `market_context_latest`
    は
    `2026-03-13 10:16:48 JST`
    生成、
    `USD/JPY=159.0825`,
    `risk_mode=neutral`,
    次の event は
    `2026-03-13 12:35 JST`
    の low impact。
  - 同 review の
    `factor_cache`
    は
    `ATR14(M1)=2.736p`,
    `ATR14(M5)=6.456p`
    で通常帯、
    `quant-market-data-feed`
    の OANDA pricing stream は
    `2026-03-13 10:13 JST`
    に `HTTP 200`
    を確認した。
  - `fills_15m=3`,
    `fills_30m=3`,
    `fills_60m=3`
    で「止まっている」のではなく、
    現在の問題は
    `PrecisionLowVol=-126.855 JPY / 20 trades`,
    `WickReversalBlend=-69.224 / 6`,
    `TickImbalance=-69.012 / 1`,
    `scalp_extrema_reversal_live=-68.985 / 43`
    のような loser lane に current 窓でもサイズが残っていたこと。
  - `run_local_feedback_cycle`
    は
    `participation_allocator=6h/12 attempts`,
    `dynamic_alloc=3d/18h half-life`
    と遅く、
    しかも
    `dynamic_alloc_worker`
    には strategy-level の
    low-sample severe loser clamp
    が無かった。
- Expected Good:
  - current 1 日窓で悪化している loser lane を
    数十分単位の feedback loop で
    `0.20` 付近まで薄くできる。
  - future loop でも
    single hard loser / few-trade burst loser
    を historical floor のまま残さない。
- Expected Bad:
  - 直後は loser lane の entry がさらに減るので、
    winner lane への置き換えが弱い間は fills が少なく見える。
  - one-off noise trade でも severe loser 条件に入る lane は
    一時的に切りすぎるリスクがある。
- Observed/Fact:
  - `scripts/run_local_feedback_cycle.py`
    の既定を
    `dynamic_alloc=1d lookback / min_trades=8 / setup_min_trades=2 / half_life=6h / min_lot_multiplier=0.20`
    と
    `participation_allocator=3h lookback / min_attempts=4 / setup_min_attempts=1 / max_units_cut=0.35 / max_units_boost=0.30 / max_probability_boost=0.15`
    へ更新した。
  - `scripts/dynamic_alloc_worker.py`
    に
    strategy-level の
    `severe_low_sample_loser`
    と
    `fast_burst_loser`
    clamp を追加した。
  - test は
    `tests/test_dynamic_alloc_worker.py`
    と
    `tests/scripts/test_run_local_feedback_cycle.py`
    で
    `24 passed`
    を確認した。
  - same 条件で
    `dynamic_alloc_worker`
    を再実行すると、
    `PrecisionLowVol=0.20`,
    `WickReversalBlend=0.16`,
    `DroughtRevert=0.20`,
    `TickImbalance=0.16`,
    `scalp_extrema_reversal_live=0.20`,
    `session_open_breakout=0.205`
    まで trim された。
- Verdict: good
- Next Action:
  - `main`
    へ push 後、
    core 4 と current loser worker を restart して
    live artifact 読み込みを揃える。
  - 次の 60-90 分は
    loser lane の fills / net_jpy / reject family
    を監視し、
    まだ赤字が残る lane は
    shared trim をこれ以上広げず
    worker-local guard / exit
    を詰める。

## 2026-03-13 11:10 JST - small positive scalp winner を winner 扱いへ寄せ、current winner trim を緩めた

- Why/Hypothesis:
  - `2026-03-13 10:49 JST`
    の最新 loser は
    `WickReversalBlend short`
    `trade_id=59389`
    / `ticket=460239`
    / `entry=159.208`
    / `close=159.227`
    / `-1.9 pips`
    / `-3.42 JPY`
    で、
    これは trend-long tape に against した loser lane だった。
  - 同時点の market context は
    `2026-03-13 10:47 JST`
    `USD/JPY=159.2265`
    / `risk_mode=neutral`
    / `next_event=2026-03-13 12:35 JST 3-Month Bill Auction(low)`
    で通常帯、
    account snapshot は
    `nav=35173.2698`
    / `margin_available=35173.2698`
    / `margin_used=0`
    / `free_margin_ratio=1.0`
    だった。
  - つまり
    「margin を使っていない」主因は risk cap ではなく、
    `PrecisionLowVol / DroughtRevert`
    の current winner long まで
    shared sizing が細くしていたことだと判断した。
  - high-turnover scalp では
    `+1.7-2.0 JPY`
    級の small winner も
    winner lane として拾わないと、
    few-trade loser trim だけが速くなって
    total notional が増えない。

- Expected Good:
  - `PrecisionLowVol / DroughtRevert`
    の current winner long setup だけは
    shared sizing で
    `0.20`
    まで潰されず、
    fills と margin use を増やせる。
  - loser strategy 全体を緩めずに、
    exact winner setup だけを
    bounded に持ち上げられる。

- Expected Bad:
  - one-trade winner noise に反応して
    exact setup を一時的に持ち上げすぎるリスクがある。
  - small positive scalp を winner 判定へ寄せるので、
    very weak edge を early boost するケースが混ざる可能性がある。

- Observed/Fact:
  - `scripts/participation_allocator.py`
    で
    setup-level winner boost の
    `profit_per_fill`
    閾値を
    `1.5-2.0 JPY`
    帯へ下げた。
  - `scripts/dynamic_alloc_worker.py`
    で
    low-sample winner relief
    を追加し、
    current loser strategy の exact winner setup には
    `0.55-0.65`
    の bounded override を出すようにした。
  - test は
    `tests/test_dynamic_alloc_worker.py`
    `tests/scripts/test_participation_allocator.py`
    `tests/scripts/test_run_local_feedback_cycle.py`
    で
    `48 passed`
    を確認した。
  - `2026-03-13 11:10 JST`
    に
    `run_local_feedback_cycle --force`
    を実行した結果、
    `participation_alloc`
    は
    `PrecisionLowVol|long|range_fade|tight_normal|...|gap:up_lean`
    と
    `...|gap:up_flat`
    を
    `boost_participation lot_multiplier=1.2549 probability_boost=0.1228`
    へ更新した。
  - same cycle で
    `DroughtRevert|long|range_fade|tight_normal|...|gap:up_lean`
    も
    `boost_participation lot_multiplier=1.3 probability_boost=0.15`
    を維持した。
  - `dynamic_alloc`
    は
    `PrecisionLowVol`
    の 2 本の winner long fingerprint と
    `DroughtRevert`
    の winner long fingerprint に
    `lot_multiplier=0.65`
    の winner-relief override を出した。
    strategy-wide base は
    `PrecisionLowVol=0.336`
    / `DroughtRevert=0.20`
    のままなので、
    broad loosening ではなく
    exact setup だけを戻している。

- Verdict: pending

- Next Action:
  - 次の
    `60-90 分`
    か
    `10-20 trade`
    で、
    `PrecisionLowVol / DroughtRevert`
    の winner long が
    以前の
    `77-92 units`
    より太く約定するかを確認する。
  - `WickReversalBlend short`
    は current loser のままなので、
    margin 拡大対象にはせず
    worker-local guard / exit 改善を別で続ける。

### 2026-03-13 11:46 JST - `scalp_ping_5s_c_live` の negative lookahead を low-activity 時だけ薄く救済

- Why/Hypothesis:
  - `2026-03-13 11:46 JST`
    時点で
    `USD/JPY 159.364/159.372`
    / spread `0.8p`
    / `ATR14(M1)=1.34p`
    / `ATR14(M5)=4.96p`
    と通常帯なのに、
    `fills_15m=0`
    / `fills_30m=0`
    / `fills_60m=1`
    で
    低稼働が継続していた。
  - `quant-scalp-ping-5s-c.log`
    の直近 skip は
    `no_signal:revert_not_found`
    と
    `lookahead_block`
    が主因で、
    特に
    `lookahead_block`
    は
    `reason=edge_negative_block`
    に集中していた。
  - `revert_not_found`
    側の threshold は既にかなり薄く、
    これ以上の blanket loosening は quality 悪化が先に来る。
    一方で
    `edge_negative_block`
    には
    `pred_move 0.30-0.40p`
    / `momentum 0.4-0.6p`
    / `range 0.4-0.6p`
    の
    thin-but-not-dead
    な候補が混ざっていた。
  - 仮説は、
    「entry が止まり過ぎている current 窓だけ、
    `scalp_ping_5s_c_live`
    の negative lookahead を
    bounded units で rescue すれば、
    低稼働を壊さずに cadence を戻せる」
    である。

- Expected Good:
  - `fills_15m/30m`
    がゼロの窓で、
    `edge_negative_block`
    起因の取り逃しを少し戻せる。
  - blanket に gate を外さず、
    `recent fills == 0`
    かつ
    `pred/momentum/range`
    が最低条件を満たす候補だけを
    `0.18-0.42x`
    の小さい units で通せる。

- Expected Bad:
  - negative edge の玉を救済するため、
    very thin edge の small loser を増やすリスクがある。
  - `recent fills == 0`
    でも地合いそのものが悪い窓では、
    cadence だけ戻って収益が伴わない可能性がある。

- Observed/Fact:
  - `workers/scalp_ping_5s/config.py`
    に
    `LOOKAHEAD_NEGATIVE_EDGE_RESCUE_*`
    を追加し、
    `ENV_PREFIX == SCALP_PING_5S_C`
    のときだけ default-on で使えるようにした。
  - `workers/scalp_ping_5s/worker.py`
    に
    `trades.db`
    ベースの recent fill count helper と
    `_maybe_rescue_negative_lookahead`
    を追加した。
    `TECH_ROUTER_ENABLED=0`
    の current C lane では、
    `reason=edge_negative_block`
    でも
    `recent fills <= 0`
    / `edge >= -0.95p`
    / `pred >= 0.24p`
    / `momentum >= 0.40p`
    / `range >= 0.40p`
    を満たす候補だけを
    bounded units で rescue する。
  - rescue が発火したときは
    `entry_thesis`
    に
    `lookahead_rescue_applied`
    / `lookahead_rescue_reason`
    / `lookahead_rescue_recent_fills`
    を残すので、
    後から勝敗を切り分けられる。
  - `ops/env/scalp_ping_5s_c.env`
    に current 運用値として
    rescue knobs を追加した。
  - test は
    `tests/workers/test_scalp_ping_5s_worker.py -k negative_lookahead_rescue`
    で
    `2 passed`
    を確認した。
  - 変更前の live 事実として、
    `quant-scalp-ping-5s-c.log`
    には
    `2026-03-13 11:45 JST`
    でも
    `entry-skip summary total=165 ... lookahead_block=41`
    が残り、
    例として
    `pred=0.318p cost=1.180p edge=-0.862p mom=0.400p range=0.400p`
    のような
    rescue 対象候補が存在した。

- Verdict: pending

- Next Action:
  - `quant-scalp-ping-5s-c`
    を再起動して
    `lookahead rescue`
    ログが出るか、
    かつ
    `fills_15m/30m`
    が
    `0/0`
    から改善するかを確認する。
  - rescue 発火玉だけを
    `trades.db`
    で後追いし、
    loser が先行するなら
    `max_neg_edge`
    をさらに浅くする。

### 2026-03-13 11:50 JST - `scalp_ping_5s_c_live` rescue 候補を order-manager min units mismatch で落とさない

- Why/Hypothesis:
  - first deploy 後、
    `quant-scalp-ping-5s-c.log`
    で
    `lookahead rescue`
    自体は即時発火した。
    ただし
    `orders.db`
    / `quant-order-manager.log`
    を見ると、
    rescue 候補は
    `entry_probability_reject`
    ではなく
    `note=entry_probability:entry_probability_below_min_units`
    で
    `-5/-6 units`
    に潰されていた。
  - 原因は
    `SCALP_PING_5S_C_MIN_UNITS=5`
    に対して、
    `quant-order-manager`
    側の
    `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=10`
    が残っていたことだった。
  - 仮説は、
    「C lane の current rescue は 5-6 units 帯で通す設計なので、
    order-manager の strategy min units を 5 に揃えれば、
    rescue 候補を無駄に潰さず cadence を戻せる」
    である。

- Expected Good:
  - `lookahead rescue`
    された C lane の 5-6 units 候補が
    `entry_probability_below_min_units`
    で落ちなくなる。
  - broad pocket ではなく
    `scalp_ping_5s_c(_live)`
    だけの mismatch 修正なので、
    shared risk guard 全体は変えずに済む。

- Expected Bad:
  - 5 units 級の very small scalp が増えるので、
    cadence は戻っても per-trade PnL への寄与は小さい。
  - low-edge rescue が増えすぎると、
    `fills` は回復しても収益が伴わない可能性は残る。

- Observed/Fact:
  - `2026-03-13 11:49:42 JST`
    に
    `lookahead rescue side=short recent_fills=0/30m edge=-0.677 pred=0.447 momentum=1.000 ... units=0.33`
    が live log に出た。
  - 同候補は
    `orders.db`
    上で
    `strategy=scalp_ping_5s_c_live`
    / `entry_probability=0.796`
    / `lookahead_rescue_applied=1`
    のまま
    `entry_probability_reject`
    となり、
    `quant-order-manager.log`
    では
    `units=-5 note=entry_probability:entry_probability_below_min_units`
    と記録された。
  - さらに
    `2026-03-13 11:50:35 JST`
    の rescue 候補も
    `entry_probability=0.638`
    / `units=-6`
    で同じ reject に当たった。
  - `ops/env/quant-order-manager.env`
    の
    `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C_LIVE`
    と
    `...SCALP_PING_5S_C`
    を
    `10 -> 5`
    に揃える。

- Verdict: pending

- Next Action:
  - `quant-order-manager`
    と
    `quant-scalp-ping-5s-c`
    を再起動して、
    次の
    `lookahead rescue`
    候補が
    `entry_probability_below_min_units`
    を跨いで filled/requested へ進むかを見る。
  - それでも reject が残るなら、
    次は
    `preserve-intent min_scale`
    ではなく
    worker-local `entry_probability`
    側を見直す。

### 2026-03-13 11:54 JST - rescued C lane に post-probability units floor を入れて execution scale を跨がせる

- Why/Hypothesis:
  - `ORDER_MIN_UNITS_STRATEGY_SCALP_PING_5S_C(_LIVE)=5`
    に揃えた後も、
    `quant-order-manager.log`
    では
    `entry_probability:entry_probability_below_min_units`
    が継続した。
  - 直近 reject は
    `units=-5/-6`
    / `entry_probability=0.53-0.79`
    帯で、
    rescue 後の raw units が small すぎて
    `entry_probability`
    scale 後に
    `5`
    を割っていた。
  - 仮説は、
    「lookahead rescue が発火した候補だけ、
    worker-local に `ceil(MIN_UNITS / entry_probability)` ベースの
    small floor を入れれば、
    order-manager の preserve-intent scale を跨げる」
    である。

- Expected Good:
  - rescued candidate が
    `5-6 units`
    のまま execution 層で潰れず、
    `7-10 units`
    帯で通る。
  - rescue 対象以外の通常 signal は触らないので、
    broad sizing loosening にはならない。

- Expected Bad:
  - low-activity rescue short が小さく通るようになるので、
    cadence は戻っても loser noise が増える可能性がある。
  - `entry_probability`
    が低い候補まで通しすぎると、
    rescue lane の PF が悪化するリスクがある。

- Observed/Fact:
  - `2026-03-13 11:53 JST`
    までの
    `orders.db`
    では、
    `lookahead_rescue_applied=1`
    の
    `scalp_ping_5s_c_live`
    short が
    `entry_probability=0.537-0.796`
    でも
    `entry_probability_reject`
    に当たり続けた。
  - `workers/scalp_ping_5s/worker.py`
    に
    `_maybe_apply_lookahead_rescue_units_floor`
    を追加し、
    `lookahead_rescue_applied`
    のときだけ
    `ceil(MIN_UNITS / entry_probability)`
    を基準に
    `MIN_UNITS * 2`
    までの bounded floor
    (`5 -> max 10 units`)
    を掛けるようにした。
  - `entry_thesis`
    に
    `lookahead_rescue_units_floor_status`
    を追加し、
    rescue floor が入った玉を後で切り出せるようにした。
  - first implementation では
    floor を
    `min_units_rescue`
    より前に当てていたため、
    raw `units<=0`
    の候補で
    `non_positive`
    となり floor が実効化しなかった。
    follow-up で
    floor 適用位置を
    `min_units_rescue`
    後へ移した。
  - test は
    `tests/workers/test_scalp_ping_5s_worker.py -k negative_lookahead_rescue`
    へ 2 本追加し、
    `0.55 prob / 6 units`
    の rescued candidate が
    `10 units`
    へ持ち上がることと、
    non-rescue 候補では不活性のままなことを固定した。

- Verdict: pending

- Next Action:
  - `quant-order-manager`
    と
    `quant-scalp-ping-5s-c`
    を再起動して、
    rescue 候補が
    `entry_probability_below_min_units`
    を抜けるか確認する。
  - 抜けた後は
    `fills_15m/30m`
    と
    rescue-tagged trade の PnL
    だけを追い、
    loser noise が多ければ
    floor cap を下げる。

### 2026-03-13 12:10 JST - `scalp_ping_5s_d_live` の `fast_flip + horizon align + m1_opposite` loser lane を worker local で遮断

- Why/Hypothesis:
  - user が提示した
    `2026-03-13 11:55:41 JST`
    の
    `-21.01 JPY`
    close は、
    `ticket=460251`
    の
    `scalp_ping_5s_d_live`
    long
    (`159.324 -> 159.270`, `-5.4p`)
    だった。
  - `entry_thesis`
    を切ると、
    `signal_mode=momentum_fflip_hz`,
    `horizon_gate=horizon_align_fflip`,
    `horizon_composite_side=long`,
    `m1_trend_gate=m1_opposite`,
    `fast_direction_flip_applied=true`
    で、
    `short->long`
    へ fast flip した後でも
    M1 が逆向きの lane が通っていた。
  - 仮説は、
    「D variant は
    `horizon_composite_side != neutral`
    かつ
    `m1_trend_gate=m1_opposite`
    の時点で reject すべきで、
    fast flip 済みの horizon-align lane も例外にしない」
    である。

- Expected Good:
  - `scalp_ping_5s_d_live`
    の
    `fast_flip`
    loser lane を止め、
    `-5p`
    級の single-loss を削る。
  - shared gate を緩めずに、
    D worker 内だけで
    non-neutral horizon / opposite M1 conflict を潰せる。

- Expected Bad:
  - D variant の fill は少し減る。
  - horizon 側へ寄せた flip lane も切るので、
    ごく一部の early recovery winner を捨てる可能性がある。

- Observed/Fact:
  - RCA window の市況は通常帯で、
    `spread=0.8p`,
    `ATR14(M1)=2.1585p`,
    `ATR14(M5)=4.6155p`
    だった。
    `logs/local_v2_stack/quant-scalp-ping-5s-b.log`
    では
    `2026-03-13 12:09-12:10 JST`
    の
    OANDA pricing が継続して
    `HTTP 200`
    を返しており、
    API 品質は問題なかった。
  - `logs/trades.db`
    の
    `scalp_ping_5s_d_live`
    7日分で、
    `horizon_composite_side != neutral`
    かつ
    `m1_trend_gate=m1_opposite`
    の close は
    `15 trades / 0勝 / -60.356 JPY`
    だった。
    うち
    `fast_direction_flip_applied=true`
    は
    `1 trade / -21.006 JPY`
    で、
    user 提示の loser と一致した。
  - `workers/scalp_ping_5s/worker.py`
    の
    `_countertrend_horizon_m1_block_reason`
    は従来
    `signal.side != horizon_side`
    のときしか block せず、
    `fast_flip`
    で horizon 側へ寄せた lane は通っていた。
  - 今回の patch で、
    D variant は
    `non-neutral horizon + m1_opposite`
    を
    `relation=align/counter`
    付きで一律 block するように変更した。
    さらに
    `market_order`
    直前にも同じ contract を late 再評価し、
    途中ルーティングで side/units が変わっても
    最終送信前に落ちる二重ガードへした。
  - test は
    `tests/workers/test_scalp_ping_5s_worker.py`
    に 2 本追加し、
    exact loser lane
    (`long + horizon long + m1_opposite + m1_score=-0.039`)
    が block されることと、
    `m1_score`
    が閾値未満の weak conflict は preserve されることを固定した。
    `tests/workers/test_scalp_ping_5s_extrema_routes.py`
    側にも route-level test を追加したが、
    環境依存 import で即時実行が安定しないため、
    今回の反映判断は worker unit test を正本とした。

- Verdict: pending

- Next Action:
  - `quant-scalp-ping-5s-d`
    再起動後に、
    同 lane が
    `countertrend_horizon_m1_block`
    skip へ変わるかを
    `logs/local_v2_stack/quant-scalp-ping-5s-d.log`
    で確認する。
  - 次の
    `60-90分`
    で
    `scalp_ping_5s_d_live`
    の
    `filled / realized_jpy / avg_pl_pips`
    を再集計し、
    D variant の赤字寄与が縮むかを見る。

### 2026-03-13 12:30 JST - 24h contract audit では照合ミスは主因ではなく、`STOP_LOSS_ORDER` が主損失だった

- Why/Hypothesis:
  - user から
    「照合のミスがたくさんあって損失につながっていないか」
    の指摘があった。
  - 仮説は 2 本で、
    1)
    `entry_thesis`
    契約欠損が大量にあり、
    entry/size/probability の食い違いが赤字を作っている、
    2)
    欠損は多くないが、
    stale `entry_units_intent`
    や `strategy_tag`
    が一部で残っている、
    である。

- Expected Good:
  - contract mismatch が主因でないなら、
    RCA を entry/SL 側へ戻せる。
  - stale intent が残っているなら、
    order_manager 側で最終 units に self-heal して、
    今後の照合ズレを runtime で潰せる。

- Expected Bad:
  - order_manager で
    `entry_units_intent`
    を強制整合すると、
    worker 側の古い意図値を後から比較したい用途には使えなくなる。
  - ただし現行 contract の正本は
    「最終送信 units」
    なので、
    stale 値を残すより安全側である。

- Observed/Fact:
  - `qr-entry-thesis-contract-check`
    を
    `--window-hours 24`
    で実行した結果は
    `overall=pass`。
    `static calls_missing=0`,
    `trades missing_rows=0 invalid_rows=0`,
    `orders missing_rows=0 invalid_rows=0`
    だった。
  - `orders.db`
    を素直に数えると
    `close_request / close_ok / close_reject_profit_buffer`
    の
    `131 rows`
    に
    `entry_thesis`
    が無いが、
    これは close 系 payload であり、
    entry contract 欠損ではない。
  - 24h 損益は
    `STOP_LOSS_ORDER 71件 / -596.871 JPY`
    が主損失で、
    `MARKET_ORDER_TRADE_CLOSE`
    は
    `37件 / +73.797 JPY`,
    `TAKE_PROFIT_ORDER`
    は
    `11件 / +115.45 JPY`
    だった。
    つまり現在の主因は照合ミスではなく、
    loser entry と hard SL 集中である。
  - ただし 1 本だけ、
    `ticket=459725`
    `scalp_macd_rsi_div_live`
    で
    `entry_units_intent=287`
    に対して
    actual units が
    `377`
    の stale intent が残っていた。
    この trade の realized は
    `-5.278 JPY`
    で、
    24h 赤字の主因ではないが、
    contract integrity 上は無視しない。
  - 対応として
    `execution/order_manager.py`
    の
    `_ensure_entry_intent_payload`
    を、
    missing 補完ではなく
    最終 `units` / `strategy_tag`
    への強制整合へ変更した。
    stale `entry_units_intent`
    や stale `strategy_tag`
    が来ても、
    order_manager payload では最終送信値に self-heal される。
  - test は
    `tests/execution/test_order_manager_log_retry.py`
    に追加し、
    stale
    `entry_units_intent=287`,
    `strategy_tag=stale_strategy`
    が
    `377` /
    `scalp_macd_rsi_div_live`
    へ上書きされることを固定した。

- Verdict: good

- Next Action:
  - 収益改善の主戦場は
    contract audit ではなく
    `STOP_LOSS_ORDER`
    集中の strategy-local RCA に戻す。
  - 以後、
    stale intent の再発は
    order_manager self-heal が受ける前提で、
    赤字 cluster は
    `PrecisionLowVol / scalp_extrema_reversal_live / WickReversalBlend / DroughtRevert`
    の entry/SL 品質側を優先して詰める。

### 2026-03-13 13:15 JST - stale `entry_units_intent` は isolated ではなく、status/trade 保存全体で発生していたので系で潰した

- Why/Hypothesis:
  - 12:30 JST 時点の contract audit は
    `missing/invalid`
    を見ていたため pass だったが、
    user 指摘どおり
    「存在はするが stale な照合値」
    が analytics を汚している可能性が残っていた。
  - 仮説は、
    1)
    `probability_scaled / submit_attempt / filled`
    の各 status で
    nested `entry_thesis.entry_units_intent`
    が actual units に追随していない、
    2)
    `position_manager`
    は missing 補完だけで、
    stale units/tag を trade 保存時に直していない、
    である。

- Expected Good:
  - `orders.db`
    と
    `trades.db`
    の
    `entry_thesis`
    が actual units / strategy に揃い、
    RCA と setup cluster が false mismatch に引っ張られなくなる。
  - どうしても矯正が必要だった row も
    `entry_units_intent_raw`
    /
    `strategy_tag_raw`
    で元値を残せる。

- Expected Bad:
  - raw intent をそのまま見たい監査では、
    上書き後の値だけ見ると confusing になり得る。
  - そのため、
    override が起きた row には
    raw 値を別 key で残し、
    order log には
    `entry_contract_corrections`
    も載せる。

- Observed/Fact:
  - 24h 再集計では、
    `orders.db`
    の
    `entry_thesis.entry_units_intent`
    と actual order units の mismatch が
    `407 rows`
    あった。
    strategy 別では
    `PrecisionLowVol 135`,
    `DroughtRevert 108`,
    `MicroLevelReactor-bounce-lower 72`,
    `WickReversalBlend 33`,
    `TickImbalance 22`
    が多かった。
  - `trades.db`
    でも
    `entry_thesis.entry_units_intent`
    と actual trade units の mismatch が
    `72 rows`
    あった。
    つまり、
    「大量欠損は無い」
    と
    「actual units と thesis が揃っている」
    は別問題だった。
  - 原因は 2 本で、
    `execution/order_manager.py`
    の
    `market_order / limit_order`
    が
    top-level payload を inject しても、
    per-status actual units で nested
    `entry_thesis`
    を毎回 canonicalize していなかったこと、
    そして
    `execution/position_manager.py`
    が
    stale units/tag を
    「missing でない」
    という理由でそのまま保存していたことだった。
  - 対応として、
    order_manager 側に
    status ごとの actual `units / side / strategy_tag`
    を使う canonicalization を追加し、
    `probability_scaled / submit_attempt / filled`
    のすべてで
    nested `entry_thesis`
    を揃えるようにした。
    さらに
    `position_manager`
    は trade ingest 時に stale
    `entry_units_intent`
    を actual trade units で上書きし、
    explicit `strategy_tag`
    を thesis より優先するようにした。
  - 既存 DB も one-off repair を実施した。
    `orders.db: updated=32575`,
    `trades.db: updated=15022`
    で、
    24h mismatch は
    `orders 407 -> 0`,
    `trades 72 -> 0`
    まで解消した。
  - test は
    `tests/execution/test_order_manager_log_retry.py`
    の targeted subset で
    `6 passed`,
    `tests/execution/test_position_manager_close.py`
    の targeted subset で
    `5 passed`
    を確認した。

- Verdict: good

- Next Action:
  - 以後は
    `entry_contract_corrections`
    が新規 row で継続発生するかを見て、
    もし残るなら
    worker 側の raw intent 更新漏れ経路まで掘る。
  - RCA はこの mismatch ノイズを外した状態で、
    ふたたび
    `STOP_LOSS_ORDER`
    cluster の quality 改善へ戻す。

## 2026-03-13 13:45 JST - current winner setup の lot 回復を強め、local feedback の dyn floor を 0.30 へ戻した

- Why/Hypothesis:
  - `2026-03-13 13:34 JST`
    の
    `pdca_profitability_latest.json`
    では
    `USD/JPY bid/ask=159.464/159.472`,
    `spread=0.8p`,
    OANDA pricing/account は
    `status=200`,
    `nav_jpy=35150.0`,
    `margin_used_jpy=0.0`,
    `open_trade_count=0`
    だった。
  - つまり margin cap ではなく、
    shared sizing が still too tight で、
    current winner setup まで薄くしていた。
  - 実際、
    変更前の
    `dynamic_alloc`
    は
    `PrecisionLowVol / DroughtRevert`
    の exact winner setup でも
    `lot_multiplier=0.65`
    止まりで、
    local feedback cycle も
    `--min-lot-multiplier 0.20`
    を固定していた。

- Expected Good:
  - loser strategy の blanket trim は維持しつつ、
    current winner setup だけは
    `0.70-1.00`
    帯まで lot を戻せる。
  - `scalp_ping_5s_c_live`
    や
    `DroughtRevert`
    の mild lane が
    `0.20`
    固定から抜け、
    margin use と fill size を少し戻せる。

- Expected Bad:
  - one-trade winner noise を拾って
    lot を戻しすぎるリスクがある。
  - そのため、
    severe loser / fast burst loser clamp 自体は維持し、
    `TickImbalance / scalp_ping_5s_d_live / WickReversalBlend`
    の hard loser 側は broad に緩めない。

- Observed/Fact:
  - `scripts/dynamic_alloc_worker.py`
    の
    low-sample winner relief
    を
    `single winner: 0.70-0.82`,
    `2-trade winner: 0.82-1.00`
    へ引き上げた。
  - `scripts/run_local_feedback_cycle.py`
    の
    `dynamic_alloc`
    既定を
    `--min-lot-multiplier 0.20 -> 0.30`
    へ更新した。
  - test は
    `tests/test_dynamic_alloc_worker.py`
    と
    `tests/scripts/test_run_local_feedback_cycle.py`
    で
    `26 passed`
    を確認した。
  - `2026-03-13 13:45 JST`
    に
    `run_local_feedback_cycle --force --job participation_allocator --job dynamic_alloc`
    を再実行し、
    `config/dynamic_alloc.json as_of=2026-03-13T04:45:04Z`
    を生成した。
  - 生成後の main lane は
    `DroughtRevert 0.20 -> 0.30`,
    `scalp_ping_5s_c_live 0.20 -> 0.30`,
    `scalp_extrema_reversal_live 0.20 -> 0.24`
    へ回復した。
    一方で
    `WickReversalBlend=0.16`,
    `scalp_ping_5s_d_live=0.16`,
    `TickImbalance=0.16`
    は据え置きだった。
  - exact winner override は
    `PrecisionLowVol long tight_normal`
    が
    `0.82`,
    `DroughtRevert long tight_normal`
    が
    `0.82`
    まで回復した。
  - 直後の factor cache は
    `2026-03-13 13:45 JST`
    で
    `M1 close=159.478 ATR=3.478p RSI=47.61 ADX=42.90`,
    `M5 close=159.524 ATR=5.870p RSI=64.69 ADX=31.34`
    だった。

- Verdict: good

- Next Action:
  - 次の
    `10-20 trades`
    で
    `PrecisionLowVol / DroughtRevert / scalp_ping_5s_c_live`
    の notional と net_jpy が改善するかを見る。
  - まだ margin use が細いままなら、
    shared broad loosen ではなく
    winner setup の relief 条件だけをさらに詰める。

## 2026-03-13 13:53 JST - post-restart 監視: runtime は正常、まだ lot 改善の live sample は不足

- Why/Hypothesis:
  - sizing 緩和後に
    actual fill size が増えたかを
    post-restart の live で確認する必要がある。
  - 市況が通常帯なのに
    fills が薄いままなら、
    margin 不足ではなく
    signal-side gate が主因の可能性が高い。

- Expected Good:
  - core 4 が正常稼働し、
    fresh artifact を読んだ状態で
    winner lane の fill が再開する。
  - dominant block family を 1 つに絞って、
    次の PDCA を worker-local に寄せられる。

- Expected Bad:
  - restart 直後は sample が薄く、
    sizing 改善の有無をまだ断定できない。
  - low activity が続く場合、
    shared sizing ではなく
    signal gate 側の詰まりが残る。

- Observed/Fact:
  - `2026-03-13 13:53 JST`
    の
    `health_snapshot`
    は
    `mechanism_integrity.ok=true`,
    `decision_latency_ms=18.8`,
    `data_lag_ms=431.8`
    で fresh だった。
  - core 4 は
    `quant-market-data-feed / quant-strategy-control / quant-order-manager / quant-position-manager`
    すべて
    `[running]`
    を維持していた。
  - factor cache は
    `M1 close=159.480 ATR=3.421p RSI=49.25 ADX=30.54`,
    `M5 close=159.454 ATR=5.951p RSI=56.41 ADX=32.80`
    で通常帯だった。
  - restart 後の new order event は
    `2026-03-13 13:52 JST`
    の
    `scalp_ping_5s_c_live entry_probability_reject`
    1 本で、
    post-restart の new filled はまだ 0 本だった。
  - `quant-scalp-ping-5s-c.log`
    では
    `lookahead_block(edge_negative_block)`
    と
    `no_signal:revert_not_found`
    が優勢で、
    `entry-skip summary total=148 no_signal=50 lookahead_block=48 no_signal:revert_not_found=42`
    を確認した。
  - `PrecisionLowVol / DroughtRevert / WickReversalBlend`
    は
    `cluster cooldown skip ... single_strategy_contained`
    を継続しており、
    pocket-wide block 自体は再発していない。

- Verdict: pending

- Next Action:
  - 次の
    `15-30分`
    で post-restart fill が still 0 のままなら、
    dominant block family は
    `scalp_ping_5s_c_live`
    の
    `lookahead_block / revert_not_found`
    と見なし、
    shared sizing ではなく
    worker-local signal gate を 1 本だけ詰める。

## 2026-03-13 14:10 JST - `scalp_ping_5s_c_live` の generic min-units rescue を post-probability floor へ拡張

- Why/Hypothesis:
  - `2026-03-13 13:59 JST`
    の local-v2 実測では、
    `USD/JPY bid=159.477 / ask=159.485 / spread=0.8p / M1 ATR=3.39p / M5 ATR=6.05p`
    で、
    `data_lag_ms=121-943`,
    `decision_latency_ms=11-67`,
    `open_trades=0`
    と市況・API 品質は通常帯だった。
  - それでも
    `fills_15m=0`, `fills_30m=4`
    で low activity が続き、
    `orders.db`
    直近6hの
    `entry_probability_reject`
    は
    `scalp_ping_5s_c_live`
    の
    `entry_probability_below_min_units=53`
    に集中していた。
  - 直近 reject sample は
    `2026-03-13 13:52:09 JST`
    の
    `entry_probability=0.824175`, `confidence=92`, `units=5`
    で、
    `lookahead_rescue_applied=0`
    の generic small probe が
    `order_manager`
    preserve-intent scale で
    `5 -> 4`
    相当に潰されていた。
  - 仮説は、
    「lookahead rescue 専用 floor だけでは current no-fill を捌けず、
    generic `min_units_rescue`
    自体にも bounded post-probability floor を入れれば、
    `entry_probability>=0.46`
    の rescue candidate を worker-local に cadence 回復できる」
    である。

- Expected Good:
  - `scalp_ping_5s_c_live`
    の rescued small probe が
    `entry_probability_below_min_units`
    へ落ちにくくなり、
    fills cadence が戻る。
  - bounded floor
    （`max MIN_UNITS*2`）
    なので、
    broad shared loosening ではなく
    strategy-local の small probe だけを薄く通せる。

- Expected Bad:
  - `entry_probability 0.46-0.55`
    帯の rescued probe が増えるため、
    loser noise が増える可能性がある。
  - cadence だけ戻っても、
    rescue-tagged trade の PF が伴わなければ
    追加 tightening が必要になる。

- Observed/Fact:
  - `orders.db`
    直近6hの
    `scalp_ping_5s_c_live`
    reject 集計は
    `53件`,
    `avg_confidence=91.8`,
    `avg_entry_probability=0.509`,
    `avg_units=-5.7`
    だった。
  - same sample を local で再計算すると、
    current threshold
    （`entry_probability>=0.46`, `confidence>=72`）
    を満たす reject のうち
    `36/53`
    は
    bounded post-probability floor
    で execution 前 reject を跨げる見込みだった。
  - `workers/scalp_ping_5s/worker.py`
    の
    `min_units_rescue`
    を
    `max(MIN_UNITS, ceil((MIN_UNITS-0.5)/entry_probability))`
    相当の bounded floor
    へ更新し、
    `entry_thesis.min_units_rescue_status`
    へ
    `rescued_post_probability_floor`
    を残すようにした。
  - regression test として、
    `0.824175 prob / 92 conf`
    の current reject 相当 sample が
    `5 -> 6 units`
    へ救済されることを
    `tests/workers/test_scalp_ping_5s_worker.py`
    に固定した。

- Verdict: pending

- Next Action:
  - `quant-scalp-ping-5s-c`
    と
    `quant-order-manager`
    を再起動し、
    次の
    `15-30分`
    で
    `scalp_ping_5s_c_live`
    の
    `entry_probability_reject(entry_probability_below_min_units)`
    が減るかを確認する。
  - 反映後は
    `orders.db`
    で
    `min_units_rescue_status='rescued_post_probability_floor'`
    相当の entry を切り出し、
    `fills_15m/30m`
    と
    `trades.db`
    の realized JPY を別建てで監視する。

## 2026-03-13 14:45 JST / local-v2: `PrecisionLowVol` を pocket-wide `profit_guard` から切り離し、current winner が loser scalp lane に巻き込まれないようにした

- Why/Hypothesis:
  - user 指摘どおり
    「entry を増やしたのに負けが嵩む」
    主因は、
    extra fills が
    `scalp_ping_5s_c_live / scalp_ping_5s_d_live / scalp_extrema_reversal_live / TickImbalance`
    の loser lane に寄る一方で、
    current market で唯一まだ勝っている
    `PrecisionLowVol`
    まで
    `scalp` pocket 全体の
    `profit_guard`
    giveback で止まっていたことだった。
  - `PrecisionLowVol`
    は current 6h で
    `4 trades / +4.484 JPY / +4.8 pips / win_rate 1.0`
    なのに、
    `orders.db`
    では
    `2026-03-13 10:15:34 JST`
    と
    `10:20:10 JST`
    に
    `profit_guard`
    block が出ていた。
  - same 180m sample の
    `trades.db`
    では
    `scalp pocket = -17.801 JPY / -4.2 pips`
    （`DroughtRevert=-7.942`, `TickImbalance=-10.183`, `PrecisionLowVol=+0.324`）
    だった一方、
    `PrecisionLowVol`
    単体は
    `+0.324 JPY / +0.2 pips`
    で giveback を起こしていなかった。
  - 仮説は
    `PrecisionLowVol`
    を
    pocket-wide guard
    ではなく
    strategy-scoped guard
    へ切り替えれば、
    loser scalp lane を緩めずに
    current winner の entry だけを増やせる、
    というもの。

- Expected Good:
  - `PrecisionLowVol`
    が
    `DroughtRevert / TickImbalance`
    の pocket drawdown に巻き込まれず、
    current range 市況での
    `profit_guard`
    block が減る。
  - broad shared loosening ではなく、
    strategy-local な guard scope 切り替えだけで
    entry 増を狙える。

- Expected Bad:
  - `PrecisionLowVol`
    自身の giveback が始まった場合でも、
    pocket loser の連座ではなく
    strategy 自身の履歴でしか止まらなくなるため、
    monitor が甘いと loser re-entry を増やす可能性がある。

- Observed/Fact:
  - `execution/order_manager.py`
    に
    strategy-specific
    `ORDER_PROFIT_GUARD_SCOPE`
    override を追加し、
    `workers/common/profit_guard.py`
    が
    `scope_override`
    を受けて
    `pocket|strategy`
    を切り替えられるようにした。
  - local-v2 の
    `quant-order-manager`
    env に
    `ORDER_PROFIT_GUARD_SCOPE_STRATEGY_PRECISIONLOWVOL=strategy`
    を入れ、
    `PrecisionLowVol`
    だけ
    strategy-scoped query
    に切り替えた。
  - 回帰として、
    `tests/test_profit_guard_prefix.py`
    に
    `PrecisionLowVol + TickImbalance`
    混在 pocket で
    pocket-scope は block、
    strategy-scope は pass
    になるケースを追加した。
  - `tests/execution/test_order_manager_log_retry.py`
    では
    `ORDER_PROFIT_GUARD_SCOPE_STRATEGY_PRECISIONLOWVOL`
    が
    `order_manager -> profit_guard`
    に
    `scope_override='strategy'`
    として渡ることを固定した。

- Verdict: pending

- Next Action:
  - `quant-order-manager`
    を再起動して反映後、
    次の
    `30-60分`
    で
    `PrecisionLowVol`
    の
    `profit_guard`
    block 件数、
    `filled`
    件数、
    realized JPY を切り出す。
  - loser lane
    （`DroughtRevert / TickImbalance / scalp_ping_5s_c_live`）
    の entry を追加で緩めずに、
    winner lane の share だけが増えるかを確認する。

## 2026-03-13 15:05 JST / local-v2: `MicroLevelReactor` の `hist_block` しきい値を dedicated env だけで下げ、current winner setup を通し直す

- Why/Hypothesis:
  - 直近24hの live winner は
    `MicroLevelReactor|micro`
    だけで
    `18 trades / +5.874 JPY / +7.8 pips / PF 1.142`
    だった。
  - ただし current no-entry の dominant block family は
    `quant-micro-levelreactor.log`
    の
    `hist_block tag=MicroLevelReactor-breakout-long strategy=MicroLevelReactor n=312 score=0.182 reason=low_recent_score`
    で、
    既定
    `MICRO_MULTI_HIST_SKIP_SCORE=0.20`
    に just under で引っかかっていた。
  - 30d setup breakdown では
    `MicroLevelReactor-breakout-long`
    自体は
    `2 trades / +32.344 JPY / win_rate 1.0`
    で、
    current winner setup が strategy-wide historical drag に巻き込まれている形だった。
  - 仮説は、
    `quant-micro-levelreactor`
    の dedicated env だけ
    `MICRO_MULTI_HIST_SKIP_SCORE`
    を
    `0.20 -> 0.18`
    に下げれば、
    winner lane を broad に緩めずに unblock できる、
    というもの。

- Expected Good:
  - `MicroLevelReactor-breakout-long`
    の current winner setup が
    `hist_block`
    で止まりにくくなり、
    micro winner の participation が戻る。
  - global micro history gate や loser micro worker までは緩まない。

- Expected Bad:
  - `MicroLevelReactor`
    全体 score
    `0.182`
    をそのまま通すため、
    winner setup 以外の marginal signal まで増える可能性がある。
  - current signal が実際には減速局面なら、
    breakout-long 再開で small loser が増える可能性がある。

- Observed/Fact:
  - `logs/trades.db`
    30d aggregate では
    `MicroLevelReactor`
    は
    `573 trades / -153.44 JPY / PF 0.932`
    と still negative だが、
    30d setup breakdown では
    `MicroLevelReactor-breakout-long`
    が
    `2 trades / +32.344 JPY`
    と strongest positive setup だった。
  - `workers/micro_runtime/worker.py`
    の history selector は
    current runtime で strategy family score を使って
    `skip = n >= HIST_MIN_TRADES and score < HIST_SKIP_SCORE`
    を判定しており、
    current log sample では
    `score=0.182`
    が just under だった。
  - 実装は
    `ops/env/quant-micro-levelreactor.env`
    に
    `MICRO_MULTI_HIST_SKIP_SCORE=0.18`
    を追加する dedicated env tuning のみとし、
    shared
    `ops/env/local-v2-stack.env`
    や code は触らなかった。

- Verdict: pending

- Next Action:
  - `quant-micro-levelreactor`
    と core 4 を再起動し、
    次の
    `30-60分`
    で
    `quant-micro-levelreactor.log`
    の
    `hist_block ... score=0.182`
    が消えるか、
    `orders.db`
    の
    `MicroLevelReactor`
    preflight / filled
    が戻るかを確認する。

## 2026-03-13 15:15 JST / local-v2: shared env の後勝ちを避けるため `MICRO_MULTI_HIST_SKIP_SCORE_OVERRIDE` を追加

- Why/Hypothesis:
  - 15:05 JST の first fix は
    `ops/env/quant-micro-levelreactor.env`
    に
    `MICRO_MULTI_HIST_SKIP_SCORE=0.18`
    を置いたが、
    restart 後の env chain は
    `base -> service -> local-v2-stack -> extra`
    で、
    shared
    `ops/env/local-v2-stack.env`
    の generic
    `MICRO_MULTI_HIST_SKIP_SCORE=0.20`
    が後勝ちしていた。
  - 実際に restart 後
    `2026-03-13 15:10:04 JST`
    の
    `quant-micro-levelreactor.log`
    で
    `hist_block tag=MicroLevelReactor-bounce-lower strategy=MicroLevelReactor n=312 score=0.182`
    が再発し、
    dedicated env だけでは unblock できていないことを確認した。
  - 仮説は、
    generic key を shared override と競合させるのではなく、
    code 側で dedicated runner 専用の
    `MICRO_MULTI_HIST_SKIP_SCORE_OVERRIDE`
    を受ければ、
    dirty な shared env を触らずに
    `MicroLevelReactor`
    だけへ threshold
    `0.18`
    を効かせられる、
    というもの。

- Expected Good:
  - `ops/env/local-v2-stack.env`
    を触らずに
    `quant-micro-levelreactor`
    だけが
    `0.18`
    threshold を使う。
  - current winner setup / adjacent setup が
    shared env の後勝ちで潰されず、
    `hist_block`
    の再発を減らせる。

- Expected Bad:
  - dedicated override key を追加することで、
    micro runtime の history gate 設定面が 1 つ増える。
  - `MicroLevelReactor`
    family 全体 score
    `0.182`
    を通す副作用自体は残るため、
    weak setup 混入の監視は必要。

- Observed/Fact:
  - `workers/micro_runtime/config.py`
    に
    `MICRO_MULTI_HIST_SKIP_SCORE_OVERRIDE`
    を optional parse として追加した。
  - `workers/micro_runtime/worker.py`
    の
    `_history_profile`
    は
    override があるときだけ
    `skip_score_threshold`
    を差し替え、
    profile にも threshold を残すようにした。
  - 回帰は
    `tests/workers/test_micro_multistrat_trend_flip.py`
    に追加し、
    `score=0.182`
    / generic
    `0.20`
    / override
    `0.18`
    で
    `skip=False`
    になることを固定した。

- Verdict: pending

- Next Action:
  - test を通したうえで
    `quant-micro-levelreactor`
    と core 4 を再起動し、
    post-deploy で
    `hist_block ... score=0.182`
    が再発するか、
    `orders.db`
    の
    `MicroLevelReactor`
    preflight / filled
    が戻るかを確認する。

## 2026-03-13 15:25 JST / local-v2: family-level `hist_block` でも recent winner setup は止めない `setup_fingerprint` protect を追加

- Why/Hypothesis:
  - 直近24hの `MicroLevelReactor` では、
    `bounce-lower`
    の exact setup 内に
    `6 trades / +27.775 JPY / win_rate 1.0`
    の current winner cluster がある一方で、
    別 fingerprint は負けていた。
  - しかし micro runtime の history gate は
    `base_tag=MicroLevelReactor`
    の family score だけで
    `skip`
    を決めており、
    winner setup も loser family に巻き込まれて止まる構造だった。
  - 仮説は、
    family-level history gate は維持しつつ、
    exact
    `setup_fingerprint`
    の closed-trade score が recent winner 条件
    （`min_trades=2`, `score>=0.58`）
    を満たすときだけ
    `skip`
    を解除すれば、
    「勝ち筋は止めない」を broad loosening なしで実現できる、
    というもの。

- Expected Good:
  - `MicroLevelReactor`
    のように
    same signal tag 内で winner / loser setup が混ざる戦略でも、
    recent winner fingerprint は family-level `hist_block` で止まらない。
  - units multiplier は family history のまま残るため、
    winner lane を通しつつ sizing の暴れは抑えられる。

- Expected Bad:
  - setup history query が 1 段増えるため、
    micro runtime の history path は少し複雑になる。
  - sample 2-3 件の early winner も protect 対象に入りうるため、
    false-positive の監視は必要。

- Observed/Fact:
  - `workers/micro_runtime/worker.py`
    に
    exact setup 用の history query / cache を追加し、
    `setup_fingerprint`
    が recent winner のときだけ
    `winner_setup_override`
    を付けて
    `skip=False`
    にするようにした。
  - setup fingerprint は runtime signal から minimal thesis を組んで
    `derive_live_setup_context`
    で生成し、
    closed trades 側は
    `json_extract(entry_thesis, '$.setup_fingerprint')`
    と
    `$.live_setup_context.setup_fingerprint`
    の exact match で集計する。
  - 回帰は
    `tests/workers/test_micro_multistrat_trend_flip.py`
    に追加し、
    `2 wins`
    setup が
    `winner_protect=True`
    になることと、
    family `skip=True`
    が
    `global+setup_winner`
    へ変わることを固定した。

- Verdict: pending

- Next Action:
  - main へ反映して
    `quant-micro-levelreactor`
    を含む active micro workers を restart し、
    live log に
    `hist_winner_override`
    が出るか、
    もしくは current winner setup が
    family-level `hist_block`
    で止まらないことを確認する。

## 2026-03-13 15:50 JST - margin full は loser ではなく exact winner lane に寄せる

- Hypothesis Key:
  - `winner_lane_exact_sizing`
- Primary Loss Driver:
  - winner under-sizing / loser over-sizing
- Mechanism Fired:
  - exact
    `MicroLevelReactor-bounce-lower`
    setup override を worker sizing に直接反映した。
  - dedicated
    `quant-micro-levelreactor`
    の実効 margin cap を
    `0.985/0.995`
    へ引き上げ、
    winner lane にだけ full-margin 意図を通した。
- Do Not Repeat Unless:
  - winner lane の
    `avg_units`
    が still under-sized で、
    dominant loss が
    `under-participation`
    のまま残る時だけ、
    同系の sizing/margin 追加緩和を再実施する。

- Why/Hypothesis:
  - live account snapshot は
    `margin_used=0`
    / `free_margin_ratio=1.0`
    で、詰まりは margin 上限ではなく
    winner lane の under-sizing だった。
  - 直近24hでは
    `MicroLevelReactor`
    が
    `18 trades / +5.874 JPY / avg_units 135.7`
    で唯一プラス圏なのに、
    `DroughtRevert`
    は
    `19 trades / -4.62 JPY / avg_units 467.5`、
    `PrecisionLowVol`
    は
    `25 trades / -127.861 JPY / avg_units 593.2`
    と loser の方が大きかった。
  - さらに micro runtime の dynamic alloc lookup は
    exact
    `MicroLevelReactor-bounce-lower`
    を見ず、
    base
    `MicroLevelReactor`
    fallback しか引かないため、
    current winner key の boost が worker sizing に乗っていなかった。
  - dedicated micro env にあった
    `MICRO_MULTI_MAX_MARGIN_USAGE`
    も、
    actual lot cap を決める
    `execution.risk_guard.allowed_lot()`
    は global
    `MAX_MARGIN_USAGE`
    を読むため、
    winner runner の full-margin 意図が sizing path に届いていなかった。

- Expected Good:
  - `MicroLevelReactor-bounce-lower`
    の exact winner score
    （current artifact で
    `score=0.735`, `lot_multiplier=1.664`）
    が worker sizing に直接反映される。
  - dedicated
    `quant-micro-levelreactor`
    だけ実効 margin cap を
    `0.985/0.995`
    へ引き上げることで、
    loser scalp を broad に緩めずに winner lane の size だけ戻せる。
  - regenerated
    `dynamic_alloc`
    では strongest winner setup override が
    `lot_multiplier=1.85`
    まで上がるため、
    current loser lane より小さいままだった
    `MicroLevelReactor`
    の average units を反転させやすくなる。

- Expected Bad:
  - exact signal tag 単位の boost は sample が薄い lane でも乗りやすくなるため、
    false-positive winner の監視が必要。
  - dedicated winner runner の actual margin cap を上げるので、
    edge が崩れた場合の giveback 速度は上がる。
  - `MomentumBurst`
    側にも actual margin cap override は入るが、
    current live で winner 実績が薄い間は cadence だけ見て追う必要がある。

- Observed/Fact:
  - `workers/micro_runtime/worker.py`
    の
    `_strategy_profile_lookup_keys`
    は
    `signal_tag`
    が
    `strategy_name-...`
    で始まるとき、
    exact tag を first lookup に追加するよう修正した。
  - `ops/env/quant-micro-levelreactor.env`
    は
    `MICRO_MULTI_BASE_UNITS=18000`,
    `MICRO_MULTI_BASE_UNITS_EQUITY_SCALE_MIN/MAX=0.32`,
    `MICRO_MULTI_STRATEGY_UNITS_MULT=MicroLevelReactor:1.55`
    へ引き上げ、
    さらに actual sizing 用に
    `MAX_MARGIN_USAGE=0.985`,
    `MAX_MARGIN_USAGE_HARD=0.995`,
    `MARGIN_SAFETY_FACTOR=0.96`
    を追加した。
  - `ops/env/quant-micro-momentumburst.env`
    にも actual margin cap override を追加したが、
    base sizing は触っていない。
  - `config/dynamic_alloc.json`
    は
    `lookback_days=1`,
    `min_trades=8`,
    `setup_min_trades=2`,
    `half_life_hours=6`,
    `target_use=0.96`,
    `max_lot_multiplier=1.85`
    で再生成した。
  - `scripts/run_local_feedback_cycle.py`
    の
    `dynamic_alloc`
    default command も同じ
    `target_use=0.96`
    /
    `max_lot_multiplier=1.85`
    へ更新し、
    120 秒周期の background cycle が old params に戻さないようにした。
  - 再生成後、
    `load_strategy_profile('MicroLevelReactor-bounce-lower', 'micro')`
    は
    `found=True`, `lot_multiplier=1.664`
    を返し、
    strongest setup override も
    `lot_multiplier=1.85`
    になっていることを確認した。

- Verdict: pending

- Next Action:
  - pytest と local-v2 restart 後、
    first
    `MicroLevelReactor-bounce-lower`
    order で
    `entry_thesis.dynamic_alloc.strategy_key=MicroLevelReactor-bounce-lower`
    と larger
    `entry_units_intent`
    が出るか確認する。
  - live account の
    `margin_used`
    が上がる一方で、
    loser scalp lane の avg units が broad に増えていないことを追う。

### 2026-03-13 18:11 JST - loser lane は entry を足す前に strategy-local な inventory stress cleanup を持たせる

- Hypothesis Key:
  - `inventory_stress_cleanup`
- Primary Loss Driver:
  - before:
    `STOP_LOSS_ORDER`
  - after as-of 2026-03-13 19:54 JST:
    `STOP_LOSS_ORDER`
- Mechanism Fired:
  - as-of 2026-03-13 19:54 JST:
    `inventory_stress_exit=0`
  - `margin_health / free_margin_low / margin_usage_high / drawdown`
    close reason も
    `0`
    件
- Do Not Repeat Unless:
  - target lane で
    `inventory_stress_exit`
    が実際に発火した、
    または dominant
    `Primary Loss Driver`
    が
    `STOP_LOSS_ORDER`
    から stale loser / margin stress 系へ変わった時だけ再調整する。

- 対象:
  - `workers/scalp_wick_reversal_blend/exit_worker.py`
  - `workers/scalp_level_reject/exit_worker.py`
  - `config/strategy_exit_protections.yaml`
  - `tests/workers/test_scalp_wick_reversal_blend_exit_worker.py`
  - `tests/workers/test_scalp_level_reject_exit_worker.py`
  - `docs/TRADE_FINDINGS.md`
  - `docs/WORKER_REFACTOR_LOG.md`
  - `docs/CURRENT_MECHANISMS.md`

- Why/Hypothesis:
  - 2026-03-13 18:11 JST 時点の local/OANDA 実測では
    `USD/JPY mid=159.328`,
    `spread=0.8 pips`,
    `M1 ATR14=3.821 pips`,
    `6m range=8.1 pips`,
    `30m range=21.4 pips`,
    `pricing/candles latency=224.5/235.0ms`
    で、
    市況や API 停止よりも strategy expectancy の悪化が主因だった。
  - 直近 30 分の orders は
    `filled=57`,
    `submit_attempt=59`,
    `rejected=2`,
    `entry_probability_reject=100`
    で、
    order path は動いているが loser lane を抱えたまま entry 側だけを詰め続けていた。
  - 24h 実績は
    `162 trades / -476.8 JPY / win_rate 28.4% / PF 0.349`、
    6h でも
    `57 trades / -160.0 JPY / win_rate 21.1% / PF 0.186`
    と改善の兆しが見えない。
  - 24h loser 上位は
    `PrecisionLowVol -134.6 JPY (26 trades)`,
    `scalp_extrema_reversal_live -88.5 JPY (55 trades)`,
    `WickReversalBlend -83.9 JPY (9 trades)`。
  - 72h の loser hold 分布は
    `PrecisionLowVol p50/p75/p90 = 23.7s / 45.7s / 78.6s`,
    `WickReversalBlend = 53.9s / 100.3s / 569.9s`,
    `scalp_extrema_reversal_live = 31.7s / 70.6s / 116.9s`
    で、
    「長時間の勝ち待ち」より
    「数十秒で stale loser が積み上がる」性質が強い。
  - Git の履歴も、
    `2025-06-21` 開始から総コミット
    `2441`、
    `2026-03-04` 以降
    `376`、
    `2026-03-01` 以降
    `396`、
    subject prefix でも
    `fix=199 tune=30 feat=46 docs=41`
    と変更速度が観測速度を上回っていた。
  - `execution/order_manager.py`
    は新規 entry の margin/risk guard が厚い一方で、
    `execution/exit_manager.py`
    は stub のまま。
    ただし AGENTS の制約上、
    common の事後 EXIT 判定は増やせないため、
    stale loser cleanup は strategy-local な dedicated exit worker に閉じる必要がある。

- Expected Good:
  - `PrecisionLowVol` /
    `WickReversalBlend` /
    `scalp_extrema_reversal_live`
    が、
    margin closeout 近辺まで losing inventory を抱え込む前に、
    strategy-local に stale loser を掃除できる。
  - 新規 entry guard をさらに厚くしなくても、
    loser lane の tail だけを減らしやすくなる。
  - 将来 winner lane の participation / lot を戻すときに、
    「負け玉在庫の積み上がり」が先にボトルネックになる状態を緩められる。

- Expected Bad:
  - account stress 下では eventual recovery 候補も早めに切るので、
    一部の戻り玉は取り逃す。
  - stress threshold が tight すぎると、
    実質 hidden SL に近づいて cadence を削る。
  - `scalp_level_reject` 実装へ寄せたため、
    extrema 以外の同 family tag へ誤適用しないかは tag ごとの YAML 監視が必要。

- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/exit_worker.py`
    と
    `workers/scalp_level_reject/exit_worker.py`
    に、
    `exit_profile.inventory_stress`
    を読む strategy-local 分岐を追加した。
  - この分岐は
    `pnl<0`
    かつ
    `min_hold/loss_pips/max_hold`
    を満たした stale loser に対してだけ動き、
    さらに
    `health_buffer`,
    `free_margin_ratio`,
    `margin_usage_ratio`,
    `unrealized_dd_ratio`
    のどれかが閾値を超えたときだけ
    `margin_health / free_margin_low / margin_usage_high / drawdown`
    で close する。
  - `config/strategy_exit_protections.yaml`
    に
    `PrecisionLowVol`,
    `WickReversalBlend`,
    `scalp_extrema_reversal_live`
    の
    `inventory_stress`
    閾値を追加した。
    既存の broad common exit は増やしていない。
  - focused test として
    `pytest tests/workers/test_scalp_wick_reversal_blend_exit_worker.py tests/workers/test_scalp_level_reject_exit_worker.py`
    を実行し、
    `11 passed`
    を確認した。

- Verdict: pending

- Next Action:
  - local-v2 restart 後、
    `logs/orders.db`
    と
    `logs/trades.db`
    で
    `margin_health / free_margin_low / margin_usage_high / drawdown`
    の close reason が target 3 戦略にだけ出ているかを確認する。
  - 次の 6h / 24h で
    target 3 戦略の
    `net_jpy`,
    `PF`,
    `hold_sec tail`
    が縮むかを比較し、
    cadence を壊していないかも同時に見る。

## 2026-03-13 21:55 JST / local-v2: `DroughtRevert` と `WickReversalBlend` の current loser setup を worker-local guard で止める

- Hypothesis Key: `drought_wick_setup_local_guard_20260313`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `0`
- Do Not Repeat Unless:
  - 直近 6h 以上で
    `DroughtRevert`
    の
    `long|range_fade|...|gap:up_flat/down_flat|volatility_compression|macro:trend_long`
    か
    `WickReversalBlend`
    の
    `long|...|gap:*_lean|volatility_compression|macro:trend_long`
    が再び dominant loser になり、
    今回の local guard が live order path で発火しても
    `STOP_LOSS_ORDER`
    の寄与が縮まらない時だけ次の tightening を入れる。

- 対象:
  - `workers/scalp_wick_reversal_blend/worker.py`
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - `docs/TRADE_FINDINGS.md`
  - `docs/WORKER_REFACTOR_LOG.md`
  - `docs/CURRENT_MECHANISMS.md`

- Change:
  - `DroughtRevert`
    の flat-gap soft-trend long guard と
    `WickReversalBlend`
    の lean-gap long guard を
    `workers/scalp_wick_reversal_blend/worker.py`
    に追加する。

- Period:
  - primary window:
    `2026-03-12 12:46 UTC - 2026-03-13 12:46 UTC`
    （`2026-03-12 21:46 JST - 2026-03-13 21:46 JST`）
  - current drag window:
    `2026-03-13 06:46 UTC - 2026-03-13 12:46 UTC`
    （`2026-03-13 15:46 JST - 2026-03-13 21:46 JST`）

- Why/Hypothesis:
  - 2026-03-13 21:42 JST の preflight 実測では
    USD/JPY
    `bid=159.322 / ask=159.330 / spread=0.8 pips`,
    `M1 ATR14=3.0 pips`,
    `range_6m=3.6 pips`,
    `range_30m=14.0 pips`,
    `data_lag_ms=472.9`,
    `decision_latency_ms=74.5`
    で、
    市況・API 品質は通常帯だった。
  - corrected SQL
    （`datetime(close_time)` / `datetime(ts)`）
    での 24h 実績は
    `124 trades / net_jpy=-340.2 / win_rate=23.4% / PF=0.248`
    で、
    上位 drag は
    `DroughtRevert=-68.8 JPY`,
    `WickReversalBlend=-66.7 JPY`,
    `scalp_extrema_reversal_live=-56.5 JPY`
    だった。
  - 直近 6h でも
    `40 trades / -101.8 JPY / PF=0.155`
    で、
    `DroughtRevert=-30.0 JPY`,
    `WickReversalBlend=-32.2 JPY`
    が current loser。
  - loser cluster は side 名義ではなく setup で偏っており、
    `DroughtRevert`
    は
    `long|range_fade|tight_normal|rsi:oversold|atr:low|gap:up_flat|volatility_compression|macro:trend_long`
    が
    `2 trades / -23.6 JPY`
    、
    `WickReversalBlend`
    は
    `long|range_fade|tight_normal|rsi:mid|atr:mid|gap:up_lean|volatility_compression|macro:trend_long`
    が
    `2 trades / -11.3 JPY`
    、
    `long|range_compression|normal_normal|rsi:mid|atr:low|gap:down_lean|volatility_compression|macro:trend_long`
    が
    `2 trades / -10.4 JPY`
    を削っていた。
  - `WickReversalBlend`
    は shared trim
    （`lot_multiplier=0.72/0.803`, `probability_offset=-0.15`）
    が既に入っていても負けており、
    shared participation では止血が足りない。
    `DroughtRevert`
    も shared boost lane ではなく worker 側の long reclaim lane が損失源だった。

- Failure Cause:
  - `DroughtRevert`
    は
    `macro:trend_long`
    下の flat-gap reclaim long が、
    deep oversold continuation ではなく
    soft-trend / mid-oversold
    probe のまま通過して
    `STOP_LOSS_ORDER`
    を連発していた。
  - `WickReversalBlend`
    は shared trim 後も、
    `volatility_compression|adx_squeeze`
    の lean-gap long reclaim が
    worker local で拒否されず、
    same fingerprint の stop を積んでいた。

- Expected Good:
  - `DroughtRevert`
    の
    mid-oversold
    flat-gap
    `macro:trend_long`
    long probe と、
    `WickReversalBlend`
    の
    lean-gap
    long probe を worker-local に落とし、
    current loser setup だけを薄くできる。
  - winner 側の
    `DroughtRevert gap:up_lean`
    や
    `WickReversalBlend gap:up_flat`
    を broad stop せず、
    exact loser fingerprint 近傍だけを先に切れる。

- Expected Bad:
  - reclaim 初動の一部 winner も落とす可能性がある。
  - とくに
    `DroughtRevert`
    は flat-gap winner も混在するため、
    `rsi 42-46`,
    `adx<=12.5`,
    `projection<=0.10`,
    `setup_quality<0.52`,
    `reversion_support<0.60`
    の soft-trend lane に限定した。
  - `WickReversalBlend`
    は lean-gap long を絞るので、
    lean winner が出るなら別の stronger reclaim 条件で戻す必要がある。

- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/worker.py`
    に
    `DroughtRevert`
    の
    flat-gap soft-trend long guard
    と
    `WickReversalBlend`
    の
    lean-gap long guard
    を追加した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に
    current loser lane を block しつつ、
    deeper oversold reclaim lane は残す回帰を追加した。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> `34 passed`
    - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> 成功

- Improvement:
  - `DroughtRevert`
    は
    `rsi 42-46 / adx<=12.5 / projection<=0.10 / flat gap`
    の soft-trend long だけを reject する。
  - `WickReversalBlend`
    は
    `0.35 <= gap_ratio < 1.20`
    の lean-gap long だけを reject し、
    `gap:up_flat`
    の winner lane は維持する。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    -> `34 passed`
  - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    -> 成功

- Verdict: pending

- Status:
  - `pending`

- Next Action:
  - restart 後の
    `logs/orders.db`
    で
    `DroughtRevert`
    と
    `WickReversalBlend`
    の current loser fingerprint が
    `filled`
    から減り、
    `preflight_start`
    以降で消えるかを 30-60 分追う。
  - 6h/24h の
    `STOP_LOSS_ORDER`
    集計で
    `DroughtRevert`
    と
    `WickReversalBlend`
    の寄与が落ちるかを再確認する。

## 2026-03-13 22:15 JST / local-v2: `WickReversalBlend` の weak countertrend short を worker-local に遮断

- Hypothesis Key: `wick_short_countertrend_guard_20260313_2215`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `0`
- Do Not Repeat Unless:
  - 直近 6h 以上で
    `WickReversalBlend`
    の
    `short|range_fade|...|volatility_compression|macro:trend_long|align:countertrend`
    が再び dominant loser のまま残り、
    今回の short guard が live order path で発火しても
    `filled`
    と
    `STOP_LOSS_ORDER`
    寄与が縮まらない時だけ次の tightening を検討する。

- 対象:
  - `workers/scalp_wick_reversal_blend/worker.py`
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - `docs/TRADE_FINDINGS.md`
  - `docs/WORKER_REFACTOR_LOG.md`
  - `docs/CURRENT_MECHANISMS.md`

- Change:
  - `WickReversalBlend`
    の
    `volatility_compression`
    下 weak countertrend short を
    worker-local guard で reject する。

- Period:
  - primary window:
    `2026-03-12 22:15 UTC - 2026-03-13 13:15 UTC`
    （`2026-03-13 07:15 JST - 2026-03-13 22:15 JST`）
  - current drag window:
    `2026-03-13 07:15 UTC - 2026-03-13 13:15 UTC`
    （`2026-03-13 16:15 JST - 2026-03-13 22:15 JST`）

- Why/Hypothesis:
  - 2026-03-13 22:00 JST 前後の preflight 実測は
    USD/JPY
    `bid=159.324 / ask=159.332 / spread=0.8 pips`,
    `M1 ATR14=3.46 pips`,
    `range_6m=8.5 pips`,
    `range_30m=11.9 pips`,
    `data_lag_ms=814.1`,
    `decision_latency_ms=14.2`
    で、
    市況・API 品質は通常帯だった。
  - corrected SQL
    （`datetime(close_time)` / `datetime(ts)`）
    では
    `24h=123 trades / -313.0 JPY / PF=0.264`,
    `6h=41 trades / -121.0 JPY / PF=0.063`,
    `1h=5 trades / -16.9 JPY / PF=0.008`
    で、
    expectancy 悪化が継続していた。
  - `WickReversalBlend`
    の 30d countertrend short
    `range_fade`
    /
    `volatility_compression`
    /
    `macro:trend_long`
    /
    `align:countertrend`
    は
    `4 trades / -35.5 JPY / 0 wins`
    で、
    live current でも
    `gap:down_flat`
    が
    `2 trades / -21.3 JPY`
    と最悪だった。
  - 具体的な loser signature は
    `projection_score=0.215`,
    `wick_quality=0.684`,
    `rsi=55.5`,
    `adx=12.9`,
    `macd_hist_pips=0.175`
    で、
    short に対して bullish continuation headwind が残ったまま fade していた。

- Failure Cause:
  - `WickReversalBlend`
    short は、
    `volatility_compression`
    でも upper-wick rejection だけで通しており、
    projection / MACD 側の bullish headwind と
    stretch 不足
    （`rsi<=58`, `adx<=20`）
    が残る weak countertrend lane を worker local で落としていなかった。

- Expected Good:
  - `projection_score>=0.10`
    かつ
    `macd_hist_pips>=0.12`
    の weak countertrend short だけを落とし、
    連続
    `STOP_LOSS_ORDER`
    の main drag を止血できる。
  - stronger short
    （高 quality / 高 ADX / より stretched な short）
    は維持し、
    `WickReversalBlend`
    全停止にはしない。

- Expected Bad:
  - 弱い short rebound winner が混ざると entry を取りこぼす。
  - そのため guard は
    `projection>=0.10`,
    `quality<=0.78`,
    `rsi<=58`,
    `adx<=20`,
    `macd_hist_pips>=0.12`
    の同時成立に限定した。

- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/worker.py`
    に
    `_wick_blend_short_countertrend_blocked`
    を追加し、
    short
    `volatility_compression`
    lane を worker-local に reject するよう更新した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に
    helper block / keep と signal block / keep の回帰を追加した。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> `38 passed`
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_policy.py`
      -> `8 passed`
    - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> 成功

- Improvement:
  - `WickReversalBlend`
    short は
    `projection>=0.10`
    と
    positive
    `macd_hist`
    が残る weak countertrend fade を reject する。
  - long lean-gap guard と独立に効かせ、
    winner になりうる stronger short lane は維持する。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    -> `38 passed`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_policy.py`
    -> `8 passed`
  - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    -> 成功

- Verdict: pending

- Status:
  - `pending`

- Next Action:
  - restart 後の
    `logs/orders.db`
    で
    `WickReversalBlend`
    の
    `short|range_fade|...|volatility_compression|macro:trend_long|align:countertrend`
    が
    `filled`
    から消えるかを 30-60 分追う。
  - 6h/24h の
    `WickReversalBlend`
    `STOP_LOSS_ORDER`
    と
    `realized_pl`
    寄与が改善するかを再確認する。

## 2026-03-13 23:40 JST / local-v2: `PrecisionLowVol` の up-lean countertrend short を worker-local に遮断

- Hypothesis Key: `precision_up_lean_countertrend_guard_20260313_2340`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `0`
- Do Not Repeat Unless:
  - 直近 6h 以上で
    `PrecisionLowVol`
    の
    `short|...|gap:up_lean|macro:trend_long|align:countertrend`
    が再び dominant loser で、
    今回の guard が live order path で発火しても
    `filled`
    と
    `STOP_LOSS_ORDER`
    の寄与が縮まらない時だけ次の tightening を検討する。

- 対象:
  - `workers/scalp_wick_reversal_blend/worker.py`
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - `docs/TRADE_FINDINGS.md`
  - `docs/WORKER_REFACTOR_LOG.md`
  - `docs/CURRENT_MECHANISMS.md`

- Change:
  - `PrecisionLowVol`
    の
    `up_lean`
    countertrend short を
    worker-local guard で reject する。

- Period:
  - primary window:
    `2026-03-12 14:34 UTC - 2026-03-13 14:34 UTC`
    （`2026-03-12 23:34 JST - 2026-03-13 23:34 JST`）
  - current drag window:
    `2026-03-13 08:34 UTC - 2026-03-13 14:34 UTC`
    （`2026-03-13 17:34 JST - 2026-03-13 23:34 JST`）

- Why/Hypothesis:
  - 2026-03-13 23:34 JST の preflight 実測は
    USD/JPY
    `bid=159.152 / ask=159.160 / spread=0.8 pips`,
    `M1 ATR14=2.73 pips`,
    `range_6m=9.5 pips`,
    `range_30m=10.6 pips`,
    `data_lag_ms=1267.0`,
    `decision_latency_ms=169.7`,
    `fills_30m=7`,
    `rejects_30m=0`
    で、
    execution 劣化ではなく expectancy 悪化だった。
  - corrected SQL
    （`datetime(close_time)` / `datetime(ts)`）
    の 6h 実績は
    `DroughtRevert=-35.4`,
    `PrecisionLowVol=-25.5`,
    `WickReversalBlend=-10.2`
    で、
    直前に触った
    `WickReversalBlend`
    より
    `PrecisionLowVol`
    の current drag が重かった。
  - `orders.db`
    直近 1h では
    `PrecisionLowVol` が
    `preflight_start=4 / filled=4`
    と live order flow を持ち、
    `DroughtRevert`
    より現時点の改善余地が大きかった。
  - `PrecisionLowVol`
    の
    `short|...|gap:up_lean|macro:trend_long|align:countertrend`
    は 30d で
    `5 losses / 0 wins / -26.6 JPY`
    だった。
    直近 loser signature も
    `projection=0.135-0.275`,
    `setup_quality=0.225-0.281`,
    `reversion_support=0.323-0.569`,
    `continuation_pressure=0.229-0.418`,
    `rsi=48.9-53.2`,
    `adx=16.3-18.2`
    に集中していた。

- Failure Cause:
  - `PrecisionLowVol`
    short は
    `short_up_lean`
    lane に size boost を持つ一方、
    `macro:trend_long`
    へ逆らう weak countertrend fade を worker local で落としていなかった。
  - 既存 guard は
    `up_flat`
    や
    high-RSI
    headwind short を主に見ており、
    `rsi 49-53`
    /
    `setup_quality < 0.30`
    の
    `up_lean`
    loser lane が current で残っていた。

- Expected Good:
  - `up_lean`
    かつ
    `macro:trend_long / align:countertrend`
    の weak short だけを落とし、
    live で流れている
    `PrecisionLowVol`
    の
    `STOP_LOSS_ORDER`
    寄与を先に削れる。
  - `WickReversalBlend`
    をさらに触らず、
    「同じ family を緩めて強める」往復を避けて別 drag へ移れる。

- Expected Bad:
  - up-lean short の中に強い reversal winner が混ざると entry を取りこぼす。
  - そのため guard は
    `setup_quality<0.30`,
    `reversion_support<0.58`,
    `continuation_pressure>=0.22`,
    `rsi<=55`,
    `projection<=0.30`
    の同時成立に限定した。

- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/worker.py`
    に
    `PrecisionLowVol`
    の
    `up_lean countertrend short`
    guard を追加した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に
    block / keep の回帰を追加した。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> `40 passed`
    - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> 成功

- Improvement:
  - `PrecisionLowVol`
    の
    `short_up_lean`
    は、
    `macro:trend_long`
    に逆らう low-quality fade だけを reject する。
  - stronger short
    （quality / reversion_support / stretch が回復した lane）
    は維持する。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    -> `40 passed`
  - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    -> 成功

- Verdict: pending

- Status:
  - `pending`

- Next Action:
  - restart 後の
    `logs/orders.db`
    で
    `PrecisionLowVol`
    の
    `short|...|gap:up_lean|macro:trend_long|align:countertrend`
    が
    `filled`
    から消えるかを 30-60 分追う。
  - 6h/24h の
    `PrecisionLowVol`
    `STOP_LOSS_ORDER`
    と
    `realized_pl`
    寄与が改善するかを再確認する。

## 2026-03-13 23:53 JST / local-v2: `PrecisionLowVol` の up-flat shallow long を worker-local に遮断

- Hypothesis Key: `precision_up_flat_shallow_long_guard_20260313_2353`
- Primary Loss Driver: `STOP_LOSS_ORDER`
- Mechanism Fired: `0`
- Do Not Repeat Unless:
  - 直近 30-60 分で
    `PrecisionLowVol`
    の
    `long|...|gap:up_flat|volatility_compression|macro:trend_long`
    が再び live loser として並び、
    今回の guard が発火しても
    `filled`
    と
    `STOP_LOSS_ORDER`
    寄与が縮まらない時だけ次の tightening を検討する。

- 対象:
  - `workers/scalp_wick_reversal_blend/worker.py`
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
  - `docs/TRADE_FINDINGS.md`
  - `docs/WORKER_REFACTOR_LOG.md`
  - `docs/CURRENT_MECHANISMS.md`

- Change:
  - `PrecisionLowVol`
    の
    `up_flat`
    shallow long を
    worker-local guard で reject する。

- Period:
  - primary window:
    `2026-03-12 14:50 UTC - 2026-03-13 14:50 UTC`
    （`2026-03-12 23:50 JST - 2026-03-13 23:50 JST`）
  - live trigger window:
    `2026-03-13 14:30 UTC - 2026-03-13 14:50 UTC`
    （`2026-03-13 23:30 JST - 2026-03-13 23:50 JST`）

- Why/Hypothesis:
  - 2026-03-13 23:49 JST の preflight 実測は
    USD/JPY
    `bid=159.322 / ask=159.330 / spread=0.8 pips`,
    `M1 ATR14=2.81 pips`,
    `range_6m=5.4 pips`,
    `range_30m=24.1 pips`,
    `data_lag_ms=482.4`,
    `decision_latency_ms=35.7`,
    `fills_30m=9`,
    `rejects_30m=0`
    で、
    市況・execution は通常帯だった。
  - `DroughtRevert`
    は 6h 集計で drag だが、
    直近 90 分の
    `orders.db`
    では live flow が薄かった。
    一方
    `PrecisionLowVol`
    は直近 20 分で
    `long|...|gap:up_flat|volatility_compression|macro:trend_long`
    が
    `2 fills`
    と
    `tight_thin/tight_normal`
    の両 lane で live に流れていた。
  - `trades.db`
    直近 20 分では
    `PrecisionLowVol`
    の同 setup が
    `-3.0 / -2.9`
    の
    `STOP_LOSS_ORDER`
    を即時に出し、
    同じ 30d cluster でも
    `3 losses / 0 wins / -8.4 JPY`
    だった。
  - loser signature は
    `projection=0.0-0.275`,
    `setup_quality=0.317-0.497`,
    `reversion_support=0.313-0.710`,
    `continuation_pressure=0.162-0.254`,
    `rsi=44.7-52.4`,
    `adx=15.1-15.7`
    で、
    deep reclaim ではなく
    `up_flat`
    の shallow long probe に寄っていた。

- Failure Cause:
  - `PrecisionLowVol`
    long は
    deep oversold negative-projection long を既に reject していたが、
    `macro:trend_long`
    の
    `up_flat`
    shallow reclaim は通していた。
  - その結果、
    `rsi 44-52`
    /
    `projection<=0.30`
    /
    `setup_quality<0.52`
    の weak long fade が
    live current で stop を積んでいた。

- Expected Good:
  - `up_flat`
    shallow long だけを落とし、
    いま live に出ている
    `PrecisionLowVol`
    long loser を即座に止められる。
  - stronger reclaim long
    は残し、
    `PrecisionLowVol`
    long 全体を止めない。

- Expected Bad:
  - 同型の sample は 3 loss のみで薄いので、
    早計な overfit の可能性がある。
  - そのため guard は
    `macro:trend_long`,
    `up_flat`,
    `continuation_pressure<=0.28`,
    `rsi>=44`,
    `projection<=0.30`,
    `setup_quality<0.52`,
    `reversion_support<0.72`
    に限定し、
    `strong_reclaim_probe`
    が立つ long は維持した。

- Observed/Fact:
  - `workers/scalp_wick_reversal_blend/worker.py`
    に
    `PrecisionLowVol`
    の
    `up_flat shallow long`
    guard を追加した。
  - `tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    に
    block / keep の回帰を追加した。
  - 検証:
    - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> `42 passed`
    - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
      -> 成功

- Improvement:
  - `PrecisionLowVol`
    は
    `up_lean` short の次に、
    `up_flat`
    shallow long も worker-local に reject する。
  - これは
    `same family`
    の broad tighten ではなく、
    直近 20 分で live に発火した別 fingerprint だけを狙う。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    -> `42 passed`
  - `python3 -m py_compile workers/scalp_wick_reversal_blend/worker.py tests/workers/test_scalp_wick_reversal_blend_signal_flow.py`
    -> 成功

- Verdict: pending

- Status:
  - `pending`

- Next Action:
  - restart 後の
    `logs/orders.db`
    で
    `PrecisionLowVol`
    の
    `long|...|gap:up_flat|volatility_compression|macro:trend_long`
    が
    `filled`
    から消えるかを 20-30 分追う。
  - 同時に
    `PrecisionLowVol`
    short
    `gap:up_lean`
    と long
    `gap:up_flat`
    の両方で
    `STOP_LOSS_ORDER`
    が減るかを 6h 集計で見直す。

## 2026-03-14 19:45 JST / trade_findings: repo history lane の repeat-risk を preflight/index に統合し、週明けは 1 lane だけに固定

- Hypothesis Key:
  - `lane_repeat_risk_preflight_20260314`
- Primary Loss Driver:
  - repo history lane の repeat-risk と reopen single-focus を preflight に出さないと、
    same family の pending lane を並列に触り、
    anti-loop が docs 止まりになること
- Mechanism Fired:
  - `none`
  - `2026-03-14 18:10 JST`
    の cross-index 追加は
    `docs/REPO_HISTORY_LANE_INDEX.md`
    までで止まり、
    `scripts/change_preflight.sh`
    と
    `scripts/trade_findings_index.py`
    からは repeat-risk / focus lane が見えなかった。
- Do Not Repeat Unless:
  - week-open の最初の live task で still 複数 family を同時に触ろうとする、
    もしくは
    `recommended_single_focus_lane`
    が stale lane を返すと確認できるまでは、
    新しい governance field を増やさず、
    まず repeat-risk の recency / active-lane heuristics を調整する。

- Change:
  - `scripts/generate_repo_history_lane_index.py`
    は lane ごとの
    `repeat_risk`,
    family repeat,
    `recommended_single_focus_lane`
    を payload と markdown へ出すよう更新した。
  - `scripts/trade_findings_index.py`
    は latest hypothesis index / unresolved entry に
    `lane_family`,
    `history_commit_count`,
    `repeat_risk`
    を混ぜ、
    `recommended_single_focus_lane`
    も出すようにした。
  - `scripts/change_preflight.sh`
    は
    `logs/repo_history_lane_index_latest.{json,md}`
    を更新し、
    query 対応の repeat-risk summary と
    single-lane focus を
    `logs/change_preflight_latest.json`
    へ入れるようにした。

- Why:
  - cross-index が docs にしか無いと、
    実際の改善着手時に operator が別導線で history を見ない限り、
    anti-loop は運用 default にならない。

- Hypothesis:
  - repeat-risk と single-focus を preflight/index の標準 artifact に入れれば、
    week-open の最初の live 改善を
    `MomentumBurst`
    1 lane に固定しやすくなり、
    adjacent
    `STOP_LOSS_ORDER`
    family への並列 tweak を避けられる。

- Why Not Same As Last Time:
  - `2026-03-14 18:10 JST`
    の
    `repo history lane cross-index`
    は history navigation を追加しただけで、
    `change_preflight_latest.json`
    /
    `trade_findings_index_latest.json`
    へ repeat-risk や
    `recommended_single_focus_lane`
    を流していなかった。
  - 今回は same docs lane の焼き直しではなく、
    preflight / index artifact と reopen single-lane protocol を追加している。

- Expected Good:
  - `scripts/change_preflight.sh`
    の query 実行だけで
    repeat_risk,
    family repeat,
    `recommended_single_focus_lane`
    が見える。
  - `trade_findings_index_latest.json`
    から latest key / unresolved を見た時点で
    `lane_family`
    と
    `history_commit_count`
    が並ぶ。
  - 週明けの最初の live task を
    `momentumburst_transition_pullback_guard_20260314`
    1 本に固定しやすくなる。

- Expected Bad:
  - history heuristic が古い lane を重く見すぎると、
    本当に今触るべき recent lane の優先度を歪める。
  - そのため reopen focus は
    `current unresolved trading lane`
    を最優先にし、
    history count は tie-break に留める。

- Promotion Gate:
  - `scripts/change_preflight.sh "momentumburst_transition_pullback_guard_20260314" 3`
    が
    `lane_repeat_risk.matches`
    と
    `recommended_single_focus_lane`
    を
    `logs/change_preflight_latest.json`
    に残すこと。
  - `python3 scripts/trade_findings_index.py`
    が
    `recommended_single_focus_lane: momentumburst_transition_pullback_guard_20260314`
    を返すこと。
  - `2026-03-16 06:00 JST`
    以降の最初の live 改善で、
    `MomentumBurst`
    以外の
    `STOP_LOSS_ORDER`
    family を同時に触らないこと。

- Escalation Trigger:
  - reopen 後も複数 family を同時に触る運用が出る、
    または focus lane が stale legacy lane へ寄るなら、
    `active-window trades / fresh current_open lane`
    を scoring に追加する。

- Period:
  - `2026-03-14 19:20-19:45 JST`

- Fact:
  - `2026-03-14 19:35 JST`
    時点でも
    `logs/oanda_account_snapshot_live.json`
    は
    `2026-03-14 06:57:58 JST`
    で止まっており、
    土曜クローズ帯のため live verdict は hold のままだった。
  - `python3 scripts/generate_repo_history_lane_index.py --query "MomentumBurst STOP_LOSS_ORDER" --limit 3`
    は
    `MomentumBurst`
    を
    `repeat_risk=severe / history_commits=78`
    で single focus と返した。
  - `python3 scripts/trade_findings_index.py`
    は
    `recommended_single_focus_lane: momentumburst_transition_pullback_guard_20260314`
    を出力した。

- Failure Cause:
  - cross-index を docs に置いただけでは、
    着手前レビューの default action を変えられない。

- Improvement:
  - repeat-risk を preflight/index artifact に組み込み、
    reopen 最初の改善は
    `MomentumBurst`
    1 lane only
    を default にする。

- Verification:
  - `python3 scripts/generate_repo_history_lane_index.py --query "MomentumBurst STOP_LOSS_ORDER" --limit 3`
  - `python3 scripts/trade_findings_index.py`
  - `python3 -m py_compile scripts/generate_repo_history_lane_index.py scripts/trade_findings_index.py`
  - `scripts/change_preflight.sh "momentumburst_transition_pullback_guard_20260314" 3`

- Verdict:
  - pending

- Status:
  - `pending`

- Next Action:
  - `2026-03-16 06:00 JST`
    再開後の最初の live task では
    `momentumburst_transition_pullback_guard_20260314`
    だけを評価対象にし、
    `PrecisionLowVol`
    /
    `WickReversalBlend`
    /
    `DroughtRevert`
    の
    `STOP_LOSS_ORDER`
    family はその窓では触らない。
  - `30-120m`
    の reopen 窓で
    `MomentumBurst-open_long|...|transition`
    の
    `filled`
    /
    `STOP_LOSS_ORDER`
    が still dominant かを先に判定する。

## 2026-03-14 13:02 JST / repo history docs: 週末クローズ帯のため live 判定は hold、docs/script は offline 継続

- Hypothesis Key:
  - `weekend_close_docs_only_hold_20260314`
- Primary Loss Driver:
  - weekend close の stale window を runtime fault と誤認し、
    docs/script-only task まで止めてしまうこと
- Mechanism Fired:
  - `none`
  - `2026-03-14 12:58-13:02 JST`
    の着手前チェックでは、
    cache stale の主因が土曜クローズ帯であることを確認した。
- Do Not Repeat Unless:
  - reopen 後も
    `tick_cache / factor_cache / oanda_account_snapshot_live.json`
    が更新再開せず、
    close window ではなく runtime fault が継続していると確認できるまでは、
    docs/script-only task を hold 側へ寄せない。

- Change:
  - `docs/REPO_HISTORY_*`
    と
    `scripts/generate_repo_history_minutes.py`
    の docs/script-only task については、
    live restart / live verdict を行わず、
    offline の commit / push までは進める運用ログとして残した。

- Why:
  - stale をすべて障害扱いすると、
    週末 close window の docs/script-only task まで止まり、
    runtime 由来ではない作業も進まなくなる。

- Hypothesis:
  - close/stale window を
    `market_hold`
    と明示し、
    docs/script-only task は offline 継続に分ければ、
    live judgment を汚さずに履歴整備だけ進められる。

- Why Not Same As Last Time:
  - この entry は runtime 改善ではなく、
    `2026-03-14`
    土曜クローズ帯での docs/script-only task を
    `market_hold`
    として切り分けた運用記録である。

- Expected Good:
  - weekend close を runtime fault と混同せずに済む。
  - docs/script-only task の commit / push を止めすぎない。

- Expected Bad:
  - 本当に runtime fault が混ざっていた場合に、
    docs-only という理由で見落とすリスクがある。
  - そのため reopen 条件を明示し、
    live restart は週明けまで持ち越す。

- Promotion Gate:
  - `tick_cache`
    と
    `factor_cache`
    と
    `oanda_account_snapshot_live.json`
    が更新再開し、
    weekend hold を解除できること。

- Escalation Trigger:
  - `2026-03-16 06:00 JST`
    以降も stale が続き、
    `pricing/stream`
    の
    `200 OK`
    後に cache 更新が戻らないなら、
    docs-only hold ではなく runtime RCA へ切り替える。

- Period:
  - 確認: JST `12:58-13:02`
  - 対象（実測）:
  `logs/health_snapshot.json`,
  `logs/tick_cache.json`,
  `logs/factor_cache.json`,
  `logs/oanda_account_snapshot_live.json`,
  `logs/local_v2_stack/quant-market-data-feed.log`
  - 対象タスク:
  `docs/REPO_HISTORY_*`
  と
  `scripts/generate_repo_history_minutes.py`
  の commit / push

- Fact:
- `scripts/collect_local_health.sh`
  は
  `2026-03-14 12:58:27 JST`
  に
  `logs/health_snapshot.json`
  を更新し、
  `mechanism_integrity=yes`
  を確認した。
- 直近の
  `tick_cache`
  は
  `2026-03-14 05:59:05 JST`
  で停止しており、
  `USD/JPY bid=159.727 / ask=159.739 / spread=1.2p`
  のまま
  `tick_age_sec=25351.3`
  だった。
- `factor_cache`
  も
  `M1=2026-03-14 05:59 JST`
  (`ATR14=0.995p`),
  `M5=2026-03-14 05:55 JST`
  (`ATR14=2.565p`)
  で止まっており、
  `m1_age_sec=25356.4`,
  `m5_age_sec=25596.4`
  だった。
- `logs/oanda_account_snapshot_live.json`
  も
  `2026-03-14 06:57:58 JST`
  で停止し、
  `age_sec=21817.8`
  だった。
- `quant-market-data-feed.log`
  では
  `2026-03-14 08:07:41-08:08:33 JST`
  に
  OANDA
  `pricing/stream`
  の
  `503 Service Unavailable`
  再接続が連続した。
  その後
  `2026-03-14 08:08:36 JST`
  と
  `2026-03-14 11:49:58 JST`
  に
  `HTTP 200 OK`
  は確認できたが、
  上記 cache / snapshot は更新再開していない。
- 直近 30 分の
  `orders.db`
  は
  `fills_30m=0`,
  `rejects_30m=0`
  だった。
- `2026-03-14`
  は
  `Saturday`
  で、
  USD/JPY spot の週末クローズ帯に当たる。
  次の通常再開は
  `2026-03-16 06:00 JST`
  （`2026-03-15 17:00 America/New_York`）
  だった。

- Failure Cause:
- stale に見えた主因は障害継続ではなく、
  `2026-03-14` 土曜の週末クローズ帯で
  live market data が進まないことだった。
- `pricing/stream` の
  `503`
  は補助的なノイズとして記録するが、
  現時点で commit/push の blocker を
  「runtime fault 復旧待ち」とみなすのは過剰。
- AGENTS の着手前チェック要件では、
  close/stale window の
  live 判定や live 反映確認は hold とする。

- Improvement:
- 本タスクは
  `docs/REPO_HISTORY_*`
  と
  `scripts/generate_repo_history_minutes.py`
  の
  docs/script-only 変更なので、
  live restart / live verdict は行わず、
  offline の commit / push までは進める。
- 運用ログ
  `logs/ops_v2_audit_20260314_1302_git_hold.json`
  には
  close window の hold 判断を退避し、
  live 再開判定は週明けへ持ち越す。

- Verification:
  - 再開条件:
  - `tick_cache` が 300 秒以内に更新されること
  - `factor_cache` の `M1/M5` timestamp が進むこと
  - `oanda_account_snapshot_live.json` が再更新されること
  - `quant-market-data-feed.log` で `pricing/stream` の `503` が収束し、
    `200 OK` 後に cache 更新が再開すること
- 今回の docs/script タスクでは、
  上記の live 再開条件を待たずに
  commit / push を進めてよい。

- Verdict:
  - pending

- Status:
  - live_hold / docs_only_proceed

- Next Action:
  - `2026-03-16 06:00 JST`
    以降に
    `tick_cache`
    /
    `factor_cache`
    /
    `oanda_account_snapshot_live.json`
    の更新再開を確認し、
    live verdict を reopen する。

## 2026-03-14 20:35 JST / local-v2: `lane_scoreboard` で lane 単位の promotion gate / auto quarantine を明示し、`participation_alloc` へ接続

- Hypothesis Key:
  - `lane_scoreboard_promotion_quarantine_20260314`
- Primary Loss Driver:
  - winner lane の昇格と loser lane の隔離が
    `participation_alloc`
    の内部 heuristics に埋もれており、
    current lane 単位の意思決定を監査・再利用しづらいこと
- Mechanism Fired:
  - `lane_scoreboard`
  - `promotion_gate`
  - `auto_quarantine`
  - `participation_alloc.setup_overrides`
  - `run_local_feedback_cycle`
- Do Not Repeat Unless:
  - lane ごとの
    `promotion/quarantine`
    が artifact に露出せず、
    または
    `participation_alloc`
    が setup override として消費できないまま残る場合を除き、
    同じ shared gate を新設する方向には戻さない。

- Change:
  - `scripts/lane_scoreboard.py`
    を追加し、
    `logs/entry_path_summary_latest.json`
    と
    `logs/trades.db`
    から
    setup-scoped lane の
    `fills / share_gap / hard_block_rate / realized_jpy / win_rate / profit_factor / stop_loss_rate`
    を集計して、
    `logs/lane_scoreboard_latest.json`
    と
    `logs/lane_scoreboard_history.jsonl`
    を生成するようにした。
  - `scripts/participation_allocator.py`
    は
    `lane_scoreboard`
    の
    `boost_participation`
    /
    `trim_units`
    を setup override として merge し、
    strategy-wide conversion と lane-specific gate を同時に扱えるようにした。
  - `scripts/run_local_feedback_cycle.py`
    に
    `lane_scoreboard`
    job を追加し、
    `entry_path_aggregator -> lane_scoreboard -> participation_allocator`
    の順で local feedback cycle に載せた。

- Why:
  - 今の repo は strategy 全体より
    current lane ごとの winner / loser 分離が重要で、
    「どの型を太らせ、どの型を細らせたか」
    を artifact 化しないと anti-loop の運用が弱い。

- Hypothesis:
  - lane scoreboard を explicit に出し、
    promotion / quarantine を
    `participation_alloc`
    の setup override へ渡せば、
    shared blanket gate を増やさずに
    current winner lane の昇格と fresh/chronic loser lane の隔離を両立できる。

- Why Not Same As Last Time:
  - これは
    `repeat_risk`
    の preflight 表示を増やしただけの変更ではなく、
    `orders.db / trades.db`
    の current setup 実測から
    lane 単位の
    `promotion_gate / auto_quarantine`
    を runtime artifact へ落とした点が違う。

- Expected Good:
  - winner lane を
    `boost_participation`
    として明示的に昇格できる。
  - loser lane を strategy 全体停止ではなく
    setup override の
    `trim_units / probability_offset`
    で隔離できる。
  - `participation_alloc`
    の判断根拠を
    `lane_scoreboard`
    から追える。

- Expected Bad:
  - lane scoreboard の threshold が浅いと、
    noisy sample を早く昇格/隔離しすぎるリスクがある。
  - そのため strategy-wide override は残しつつ、
    scoreboard 側は setup override に限定して merge する。

- Promotion Gate:
  - market reopen 後に
    `lane_scoreboard_latest.json`
    で current lane の
    `promotion/quarantine`
    が生成され、
    `config/participation_alloc.json`
    の setup override と整合すること。

- Escalation Trigger:
  - reopen 後も current loser lane が
    `hold`
    のまま残る、
    または winner lane が strategy-wide loser に潰されるなら、
    threshold 調整ではなく
    lane identity / closed-trade join / lookback 設計を再検討する。

- Period:
  - 実装・検証:
    `2026-03-14 19:58-20:35 JST`
  - 対象:
    `scripts/lane_scoreboard.py`,
    `scripts/participation_allocator.py`,
    `scripts/run_local_feedback_cycle.py`,
    `tests/scripts/test_lane_scoreboard.py`,
    `tests/scripts/test_participation_allocator.py`,
    `tests/scripts/test_run_local_feedback_cycle.py`

- Fact:
  - `lane_scoreboard`
    は
    setup-scoped lane ごとに
    `promotion_gate`
    と
    `quarantine_gate`
    を持ち、
    `boost_participation / trim_units / hold`
    を explicit に出すようになった。
  - `participation_allocator`
    は
    strategy-wide conversion で作る setup override に加えて、
    `lane_scoreboard`
    の同一 setup key override を merge し、
    scoreboard 側の lane decision を優先する。
  - `run_local_feedback_cycle`
    は
    `lane_scoreboard`
    job を既定 ON で持ち、
    `logs/lane_scoreboard_latest.json`
    を local feedback artifact として扱う。

- Failure Cause:
  - 既存の
    `participation_alloc`
    は結果として winner/loser を押し引きしていたが、
    lane 単位の gate が explicit artifact ではなかったため、
    current lane の昇格/隔離を review しにくかった。

- Improvement:
  - lane scoreboard を追加し、
    runtime 側は新しい shared gate ではなく
    existing `participation_alloc`
    setup override へ接続する構成へ整理した。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/scripts/test_lane_scoreboard.py tests/scripts/test_participation_allocator.py tests/scripts/test_run_local_feedback_cycle.py`
  - `python3 -m py_compile scripts/lane_scoreboard.py scripts/participation_allocator.py scripts/run_local_feedback_cycle.py`
  - `python3 scripts/lane_scoreboard.py --output /tmp/qr_lane_scoreboard.json --history /tmp/qr_lane_scoreboard.jsonl`
  - `python3 scripts/participation_allocator.py --lane-scoreboard /tmp/qr_lane_scoreboard.json --output /tmp/qr_participation_alloc.json`

- Verdict:
  - pending

- Status:
  - market_hold / offline_verified

- Next Action:
  - `2026-03-16 06:00 JST`
    以降に
    `lane_scoreboard_latest.json`
    と
    `config/participation_alloc.json`
    を reopen 窓で確認し、
    `MomentumBurst`
    の single-focus lane が
    `promote`
    か
    `quarantine`
    のどちらへ寄るかを first check にする。

## 2026-03-14 20:55 JST / docs: 日次 `winner lane review` 用の固定 prompt を repo に追加

- Hypothesis Key:
  - `winner_lane_review_prompt_daily_20260314`
- Primary Loss Driver:
  - 毎日の lane review を都度その場で書くと、
    参照 artifact と判定軸が揺れて
    `promote / hold / quarantine / graduate_to_strategy`
    の判断がぶれやすいこと
- Mechanism Fired:
  - `none`
  - docs/process-only change
- Do Not Repeat Unless:
  - 固定 prompt では拾えない review 観点が明確に出るまでは、
    別の ad-hoc prompt を増やさず
    `docs/prompts/WINNER_LANE_REVIEW_DAILY.md`
    を更新する。

- Change:
  - `docs/prompts/WINNER_LANE_REVIEW_DAILY.md`
    を追加し、
    `change_preflight`
    /
    `trade_findings_index`
    /
    `lane_scoreboard`
    /
    `participation_alloc`
    /
    `REPO_HISTORY_LANE_INDEX`
    を前提にした daily review prompt を固定化した。
  - `docs/AGENT_COLLAB_HUB.md`
    と
    `docs/INDEX.md`
    から辿れるようにした。

- Why:
  - 毎日やる review は、その場の思いつきではなく
    同じ入力と同じ出力形式で回した方がよい。

- Hypothesis:
  - fixed prompt を repo に置けば、
    daily review の品質と比較可能性が上がる。

- Why Not Same As Last Time:
  - これは
    `lane_scoreboard`
    本体の実装ではなく、
    その artifact を毎日どう読むかを固定する docs/process change である。

- Expected Good:
  - daily review の観点が揺れにくくなる。
  - `single next action`
    を 1 本に絞る運用を維持しやすくなる。

- Expected Bad:
  - prompt が固定されすぎると、
    新しい failure mode を拾いにくくなる。
  - その場合は prompt を更新し、
    別 prompt を乱立させない。

- Promotion Gate:
  - 次回以降の review が
    `docs/prompts/WINNER_LANE_REVIEW_DAILY.md`
    を参照して行われ、
    `promote / hold / quarantine / graduate_to_strategy`
    の4区分で返ること。

- Escalation Trigger:
  - daily review で必要な artifact や判定欄が不足するなら、
    prompt の修正ではなく
    `lane_scoreboard / trade_findings_index`
    側の出力を増やす。

- Period:
  - 実装:
    `2026-03-14 20:45-20:55 JST`
  - 対象:
    `docs/prompts/WINNER_LANE_REVIEW_DAILY.md`,
    `docs/AGENT_COLLAB_HUB.md`,
    `docs/INDEX.md`

- Fact:
  - repo 内に
    `winner lane review`
    専用の fixed prompt はまだ無かった。
  - 追加後は
    `docs/AGENT_COLLAB_HUB.md`
    と
    `docs/INDEX.md`
    から参照可能になった。

- Failure Cause:
  - review の観点が docs として固定されていなかった。

- Improvement:
  - daily review 用 prompt を repo に追加し、
    参照導線も docs へ組み込んだ。

- Verification:
  - docs 追加のみのため runtime test はなし。
  - 参照導線は
    `docs/AGENT_COLLAB_HUB.md`
    と
    `docs/INDEX.md`
    で確認した。

- Verdict:
  - good

- Status:
  - docs_ready

- Next Action:
  - 次回の daily review から
    `docs/prompts/WINNER_LANE_REVIEW_DAILY.md`
    をそのまま使う。

## 2026-03-15 23:35 JST / preflight: weekend close を `market_closed_hold` として明示し、stale fault と混同しない

- Hypothesis Key:
  - `weekend_close_preflight_hold_20260315`
- Primary Loss Driver:
  - weekend close の stale cache を runtime fault と誤認し、
    reopen 前の改善優先度を誤ること
- Mechanism Fired:
  - `none`
  - `2026-03-15 23:06 JST`
    の `change_preflight`
    は日曜クローズ帯なのに
    `tick_stale=148035.7s`
    /
    `data_lag_high=15263.3ms`
    をそのまま warning に載せていた。
- Do Not Repeat Unless:
  - `2026-03-16 07:00 JST`
    以降の reopen 後も
    `market_open=true`
    に戻らず、
    `tick_cache / factor_cache / oanda_account_snapshot_live.json`
    の更新再開が確認できないときだけ、
    weekend hold ではなく runtime RCA を再度開く。

- Change:
  - `scripts/change_preflight.sh`
    に
    `utils.market_hours.is_market_open()/seconds_until_open()`
    を組み込み、
    closed 窓では
    `preflight_status=market_closed_hold`
    と
    `warnings=market_closed:*_to_open`
    を返すようにした。
  - `scripts/improvement_gate.py`
    は
    `artifact.market.market_open=false`
    を見て
    `market_hold_review_only`
    を返すようにした。
  - `tests/scripts/test_improvement_gate.py`
    に
    closed 窓判定の回帰テストを追加した。

- Why:
  - 週末クローズ中の stale tick は通常故障ではない。
  - それを runtime fault と同じ警告で出すと、
    pending lane の再検証待ちと
    本当の障害対応が混ざる。

- Hypothesis:
  - preflight / proposal gate が
    closed 窓を
    `market_closed_hold`
    と明示すれば、
    reopen 前の false triage を減らせる。

- Why Not Same As Last Time:
  - `weekend_close_docs_only_hold_20260314`
    は docs/script-only task の hold 運用記録だった。
  - 今回は
    `change_preflight.sh`
    と
    `scripts/improvement_gate.py`
    の executable 判定自体を変更しており、
    decision surface は
    `docs-only運用`
    ではなく
    `improvement preflight runtime governance`
    である。

- Expected Good:
  - closed 窓では
    `market_open=no`
    /
    `seconds_until_open=*`
    /
    `market_closed_hold`
    が明示され、
    stale cache を理由に新規 tweak を始めにくくなる。
  - reopen 後にだけ
    本当の stale fault を見に行ける。

- Expected Bad:
  - 本当に closed 窓中に runtime fault が混ざっていても、
    `market_closed`
    という大きなラベルで一段隠れる可能性がある。
  - そのため
    `mechanism_integrity_fail`
    は別 warning として残し、
    reopen 後の Promotion Gate を明示する。

- Promotion Gate:
  - reopen 後の
    `change_preflight`
    で
    `market_open=true`
    になり、
    `market_closed:*_to_open`
    が消えること。

- Escalation Trigger:
  - `2026-03-16 07:00 JST`
    を過ぎても
    `market_open=false`
    のままか、
    `market_open=true`
    に戻った後も
    `tick_stale>300s`
    か
    `data_lag_ms>1500`
    が継続すること。

- Period:
  - 実装/検証:
    `2026-03-15 23:20-23:35 JST`
  - 対象:
    `scripts/change_preflight.sh`,
    `scripts/improvement_gate.py`,
    `tests/scripts/test_improvement_gate.py`,
    `logs/change_preflight_latest.json`,
    `logs/improvement_gate_latest.{json,md}`

- Fact:
  - `date`
    実測は
    `2026-03-15 23:27 JST (Sunday)`。
  - `tick_cache.json`
    の最終 tick は
    `2026-03-14 05:59 JST`
    で、
    土曜クローズ帯の停止と整合していた。
  - 修正後の
    `change_preflight`
    は
    `market_open=no`
    /
    `seconds_until_open=26951.5`
    /
    `preflight_status=market_closed_hold`
    を返した。

- Failure Cause:
  - preflight が
    `market_hours`
    を見ずに
    stale tick / lag を常に runtime warning として扱っていた。

- Improvement:
  - closed 窓では
    `market_closed_hold`
    を first-class な hold 理由として扱うようにした。

- Verification:
  - `PYTHONPATH=. python3 -m pytest -q tests/scripts/test_improvement_gate.py`
    -> `6 passed`
  - `scripts/change_preflight.sh "weekend close hold verification" 3`
    -> `market_open=no`, `preflight_status=market_closed_hold`

- Verdict:
  - good

- Status:
  - tooling_ready

- Next Action:
  - `2026-03-16 07:00 JST`
    以降に
    `change_preflight`
    を再実行し、
    `market_open=true`
    と cache 更新再開を確認してから
    pending lane の live reopen を行う。

## 2026-03-16 08:22 JST / local-v2: autorecover default profile を `trade_min` に揃え、ENOSPC でも worker 復旧を止めない

- Hypothesis Key:
  - `trade_min_autorecover_profile_parity_20260316`
- Primary Loss Driver:
  - `trade_min`
    専用 worker が launchd autorecover の監視対象から外れ、
    停止後に復帰しないこと
- Mechanism Fired:
  - `scripts/status_local_v2_launchd.sh`
    実測で、
    launchd agent は
    `watchdog --once --profile 'trade_cover'`
    を実行していた。
  - 同時刻の
    `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    では
    `quant-scalp-precision-lowvol`,
    `quant-scalp-vwap-revert`,
    `quant-scalp-drought-revert`,
    `quant-session-open`
    が `stale_pid_file` で停止していた。
  - `logs/local_v2_autorecover.log`
    には
    `2026-03-16 08:11 JST`
    / `08:12 JST`
    の
    `scripts/local_v2_stack.sh: line 1115/913/827: cannot create temp file for here document`
    が残っていた。
- Do Not Repeat Unless:
  - launchd の configured profile が
    `trade_min`
    で、
    `local_v2_autorecover.log`
    から here-doc temp file failure が消えた後も、
    同じ `trade_min` worker 群が stop のまま残るときだけ
    次の persistence RCA を開く。

- Change:
  - `scripts/local_v2_stack.sh`
    の recovery/status critical path で使っていた
    bash here-doc を process substitution へ置き換え、
    temp-file-free にした。
  - `scripts/local_v2_autorecover_once.sh`
    の
    `market_sanity_ready`
    は
    `/tmp`
    一時ファイルを使わずに理由を受けるようにした。
  - `scripts/local_v2_watchdog.sh`,
    `scripts/local_v2_autorecover_once.sh`,
    `scripts/install_local_v2_launchd.sh`
    の既定 profile を
    `trade_min`
    へ統一した。
  - `scripts/status_local_v2_launchd.sh`
    は
    `configured_profile`
    を表示し、
    `trade_min`
    以外を drift warning として出すようにした。
  - launchd agent を
    `--profile trade_min`
    で再インストールし、
    `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env`
    で stop していた worker を復旧した。

- Why:
  - 現行実運用は
    `trade_min`
    だが、
    常駐 autorecover が
    `trade_cover`
    を監視していると、
    `trade_min`
    専用 worker は停止しても自動復旧しない。
  - 加えて、
    recovery/status 自体が shell temp file へ依存すると、
    空き容量逼迫時に
    `stack up`
    が失敗して stop 状態を長引かせる。

- Hypothesis:
  - autorecover / watchdog / launchd install の default profile を
    `trade_min`
    に揃え、
    `local_v2_stack.sh`
    の critical path を temp-file-free にすれば、
    今回のような
    `trade_min` worker の silent stop を再発させにくくできる。

- Why Not Same As Last Time:
  - `2026-03-12 20:58 JST`
    /
    `21:05 JST`
    の entry は、
    stop していた dedicated worker を前提に
    `strategy_feedback`
    の active discovery を直したものだった。
  - 今回の decision surface は
    `strategy_feedback coverage`
    ではなく、
    launchd / watchdog /
    `local_v2_stack`
    の
    `worker persistence contract`
    そのもの。

- Expected Good:
  - launchd / watchdog の default だけで
    `trade_min`
    専用 worker が復旧対象から漏れなくなる。
  - 低空き容量時も、
    here-doc temp file failure だけで
    `status/up/reconcile`
    が止まりにくくなる。
  - `scripts/status_local_v2_launchd.sh`
    の warning で profile drift を早く見つけられる。

- Expected Bad:
  - filesystem が完全に枯渇した場合は、
    log 書き込みや plist 更新まで含めて別の
    disk RCA
    が必要になる。
  - `trade_min`
    に default を寄せたため、
    `trade_cover`
    を意図的に使う端末では明示
    `--profile trade_cover`
    が必要になる。

- Promotion Gate:
  - `scripts/status_local_v2_launchd.sh`
    が
    `configured_profile=trade_min`
    を返し、
    `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    で
    `trade_min`
    全 service が `running`
    を返すこと。

- Escalation Trigger:
  - `local_v2_autorecover.log`
    に
    `cannot create temp file for here document`
    が再出現すること。
  - または、
    launchd が
    `trade_min`
    に揃った後も
    同じ worker 群が
    `stopped/stale_pid_file`
    のまま残ること。

- Period:
  - 調査:
    `2026-03-16 08:04-08:18 JST`
  - 実装/反映:
    `2026-03-16 08:18-08:22 JST`
  - 対象:
    `scripts/local_v2_stack.sh`,
    `scripts/local_v2_watchdog.sh`,
    `scripts/local_v2_autorecover_once.sh`,
    `scripts/install_local_v2_launchd.sh`,
    `scripts/status_local_v2_launchd.sh`,
    `tests/scripts/test_local_v2_launchd_scripts.py`

- Fact:
  - `2026-03-16 08:05 JST`
    の
    `scripts/status_local_v2_launchd.sh`
    は
    `configured_profile=trade_cover`
    と drift warning を返した。
  - 同時刻の
    `trade_min`
    status では
    `quant-scalp-precision-lowvol`,
    `quant-scalp-vwap-revert`,
    `quant-scalp-drought-revert`,
    `quant-session-open`
    が停止していた。
  - `2026-03-16 08:22 JST`
    に launchd を再インストール後、
    `scripts/status_local_v2_launchd.sh`
    は
    `configured_profile=trade_min`
    を返した。
  - 同時刻の
    `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env`
    は stop していた 4 worker と exit worker を起動し、
    直後の status で全 service が
    `running`
    を返した。

- Failure Cause:
  - launchd autorecover の configured profile drift
    （`trade_cover`）
    と、
    `local_v2_stack.sh`
    / `local_v2_autorecover_once.sh`
    の shell temp file 依存が重なって、
    `trade_min`
    worker の stop を長引かせた。

- Improvement:
  - autorecover / watchdog / launchd install を
    `trade_min`
    既定へ統一し、
    recovery/status critical path から temp-file 依存を除去した。

- Verification:
  - `bash -n scripts/local_v2_stack.sh scripts/local_v2_watchdog.sh scripts/local_v2_autorecover_once.sh scripts/install_local_v2_launchd.sh scripts/status_local_v2_launchd.sh`
  - `PYTHONPATH=. python3 -m pytest -q tests/scripts/test_local_v2_launchd_scripts.py`
    -> `2 passed`
  - `scripts/status_local_v2_launchd.sh`
    変更前:
    `configured_profile=trade_cover`
  - `scripts/install_local_v2_launchd.sh --profile trade_min --env ops/env/local-v2-stack.env`
  - `scripts/status_local_v2_launchd.sh`
    変更後:
    `configured_profile=trade_min`
  - `scripts/local_v2_stack.sh up --profile trade_min --env ops/env/local-v2-stack.env`
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env`
    -> 全 service `running`

- Verdict:
  - good

- Status:
  - done

- Next Action:
  - `local_v2_autorecover.log`
    を reopen 後 1 セッション監視し、
    `configured_profile=trade_min`
    のまま
    `trade_min`
    専用 worker が stop しないことを確認する。

## 2026-03-16 09:28 JST / local-v2: micro runtime の cooldown は `dispatch success` 時だけ進め、`res=None` で cadence を落とさない

- Hypothesis Key:
  - `micro_runtime_dispatch_success_cooldown_20260316`
- Primary Loss Driver:
  - `participation_loss`
- Mechanism Fired:
  - `dispatch_none_cooldown_burn`
  - `workers/micro_runtime/worker.py`
    は
    `market_order()`
    直後に
    `_STRATEGY_LAST_TS`
    を無条件更新していた。
    さらに
    `logs/local_v2_stack/quant-micro-momentumburst.log`
    には
    `2026-03-06 07:18 JST`
    と
    `18:06 JST`
    の
    `res=none`
    が残っており、
    no-fill path が実際に存在した。
- Do Not Repeat Unless:
  - post-deploy の live log で
    `res=none cooldown_updated=0`
    を確認した後も、
    micro runtime worker が still no-fill path で cadence を失っているときだけ、
    cooldown key 分離や reject-family 別の cadence 制御へ進む。

- Change:
  - `workers/micro_runtime/worker.py`
    に
    `_record_strategy_dispatch()`
    を追加し、
    truthy dispatch result のときだけ
    `_STRATEGY_LAST_TS`
    を更新するようにした。
  - 既存の
    `sent units ... res=...`
    log に
    `cooldown_updated=0/1`
    を足し、
    dispatch の成否と cooldown 消費を同時に監査できるようにした。
  - `tests/strategies/test_momentum_burst.py`
    に
    dispatch success 時だけ
    cooldown timestamp が進む回帰を追加した。

- Why:
  - `MomentumBurst`
    の new loser-lane tightening は
    `scripts/improvement_preflight.sh`
    で
    `review_existing_pending`
    になり、
    current single-focus lane には積めなかった。
  - 一方で
    shared
    `micro_runtime`
    には、
    loser-lane 改修とは独立に
    participation を削る execution bug が残っていた。
  - これは同日
    `range_fade`
    の別 lane を増やす話ではなく、
    current live cadence を壊す runtime bug の是正なので、
    `improvement_preflight`
    でも
    `allow_new_lane`
    を返した。

- Hypothesis:
  - `market_order() -> None`
    の pre-order reject / no-fill で strategy cooldown を進めないようにすれば、
    dedicated micro worker は次の valid setup を待ち時間無しで再評価でき、
    false cooldown による participation loss を減らせる。

- Why Not Same As Last Time:
  - `momentumburst_no_signal_diagnostics_20260316`
    は
    `MomentumBurst`
    の signal absence を読むための diagnostics 追加だった。
  - 今回は
    `MomentumBurst`
    family の新しい loser-lane tightening ではなく、
    shared
    `micro_runtime`
    の execution / cooldown bug を直している。

- Expected Good:
  - `res=None`
    の no-fill path で
    `MomentumBurst`
    を含む dedicated micro worker が cooldown を無駄に消費しない。
  - live log で
    `cooldown_updated=0`
    を確認でき、
    next RCA で
    no-signal と post-signal no-fill を分離しやすくなる。

- Expected Bad:
  - persistent reject path では、
    同一 setup の再送頻度が一時的に上がる可能性がある。
  - そのため今回は cooldown key の細分化までは行わず、
    まず
    `dispatch success`
    のときだけ timestamp を進める最小修正に留める。

- Promotion Gate:
  - `quant-micro-momentumburst.log`
    など micro runtime worker の live log で
    `res=none cooldown_updated=0`
    と
    `res=<ticket> cooldown_updated=1`
    の両方を確認できること。
  - post-deploy で
    micro worker が crash せず、
    `tests/strategies/test_momentum_burst.py`
    の回帰が通ること。

- Escalation Trigger:
  - post-deploy でも
    `res=none cooldown_updated=0`
    が同じ strategy / setup で連発し、
    `fills_30m`
    が伸びない場合は、
    cooldown bug ではなく
    `strategy_entry / order_manager`
    の reject family を single-focus RCA に切り替える。

- Period:
  - 調査:
    `2026-03-16 09:19-09:28 JST`
  - 実装/検証:
    `2026-03-16 09:28 JST` 以降

- Fact:
  - `scripts/change_preflight.sh "momentumburst_participation_triage_20260316" 3`
    は
    `2026-03-16 09:23 JST`
    時点で
    `market_open=yes`,
    `spread=0.8p`,
    `fills_15m=3`,
    `fills_30m=9`,
    `preflight_status=ok`
    だった。
  - `scripts/improvement_preflight.sh "micro runtime cooldown after dispatch-none 2026-03-16" ...`
    は
    `allow_new_lane`
    を返し、
    unresolved overlap 無しと判定した。
  - `logs/local_v2_stack/quant-micro-momentumburst.log`
    には
    `2026-03-06 07:18 JST`
    と
    `18:06 JST`
    の
    `res=none`
    が複数あり、
    no-fill dispatch が historical に存在した。
  - dedicated
    `quant-micro-momentumburst`
    の env は
    `MICRO_MULTI_STRATEGY_COOLDOWN_SEC=90`
    なので、
    false cooldown 1 回あたりの逸失が重い。

- Failure Cause:
  - shared
    `micro_runtime`
    は
    `market_order()`
    の成否を区別せず、
    call しただけで strategy cooldown を進めていた。
  - そのため
    no-fill / reject で実際の建玉が無いのに、
    dedicated micro worker は next setup を local cooldown で見送る構造だった。

- Improvement:
  - `micro_runtime`
    の cooldown 更新を
    successful dispatch
    に限定し、
    no-fill では cadence を落とさないようにした。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/strategies/test_momentum_burst.py`
  - `python3 -m py_compile workers/micro_runtime/worker.py tests/strategies/test_momentum_burst.py`
  - `scripts/local_v2_stack.sh restart --profile trade_min --env ops/env/local-v2-stack.env --services quant-micro-momentumburst,quant-micro-levelreactor,quant-micro-trendretest,quant-micro-rangebreak`
  - `scripts/local_v2_stack.sh status --profile trade_min --env ops/env/local-v2-stack.env --services quant-micro-momentumburst,quant-micro-levelreactor,quant-micro-trendretest,quant-micro-rangebreak`
  - post-deploy live log で
    `cooldown_updated`
    を確認する。

- Verdict:
  - pending

- Status:
  - `pending_live_dispatch_validation`

- Next Action:
  - post-deploy の
    `quant-micro-momentumburst.log`
    /
    `quant-micro-levelreactor.log`
    で
    `res=none cooldown_updated=0`
    を first check とし、
    false cooldown が消えたことを確認する。
  - その後も
    `MomentumBurst`
    が無発火なら、
    single-focus lane は引き続き
    `momentumburst_no_signal_diagnostics_20260316`
    の live reason で判断する。

## 2026-03-16 09:17 JST / local-v2: `MomentumBurst` single-focus lane の no-signal diagnostics を strategy-local に露出

- Hypothesis Key:
  - `momentumburst_no_signal_diagnostics_20260316`
- Primary Loss Driver:
  - single-focus pending lane の
    `MomentumBurst`
    で、
    「なぜ今 signal が出ていないか」が live log に無く、
    `review_existing_pending`
    のまま blind retighten へ流れること
- Mechanism Fired:
  - `momentumburst_no_signal_diagnostic`
  - `2026-03-16 09:11 JST`
    時点の
    `quant-micro-momentumburst.log`
    は
    `allowlist applied: MomentumBurst`
    の反復だけで、
    no-signal reason を残していなかった。
- Do Not Repeat Unless:
  - `momentumburst_no_signal`
    が live log に出るようになった後も、
    `MomentumBurst`
    の current lane を判定するのに必要な
    `pullback / mtf / trend / rsi / quality / context`
    のどれで止まっているかが still 分からないときだけ、
    diagnostics field を追加する。

- Change:
  - `strategies/micro/momentum_burst.py`
    に
    `MomentumBurstMicro.diagnostic(fac)`
    を追加し、
    `long/short`
    の
    `base / pullback / mtf / trend / price / rsi / indicator / context`
    判定を dict で返すようにした。
  - `workers/micro_runtime/worker.py`
    は
    `MomentumBurst`
    が `cand=None`
    を返したときだけ
    `momentumburst_no_signal`
    を 120 秒に 1 回 rate-limit して出すようにした。
  - 同時に
    `_allowed_strategies()`
    の allowlist 結果を env 値ごとに cache し、
    `allowlist applied`
    を毎ループ出さないようにした。
  - `tests/strategies/test_momentum_burst.py`
    に
    `transition long` の
    `pullback_guard`
    block / keep を
    `diagnostic()`
    でも確認する回帰を追加した。

- Why:
  - `scripts/improvement_preflight.sh`
    実測では
    `DroughtRevert`
    と
    `scalp_extrema_reversal_live`
    の新規 tweak は
    どちらも
    `review_existing_pending`
    で block された。
  - 同じ gate は reopen first-focus として
    `MomentumBurst`
    を指しているが、
    直近 24h の
    `orders.db / trades.db`
    では
    `MomentumBurst`
    の fresh sample が無く、
    `lane_scoreboard / participation_alloc / entry_path_summary`
    にも lane が出ていなかった。
  - この状態で trade logic をさらに触ると、
    anti-loop に反して
    「観測なしの tightening」
    を積むことになる。

- Hypothesis:
  - `MomentumBurst`
    の no-signal reason を strategy-local に露出すれば、
    current pending lane が
    `pullback_guard`
    で詰まっているのか、
    そもそも
    `base setup`
    が無いのかを live で切り分けられ、
    adjacent loser family を blind に触らずに次の一手を決められる。

- Why Not Same As Last Time:
  - `momentumburst_transition_pullback_guard_20260314`
    は
    `MomentumBurst`
    の trade logic 自体を tighten した entry だった。
  - 今回は
    新しい tighten ではなく、
    その pending lane を reopen 窓で判定するための
    diagnostics / log hygiene を追加している。

- Expected Good:
  - `quant-micro-momentumburst.log`
    で
    `long_transition_pullback_guard`
    なのか
    `long_base_conditions`
    なのかが見える。
  - `allowlist applied`
    のループログが止まり、
    single-focus lane の判定に必要な log だけが残る。
  - `MomentumBurst`
    を触るなら
    `MomentumBurst`
    family の中だけで次の change を決められる。

- Expected Bad:
  - no-signal window が長いと diagnostics log も増える。
  - そのため
    `momentumburst_no_signal`
    は 120 秒 rate-limit に留め、
    per-loop log にはしない。

- Promotion Gate:
  - `quant-micro-momentumburst.log`
    に
    `momentumburst_no_signal`
    が出て、
    `reason`
    と
    `long/short`
    の block summary を live で読めること。
  - 次の
    `MomentumBurst`
    signal が出た場合でも、
    既存 entry tag / entry_thesis contract を壊していないこと。

- Escalation Trigger:
  - `momentumburst_no_signal`
    の同一 reason が
    reopen 後 30-120 分続き、
    かつ
    `MomentumBurst`
    fills が still `0`
    のままなら、
    adjacent
    `STOP_LOSS_ORDER`
    family ではなく
    `MomentumBurst`
    family 自身の
    `improvement_preflight`
    へ進む。

- Period:
  - 調査:
    `2026-03-16 08:55-09:11 JST`
  - 実装/検証:
    `2026-03-16 09:11-09:17 JST`
  - 対象:
    `strategies/micro/momentum_burst.py`,
    `workers/micro_runtime/worker.py`,
    `tests/strategies/test_momentum_burst.py`

- Fact:
  - `scripts/improvement_preflight.sh "live reopen loser-lane triage 2026-03-16" ...`
    は
    `DroughtRevert`
    と
    `scalp_extrema_reversal_live`
    の両候補を
    `review_existing_pending`
    と判定した。
  - `logs/strategy_feedback.json`
    の
    `MomentumBurst`
    は
    `trades=14`,
    `win_rate=0.286`,
    `profit_factor=0.611`
    で、
    `flow_regime=transition`
    の setup override は
    `entry_probability_multiplier=0.875`,
    `entry_units_multiplier=0.836`
    だった。
  - 一方で
    `2026-03-16 09:11 JST`
    時点の
    `orders.db / trades.db`
    直近 24h に
    `MomentumBurst`
    の fresh order/trade は無く、
    `lane_scoreboard_latest.json`
    /
    `participation_alloc.json`
    /
    `entry_path_summary_latest.json`
    にも
    `MomentumBurst`
    lane は出ていなかった。
  - `tests/strategies/test_momentum_burst.py`
    は
    `45 passed`
    を確認した。

- Failure Cause:
  - current single-focus lane に対して、
    signal absence を説明する strategy-local diagnostics が無かった。
  - さらに
    `allowlist applied`
    が毎ループ出て、
    live 判定に必要な log を埋めていた。

- Improvement:
  - `MomentumBurst`
    の no-signal diagnostics を strategy-local に出し、
    live で
    `pullback / mtf / quality`
    のどこで止まっているかを読めるようにした。

- Verification:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/strategies/test_momentum_burst.py`
    -> `45 passed`
  - `python3 -m py_compile strategies/micro/momentum_burst.py workers/micro_runtime/worker.py tests/strategies/test_momentum_burst.py`
    -> 成功

- Verdict:
  - pending

- Status:
  - `pending_live_reopen_diag`

- Next Action:
  - `quant-micro-momentumburst`
    を反映後、
    `logs/local_v2_stack/quant-micro-momentumburst.log`
    で
    `momentumburst_no_signal`
    を確認する。
  - その reason が
    `long_transition_pullback_guard`
    に偏るなら
    `MomentumBurst`
    family の escalation を、
    `long_base_conditions`
    に偏るなら
    trade logic を増やさず market mismatch として扱う。
