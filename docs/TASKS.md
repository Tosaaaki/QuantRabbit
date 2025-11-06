# TASKS – Repository Task Board

本リポジトリの全タスクを一元管理する台帳。タスクが発生したら本ファイルへ逐次追記し、作業中は本ファイルを参照しながら進め、完了後はアーカイブへ移してください。詳細ポリシーは `AGENTS.md` の「11. タスク運用ルール（共通）」を参照。

## 運用ルール（要約）

- 発生: 本ファイル「Open Tasks」にテンプレートで追加。
- 進行: ステータス/Plan/Notes を更新し、PR/コミットには `[Task:<ID>]` を含める。
- 完了: 「Archive」へ移動し、完了日・PR/コミットリンク・要約を追記。
- 例外: オンライン自動チューニング関連は `docs/autotune_taskboard.md` を主に使用（必要に応じて相互リンク）。

## テンプレート

以下をコピーして使用:

```md
- [ ] ID: T-YYYYMMDD-001
  Title: <短い件名>
  Status: todo | in-progress | review | done
  Priority: P1 | P2 | P3
  Owner: <担当>
  Scope/Paths: <例> AGENTS.md, docs/TASKS.md
  Context: <Issue/PR/仕様リンク>
  Acceptance:
    - <受入条件1>
    - <受入条件2>
  Plan:
    - <主要ステップ1>
    - <主要ステップ2>
  Notes:
    - <補足>
```

## Open Tasks

（ここに新規タスクを追加）

- [ ] ID: T-20251103-001
  Title: H1トレンドワーカーの追加実装
  Status: in-progress
  Priority: P1
  Owner: tossaki
  Scope/Paths: market_data/candle_fetcher.py, indicators/factor_cache.py, workers/trend_h1/, main.py, docs/TASKS.md
  Context: 中期トレンド用の常時稼働ワーカー不足（AGENT.me 仕様 §3.5, 提案メモ 2025-11-03）
  Acceptance:
    - H1足の因子が取得・キャッシュされ、ワーカーが利用できる
    - TrendH1ワーカーがMovingAverageCrossを用いて売買指示を生成し、Risk/Exit連携済み
    - main.py からワーカーの起動制御が可能で、環境変数でON/OFFできる
    - リプレイ/紙上テスト手順を README or コメントで案内
  Plan:
    - H1 timeframe を candle/factor パイプラインに追加
    - workers/trend_h1/ に worker.py, config.py, __init__.py を整備
    - main.py での起動・設定導線を追加
  Notes:
    - 既存のTrendMAロジックを再利用し、シンプルなステージ配分で初期運用
    - リプレイ実行例: `python3 scripts/replay_trendma.py --ticks-glob 'tmp/ticks_USDJPY_202510*.jsonl' --out tmp/replay_trend_h1.json`
    - 上記データセットで +4.4pips / 3 trades（win率 66.7%）を確認

- [ ] ID: T-20251104-001
  Title: Impulse Momentum S5 派生ワーカーの実装
  Status: todo
  Priority: P1
  Owner: tossaki
  Scope/Paths: workers/impulse_momentum_s5/, main.py, scripts/replay_workers.py, docs/TASKS.md
  Context: impulse_break_s5 の好成績部分（強トレンド＋瞬間変動）を抽出した派生ワーカーで期待値を引き上げる
  Acceptance:
    - impulse_momentum_s5 の config/worker が追加され、環境変数で ON/OFF 可能
    - main.py から起動され、PositionManager/Risk/Exit 連携が整合
    - 瞬間変動・スプレッド・方向整合ガードが実装され、RR>1 の利確/損切りが設定されている
    - scripts/replay_workers.py で実ティックリプレイが可能になり、10月データで検証ログを取得
  Plan:
    - config.py を作成し、時間帯・瞬間変動・トレンド整合のデフォルト値を定義
    - worker.py を実装し、単段エントリ＋BE/トレール制御を組み込む
    - main.py にワーカー起動ルートと環境変数を追加
    - リプレイスクリプトに worker 名を追加し、実ティックで検証
  Notes:
    - 曜日フィルタは導入せず、ボラティリティとレジーム整合で制御する

- [ ] ID: T-20251104-002
  Title: Mirror Spike Tight 派生ワーカーの実装
  Status: todo
  Priority: P2
  Owner: tossaki
  Scope/Paths: workers/mirror_spike_tight/, main.py, scripts/replay_workers.py, docs/TASKS.md
  Context: mirror_spike_s5 の薄利傾向を、トレンド整合＋高ボラ条件のみで狙う派生を作り平均損益を改善
  Acceptance:
    - mirror_spike_tight の config/worker が追加され、stage=1 固定で RR>1 設定
    - H1/M1 トレンド整合と瞬間変動・スプレッド閾値ガードが組み込まれている
    - main.py から起動制御でき、env で切替可能
    - リプレイスクリプトで再生でき、10月実ティックで検証結果を得る
  Plan:
    - config.py にガード閾値とステージ設定を定義
    - worker.py を実装し、条件を満たす高精度スパイクのみエントリ
    - main.py に登録し、環境変数フラグを追加
    - scripts/replay_workers.py に worker 名を追加し、実ティックでA/B検証
  Notes:
    - ステージ拡張やナンピンは許可せず、BE移行と部分利確を組み合わせる

- [ ] ID: T-20251104-003
  Title: Maker型1pipスカルパーワーカーの実装
  Status: in-progress
  Priority: P1
  Owner: tossaki
  Scope/Paths: market_data/orderbook_state.py, analytics/cost_guard.py, workers/onepip_maker_s1/, main.py, docs/TASKS.md
  Context: 1pip利確多数回の実装方針（板主導メイカー戦略）を別ワーカーとして実装する（AGENT.me 仕様 §3.5.1、提案メモ 2025-11-04）
  Acceptance:
    - L2板情報キャッシュとレイテンシ推定を扱う orderbook_state モジュールが追加され、最新スナップショットを参照できる
    - 取引コスト c を推定するコストガードが実装され、latest fill から c を更新・参照できる
    - workers/onepip_maker_s1 worker が shadow モードで稼働し、post-only・TTLロジックの骨組みとメトリクス出力が揃う
    - main.py にワーカー起動導線が追加され、環境変数で有効化/シャドウ切替が可能
  Plan:
    - orderbook_state/cost_guard の土台を実装し、将来のL2ストリームから更新できるようにする
    - onepip_maker_s1 worker を shadow モードで実装し、条件ログとシグナルカウントを記録
    - main.py へワーカー登録と環境変数フラグを追加し、既存スプレッド・レンジガードと連携
  Notes:
    - 実オーダー発注は post-only 対応と L2遅延監視が整ってから enable する想定
    - シャドウ結果を `logs/onepip_maker_s1_shadow.jsonl` に出力し、EV/コスト検証に活用する
    - 2025-11-04: tick stream から top-of-book を orderbook_state へ自動投入し、コストガードはログから定期リフレッシュする構成を追加済み
    - 2025-11-04: `execution/order_manager.limit_order` を追加し、env で shadow を切った際に post-only limit を即時発注できる骨組みまで実装済み（SL/TP/TTL は env 係数で制御、TTL<1s 時は自動キャンセルタスクが動く）
    - 2025-11-04: Risk Guard/口座スナップショット連携と `onepip_maker_*` メトリクス記録を追加し、ライブ化チェックリストを README へ記載済み

## Archive

完了タスクを以下形式で移動・記録:

```md
- [x] ID: T-YYYYMMDD-001
  Title: <短い件名>
  Status: done
  Priority: P2
  Owner: <担当>
  Scope/Paths: <対象>
  Context: <Issue/PR/仕様リンク>
  Acceptance:
    - <満たした条件1>
  Completed: YYYY-MM-DD
  PR: <PRリンク or コミットSHA>
  Summary: <1〜3 行で要約>
```

---

- [x] ID: T-20251102-001
  Title: gcloud doctor/installer とデプロイ前診断導入
  Status: done
  Priority: P2
  Owner: maint
  Scope/Paths: scripts/install_gcloud.sh, scripts/gcloud_doctor.sh, scripts/deploy_to_vm.sh, docs/GCP_DEPLOY_SETUP.md, README.md
  Context: デプロイ不能・gcloud 未導入の再発防止（IAP/OS Login を含む）
  Acceptance:
    - gcloud 未導入時に install スクリプトで導入可能
    - doctor で project/API/OS Login/インスタンス/SSH を検査できる
    - deploy スクリプトが前提不備で早期失敗・誘導
    - セットアップ手順ドキュメントがある
  Completed: 2025-11-02
  PR: <pending>
  Summary: gcloud インストーラ・プリフライト（doctor）とドキュメントを追加し、deploy スクリプトに前提チェックを組み込みました。

- [x] ID: T-20251102-002
  Title: ヘッドレス運用（サービスアカウント）対応
  Status: done
  Priority: P2
  Owner: maint
  Scope/Paths: scripts/gcloud_doctor.sh, scripts/deploy_to_vm.sh, docs/GCP_DEPLOY_SETUP.md, AGENTS.md
  Context: アクティブアカウント不在でも GCP/VM を操作できるようにする
  Acceptance:
    - SA キー指定で doctor/deploy が動作（-K）
    - 必要時 SA インパーソネートにも対応（-A）
    - ドキュメントに手順とロール要件を明記
  Completed: 2025-11-02
  PR: <pending>
  Summary: doctor/deploy に SA キー自動有効化とインパーソネートを追加し、ヘッドレス運用手順を docs に追記しました。

- [x] ID: T-20251102-003
  Title: AGENTS.md に quantrabbit クイックコマンドを追加
  Status: done
  Priority: P3
  Owner: maint
  Scope/Paths: AGENTS.md
  Context: 実運用の即時参照用に、Doctor/Deploy/IAP/SA の実値例を集約
  Acceptance:
    - AGENTS.md 10.3 に quantrabbit 固定のコマンド例がある
    - 旧来の冗長/重複手順は削除済み
  Completed: 2025-11-02
  PR: <pending>
  Summary: プロジェクト/ゾーン/インスタンス実値のクイックコマンドを AGENTS.md に追記。
