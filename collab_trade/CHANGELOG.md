# 共同トレード CHANGELOG

## 2026-03-21
- T03:00 **テクニカル一覧コードパス完全修正** — 全指標のコード列を`indicators/calc_core.py`→`collab_trade/indicators/calc_core.py`に統一。Microモメンタムに「共同トレード中は使用不可」注記追加
- T02:30 **導線の全面検証・修正** — メモリ3ファイル(trigger/autonomy/project)の古い参照パスを修正(logs/collab_state.md→collab_trade/state.md、docs/TRADE_LOG_COLLAB→daily/trades.md)。CLAUDE.mdにトレード実行フロー図追加。テクニカル一覧のコードパスをcollab_trade/indicators/に修正。スキルの記録先からlive_trade_log.txt除去。MEMORY.mdインデックス更新
- T02:00 **テクニカルエンジン導入** — `indicators/`にcalc_core.py+divergence.pyをコピー（本体に影響せずパラメータ自由にいじれる）。quick_calc.py新設（`python3 collab_trade/indicators/quick_calc.py USD_JPY M5 50`で即テクニカル分析）。CLAUDE.mdテクニカル一覧を84指標の完全版に書き直し
- T01:30 **ディレクトリ再設計** — CLAUDE.md全面書き直し（手法/ルール/失敗パターン/テクニカル一覧/ペア別ノート/セッション時間帯を統合）。daily/YYYY-MM-DD/trades.md+notes.md構造、summary.md全日統括を新設。2026-03-20データを新構造に移行
- T01:00 **`/collab-trade`スキル作成 + 導線整備** — `.claude/skills/collab-trade.md`新設。feedback_collab_trade_trigger.mdをcollab_trade/CLAUDE.md正本に更新。ルートCLAUDE.mdに導線追加
- T00:30 **共同トレード専用ディレクトリ作成** — collab_trade/CLAUDE.md、state.md新設。ルートCLAUDE.mdに「共同トレード」コマンドと運用の鉄則追記
- T00:00 **Session3反省記録** — 後半の質低下分析（BGタスク乱発→コンテキスト破壊、利確遅延、受け身bot化）。対策をfeedback_collab_autonomy.mdとTRADE_LOG_COLLAB_20260320.mdに記録

## 2026-03-20
- 共同トレード初回実施。3セッション合計+1,760円(+11.9%)
