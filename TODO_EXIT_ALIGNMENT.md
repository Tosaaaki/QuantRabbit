## EXIT 整合性チェック ToDo

共通方針
- もうポケット単位で考えない。mainはWORKER_ONLY_MODEで発注・EXITを行わず、各ワーカーがエントリー/EXITまで完結させる。
- micro_core（ホールド比率ガード連発でブロック）は使わない前提。必要なら明示的に止める or 無効化環境をセットし、再度使う場合は個別に指示する。
- ポケット共通EXITは使わない。エントリーしたワーカー専用のEXITのみ有効。
- ExitManager 自動EXITは `EXIT_DISABLE_AUTO_EXIT=1`（scalp fast_cut/kill も無効化）前提で整合を見る。
- macro_exit/micro_exit/scalp_exit の共通出口ループは `DISABLE_POCKET_EXIT`（デフォルトON）で停止。

進め方（ワーカーごとに繰り返し）
- 対象ワーカー一覧を確定（systemd/strategies から列挙し、優先度付け）
- ワーカーのエントリー条件を抽出（指標/閾値/時間帯/ボラ/ステージ/メタ付与）
- ワーカーのEXIT条件を抽出（CLOSEシグナル、TP/SL、クールダウン/再入場制約）
- エントリーとEXITの整合チェック（優位性崩れの検知有無、閾値逆転、RR/メタ矛盾）
- 不整合の修正案作成（条件揃え、メタ/TP/SLの標準化方針）
- 修正実装（ワーカー単位でパッチ、必要なら ExitManager のトグル/メタ調整）
- 動作確認（単体テスト or リプレイでエントリー→EXITの流れを確認）
- 本番反映とモニタ（デプロイ＋短時間のログ監視）
- 作業ごとにローカルでコミット＋リモートへプッシュし、VM反映も都度行う（未コミット/未デプロイ残しを避ける）

メモ
- いまは `EXIT_DISABLE_AUTO_EXIT=1` で ExitManager の自動EXITは全無効。各ワーカーのEXITが唯一のクローズ経路になっている前提で整合を見る。

### H1Momentum（macro）
- エントリー（workers/macro_h1momentum/worker.py + strategies/trend/h1_momentum.py）
  - H1: ADX>=18, MA10/MA20ギャップ>=4.5p、ATR_pips>=2.0、EMA12/EMA24順行、RSI極端(>=76.5 or <=23.5)で除外
  - M1: RSIロング側<76/ショート側>24で極端回避
  - SL: ATR*2.1 を 18〜36p にクランプ、TP: SL×(0.85〜1.05) もしくは ATR*1.3 以上
  - サイズ: BASE_ENTRY_UNITS×TPスケール×confidenceスケール×cap、リスク上限とfree_marginで制限、capはrange/perf/marginを考慮
  - tag: H1Momentum-bull/bear, hard_stop_pipsもメタに含む
- EXIT（workers/macro_h1momentum/exit_worker.py）
  - PnL/時間/RSI/レンジ判定主体。range_mode検知（ADX<=22,BBW<=0.22,ATR<=6.5）
  - 即時ハードストップ: pnl<=-stop_loss(通常3.0p, range 2.4p, ATR冷えでやや緩む)
  - 時間: pnl<0 かつ hold>=NEG_HOLD_SEC(1500s) → time_cut（allow_negative_exitがTrue）、pnl<=profit_take*0.7 で max_hold(3h/2.5h)超え → time_stop
  - RSIフェード: ロングRSI<=44/ショートRSI>=56 で負域ならカット（ATR高なら許可）
  - 利確系: lock_trigger(2.2p)/bufferでBEロック、trail_start(6.5p)からtrail_backoff(2.0p)、profit_take(5.0p)基準のtake/range_take、RSI_take(72/28)ヒット、VWAPギャップ<=1.1pで±pnlに応じてtake/cut
- 整合メモ
  - エントリーはH1トレンド構造依存だが、EXITは構造（MAクロス/ADX低下）を見ずPnLとRSI/時間中心。H1勢い消滅（MA10/MA20クロス逆転、ADX低下）でEXITしないケースがありうる。
  - エントリーSLは18〜36pだが、EXITのstop_lossは3p前後ではるかにタイト。ハードSLとEXITハードストップの乖離が大きく、RR設計と整合していない可能性。
  - range判定はEXIT側で再評価しており、entry時のrange判定（detect_range_mode）との閾値違いがあり得る。
- 追加で確認したい点（要決め）
  - H1構造崩れ（MAクロス逆転/ADX<しきい値）をEXITトリガに含めるか？
  - stop_loss/lock/trailのpipsをエントリーSL/TP比に合わせて再設計するか？
  - allow_negative_exit=False運用にするか（現状True）と、その際のmax_hold/time_cutの扱い。

### M1Scalper（scalp）
- エントリー（strategies/scalping/m1_scalper.py）
  - ゲート: ATR_pips>=1.2（tactical時は別値）、vol5>=0.35、ADX>=12。低ボラレンジ（ADX<18 & BBW<0.0016 & ATR<2.4）はBB/VWAP近接のみ許可。
  - トレンド判定: EMAモメンタム/価格乖離で trend_up/down/strong_up/down を判定し、逆張りを抑制。
  - TP/SL: tp_dyn≈ATR*3 を [5,9]pにクランプ、sl_dyn≈min(ATR*2, 0.95*TP) を >=4p。fast_cut≈0.85*ATR, fast_cut_time≈12*ATR秒（メタ付与）。
  - メタ: fast_cut_pips/time/hard_mult=1.6、kill_switch付与。tag: buy-dip/sell-rally/trend-long/short/nwave派生。
- EXIT
  - ポケット共通EXITは使わない。専用Exitワーカーも置かず、エントリーしたワーカー（M1Scalper）が設定するSL/TPのみで完結させる方針。
  - ExitManager自動EXITは `EXIT_DISABLE_AUTO_EXIT=1`、fast_cut/kill メタは環境変数で無効化中。
- 整合ギャップ
  - fast_cut/kill 無効化により、低ボラ/レンジの早期撤退や構造崩れEXITなし。ホールドリスクは許容前提。
  - 必要なら将来、各エントリーワーカー内にCLOSEロジックを内包させる形で追加する（ポケット共通EXITや専用EXITワーカーは置かない）。

### TrendMA（macro）
- エントリー（strategies/trend/ma_cross.py）
  - H1/H4のMA10/MA20ギャップとADXでトレンド判定。ATR/BBWレンジ抑制、MACDフィルタ、MTFスロープ整合性。逆方向クールダウン30分。
  - SL/TP: projection +構造/VWAP/スイング高安で補正。SLはデフォで10p以上、TPはSL比1.08以上、MTF整合でスケール。
  - メタ: strategy_tag=TrendMA系。hard_stop_pipsなどは明示無し（構造から計算）。
- EXIT（workers/macro_trendma/exit_worker.py）
  - PnL/時間/RSI/レンジ/VWAP + H1構造崩れ（MA逆転/ADX低下＋ギャップ収縮）でCLOSE。range_mode閾値: ADX<=22, BBW<=0.22, ATR<=6.5。
  - エントリーメタがあれば stop/lock/trail/profit をスケール（hard_stop*0.4 など）。
  - stop_lossの最低値を拡張（エントリーSLとの整合を少し改善）。
- 整合ギャップ
  - range判定閾値がエントリー側のレンジ抑制と完全一致しているかは未確認（必要なら揃える）。

### Scalp multi（RangeFader / PulseBreak / ImpulseRetraceScalp）
- エントリー: strategies/scalping/* で各戦略（BB/レンジ/ブレイク/逆張り）。
- EXIT（workers/scalp_multistrat/exit_worker.py）
  - PnL/時間/RSI/レンジ/VWAP。range閾値共有化済み、エントリーメタで stop/lock/trail/profit をスケール（hard_stop*0.5 等）。
  - 構造崩れ検知は未導入（必要なら各戦略ごとに条件を持ち込む余地あり）。
- 整合ギャップ
  - 戦略ごとの優位性崩れ（BBタッチ解消、ブレイク失敗、逆行モメ減衰など）をEXIT側で見ていない。必要なら個別に追加。

### Impulse系 S5（impulse_momentum_s5 / impulse_retest_s5 / impulse_break_s5）
- エントリー: S5バケットのモメ/リトレース/ブレイク系。詳細要抽出。
- EXIT: 専用Exitワーカーあり（PnL/時間中心）。エントリーメタ連携や構造崩れ検知の有無は未確認。
- 次アクション: エントリー条件とEXIT閾値・構造崩れの整合を確認し、必要ならエントリーメタ参照スケール＆構造崩れCLOSEを追加。

### FastScalp（scalp_fast）
- EXIT（workers/fast_scalp/exit_worker.py）
  - PnL/時間/RSI/レンジ/VWAP。range閾値共有化、エントリーメタで stop/lock/trail/profit をスケール（hard_stop*0.5 等）。M1 MA逆転/ADX低下＋ギャップ縮小で structure_break。
- 整合ギャップ
  - なし（構造崩れも追加済み）。必要なら閾値微調整。

### Pullback 系 S5（pullback_s5 / pullback_runner_s5 / pullback_scalp）
- EXIT: エントリーメタ（hard_stop/tp_hint）で stop/lock/trail/profit をスケール、lock_buffer を stop_loss に合わせて拡張。レンジ閾値は ADX<=22, BBW<=0.20, ATR<=6 で統一。
- 構造崩れ: M1 MA10/MA20 逆転 or ADX<14&ギャップ<2p で structure_break クローズ。

### SqueezeBreak S5 / VWAP Magnet S5
- EXIT: pullback系と同様にエントリーメタスケール + lock_buffer 拡張。レンジ閾値を共通化。
- 構造崩れ: M1 MA逆転/ADX低下で structure_break。

### MicroMultiStrat
- EXIT: エントリーメタで stop/lock/trail/profit をスケール、lock_buffer を stop_loss 連動に。レンジ閾値共通化、構造崩れ（M1 MA逆転/ADX<14&gap<2）を追加。

### LondonMomentum / Donchian55 / ManualSwing（macro）
- EXIT: hard_stop/tp_hint をスケールに反映し、lock_buffer を stop_loss 連動で拡張。
- 構造崩れ: H1 MA10/MA20 逆転 or ADX<16 & gap<3.5p で structure_break。

### SimpleExit ワーカー（onepip/mirror_spike/trend_h1/impulse_break_s5）
- 修正: hard_stop/tp_hint を保持する TradeState へ拡張、エントリーメタで stop/take/trail/lock をスケール。構造崩れオプションを実装（impulse_break_s5 で有効化）。

### M1Scalper
- EXIT: 専用ワーカー追加（workers/scalp_m1scalper/exit_worker.py）。SimpleExitベースでエントリーメタ追随＋構造崩れ（M1 MA10/MA20逆転＋ADX低下）で撤退。ポケット共通EXITは使わない。

### 新規ToDo: TP最優先・仮想SL化
- 目的: 全ワーカーで「TP到達を第一目標」「雲行き悪化時のみ微益/微損撤退」、実SLは置かず仮想SL/メタSLで運用しRRを維持。
- 方針ドラフト
  - 物理SLは送らず（or極小）、EXIT側で仮想SL/構造崩れ/レンジ圧縮をトリガに微損撤退。
  - TPはエントリーTP未満に落とさない。trail/lock開始はtp_pips比で設定。
  - RR確認: 仮想SL距離をtp_pipsの≥0.7倍程度にし、ロット計算は entry_hard_stop をリスクガードに渡す（実発注はTPのみ）。
- 早逃げ: 構造崩れ(MA逆転+ADX低下+gap小)とボラ圧縮時のみ微益/微損でクローズ。負域許容は限定。
- 対応順（提案）: 1) Scalp系全般（fast_scalp, scalp_multistrat, m1scalper, pullback/squeeze/vwap/impulse系）→ 2) MicroMulti → 3) TrendMA/H1Momentum/Donchian/London/ManualSwing。
- 進め方: ワーカーごとに「物理SL送信を外し、仮想SL/構造崩れ/レンジ圧縮のEXITに切替」「trail/lock閾値をtp基準に再設計」「リスク計算でのhard_stopは維持」を適用。完了したらここでチェックを外して整理する。

### 使用テクニカル（参照一覧）
- トレンド/モメンタム: MA10/20（H1/H4/M1）、EMA12/24、MA/EMAスロープ・乖離、MAクロス（構造崩れ判定）、ADX、MACD/ヒスト、ROC5/10、Chaikin Vol、DMI(+DI/-DI)。
- ボラ/帯: ATR_pips、BBW、KC幅、Donchian幅。
- オシレーター: RSI、StochRSI、CCI。
- レベル系: スイング高安距離、クラスタ高低ギャップ（cluster_high_gap/low_gap）、Donchian高安、VWAP乖離(vwap_gap)。
- パターン/構造: N波（M1Scalper）、スパイク/リバーサル（mirror_spike）、レンジ判定(range_mode: ADX/BBW/ATR)、構造崩れ（MA逆転+ADX低下+ギャップ縮小）。
- Ichimoku: cloud_pos、span_a_gap、span_b_gap。
- 価格・ティック: candle body pips、tick_rate/vol_5m、momentum_pips/short_momentum_pips。
- その他ゲート: スプレッドゲート、ニュースブロック、市場時間帯、PF/勝率によるcap調整。
  - TODO: パターン活用範囲を各ワーカーに明示し、必要ならシグナル/EXIT条件に統合する（雲も含む）。
    - 対象パターン一覧（統一参照用）:
      - 反転系: ダブルトップ/ダブルボトム、トリプルトップ/トリプルボトム、三尊(HS)/逆三尊(iHS)、M/Wボトム、スパイク/V字反転、エンゴルフィング（大陽/大陰）※必要なら。
      - 継続/ブレイク系: 旗(フラッグ)/ペナント、上昇・下降ウェッジ、三角保ち合い（上昇・下降・対称）、レンジ矩形、カップ・アンド・ハンドル、ラウンディングトップ/ボトム。
      - N波: M1Scalperで既存利用。必要なら他ワーカーへの転用も検討。
