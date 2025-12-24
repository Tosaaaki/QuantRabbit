## EXIT 整合性チェック ToDo

共通方針
- ポケット共通EXITは使わない。エントリーしたワーカー専用のEXITのみ有効。
- ExitManager 自動EXITは `EXIT_DISABLE_AUTO_EXIT=1`（scalp fast_cut/kill も無効化）前提で整合を見る。

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
