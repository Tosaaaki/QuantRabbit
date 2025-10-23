#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_scalp.py
- ランダムサーチ＋ウォークフォワード評価でスカルプ戦略のパラメータをチューニング
- backtest_scalp.py に --json-out / --params-json / --strategies を追加するパッチ前提（同梱パッチ参照）
- 既存の backtest_scalp.py を import できる場合は run_backtest() を直接呼ぶが、
  そうでなければサブプロセスで backtest_scalp.py を起動して JSON を拾う
"""
import argparse, json, os, random, subprocess, sys, time, pathlib, datetime
from typing import Dict, List, Any, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs"
DEFAULT_CANDLES_GLOB = "candles_M1_*.json"
BACKTEST_SCRIPT = REPO_ROOT / "scripts" / "backtest_scalp.py"
PARAM_SPACE_PATH = REPO_ROOT / "configs" / "scalp_param_space.json"
ACTIVE_PARAMS_PATH = REPO_ROOT / "configs" / "scalp_active_params.json"  # 実運用で読み込む想定ファイル
DEFAULT_DB_PATH = REPO_ROOT / "logs" / "autotune.db"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from autotune.database import get_connection, record_run
except Exception:  # pragma: no cover - optional dependency during bootstrap
    get_connection = None  # type: ignore
    record_run = None  # type: ignore

StrategyParams = Dict[str, Dict[str, Any]]
ResultDict = Dict[str, Any]

def _load_param_space(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _coerce(value, spec):
    t = spec.get("type")
    if t == "int":
        return int(round(value))
    if t == "float":
        return float(value)
    if t == "choice":
        return value
    return value

def sample_params_for_strategy(space: Dict[str, Any], seed=None) -> Dict[str, Any]:
    rng = random.Random(seed)
    params = {}
    for k, spec in space.items():
        t = spec.get("type")
        if t == "int":
            params[k] = rng.randint(int(spec["min"]), int(spec["max"]))
        elif t == "float":
            lo, hi = float(spec["min"]), float(spec["max"])
            params[k] = round(rng.uniform(lo, hi), 2)
        elif t == "choice":
            params[k] = rng.choice(list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported type in param space: {t}")
    return params

def list_candle_files(candles_dir: pathlib.Path, pattern: str) -> List[pathlib.Path]:
    files = sorted(candles_dir.glob(pattern))
    return [p for p in files if p.is_file()]

def _try_import_backtester():
    # 可能なら import して直接呼び出す（パッチ適用後の run_backtest を期待）
    try:
        sys.path.insert(0, str(REPO_ROOT))
        import scripts.backtest_scalp as backtest_mod  # type: ignore
        if hasattr(backtest_mod, "run_backtest"):
            return backtest_mod
    except Exception:
        return None
    return None

def run_backtest_once(candles_path: pathlib.Path, strategies_params: StrategyParams, strategies: List[str]) -> ResultDict:
    """
    backtest_scalp.py を 1 回実行して JSON 結果を取得
    """
    backtester = _try_import_backtester()
    if backtester is not None:
        # 直接呼ぶ
        try:
            return backtester.run_backtest(
                candles_path=str(candles_path),
                params_overrides=strategies_params,
                strategies=strategies
            )
        except Exception as e:
            print(f"[WARN] run_backtest() direct call failed -> fallback to subprocess: {e}", file=sys.stderr)

    # サブプロセス実行：--params-json / --json-out を使う（パッチ適用が必要）
    tmp_json = candles_path.parent / f".bt_{candles_path.stem}_{int(time.time()*1000)}.json"
    with open(tmp_json, "w") as f:
        pass  # pre-create
    params_json_path = candles_path.parent / f".params_{candles_path.stem}_{int(time.time()*1000)}.json"
    with open(params_json_path, "w") as f:
        json.dump(strategies_params, f, ensure_ascii=False)
    try:
        cmd = [
            sys.executable, str(BACKTEST_SCRIPT),
            "--candles", str(candles_path),
            "--strategies", ",".join(strategies),
            "--params-json", str(params_json_path),
            "--json-out", str(tmp_json)
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print("[ERR] backtest subprocess failed:", r.stdout, r.stderr, file=sys.stderr)
            raise RuntimeError("backtest subprocess failed")
        with open(tmp_json, "r") as f:
            data = json.load(f)
        return data
    finally:
        # 消しすぎるとデバッグできないので結果 JSON は残す。パラメのみ削除
        try:
            params_json_path.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass

def compute_metrics(result: ResultDict) -> Dict[str, Any]:
    """
    result の仕様（パッチ適用版の例）:
    {
      "date": "2025-10-22",
      "summary": {"profit_pips": 3.0, "trades": 10, "win_rate": 0.40, "profit_factor": 1.07, "max_dd_pips": 8.4},
      "by_strategy": {"M1Scalper": {...}, "PulseBreak": {...}},
      "trades": [...]  # 任意
    }
    """
    s = result.get("summary") or {}
    return {
        "profit_pips": float(s.get("profit_pips", 0.0)),
        "trades": int(s.get("trades", 0)),
        "win_rate": float(s.get("win_rate", 0.0)),
        "profit_factor": float(s.get("profit_factor", 0.0)),
        "max_dd_pips": float(s.get("max_dd_pips", 0.0))
    }

def merge_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 単純集計 + 簡易 PF 計算（近似）。DD は max を採用（保守的）
    total_pips = sum(m["profit_pips"] for m in metrics_list)
    total_trades = sum(m["trades"] for m in metrics_list)
    avg_win_rate = sum(m["win_rate"] for m in metrics_list) / max(1, len(metrics_list))
    # PF は平均を採用（各日複利の簡易近似）
    avg_pf = sum(m["profit_factor"] for m in metrics_list) / max(1, len(metrics_list))
    max_dd = max((m["max_dd_pips"] for m in metrics_list), default=0.0)
    return {
        "profit_pips": round(total_pips, 2),
        "trades": int(total_trades),
        "win_rate": round(avg_win_rate, 4),
        "profit_factor": round(avg_pf, 4),
        "max_dd_pips": round(max_dd, 2)
    }

def passes_gates(agg: Dict[str, Any]) -> bool:
    # 最低限のゲート（保守的）
    if agg["trades"] < 8:  # 試行回数が少なすぎると信用しない
        return False
    if agg["profit_factor"] < 1.05:
        return False
    if agg["max_dd_pips"] > 12.0:
        return False
    return True

def score(agg: Dict[str, Any]) -> float:
    # ゲートを超えた上でのスコアリング（PF 重視、過度の DD にペナルティ）
    pf = min(agg["profit_factor"], 1.6)
    wr = agg["win_rate"]
    dd = agg["max_dd_pips"]
    base = pf
    bonus = 0.15 * max(0.0, wr - 0.5)  # 50%超から加点
    penalty = 0.02 * max(0.0, dd - 6.0)  # DD>6pips から緩やかに減点
    return base + bonus - penalty

def walk_forward_split(files: List[pathlib.Path], valid_ratio: float = 0.3) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
    n = len(files)
    if n <= 1:
        return files, []
    k = max(1, int(n * (1 - valid_ratio)))
    return files[:k], files[k:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles-dir", default=str(DEFAULT_LOG_DIR), help="logs ディレクトリのパス")
    ap.add_argument("--glob", default=DEFAULT_CANDLES_GLOB, help="candles ファイルの glob")
    ap.add_argument("--dates", default="", help="'20251020,20251021' のように日付指定。空なら全て")
    ap.add_argument("--strategies", default="M1Scalper,PulseBreak,RangeFader")
    ap.add_argument("--trials-per-strategy", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-best", action="store_true", help="改善あれば configs/scalp_active_params.json を上書き")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "logs" / "tuning"))
    ap.add_argument(
        "--record-db",
        default=str(DEFAULT_DB_PATH),
        help="結果を記録する SQLite DB パス。空文字で無効化",
    )
    args = ap.parse_args()

    candles_dir = pathlib.Path(args.candles_dir)
    all_files = list_candle_files(candles_dir, args.glob)
    if args.dates:
        target_dates = set(args.dates.split(","))
        files = [p for p in all_files if any(d in p.name for d in target_dates)]
    else:
        files = all_files[-10:]  # 直近 10 日ぶん程度

    if not files:
        print("[ERR] candles ファイルが見つかりません。", file=sys.stderr)
        sys.exit(2)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    random.seed(args.seed)

    space_all = _load_param_space(PARAM_SPACE_PATH)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_files, valid_files = walk_forward_split(files, valid_ratio=0.3)

    best_by_strategy = {}
    logs = []

    for strat in strategies:
        space = space_all.get(strat)
        if not space:
            print(f"[WARN] パラメータ空間が未定義: {strat}", file=sys.stderr)
            continue

        best = None
        for t in range(args.trials_per_strategy):
            params = sample_params_for_strategy(space, seed=(args.seed + t))
            # 全ストラテジの params 辞書に載せる
            merged_params = {strat: params}

            # 学習期間で評価
            train_metrics = []
            for cp in train_files:
                r = run_backtest_once(cp, merged_params, [strat])
                m = compute_metrics(r)
                train_metrics.append(m)
            agg_train = merge_metrics(train_metrics)

            if not passes_gates(agg_train):
                logs.append({"strategy": strat, "trial": t, "phase": "train", "params": params, "agg": agg_train, "status":"reject_gate"})
                continue

            # 検証期間で確認
            valid_metrics = []
            for cp in valid_files:
                r = run_backtest_once(cp, merged_params, [strat])
                m = compute_metrics(r)
                valid_metrics.append(m)
            agg_valid = merge_metrics(valid_metrics) if valid_metrics else agg_train

            sc = score(agg_valid)
            rec = {"strategy": strat, "trial": t, "params": params, "train": agg_train, "valid": agg_valid, "score": sc}
            logs.append({**rec, "status":"accepted"})

            if best is None or sc > best["score"]:
                best = rec

        if best:
            best_by_strategy[strat] = best

    # 出力
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = outdir / f"tuning_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp_utc": ts,
            "files_train": [p.name for p in train_files],
            "files_valid": [p.name for p in valid_files],
            "best_by_strategy": best_by_strategy,
            "logs": logs[-500:]  # ログは直近 500 件だけ
        }, f, ensure_ascii=False, indent=2)

    print(f"[INFO] tuning result saved: {out_path}")

    if args.record_db and best_by_strategy and get_connection and record_run:
        try:
            conn = get_connection(pathlib.Path(args.record_db))
            for strat, rec in best_by_strategy.items():
                record_run(
                    conn,
                    run_id=ts,
                    strategy=strat,
                    params=rec["params"],
                    train=rec["train"],
                    valid=rec["valid"],
                    score=rec["score"],
                    source_file=str(out_path),
                )
            conn.close()
            print(f"[INFO] recorded tuning results into {args.record_db}")
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] failed to record tuning result: {exc}", file=sys.stderr)

    if args.write_best and best_by_strategy:
        # 既存 active を読み込み、更新（存在しない場合は新規作成）
        active = {}
        if ACTIVE_PARAMS_PATH.exists():
            try:
                with open(ACTIVE_PARAMS_PATH, "r") as f:
                    active = json.load(f)
            except Exception:
                active = {}
        for strat, rec in best_by_strategy.items():
            active[strat] = rec["params"]
        tmp = ACTIVE_PARAMS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(active, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ACTIVE_PARAMS_PATH)
        print(f"[INFO] updated active params -> {ACTIVE_PARAMS_PATH}")

if __name__ == "__main__":
    main()
