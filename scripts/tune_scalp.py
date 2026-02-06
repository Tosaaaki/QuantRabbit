#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_scalp.py
- ランダムサーチ＋ウォークフォワード評価でスカルプ戦略のパラメータをチューニング
- backtest_scalp.py に --json-out / --params-json / --strategies を追加するパッチ前提（同梱パッチ参照）
- 既存の backtest_scalp.py を import できる場合は run_backtest() を直接呼ぶが、
  そうでなければサブプロセスで backtest_scalp.py を起動して JSON を拾う
"""
from __future__ import annotations

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


def _load_autotune_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        from utils.secrets import get_secret
    except Exception:
        return ""
    for key in (name, name.lower()):
        try:
            candidate = get_secret(key)
        except Exception:
            continue
        if candidate:
            return str(candidate).strip()
    return ""


from autotune.database import AUTOTUNE_BQ_TABLE, record_run_bigquery

StrategyParams = Dict[str, Dict[str, Any]]
ResultDict = Dict[str, Any]

PROFILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "scalp": {
        "label": "Scalping (M1)",
        "timeframe": "M1",
        "strategies": ["PulseBreak", "RangeFader"],
        "candles_glob": "candles_M1_*.json",
        "valid_ratio": 0.3,
        "trials_per_strategy": 40,
        "min_trades": 8,
        "profit_factor_min": 1.05,
        "max_dd_pips": 12.0,
        "file_lookback": 12,
        "dd_penalty": 0.02,
        "pf_cap": 1.6,
        "dd_anchor_pips": 6.0,
    },
    "micro": {
        "label": "Micro (M5)",
        "timeframe": "M5",
        "strategies": ["BB_RSI"],
        "candles_glob": "candles_M1_*.json",
        "valid_ratio": 0.3,
        "trials_per_strategy": 35,
        "min_trades": 6,
        "profit_factor_min": 1.05,
        "max_dd_pips": 18.0,
        "file_lookback": 14,
        "dd_penalty": 0.015,
        "pf_cap": 1.7,
        "dd_anchor_pips": 10.0,
    },
    "macro": {
        "label": "Macro (H4)",
        "timeframe": "H4",
        "strategies": ["TrendMA", "Donchian55"],
        "candles_glob": "candles_M1_*.json",
        "valid_ratio": 0.4,
        "trials_per_strategy": 28,
        "min_trades": 4,
        "profit_factor_min": 1.08,
        "max_dd_pips": 240.0,
        "file_lookback": 20,
        "dd_penalty": 0.01,
        "pf_cap": 1.8,
        "dd_anchor_pips": 120.0,
    },
}

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
            precision = int(spec.get("precision", 2))
            params[k] = round(rng.uniform(lo, hi), precision)
        elif t == "choice":
            params[k] = rng.choice(list(spec["choices"]))
        else:
            raise ValueError(f"Unsupported type in param space: {t}")
    return params

def list_candle_files(
    candles_dir: pathlib.Path, pattern: str, limit: int | None = None
) -> List[pathlib.Path]:
    files = sorted(candles_dir.glob(pattern))
    filtered = [p for p in files if p.is_file()]
    if limit is not None and limit > 0:
        return filtered[-limit:]
    return filtered

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

def run_backtest_once(
    candles_path: pathlib.Path,
    strategies_params: StrategyParams,
    strategies: List[str],
    *,
    timeframe: str,
) -> ResultDict:
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
                strategies=strategies,
                timeframe=timeframe,
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
            sys.executable,
            str(BACKTEST_SCRIPT),
            "--candles",
            str(candles_path),
            "--strategies",
            ",".join(strategies),
            "--params-json",
            str(params_json_path),
            "--json-out",
            str(tmp_json),
            "--timeframe",
            timeframe,
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
      "by_strategy": {"PulseBreak": {...}, "RangeFader": {...}},
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

def passes_gates(agg: Dict[str, Any], profile_cfg: Dict[str, Any]) -> bool:
    min_trades = int(profile_cfg.get("min_trades", 8))
    pf_min = float(profile_cfg.get("profit_factor_min", 1.05))
    dd_limit = float(profile_cfg.get("max_dd_pips", 12.0))

    if agg["trades"] < min_trades:
        return False
    if agg["profit_factor"] < pf_min:
        return False
    if agg["max_dd_pips"] > dd_limit:
        return False
    return True


def score(agg: Dict[str, Any], profile_cfg: Dict[str, Any]) -> float:
    pf_cap = float(profile_cfg.get("pf_cap", 1.6))
    dd_penalty = float(profile_cfg.get("dd_penalty", 0.02))
    dd_anchor = float(profile_cfg.get("dd_anchor_pips", 6.0))

    pf = min(agg["profit_factor"], pf_cap)
    wr = agg["win_rate"]
    dd = agg["max_dd_pips"]
    base = pf
    bonus = 0.15 * max(0.0, wr - 0.5)
    penalty = dd_penalty * max(0.0, dd - dd_anchor)
    return base + bonus - penalty

def walk_forward_split(files: List[pathlib.Path], valid_ratio: float = 0.3) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
    n = len(files)
    if n <= 1:
        return files, []
    k = max(1, int(n * (1 - valid_ratio)))
    return files[:k], files[k:]

def main():
    profile_choices = ", ".join(PROFILE_CONFIGS.keys())
    ap = argparse.ArgumentParser()
    ap.add_argument("--candles-dir", default=str(DEFAULT_LOG_DIR), help="logs ディレクトリのパス")
    ap.add_argument("--profile", default="scalp", help=f"対象プロファイル（{profile_choices} または 'all'）")
    ap.add_argument("--glob", default="", help="candles ファイルの glob。空ならプロファイル既定を使用")
    ap.add_argument("--dates", default="", help="'20251020,20251021' のように日付指定。空なら最新を自動選択")
    ap.add_argument("--strategies", default="", help="カンマ区切りで明示指定（単一プロファイル時のみ有効）")
    ap.add_argument("--trials-per-strategy", type=int, default=0, help="各ストラテジーの試行数。0 ならプロファイル既定")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-best", action="store_true", help="改善あれば configs/scalp_active_params.json を上書き")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "logs" / "tuning"))
    ap.add_argument(
        "--record-db",
        default="",
        help="SQLite 記録は無効（BQのみ）。空文字固定。",
    )
    ap.add_argument(
        "--bq-table",
        default="",
        help="結果を BigQuery に記録するテーブル (project.dataset.table)。空文字で無効化",
    )
    args = ap.parse_args()

    bq_table = args.bq_table or AUTOTUNE_BQ_TABLE
    if args.record_db:
        print("[ERR] --record-db is disabled (BigQuery only).", file=sys.stderr)
        sys.exit(2)
    if not bq_table:
        print("[ERR] --bq-table or AUTOTUNE_BQ_TABLE is required.", file=sys.stderr)
        sys.exit(2)
    candles_dir = pathlib.Path(args.candles_dir)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    space_all = _load_param_space(PARAM_SPACE_PATH)

    profile_input = args.profile.lower().strip()
    if profile_input == "all":
        profile_names = list(PROFILE_CONFIGS.keys())
    else:
        profile_names = [name.strip() for name in args.profile.split(",") if name.strip()]
    if not profile_names:
        print("[ERR] プロファイルが指定されていません", file=sys.stderr)
        sys.exit(2)
    for name in profile_names:
        if name not in PROFILE_CONFIGS:
            print(f"[ERR] 未知のプロファイルです: {name}", file=sys.stderr)
            sys.exit(2)

    user_strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if user_strategies and len(profile_names) > 1:
        print("[ERR] --strategies は単一プロファイル指定時のみ利用できます", file=sys.stderr)
        sys.exit(2)

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    aggregated_best: Dict[str, Dict[str, Any]] = {}
    profile_results: Dict[str, Dict[str, Any]] = {}

    for idx, profile_name in enumerate(profile_names):
        cfg = PROFILE_CONFIGS[profile_name]
        pattern = args.glob or cfg.get("candles_glob", DEFAULT_CANDLES_GLOB)
        files_all = list_candle_files(candles_dir, pattern)
        if args.dates:
            target_dates = set(d.strip() for d in args.dates.split(",") if d.strip())
            files = [p for p in files_all if any(d in p.name for d in target_dates)]
        else:
            lookback = cfg.get("file_lookback")
            files = files_all[-int(lookback) :] if lookback else files_all

        if not files:
            print(f"[WARN] profile={profile_name}: candles が見つかりません (pattern={pattern})", file=sys.stderr)
            continue

        valid_ratio = float(cfg.get("valid_ratio", 0.3))
        train_files, valid_files = walk_forward_split(files, valid_ratio=valid_ratio)

        selected_strategies = user_strategies or cfg.get("strategies", [])
        if not selected_strategies:
            print(f"[WARN] profile={profile_name}: strategies が未設定のためスキップします", file=sys.stderr)
            continue

        trials = args.trials_per_strategy or int(cfg.get("trials_per_strategy", 40))
        best_by_strategy: Dict[str, Dict[str, Any]] = {}
        logs: List[Dict[str, Any]] = []

        for strat in selected_strategies:
            space = space_all.get(strat)
            if not space:
                print(f"[WARN] パラメータ空間が未定義: {strat}", file=sys.stderr)
                continue

            best: Dict[str, Any] | None = None
            for t in range(trials):
                seed_val = args.seed + idx * 1000 + t
                params = sample_params_for_strategy(space, seed=seed_val)
                merged_params = {strat: params}

                train_metrics = []
                for cp in train_files:
                    result = run_backtest_once(cp, merged_params, [strat], timeframe=cfg["timeframe"])
                    train_metrics.append(compute_metrics(result))
                agg_train = merge_metrics(train_metrics)

                if not passes_gates(agg_train, cfg):
                    logs.append(
                        {
                            "profile": profile_name,
                            "timeframe": cfg["timeframe"],
                            "strategy": strat,
                            "trial": t,
                            "phase": "train",
                            "params": params,
                            "agg": agg_train,
                            "status": "reject_gate",
                        }
                    )
                    continue

                valid_metrics = []
                for cp in valid_files:
                    result = run_backtest_once(cp, merged_params, [strat], timeframe=cfg["timeframe"])
                    valid_metrics.append(compute_metrics(result))
                agg_valid_raw = merge_metrics(valid_metrics) if valid_metrics else dict(agg_train)

                train_payload = dict(agg_train)
                train_payload.update({"profile": profile_name, "timeframe": cfg["timeframe"], "window": "train"})
                valid_payload = dict(agg_valid_raw)
                valid_payload.update({"profile": profile_name, "timeframe": cfg["timeframe"], "window": "valid"})

                sc = score(agg_valid_raw, cfg)
                rec = {
                    "strategy": strat,
                    "trial": t,
                    "params": params,
                    "train": train_payload,
                    "valid": valid_payload,
                    "score": sc,
                    "profile": profile_name,
                    "timeframe": cfg["timeframe"],
                }
                logs.append({**rec, "status": "accepted"})

                if best is None or sc > best["score"]:
                    best = rec

            if best:
                best_by_strategy[strat] = best

        profile_results[profile_name] = {
            "timeframe": cfg["timeframe"],
            "files_train": [p.name for p in train_files],
            "files_valid": [p.name for p in valid_files],
            "best_by_strategy": best_by_strategy,
            "logs": logs[-500:],
        }

        for strat, rec in best_by_strategy.items():
            existing = aggregated_best.get(strat)
            if not existing or rec["score"] > existing["score"]:
                aggregated_best[strat] = rec

    if not aggregated_best:
        print("[ERR] 有効なチューニング結果が得られませんでした", file=sys.stderr)
        sys.exit(3)

    out_payload = {
        "timestamp_utc": ts,
        "profiles": profile_results,
        "best_by_strategy": aggregated_best,
    }

    out_path = outdir / f"tuning_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] tuning result saved: {out_path}")

    try:
        for strat, rec in aggregated_best.items():
            run_id = f"{rec['profile']}-{ts}"
            record_run_bigquery(
                run_id=run_id,
                strategy=strat,
                params=rec["params"],
                train=rec["train"],
                valid=rec["valid"],
                score=rec["score"],
                source_file=str(out_path),
                table_override=bq_table,
            )
        print(f"[INFO] recorded tuning results into BigQuery table {bq_table}")
    except Exception as exc:  # pragma: no cover
        print(f"[ERR] failed to record tuning result to BigQuery: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.write_best:
        active: Dict[str, Any] = {}
        if ACTIVE_PARAMS_PATH.exists():
            try:
                with open(ACTIVE_PARAMS_PATH, "r") as f:
                    active = json.load(f)
            except Exception:
                active = {}
        for strat, rec in aggregated_best.items():
            active[strat] = rec["params"]
        tmp = ACTIVE_PARAMS_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(active, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ACTIVE_PARAMS_PATH)
        print(f"[INFO] updated active params -> {ACTIVE_PARAMS_PATH}")


if __name__ == "__main__":
    main()
