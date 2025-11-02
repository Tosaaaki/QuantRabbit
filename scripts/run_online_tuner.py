#!/usr/bin/env python3
import argparse

from autotune.online_tuner import OnlineTuner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs-glob', required=True, help='e.g., logs/trades/*.jsonl or tmp/exit_eval*.csv')
    ap.add_argument('--presets', default='config/tuning_presets.yaml')
    ap.add_argument('--overrides-out', default='config/tuning_overrides.yaml')
    ap.add_argument('--history-dir', default='config/tuning_history')
    ap.add_argument('--minutes', type=int, default=15)
    ap.add_argument('--shadow', action='store_true')
    args = ap.parse_args()

    tuner = OnlineTuner(presets_path=args.presets, overrides_out=args.overrides_out,
                        history_dir=args.history_dir, minutes=args.minutes)
    hist = tuner.run_once(args.logs_glob, shadow=args.shadow)
    print(f'[online_tuner] done. history={hist} shadow={args.shadow}')

if __name__ == '__main__':
    main()
