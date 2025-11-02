#!/usr/bin/env python3
import argparse
import sys

from utils.yaml_merge import main as _merge_main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='config/tuning_presets.yaml')
    ap.add_argument('--over', default='config/tuning_overrides.yaml')
    ap.add_argument('--out', default='config/tuning_overlay.yaml')
    args = ap.parse_args()
    # reuse yaml_merge logic
    sys.argv = [
        "yaml_merge",
        "--base",
        args.base,
        "--over",
        args.over,
        "--out",
        args.out,
    ]
    _merge_main()

if __name__ == '__main__':
    main()
