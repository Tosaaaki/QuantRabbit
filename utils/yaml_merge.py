#!/usr/bin/env python3
import argparse
import os
import sys

import yaml

def deep_update(base, over):
    for k, v in over.items():
        if isinstance(v, dict):
            base[k] = deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--over', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.base, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    with open(args.over, "r", encoding="utf-8") as f:
        over = yaml.safe_load(f) or {}
    merged = deep_update(base, over)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    print(f"[yaml_merge] wrote {args.out}")

if __name__ == '__main__':
    main()
