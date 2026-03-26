#!/usr/bin/env python3
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.inp, parse_dates=["entry_ts", "exit_ts"])
    total = len(df)
    if total == 0:
        print("No trades.")
        return
    ev = df["pips"].mean()
    counts = df["reason"].value_counts(normalize=True).mul(100).round(1).to_dict()
    print(f"Trades={total}, EV(pips)={ev:.3f}")
    print("Reason%:", counts)
    print(
        "Median events=",
        df["events"].median(),
        "Median grace(ms)=",
        df["grace_used_ms"].median(),
    )


if __name__ == "__main__":
    main()
