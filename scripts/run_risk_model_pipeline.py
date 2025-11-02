#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from analytics.risk_model import RiskModelPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vertex/BigQuery risk model pipeline.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training step.")
    parser.add_argument("--no-publish", action="store_true", help="Do not publish results to Pub/Sub.")
    parser.add_argument(
        "--state",
        dest="state_path",
        help="Override state output path (default: logs/risk_scores.json).",
    )
    args = parser.parse_args()

    pipeline = RiskModelPipeline(state_path=args.state_path)
    scores = pipeline.run(train=not args.skip_train, publish=not args.no_publish)
    print(json.dumps({"count": len(scores), "scores": [s.__dict__ for s in scores]}, indent=2))


if __name__ == "__main__":
    main()
