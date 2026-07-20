#!/usr/bin/env python3
"""Deep-scan and seal the fixed DOJO long-horizon M5/M1 source roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from quant_rabbit.dojo_long_horizon_source_manifest import (  # noqa: E402
    DojoLongHorizonSourceManifestError,
    build_long_horizon_source_manifest,
    write_long_horizon_source_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m5-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = build_long_horizon_source_manifest(
            m5_root=args.m5_root,
            m1_root=args.m1_root,
        )
        write_long_horizon_source_manifest(args.output, manifest)
    except DojoLongHorizonSourceManifestError as exc:
        print(
            json.dumps(
                {
                    "status": "REJECTED",
                    "error": str(exc),
                    "live_permission": False,
                    "order_authority": "NONE",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "SEALED",
                "output": str(args.output),
                "source_manifest_sha256": manifest["source_manifest_sha256"],
                "binding_count": manifest["binding_count"],
                "physical_shard_count": manifest["physical_shard_count"],
                "plan_digest_inputs": manifest["plan_digest_inputs"],
                "live_permission": False,
                "order_authority": "NONE",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
