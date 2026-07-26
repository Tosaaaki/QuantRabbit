#!/usr/bin/env python3
"""Run one trusted paper-only replay worker.

The Ed25519 private-key path is accepted only from a fixed environment
variable.  The worker itself requires that key to be a 0600 regular file owned
by the current uid and located outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from quant_rabbit.dojo_replay_worker import (
    TrustedReplayWorkerError,
    run_trusted_replay_worker,
)


PRIVATE_KEY_PATH_ENV = "QR_DOJO_REPLAY_WORKER_PRIVATE_KEY_PATH"
REPOSITORY_ROOT = Path(__file__).resolve(strict=True).parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one externally signed exact-bid/ask paper replay."
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--worker-config", required=True, type=Path)
    args = parser.parse_args(argv)
    private_key_value = os.environ.get(PRIVATE_KEY_PATH_ENV)
    if not private_key_value:
        parser.error(f"{PRIVATE_KEY_PATH_ENV} is required")
    private_key_path = Path(private_key_value)
    if not private_key_path.is_absolute():
        parser.error(f"{PRIVATE_KEY_PATH_ENV} must be an absolute path")

    try:
        result = run_trusted_replay_worker(
            REPOSITORY_ROOT,
            candidate_id=args.candidate_id,
            worker_config_path=args.worker_config,
            private_key_path=private_key_path,
        )
    except TrustedReplayWorkerError as exc:
        parser.error(str(exc))
    sys.stdout.write(
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
