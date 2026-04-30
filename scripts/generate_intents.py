#!/usr/bin/env python3
from __future__ import annotations

import sys

from quant_rabbit.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate-intents", *sys.argv[1:]]))
