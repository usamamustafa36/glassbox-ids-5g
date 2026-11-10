#!/usr/bin/env python3
"""Explainable Intrusion Detection for 5G - command-line entry point.

Usage:
    python run.py --help
    python run.py verify
    python run.py all

Works identically on Windows, macOS and Linux. No environment variables and no
working-directory assumptions: everything resolves from this file's location.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from xids5g.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
