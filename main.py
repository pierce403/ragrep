#!/usr/bin/env python3
"""Local launcher for the packaged CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ragrep.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
