"""Shared fixtures for annotation tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path for non-installed package
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
