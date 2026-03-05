"""Runtime bootstrap utilities for CLI scripts."""

import sys
from pathlib import Path


def add_src_to_path() -> None:
    """Ensure `src/` is importable when running scripts directly."""
    project_root = Path(__file__).resolve().parents[1]
    src_dir = project_root / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
