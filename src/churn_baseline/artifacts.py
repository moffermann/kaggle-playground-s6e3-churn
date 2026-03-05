"""I/O utilities for model artifacts and reports."""

import json
from pathlib import Path
from typing import Any, Dict


def ensure_parent_dir(path: str | Path) -> Path:
    """Ensure parent directory exists and return Path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    """Write a JSON payload with UTF-8 encoding and stable formatting."""
    out_path = ensure_parent_dir(path)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
