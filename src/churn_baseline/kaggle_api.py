"""Kaggle API helpers with robust token handling for KGAT_* keys."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from kaggle.api.kaggle_api_extended import KaggleApi


def _read_kaggle_credentials() -> Tuple[Optional[str], Optional[str]]:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return None, None
    data = json.loads(kaggle_json.read_text(encoding="utf-8"))
    username = data.get("username")
    key = data.get("key")
    if isinstance(username, str):
        username = username.strip()
    if isinstance(key, str):
        key = key.strip()
    return username or None, key or None


def _resolve_credentials() -> Tuple[Optional[str], Optional[str]]:
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if isinstance(username, str):
        username = username.strip()
    if isinstance(key, str):
        key = key.strip()
    if username and key:
        return username, key
    file_username, file_key = _read_kaggle_credentials()
    return username or file_username, key or file_key


def build_authenticated_api() -> KaggleApi:
    """Build KaggleApi with support for both legacy and KGAT tokens."""
    username, key = _resolve_credentials()
    if not key:
        raise RuntimeError("Kaggle credentials not found. Configure ~/.kaggle/kaggle.json")

    api = KaggleApi()
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key
        if username:
            os.environ["KAGGLE_USERNAME"] = username
            api.config_values["username"] = username
        api.config_values["token"] = key
        api.config_values["auth_method"] = "access_token"
    else:
        if username:
            os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        api.authenticate()
    return api


def submit_file(
    competition: str,
    file_path: str | Path,
    message: str,
) -> Dict[str, Any]:
    """Submit a CSV file to a Kaggle competition."""
    api = build_authenticated_api()
    response = api.competition_submit(str(file_path), message, competition)
    return {
        "message": getattr(response, "message", None),
        "ref": getattr(response, "ref", None),
    }
