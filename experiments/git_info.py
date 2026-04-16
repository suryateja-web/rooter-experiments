from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit(repo_path: str | Path) -> str:
    path = Path(repo_path)
    if not path.exists():
        return "missing"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - best-effort metadata.
        return "unknown"


def git_dirty(repo_path: str | Path) -> bool:
    path = Path(repo_path)
    if not path.exists():
        return False
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 - best-effort metadata.
        return False
    return bool(output.strip())

