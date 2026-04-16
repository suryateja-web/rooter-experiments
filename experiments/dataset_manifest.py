from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppRunSelection:
    session: dict[str, Any]
    app_run: dict[str, Any]


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def iter_app_runs(manifest: dict[str, Any]) -> list[AppRunSelection]:
    selections: list[AppRunSelection] = []
    for session in manifest.get("sessions", []):
        for app_run in session.get("app_runs", []):
            selections.append(AppRunSelection(session=session, app_run=app_run))
    return selections


def filter_app_runs(
    selections: list[AppRunSelection],
    selection_config: dict[str, Any],
) -> list[AppRunSelection]:
    session_ids = set(selection_config.get("session_ids") or [])
    app_run_ids = set(selection_config.get("app_run_ids") or [])
    collection_folders = set(selection_config.get("collection_folders") or [])
    detector_families = set(selection_config.get("detector_model_families") or [])

    filtered = []
    for item in selections:
        session = item.session
        app_run = item.app_run
        if session_ids and session.get("session_id") not in session_ids:
            continue
        if app_run_ids and app_run.get("run_id") not in app_run_ids:
            continue
        if collection_folders and session.get("collection_folder") not in collection_folders:
            continue
        if detector_families and app_run.get("detector_model_family") not in detector_families:
            continue
        filtered.append(item)

    return filtered

