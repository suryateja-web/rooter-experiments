#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.dataset_manifest import filter_app_runs, iter_app_runs, load_manifest
from experiments.git_info import git_commit, git_dirty


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def slug(value: str) -> str:
    return "_".join(
        "".join(char.lower() if char.isalnum() else "_" for char in value).split("_")
    )


def flatten_params(value: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flattened.update(flatten_params(item, name))
        else:
            flattened[name] = str(item)
    return flattened


def clean_param_value(value: Any) -> str:
    return "" if value is None else str(value)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def build_command(template: str, values: dict[str, str]) -> list[str]:
    rendered = template.format(**values)
    return shlex.split(rendered)


def render_run(
    config: dict[str, Any],
    session: dict[str, Any],
    app_run: dict[str, Any],
) -> tuple[str, Path, list[str]]:
    output_root = Path(config["output_root"])
    postprocessor = config["postprocessor"]

    local_run_name = (
        f"{slug(session['session_id'])}__{slug(app_run['run_id'])}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = output_root / local_run_name

    command_values = {
        "raw_json_path": app_run["raw_json_path"],
        "frames_path": session["frames_path"],
        "output_dir": str(output_dir),
        "session_id": session["session_id"],
        "app_run_id": app_run["run_id"],
    }
    command = build_command(postprocessor["command_template"], command_values)

    return local_run_name, output_dir, command


def run_one(config: dict[str, Any], session: dict[str, Any], app_run: dict[str, Any]) -> None:
    dataset_repo_path = config["dataset_repo_path"]
    postprocessor_repo_path = config["postprocessor_repo_path"]
    postprocessor = config["postprocessor"]
    local_run_name, output_dir, command = render_run(config, session, app_run)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "session_id": session["session_id"],
        "app_run_id": app_run["run_id"],
        "raw_json_path": app_run["raw_json_path"],
        "frames_path": session["frames_path"],
        "output_dir": str(output_dir),
        "command": command,
        "config": config,
    }
    write_json(output_dir / "run_input.json", metadata)

    with mlflow.start_run(run_name=local_run_name):
        mlflow.set_tags(
            {
                "runner": "postprocessor",
                "session_id": session["session_id"],
                "app_run_id": app_run["run_id"],
                "collection_folder": session.get("collection_folder", ""),
                "detector_model_family": app_run.get("detector_model_family", ""),
                "ocr_model_family": app_run.get("ocr_model_family", ""),
                "algo_variant": app_run.get("algo_variant", ""),
                "dataset_git_commit": git_commit(dataset_repo_path),
                "dataset_git_dirty": str(git_dirty(dataset_repo_path)),
                "postprocessor_git_commit": git_commit(postprocessor_repo_path),
                "postprocessor_git_dirty": str(git_dirty(postprocessor_repo_path)),
            }
        )
        mlflow.log_params(flatten_params(config.get("params", {})))
        mlflow.log_params(
            {
                "postprocessor.name": clean_param_value(postprocessor.get("name", "")),
                "postprocessor.version": clean_param_value(
                    postprocessor.get("version", ""),
                ),
                "frame_count": clean_param_value(session.get("frame_count", 0)),
                "app_run.json_entries": clean_param_value(app_run.get("json_entries")),
                "app_run.total_detections": clean_param_value(
                    app_run.get("total_detections"),
                ),
                "app_run.matched_frame_count": clean_param_value(
                    app_run.get("matched_frame_count"),
                ),
            }
        )
        mlflow.log_artifact(output_dir / "run_input.json")

        started_at = datetime.now(timezone.utc).isoformat()
        process = subprocess.run(
            command,
            cwd=postprocessor_repo_path,
            text=True,
            capture_output=True,
            check=False,
        )
        completed_at = datetime.now(timezone.utc).isoformat()

        (output_dir / "stdout.txt").write_text(process.stdout)
        (output_dir / "stderr.txt").write_text(process.stderr)
        write_json(
            output_dir / "run_result.json",
            {
                "returncode": process.returncode,
                "started_at": started_at,
                "completed_at": completed_at,
            },
        )

        mlflow.log_metric("returncode", process.returncode)
        mlflow.log_artifacts(str(output_dir))

        if process.returncode != 0:
            raise RuntimeError(
                f"Postprocessor failed for {session['session_id']} / "
                f"{app_run['run_id']} with return code {process.returncode}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    manifest = load_manifest(config["dataset_manifest_path"])
    selections = filter_app_runs(
        iter_app_runs(manifest),
        config.get("selection", {}),
    )

    print(f"Selected app runs: {len(selections)}")
    for item in selections:
        _local_run_name, output_dir, command = render_run(
            config,
            item.session,
            item.app_run,
        )
        print(f"{item.session['session_id']} / {item.app_run['run_id']}")
        print(f"  output_dir: {output_dir}")
        print(f"  command: {' '.join(shlex.quote(part) for part in command)}")

    if args.dry_run:
        return

    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])

    for item in selections:
        print(f"Running {item.session['session_id']} / {item.app_run['run_id']}")
        run_one(config, item.session, item.app_run)


if __name__ == "__main__":
    main()
