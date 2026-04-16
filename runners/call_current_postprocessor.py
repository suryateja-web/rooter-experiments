#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from os import path as os_path
from pathlib import Path
from typing import Any


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def flatten_detections(raw: Any, natural_key) -> tuple[list[dict], list[str]]:
    if not isinstance(raw, list):
        raise ValueError("Detections JSON must be a list")
    if not raw:
        return [], []

    first = raw[0]
    if isinstance(first, dict) and "detections" in first:
        flattened: list[dict] = []
        frames: list[str] = []
        seen = set()
        for frame in raw:
            if not isinstance(frame, dict):
                continue
            frame_name = frame.get("fileName") or frame.get("image") or ""
            frame_base = os_path.basename(str(frame_name)) if frame_name else ""
            if frame_base and frame_base not in seen:
                seen.add(frame_base)
                frames.append(frame_base)
            dets = frame.get("detections") or []
            if not frame_base or not isinstance(dets, list):
                continue
            for det in dets:
                if not isinstance(det, dict):
                    continue
                det_copy = dict(det)
                det_copy["image"] = frame_base
                flattened.append(det_copy)
        return flattened, sorted(frames, key=natural_key)

    detections = [item for item in raw if isinstance(item, dict)]
    frames = []
    seen = set()
    for detection in detections:
        image = detection.get("image") or detection.get("fileName") or ""
        frame_base = os_path.basename(str(image)) if image else ""
        if frame_base and frame_base not in seen:
            seen.add(frame_base)
            frames.append(frame_base)
        if frame_base:
            detection["image"] = frame_base
    return detections, sorted(frames, key=natural_key)


def run_current_postprocessor(
    input_path: Path,
    postprocessor_repo: Path,
    debug_dir: Path | None = None,
) -> dict:
    sys.path.insert(0, str(postprocessor_repo))

    from lambda_session_processor.match_pipeline import load_default_rules, run_pipeline
    from lambda_session_processor.match_pipeline.apply_sort import natural_key

    raw = json.loads(input_path.read_text())
    detections, frame_order = flatten_detections(raw, natural_key)
    if not detections and not frame_order:
        return {"matches": [], "frames_total": 0}

    target_states, rules = load_default_rules()
    if not target_states or not rules:
        raise RuntimeError("Match rules are not configured.")

    def ocr_hook(det: dict, frame: str) -> str:
        text = det.get("ocrText")
        return text if isinstance(text, str) else ""

    result = run_pipeline(
        detections,
        target_states,
        rules,
        ocr_hook=ocr_hook,
        frame_order_override=frame_order,
        debug_dir=debug_dir,
    )
    metrics = result.get("metrics") or {}
    return {
        "matches": metrics.get("matches", []),
        "frames_total": metrics.get("frames_total", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw app detection/OCR JSON path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--postprocessor-repo",
        default="/home/ec2-user/rooter-passport-postprocessor",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write postprocessor intermediate debug artifacts",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    postprocessor_repo = Path(args.postprocessor_repo).resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not postprocessor_repo.exists():
        raise FileNotFoundError(postprocessor_repo)

    started_at = datetime.now(timezone.utc).isoformat()
    debug_dir = output_dir / "postprocessor_debug" if args.debug else None
    result = run_current_postprocessor(input_path, postprocessor_repo, debug_dir=debug_dir)
    completed_at = datetime.now(timezone.utc).isoformat()

    matches = result.get("matches", []) if isinstance(result, dict) else []
    frames_total = result.get("frames_total", 0) if isinstance(result, dict) else 0

    write_json(output_dir / "result.json", result)
    write_json(
        output_dir / "summary.json",
        {
            "input_path": str(input_path),
            "postprocessor_repo": str(postprocessor_repo),
            "started_at": started_at,
            "completed_at": completed_at,
            "frames_total": frames_total,
            "match_count": len(matches) if isinstance(matches, list) else 0,
            "debug_enabled": args.debug,
            "debug_dir": str(debug_dir) if debug_dir else None,
        },
    )


if __name__ == "__main__":
    main()
