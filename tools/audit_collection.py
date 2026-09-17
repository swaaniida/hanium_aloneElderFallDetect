from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit collected TSSTG pose events")
    parser.add_argument("subject_root", type=Path)
    return parser.parse_args()


def main() -> None:
    root = parse_args().subject_root
    events_path = root / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(event["plan_index"]) for event in events]
    missing_files = []
    bad_hashes = []
    bad_samples = []
    duplicate_ids = []
    durations = []
    valid_ratios = []
    seen_ids = set()

    for event in events:
        event_id = event["event_id"]
        if event_id in seen_ids:
            duplicate_ids.append(event_id)
        seen_ids.add(event_id)

        pose_path = root / event["pose_file"]
        if not pose_path.exists():
            missing_files.append(event_id)
            continue
        digest = hashlib.sha256(pose_path.read_bytes()).hexdigest()
        if digest != event["pose_sha256"]:
            bad_hashes.append(event_id)

        with np.load(pose_path, allow_pickle=False) as data:
            count = len(data["timestamps_ns"])
            pose_hz = float(data["pose_hz"]) if "pose_hz" in data else 25.0
            duration = (
                int(data["timestamps_ns"][-1]) - int(data["timestamps_ns"][0])
            ) / 1e9
            durations.append(duration)
            valid_ratios.append(float(np.mean(data["pose_valid"])))
            if count != 100 or abs(pose_hz - 25.0) > 1e-6:
                bad_samples.append((event_id, count, pose_hz))

    report = {
        "events": len(events),
        "plan_index_min": min(indices) if indices else None,
        "plan_index_max": max(indices) if indices else None,
        "unique_plan_indices": len(set(indices)),
        "locations": dict(collections.Counter(e["location"] for e in events)),
        "location_labels": {
            f"{location}/{label}": count
            for (location, label), count in sorted(
                collections.Counter(
                    (e["location"], e["label"]) for e in events
                ).items()
            )
        },
        "location_actions": {
            f"{location}/{action}": count
            for (location, action), count in sorted(
                collections.Counter(
                    (e["location"], e["action_code"]) for e in events
                ).items()
            )
        },
        "missing_files": missing_files,
        "bad_hashes": bad_hashes,
        "bad_samples": bad_samples,
        "duplicate_event_ids": duplicate_ids,
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "mean": sum(durations) / len(durations) if durations else None,
            "max": max(durations) if durations else None,
        },
        "valid_pose_ratio": {
            "mean": sum(valid_ratios) / len(valid_ratios) if valid_ratios else None,
            "min": min(valid_ratios) if valid_ratios else None,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    expected_indices = set(range(1, 201))
    valid = (
        len(events) == 200
        and set(indices) == expected_indices
        and not missing_files
        and not bad_hashes
        and not bad_samples
        and not duplicate_ids
    )
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
