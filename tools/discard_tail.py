from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def discard_tail(root: Path, first_index: int, last_index: int) -> Path:
    events_path = root / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    discard_ids = {
        f"{events[0]['subject_id']}_E{index:03d}"
        for index in range(first_index, last_index + 1)
    }
    actual_tail_ids = {event["event_id"] for event in events[first_index - 1:]}
    if actual_tail_ids != discard_ids or len(events) != last_index:
        raise RuntimeError(
            "For safety, only the exact current tail can be discarded; "
            f"expected {sorted(discard_ids)}, got {sorted(actual_tail_ids)}"
        )

    backup = root / time.strftime(
        f"discard_backup_E{first_index:03d}_E{last_index:03d}_%Y%m%d_%H%M%S"
    )
    backup_samples = backup / "samples"
    backup_samples.mkdir(parents=True)
    shutil.copy2(events_path, backup / events_path.name)

    for event in events[first_index - 1:]:
        source = root / event["pose_file"]
        shutil.move(str(source), str(backup_samples / source.name))

    for name in ("latest_review.json", "latest_review.mp4", "latest_buffer.mp4"):
        source = root / name
        if source.exists():
            shutil.move(str(source), str(backup / name))

    kept = events[:first_index - 1]
    temporary = events_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(
            json.dumps(event, ensure_ascii=False) + "\n" for event in kept
        ),
        encoding="utf-8",
    )
    temporary.replace(events_path)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    args = parser.parse_args()
    backup = discard_tail(args.root, args.first, args.last)
    print(f"Discard complete. Backup: {backup}")


if __name__ == "__main__":
    main()
