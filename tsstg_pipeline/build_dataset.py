from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


WINDOW_FRAMES = 30
CLASS_NAMES = np.asarray(["NON_FALL", "FALL"])
COCO_13_INDICES = np.asarray(
    [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    dtype=np.int64,
)
SPLIT_BY_REPETITION = {
    **{repetition: "train" for repetition in range(1, 15)},
    15: "validation",
    16: "validation",
    **{repetition: "test" for repetition in range(17, 21)},
}


def assign_event_splits(
    events: list[dict], split_mode: str, split_seed: int
) -> dict[str, str]:
    if split_mode == "pilot100":
        groups: dict[str, list[dict]] = {}
        for event in events:
            groups.setdefault(event["action_code"], []).append(event)
        if len(groups) != 10 or any(len(group) != 10 for group in groups.values()):
            counts = {key: len(value) for key, value in groups.items()}
            raise ValueError(
                "pilot100 requires exactly 10 events for each of 10 subtypes; "
                f"got {counts}"
            )
        rng = np.random.default_rng(split_seed)
        assignments = {}
        for action_code in sorted(groups):
            group = sorted(groups[action_code], key=lambda event: event["event_id"])
            shuffled = [group[index] for index in rng.permutation(len(group))]
            for event in shuffled[:8]:
                assignments[event["event_id"]] = "train"
            assignments[shuffled[8]["event_id"]] = "validation"
            assignments[shuffled[9]["event_id"]] = "test"
        return assignments

    if split_mode == "location_stratified":
        locations = sorted({event["location"] for event in events})
        if len(locations) != 2:
            raise ValueError(
                "location_stratified requires exactly two locations; "
                f"got {locations}"
            )
        assignments = {}
        for location in locations:
            location_events = [
                event for event in events if event["location"] == location
            ]
            # Resetting the same seed per location makes these assignments
            # identical to building each 100-event location independently.
            assignments.update(
                assign_event_splits(location_events, "pilot100", split_seed)
            )
        return assignments

    assignments = {}
    for event in events:
        repetition = int(event["repetition"])
        if repetition not in SPLIT_BY_REPETITION:
            raise ValueError(f"Invalid repetition for full200: {repetition}")
        assignments[event["event_id"]] = SPLIT_BY_REPETITION[repetition]
    return assignments


def moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return values.astype(np.float32, copy=True)
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid").astype(
        np.float32
    )


def interpolate_keypoints(
    xy: np.ndarray,
    confidence: np.ndarray,
    pose_valid: np.ndarray,
    minimum_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate missing joint coordinates without inventing confidence."""
    xy = np.asarray(xy, np.float32).copy()
    confidence = np.asarray(confidence, np.float32).copy()
    pose_valid = np.asarray(pose_valid, bool)
    confidence[~pose_valid] = 0.0
    frame_indices = np.arange(len(xy), dtype=np.float32)

    for joint in range(xy.shape[1]):
        valid = (
            pose_valid
            & np.isfinite(xy[:, joint]).all(axis=1)
            & (confidence[:, joint] >= minimum_confidence)
        )
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            xy[:, joint] = 0.0
            confidence[:, joint] = 0.0
            continue
        for coordinate in range(2):
            xy[:, joint, coordinate] = np.interp(
                frame_indices,
                valid_indices.astype(np.float32),
                xy[valid_indices, joint, coordinate],
            )
        confidence[~valid, joint] = 0.0
    return xy, confidence


def pose_motion_signals(
    xy: np.ndarray,
    confidence: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Create camera-scale-independent signals using only joint positions."""
    width, height = image_size
    normalized = xy.copy()
    normalized[:, :, 0] /= float(width)
    normalized[:, :, 1] /= float(height)

    visible = confidence > 0.0
    body_height = np.empty(len(xy), np.float32)
    for frame in range(len(xy)):
        y = normalized[frame, visible[frame], 1]
        if len(y) >= 4:
            body_height[frame] = max(
                float(np.percentile(y, 90) - np.percentile(y, 10)), 0.05
            )
        else:
            body_height[frame] = np.nan
    finite_height = np.isfinite(body_height)
    if not finite_height.any():
        body_height[:] = 0.25
    else:
        body_height = np.interp(
            np.arange(len(body_height)),
            np.flatnonzero(finite_height),
            body_height[finite_height],
        ).astype(np.float32)
    body_height = moving_average(body_height)

    # Shoulders and hips remain stable when hands/feet are briefly occluded.
    torso_y = np.median(normalized[:, [5, 6, 11, 12], 1], axis=1)
    torso_y = moving_average(torso_y)
    downward_velocity = np.zeros(len(xy), np.float32)
    downward_velocity[1:] = np.diff(torso_y) / np.maximum(
        body_height[:-1], 0.05
    )
    downward_velocity = moving_average(downward_velocity, 3)

    displacement = np.linalg.norm(np.diff(normalized, axis=0), axis=2)
    motion = np.zeros(len(xy), np.float32)
    motion[1:] = np.median(displacement, axis=1) / np.maximum(
        body_height[:-1], 0.05
    )
    motion = moving_average(motion, 3)

    shoulder_mid = normalized[:, [5, 6]].mean(axis=1)
    hip_mid = normalized[:, [11, 12]].mean(axis=1)
    torso_vector = shoulder_mid - hip_mid
    horizontal_posture = np.abs(torso_vector[:, 0]) / np.maximum(
        np.linalg.norm(torso_vector, axis=1), 1e-6
    )
    horizontal_posture = moving_average(horizontal_posture)
    return {
        "normalized_xy": normalized,
        "body_height": body_height,
        "torso_y": torso_y,
        "downward_velocity": downward_velocity,
        "motion": motion,
        "horizontal_posture": horizontal_posture,
    }


def estimate_fall_markers(signals: dict[str, np.ndarray]) -> tuple[int, int]:
    velocity = signals["downward_velocity"]
    torso_y = signals["torso_y"]
    peak = int(np.argmax(velocity))
    threshold = max(float(velocity[peak]) * 0.20, 0.004)

    fall_start = peak
    quiet_count = 0
    for frame in range(peak - 1, -1, -1):
        if velocity[frame] <= threshold:
            quiet_count += 1
            if quiet_count >= 3:
                fall_start = frame + 3
                break
        else:
            quiet_count = 0

    impact_search_end = min(len(torso_y), peak + WINDOW_FRAMES)
    impact = peak + int(np.argmax(torso_y[peak:impact_search_end]))
    return fall_start, impact


def choose_window(
    signals: dict[str, np.ndarray],
    label: str,
) -> tuple[int, float, int, int]:
    frame_count = len(signals["torso_y"])
    if frame_count < WINDOW_FRAMES:
        raise ValueError(
            f"Need at least {WINDOW_FRAMES} frames, got {frame_count}"
        )
    fall_start, impact = estimate_fall_markers(signals)
    scores = []
    for start in range(frame_count - WINDOW_FRAMES + 1):
        stop = start + WINDOW_FRAMES
        torso_y = signals["torso_y"][start:stop]
        body_height = float(np.median(signals["body_height"][start:stop]))
        descent = float(torso_y[-5:].mean() - torso_y[:5].mean()) / max(
            body_height, 0.05
        )
        peak_down = float(
            np.max(signals["downward_velocity"][start:stop])
        )
        posture = signals["horizontal_posture"][start:stop]
        posture_change = float(posture[-5:].mean() - posture[:5].mean())
        motion = float(np.mean(signals["motion"][start:stop]))
        if label == "FALL":
            includes_markers = start <= fall_start <= impact < stop
            score = (
                2.5 * max(descent, 0.0)
                + 1.5 * max(peak_down, 0.0)
                + 0.8 * max(posture_change, 0.0)
                + 0.25 * motion
                + (0.5 if includes_markers else 0.0)
            )
        else:
            # Pick the performed action, not the idle preparation/recovery part.
            score = motion + 0.15 * abs(descent) + 0.10 * abs(posture_change)
        scores.append(score)
    start = int(np.argmax(np.asarray(scores)))
    if label != "FALL":
        fall_start = impact = -1
    return start, float(scores[start]), fall_start, impact


def make_tsstg_feature(
    xy: np.ndarray,
    confidence: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Match ActionsEstLoader preprocessing and produce (T, 14, 3)."""
    points = np.concatenate(
        (xy[:, COCO_13_INDICES], confidence[:, COCO_13_INDICES, None]),
        axis=2,
    ).astype(np.float32)
    points[:, :, 0] /= float(image_size[0])
    points[:, :, 1] /= float(image_size[1])

    for frame in range(len(points)):
        coordinate_min = points[frame, :, :2].min(axis=0)
        coordinate_max = points[frame, :, :2].max(axis=0)
        span = np.maximum(coordinate_max - coordinate_min, 1e-6)
        points[frame, :, :2] = (
            (points[frame, :, :2] - coordinate_min) / span
        ) * 2.0 - 1.0

    neck = ((points[:, 1] + points[:, 2]) / 2.0)[:, None, :]
    return np.concatenate((points, neck), axis=1).astype(np.float32)


def read_events(subject_root: Path) -> list[dict]:
    events_path = subject_root / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Event manifest not found: {events_path}")
    events = []
    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON at {events_path}:{line_number}"
            ) from error
    return events


def save_split(output_root: Path, split: str, rows: list[dict]) -> None:
    features = np.stack([row.pop("feature") for row in rows])
    labels = np.asarray([row["label_index"] for row in rows], np.int64)
    np.savez_compressed(
        output_root / f"{split}.npz",
        features=features,
        labels=labels,
        labels_one_hot=np.eye(len(CLASS_NAMES), dtype=np.float32)[labels],
        event_ids=np.asarray([row["event_id"] for row in rows]),
        class_names=CLASS_NAMES,
        pose_hz=np.asarray(25.0, np.float32),
        window_frames=np.asarray(WINDOW_FRAMES, np.int32),
    )


def build_dataset(
    subject_root: Path,
    output_root: Path,
    minimum_confidence: float,
    split_mode: str,
    split_seed: int,
    locations: set[str] | None = None,
    exclude_train_event_ids: set[str] | None = None,
) -> None:
    events = read_events(subject_root)
    if locations:
        events = [event for event in events if event.get("location") in locations]
    exclude_train_event_ids = exclude_train_event_ids or set()
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_split = {"train": [], "validation": [], "test": []}
    event_splits = assign_event_splits(events, split_mode, split_seed)

    known_event_ids = {event["event_id"] for event in events}
    unknown_exclusions = exclude_train_event_ids - known_event_ids
    if unknown_exclusions:
        raise ValueError(
            f"Excluded event IDs are not in the selected input: {sorted(unknown_exclusions)}"
        )
    invalid_exclusions = {
        event_id
        for event_id in exclude_train_event_ids
        if event_splits[event_id] != "train"
    }
    if invalid_exclusions:
        raise ValueError(
            "Only train events may be excluded; got "
            f"{sorted(invalid_exclusions)}"
        )

    for event in events:
        repetition = int(event["repetition"])
        expected_split = event_splits[event["event_id"]]
        if event["event_id"] in exclude_train_event_ids:
            continue
        if (
            split_mode == "full200"
            and event.get("split", expected_split) != expected_split
        ):
            raise ValueError(
                f"{event['event_id']} has an invalid split for repetition "
                f"{repetition}"
            )
        pose_path = subject_root / event["pose_file"]
        with np.load(pose_path, allow_pickle=False) as pose:
            xy = pose["keypoints_xy"]
            confidence = pose["keypoints_conf"]
            pose_valid = pose["pose_valid"]
            image_size = tuple(
                int(value) for value in pose.get("image_size", [640, 480])
            )
            pose_hz = float(pose.get("pose_hz", 25.0))
        if abs(pose_hz - 25.0) > 1e-6 or len(xy) != 100:
            raise ValueError(
                f"{event['event_id']} must contain exactly 100 poses at 25 Hz; "
                f"got {len(xy)} at {pose_hz:g} Hz"
            )

        xy, confidence = interpolate_keypoints(
            xy, confidence, pose_valid, minimum_confidence
        )
        signals = pose_motion_signals(xy, confidence, image_size)
        window_start, score, fall_start, impact = choose_window(
            signals, event["label"]
        )
        window_stop = window_start + WINDOW_FRAMES
        feature = make_tsstg_feature(
            xy[window_start:window_stop],
            confidence[window_start:window_stop],
            image_size,
        )
        row = {
            "event_id": event["event_id"],
            "subject_id": event["subject_id"],
            "action_code": event["action_code"],
            "event_label": event["label"],
            "label_index": 1 if event["label"] == "FALL" else 0,
            "repetition": repetition,
            "split": expected_split,
            "window_start": window_start,
            "window_stop_exclusive": window_stop,
            "fall_start_estimate": fall_start,
            "impact_estimate": impact,
            "window_contains_both_estimates": (
                event["label"] == "FALL"
                and window_start <= fall_start <= impact < window_stop
            ),
            "selection_score": score,
            "valid_pose_ratio": event.get("valid_pose_ratio", ""),
            "position": event.get("position", ""),
            "view": event.get("view", ""),
            "lighting": event.get("lighting", ""),
            "outfit": event.get("outfit", ""),
            "fall_side": event.get("fall_side", ""),
            "location": event.get("location", ""),
            "feature": feature,
        }
        rows_by_split[expected_split].append(row)

    if split_mode == "pilot100":
        base_expected_counts = {"train": 80, "validation": 10, "test": 10}
    elif split_mode == "location_stratified":
        base_expected_counts = {"train": 160, "validation": 20, "test": 20}
    else:
        base_expected_counts = {"train": 140, "validation": 20, "test": 40}
    expected_counts = dict(base_expected_counts)
    expected_counts["train"] -= len(exclude_train_event_ids)
    for split, expected in expected_counts.items():
        expected_input_total = sum(base_expected_counts.values())
        if len(events) == expected_input_total and len(rows_by_split[split]) != expected:
            raise ValueError(
                f"Expected {expected} {split} events, got {len(rows_by_split[split])}"
            )

    csv_rows = []
    for split, rows in rows_by_split.items():
        if rows:
            csv_rows.extend([{k: v for k, v in row.items() if k != "feature"} for row in rows])
            save_split(output_root, split, rows)
    if csv_rows:
        with (output_root / "windows.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    summary = {
        "source": str(subject_root),
        "locations": sorted(locations) if locations else sorted(
            {event.get("location", "") for event in events}
        ),
        "split_mode": split_mode,
        "split_seed": split_seed if split_mode != "full200" else None,
        "split_strategy": (
            "seeded random 8/1/1 within each action_code"
            if split_mode == "pilot100"
            else (
                "seeded random 8/1/1 within each location and action_code"
                if split_mode == "location_stratified"
                else "fixed by repetition: 1-14/15-16/17-20"
            )
        ),
        "excluded_train_event_ids": sorted(exclude_train_event_ids),
        "class_names": CLASS_NAMES.tolist(),
        "window_frames": WINDOW_FRAMES,
        "pose_hz": 25.0,
        "feature_shape": [WINDOW_FRAMES, 14, 3],
        "split_counts": {
            split: len(rows) for split, rows in rows_by_split.items()
        },
        "note": "bbox is not included in TSSTG features or window selection",
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-split 30-frame binary TSSTG datasets"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Subject directory containing events.jsonl and samples/",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.20)
    parser.add_argument(
        "--split-mode",
        choices=("pilot100", "location_stratified", "full200"),
        default="full200",
        help=(
            "pilot100: one-location 80/10/10; location_stratified: "
            "two-location 160/20/20; full200: fixed repetitions 140/20/40"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--location",
        action="append",
        choices=("location_1", "location_2"),
        help="Build only the selected location; may be supplied more than once",
    )
    parser.add_argument(
        "--exclude-train-event-id",
        action="append",
        default=[],
        help="Exclude a known low-quality train event without changing validation/test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        args.input,
        args.output,
        args.minimum_confidence,
        args.split_mode,
        args.split_seed,
        set(args.location) if args.location else None,
        set(args.exclude_train_event_id),
    )


if __name__ == "__main__":
    main()
