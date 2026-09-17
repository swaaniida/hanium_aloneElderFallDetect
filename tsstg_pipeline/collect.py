from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NCNN_MODEL = REPO_ROOT / "models" / "pose" / "yolo26n-pose_ncnn_model"
HOME_NCNN_MODEL = Path.home() / "yolo26n-pose_ncnn_model"
DEFAULT_MODEL = (
    LOCAL_NCNN_MODEL if LOCAL_NCNN_MODEL.exists() else HOME_NCNN_MODEL
)
WINDOW_NAME = "TSSTG binary pose collector"
IMAGE_SIZE = (640, 480)
CAMERA_FPS = 30
POSE_HZ = 25.0
COUNTDOWN_SECONDS = 2.0
CAPTURE_SECONDS = 4.0
CAPTURE_FRAME_COUNT = int(POSE_HZ * CAPTURE_SECONDS)
POSE_SAMPLE_INTERVAL_NS = round(1_000_000_000 / POSE_HZ)
MAX_SAMPLE_TIME_ERROR_NS = 30_000_000
PERSON_CONFIDENCE = 0.35

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7),
    (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# 5 fall + 5 hard/easy non-fall groups, 20 repetitions each = 200 events.
ACTION_PLAN = [
    ("F01_FORWARD", "FALL", "Fall forward", 20),
    ("F02_BACKWARD", "FALL", "Fall backward", 20),
    ("F03_SIDE", "FALL", "Fall sideways (alternate left/right)", 20),
    ("F04_WALK_TURN", "FALL", "Fall while walking or turning", 20),
    ("F05_SLOW_SLIDE", "FALL", "Slow collapse or wall slide", 20),
    ("N01_WALK_STAND", "NON_FALL", "Walk, stop, and turn", 20),
    ("N02_SIT_LIE", "NON_FALL", "Sit on chair or lie on bed normally", 20),
    ("N03_BEND_SQUAT", "NON_FALL", "Pick up, squat, or kneel", 20),
    ("N04_FAST_SAFE", "NON_FALL", "Sit/lie quickly or stumble and recover", 20),
    ("N05_FLOOR_MOVE", "NON_FALL", "Sit/lie/move/get up on floor", 20),
]

# Conditions are assigned by repetition so every subtype has the same balance.
# Repetitions 17-20 are the held-out test set and deliberately use outfit_2.
POSITIONS = [
    "center", "left", "right", "center", "near",
    "center", "left", "right", "center", "far",
    "center", "left", "right", "center", "near",
    "center", "left", "right", "center", "far",
]
VIEWS = ["front", "back", "left_side", "right_side"] * 5
LIGHTING = [
    "normal", "normal", "dim", "normal", "normal",
    "normal", "dim", "normal", "normal", "dim",
    "normal", "normal", "normal", "dim", "normal",
    "normal", "dim", "normal", "normal", "dim",
]
OUTFITS = ["outfit_1"] * 16 + ["outfit_2"] * 4

TSSTG_SPLIT_BY_REPETITION = {
    **{repetition: "train" for repetition in range(1, 15)},
    15: "validation",
    16: "validation",
    **{repetition: "test" for repetition in range(17, 21)},
}


@dataclass
class MouseState:
    left: bool = False
    right: bool = False


def mouse_callback(event, _x, _y, _flags, state: MouseState) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        state.left = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.right = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_ncnn_threads(num_threads: int) -> None:
    """Set NCNN threads before convolution weights are packed."""
    from ultralytics.nn.backends.ncnn import NCNNBackend
    from ultralytics.utils import LOGGER, YAML
    from ultralytics.utils.checks import check_requirements

    def load_model(self, weight: str | Path) -> None:
        LOGGER.info(f"Loading {weight} for NCNN inference...")
        check_requirements("ncnn", cmds="--no-deps")
        import ncnn as pyncnn

        self.pyncnn = pyncnn
        self.net = pyncnn.Net()
        self.net.opt.num_threads = num_threads
        self.net.opt.use_vulkan_compute = False
        param_path = Path(weight)
        if not param_path.is_file():
            param_path = next(param_path.glob("*.param"))
        self.net.load_param(str(param_path))
        self.net.load_model(str(param_path.with_suffix(".bin")))
        metadata_path = param_path.parent / "metadata.yaml"
        if metadata_path.exists():
            self.apply_metadata(YAML.load(metadata_path))

    NCNNBackend.load_model = load_model


def make_plan() -> list[dict]:
    actions = {action[0]: action for action in ACTION_PLAN}
    # Both locations collect the same ten subtypes, ten events per subtype.
    subtype_order = [
        "F01_FORWARD", "N01_WALK_STAND",
        "F02_BACKWARD", "N02_SIT_LIE",
        "F03_SIDE", "N03_BEND_SQUAT",
        "F04_WALK_TURN", "N04_FAST_SAFE",
        "F05_SLOW_SLIDE", "N05_FLOOR_MOVE",
    ]
    location_blocks = {
        "location_1": [
            (action_code, range(1, 11)) for action_code in subtype_order
        ],
        "location_2": [
            (action_code, range(11, 21)) for action_code in subtype_order
        ],
    }
    plan = []
    for location, blocks in location_blocks.items():
        for action_code, repetitions in blocks:
            _, label, description, _ = actions[action_code]
            for repetition in repetitions:
                plan.append({
                    "action_code": action_code,
                    "label": label,
                    "description": description,
                    "repetition": repetition,
                    "split": TSSTG_SPLIT_BY_REPETITION[repetition],
                    "position": POSITIONS[repetition - 1],
                    "view": VIEWS[repetition - 1],
                    "lighting": LIGHTING[repetition - 1],
                    "outfit": OUTFITS[repetition - 1],
                    "location": location,
                    "fall_side": (
                        "left" if repetition % 2 else "right"
                    ) if action_code == "F03_SIDE" else "not_applicable",
                })
    for index, item in enumerate(plan, 1):
        item["plan_index"] = index
    return plan


def put_lines(image: np.ndarray, lines: list[str], color=(0, 255, 255)) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, color, 1, cv2.LINE_AA,
        )
        y += 27


def draw_pose(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    threshold: float = 0.2,
) -> None:
    for a, b in SKELETON_EDGES:
        if keypoints_conf[a] >= threshold and keypoints_conf[b] >= threshold:
            pa = tuple(np.rint(keypoints_xy[a]).astype(int))
            pb = tuple(np.rint(keypoints_xy[b]).astype(int))
            cv2.line(image, pa, pb, (80, 255, 80), 2, cv2.LINE_AA)
    for point, confidence in zip(keypoints_xy, keypoints_conf):
        if confidence >= threshold:
            cv2.circle(
                image, tuple(np.rint(point).astype(int)), 4,
                (0, 200, 255), -1, cv2.LINE_AA,
            )


def skeleton_canvas(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    title: str,
) -> np.ndarray:
    canvas = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 24, np.uint8)
    draw_pose(canvas, keypoints_xy, keypoints_conf)
    put_lines(canvas, [title], (100, 255, 100))
    return canvas


def extract_pose(model: YOLO, image: np.ndarray):
    result = model.predict(
        source=image,
        imgsz=320,
        conf=PERSON_CONFIDENCE,
        verbose=False,
    )[0]
    if (
        result.boxes is None
        or result.keypoints is None
        or len(result.boxes) == 0
    ):
        return (
            np.full((17, 2), np.nan, np.float32),
            np.zeros(17, np.float32),
            np.full(4, np.nan, np.float32),
            False,
        )
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    index = int(np.argmax(areas))
    xy = result.keypoints.xy[index].cpu().numpy().astype(np.float32)
    if result.keypoints.conf is None:
        conf = np.ones(17, np.float32)
    else:
        conf = result.keypoints.conf[index].cpu().numpy().astype(np.float32)
    return xy, conf, boxes[index], True


def open_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, IMAGE_SIZE[0], IMAGE_SIZE[1],
        rs.format.bgr8, CAMERA_FPS,
    )
    pipeline.start(config)
    for _ in range(20):
        pipeline.wait_for_frames()
    return pipeline


def next_color_frame(pipeline) -> np.ndarray:
    while True:
        color = pipeline.wait_for_frames().get_color_frame()
        if color:
            return np.asanyarray(color.get_data()).copy()


def next_color_sample(pipeline) -> tuple[np.ndarray, int, int]:
    """Return one camera frame with host-monotonic time and camera frame id."""
    while True:
        color = pipeline.wait_for_frames().get_color_frame()
        received_ns = time.monotonic_ns()
        if color:
            return (
                np.asanyarray(color.get_data()).copy(),
                received_ns,
                int(color.get_frame_number()),
            )


def select_25hz_frames(
    raw_frames: list[np.ndarray],
    raw_timestamps_ns: np.ndarray,
    raw_frame_numbers: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Select exactly 100 unique camera frames nearest a uniform 25 Hz grid."""
    target_timestamps_ns = (
        int(raw_timestamps_ns[0])
        + np.arange(CAPTURE_FRAME_COUNT, dtype=np.int64)
        * POSE_SAMPLE_INTERVAL_NS
    )
    selected_indices = np.searchsorted(raw_timestamps_ns, target_timestamps_ns)
    selected_indices = np.clip(selected_indices, 1, len(raw_timestamps_ns) - 1)
    before = selected_indices - 1
    choose_before = (
        target_timestamps_ns - raw_timestamps_ns[before]
        <= raw_timestamps_ns[selected_indices] - target_timestamps_ns
    )
    selected_indices = np.where(choose_before, before, selected_indices)

    selected_frame_numbers = raw_frame_numbers[selected_indices]
    if (
        len(np.unique(selected_indices)) != CAPTURE_FRAME_COUNT
        or len(np.unique(selected_frame_numbers)) != CAPTURE_FRAME_COUNT
    ):
        raise RuntimeError(
            "Camera dropped too many frames to form 100 unique 25 Hz samples; retry"
        )
    time_errors_ns = np.abs(
        raw_timestamps_ns[selected_indices] - target_timestamps_ns
    )
    if int(time_errors_ns.max()) > MAX_SAMPLE_TIME_ERROR_NS:
        raise RuntimeError(
            "Camera timing gap exceeded 30 ms; retry the sample "
            f"(max={time_errors_ns.max() / 1e6:.1f} ms)"
        )
    return (
        [raw_frames[int(index)] for index in selected_indices],
        raw_timestamps_ns[selected_indices],
        target_timestamps_ns,
        selected_frame_numbers,
    )


def save_latest_buffer(frames: list[np.ndarray], output_path: Path) -> None:
    """Save only the newest unannotated 4-second RGB buffer."""
    pending_path = output_path.with_name(f".{output_path.name}.pending.mp4")
    pending_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(pending_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        POSE_HZ,
        IMAGE_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open buffer video writer: {pending_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    output_path.unlink(missing_ok=True)
    pending_path.replace(output_path)


def show_live_until_click(
    pipeline,
    item: dict,
    completed: int,
    total: int,
    mouse: MouseState,
) -> bool:
    mouse.left = mouse.right = False
    while True:
        image = next_color_frame(pipeline)
        put_lines(image, [
            f"READY {completed}/{total}",
            f"NEXT: {item['action_code']} ({item['label']})",
            item["description"],
            (
                f"location repeat {((item['repetition'] - 1) % 10) + 1}/10 "
                f"| total {item['repetition']}/20 | split: {item['split']}"
            ),
            (
                f"condition: {item['position']} / {item['view']} / "
                f"{item['lighting']} / {item['outfit']}"
            ),
            f"location: {item['location']}",
            "LEFT: start 2s countdown | Q: quit",
        ])
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if mouse.left:
            mouse.left = False
            return True


def countdown(pipeline, item: dict) -> bool:
    started = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - started
        remaining = max(0.0, COUNTDOWN_SECONDS - elapsed)
        image = next_color_frame(pipeline)
        put_lines(image, [
            item["description"],
            f"START IN {remaining:.1f} s",
            "Perform when RECORD appears | Q: quit",
        ], (0, 255, 255))
        cv2.imshow(WINDOW_NAME, image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
        if elapsed >= COUNTDOWN_SECONDS:
            return True


def capture_sample(
    pipeline,
    model: YOLO,
    item: dict,
    review_video: Path,
) -> dict | None:
    # The previous accepted review remains until this sample is accepted.
    review_video.unlink(missing_ok=True)

    raw_frames = []
    raw_timestamps_ns = []
    raw_frame_numbers = []
    started_ns = 0
    last_target_ns = 0
    while True:
        image, timestamp_ns, frame_number = next_color_sample(pipeline)
        if not raw_frames:
            started_ns = timestamp_ns
            last_target_ns = (
                started_ns
                + (CAPTURE_FRAME_COUNT - 1) * POSE_SAMPLE_INTERVAL_NS
            )
        raw_frames.append(image)
        raw_timestamps_ns.append(timestamp_ns)
        raw_frame_numbers.append(frame_number)
        remaining = max(0.0, (last_target_ns - timestamp_ns) / 1e9)
        preview = image.copy()
        put_lines(preview, [
            f"RECORD {item['action_code']} {remaining:.1f}s",
            "RGB is buffered in RAM only; pose processing follows",
        ], (0, 0, 255))
        cv2.imshow(WINDOW_NAME, preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None
        # Keep one camera frame after the last target so nearest-neighbour
        # resampling has a sample on both sides of the target when possible.
        if timestamp_ns >= last_target_ns + round(1e9 / CAMERA_FPS):
            break

    try:
        frames, timestamps_ns, target_timestamps_ns, frame_numbers = (
            select_25hz_frames(
                raw_frames,
                np.asarray(raw_timestamps_ns, np.int64),
                np.asarray(raw_frame_numbers, np.int64),
            )
        )
    except RuntimeError as error:
        warning = raw_frames[-1].copy()
        put_lines(warning, ["CAPTURE TIMING FAILED", str(error)], (0, 0, 255))
        cv2.imshow(WINDOW_NAME, warning)
        cv2.waitKey(1500)
        return {}

    save_latest_buffer(
        frames,
        review_video.parent / "latest_buffer.mp4",
    )

    writer = cv2.VideoWriter(
        str(review_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        POSE_HZ,
        (IMAGE_SIZE[0] * 2, IMAGE_SIZE[1]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {review_video}")

    all_xy = []
    all_conf = []
    all_bbox = []
    all_valid = []
    try:
        for sequence, image in enumerate(frames):
            xy, conf, bbox, valid = extract_pose(model, image)
            annotated = image.copy()
            if valid:
                draw_pose(annotated, xy, conf)
                x1, y1, x2, y2 = np.rint(bbox).astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 160, 0), 2)
                pose_view = skeleton_canvas(xy, conf, "POSE ONLY")
            else:
                pose_view = np.full_like(image, 24)
                put_lines(pose_view, ["POSE NOT DETECTED"], (0, 0, 255))
            put_lines(annotated, [
                f"PROCESS {item['action_code']}",
                f"pose {sequence + 1}/{CAPTURE_FRAME_COUNT}",
            ], (0, 0, 255))
            combined = np.hstack((annotated, pose_view))
            writer.write(combined)
            cv2.imshow(WINDOW_NAME, combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None

            all_xy.append(xy)
            all_conf.append(conf)
            all_bbox.append(bbox)
            all_valid.append(valid)
    finally:
        writer.release()

    return {
        "timestamps_ns": timestamps_ns,
        "target_timestamps_ns": target_timestamps_ns,
        "camera_frame_number": frame_numbers,
        "frame_seq": np.arange(CAPTURE_FRAME_COUNT, dtype=np.int64),
        "image_size": np.asarray(IMAGE_SIZE, np.int32),
        "pose_hz": np.asarray(POSE_HZ, np.float32),
        "keypoints_xy": np.asarray(all_xy, np.float32),
        "keypoints_conf": np.asarray(all_conf, np.float32),
        "bbox_xyxy": np.asarray(all_bbox, np.float32),
        "pose_valid": np.asarray(all_valid, bool),
    }


def review_loop(review_video: Path, mouse: MouseState) -> str:
    mouse.left = mouse.right = False
    capture = cv2.VideoCapture(str(review_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open review video: {review_video}")
    frame_interval_ms = max(1, round(1000 / POSE_HZ))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            put_lines(frame, [
                "REVIEW (loop)",
                "LEFT: accept/next | RIGHT: discard/retry | Q: quit",
            ], (255, 255, 0))
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(frame_interval_ms) & 0xFF
            if key == ord("q"):
                return "quit"
            if mouse.left:
                mouse.left = False
                return "accept"
            if mouse.right:
                mouse.right = False
                return "retry"
    finally:
        capture.release()


def save_accepted_sample(
    root: Path,
    subject_id: str,
    item: dict,
    pose: dict,
    review_video: Path,
) -> None:
    event_id = f"{subject_id}_E{item['plan_index']:03d}"
    sample_path = root / "samples" / f"{event_id}.npz"
    np.savez_compressed(sample_path, **pose)
    duration = (
        (int(pose["timestamps_ns"][-1]) - int(pose["timestamps_ns"][0])) / 1e9
        if len(pose["timestamps_ns"]) > 1 else 0.0
    )
    valid_ratio = float(np.mean(pose["pose_valid"])) if len(pose["pose_valid"]) else 0.0
    record = {
        **item,
        "event_id": event_id,
        "subject_id": subject_id,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pose_file": str(sample_path.relative_to(root)),
        "pose_count": int(len(pose["timestamps_ns"])),
        "duration_seconds": duration,
        "valid_pose_ratio": valid_ratio,
        "pose_sha256": sha256_file(sample_path),
    }
    with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Only the newest review video is retained and is overwritten next time.
    latest = root / "latest_review.mp4"
    latest.unlink(missing_ok=True)
    shutil.move(str(review_video), str(latest))
    (root / "latest_review.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_completed(root: Path) -> set[int]:
    events_path = root / "events.jsonl"
    if not events_path.exists():
        return set()
    completed = set()
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            completed.add(int(json.loads(line)["plan_index"]))
    return completed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", required=True, help="Pseudonymous ID, e.g. S001")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "collected_pose_data")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ncnn-threads", type=int, default=2)
    parser.add_argument(
        "--location",
        choices=("location_1", "location_2"),
        required=True,
        help="Collect only one planned 100-event location session",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output / args.subject_id
    (root / "samples").mkdir(parents=True, exist_ok=True)
    review_video = root / ".pending_review.mp4"
    plan = make_plan()
    plan_path = root / "collection_plan.json"
    if plan_path.exists():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise RuntimeError(
                f"Existing collection plan is incompatible: {plan_path}. "
                "Use a new subject ID or output directory."
            )
    else:
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    completed = load_completed(root)

    if args.location == "location_2":
        location_1_indices = {
            item["plan_index"]
            for item in plan
            if item["location"] == "location_1"
        }
        missing_location_1 = location_1_indices - completed
        if missing_location_1:
            raise RuntimeError(
                "Complete location_1 before starting location_2; "
                f"{len(missing_location_1)} events remain"
            )
    active_plan = [
        item for item in plan if item["location"] == args.location
    ]
    active_indices = {item["plan_index"] for item in active_plan}
    completed_active = completed & active_indices

    configure_ncnn_threads(args.ncnn_threads)
    model = YOLO(str(args.model))
    pipeline = open_camera()
    mouse = MouseState()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, mouse)

    try:
        for item in active_plan:
            if item["plan_index"] in completed:
                continue
            while True:
                if not show_live_until_click(
                    pipeline,
                    item,
                    len(completed_active),
                    len(active_plan),
                    mouse,
                ):
                    return
                if not countdown(pipeline, item):
                    return
                pose = capture_sample(pipeline, model, item, review_video)
                if pose is None:
                    return
                if not pose:
                    continue
                decision = review_loop(review_video, mouse)
                if decision == "quit":
                    return
                if decision == "retry":
                    review_video.unlink(missing_ok=True)
                    continue
                save_accepted_sample(
                    root, args.subject_id, item, pose, review_video
                )
                completed.add(item["plan_index"])
                completed_active.add(item["plan_index"])
                break
        done = np.full((480, 960, 3), 24, np.uint8)
        completion_text = f"{args.location.upper()} 100 SAMPLES COMPLETE"
        put_lines(done, [completion_text, "Press any key to close"])
        cv2.imshow(WINDOW_NAME, done)
        cv2.waitKey(0)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
