from __future__ import annotations

import base64
import os
import traceback
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
import torch
import paho.mqtt.client as mqtt
from flask import Flask, Response
from ultralytics import YOLO


# ============================================================
# 경로 설정
# ============================================================

PROJECT_DIR = Path.home() / "Human-Falling-Detect-Tracks"
YOLO_MODEL_PATH = Path.home() / "yolo26n-pose.pt"
TSSTG_WEIGHT_PATH = (
    PROJECT_DIR / "Models" / "TSSTG" / "tsstg-model.pth"
)

sys.path.insert(0, str(PROJECT_DIR))

from ActionsEstLoader import TSSTG  # noqa: E402


# ============================================================
# 추론 설정
# ============================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FPS = 30

YOLO_IMAGE_SIZE = 320
PERSON_CONFIDENCE = 0.35
KEYPOINT_CONFIDENCE = 0.20

SEQUENCE_LENGTH = 30
WEB_PORT = 5000

DEVICE = "cpu"

# ============================================================
# 외부 연동 설정
# ============================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")
MQTT_BROKER  = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT    = int(os.getenv("MQTT_PORT", "1883"))
ELDER_ID     = os.getenv("ELDER_ID", "elder_001")

MQTT_TOPIC_FRAME  = "senior_care/camera/frame"
MQTT_TOPIC_FALL   = "senior_care/fall"
MQTT_TOPIC_ACTION = "senior_care/action"

FALL_COOLDOWN   = 10        # 낙상 알림 쿨다운 (초)
MQTT_FRAME_INTERVAL = 0.1   # MQTT 프레임 publish 간격 (초, 약 10fps)

_last_fall_time   = 0.0
_last_mqtt_frame  = 0.0

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# 동일한 사람인지 판단하기 위한 bbox IoU
TRACK_IOU_THRESHOLD = 0.20

# 사람을 놓쳤을 때 버퍼를 유지할 최대 프레임
MAX_MISSING_FRAMES = 10


# ============================================================
# COCO skeleton 연결선
# ============================================================

SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


# ============================================================
# 도우미 함수
# ============================================================

def bbox_iou(
    box_a: np.ndarray,
    box_b: np.ndarray,
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )
    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection = (
        intersection_width * intersection_height
    )

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )
    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def get_bbox_depth(
    depth_frame: rs.depth_frame,
    bbox: np.ndarray,
) -> float | None:
    depth_image = np.asanyarray(
        depth_frame.get_data()
    )

    image_height, image_width = depth_image.shape

    x1, y1, x2, y2 = bbox.astype(int)

    x1 = max(0, min(x1, image_width - 1))
    x2 = max(0, min(x2, image_width))
    y1 = max(0, min(y1, image_height - 1))
    y2 = max(0, min(y2, image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1

    # bbox 가장자리에 포함된 배경을 줄이기 위해 중앙 영역 사용
    margin_x = int(width * 0.20)
    margin_y = int(height * 0.20)

    roi_x1 = x1 + margin_x
    roi_x2 = x2 - margin_x
    roi_y1 = y1 + margin_y
    roi_y2 = y2 - margin_y

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None

    roi = depth_image[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ]

    valid = roi[roi > 0]

    if valid.size == 0:
        return None

    depth_m = float(
        np.median(valid) * depth_frame.get_units()
    )

    if not 0.2 < depth_m < 10.0:
        return None

    return depth_m


def draw_pose(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
) -> None:
    for start_index, end_index in SKELETON_EDGES:
        if (
            keypoints_conf[start_index]
            < KEYPOINT_CONFIDENCE
            or keypoints_conf[end_index]
            < KEYPOINT_CONFIDENCE
        ):
            continue

        start_point = tuple(
            keypoints_xy[start_index].astype(int)
        )
        end_point = tuple(
            keypoints_xy[end_index].astype(int)
        )

        cv2.line(
            image,
            start_point,
            end_point,
            (0, 255, 255),
            2,
        )

    for index, point in enumerate(keypoints_xy):
        if (
            keypoints_conf[index]
            < KEYPOINT_CONFIDENCE
        ):
            continue

        cv2.circle(
            image,
            tuple(point.astype(int)),
            4,
            (0, 0, 255),
            -1,
        )


def action_color(
    action_name: str,
) -> tuple[int, int, int]:
    if action_name == "Fall Down":
        return 0, 0, 255

    if action_name == "Lying Down":
        return 0, 165, 255

    if action_name in {"Sit down", "Stand up"}:
        return 255, 255, 0

    return 0, 255, 0


def choose_person(
    boxes: np.ndarray,
    previous_bbox: np.ndarray | None,
) -> int:
    """
    현재 baseline은 한 사람을 추적한다.

    이전 bbox가 있으면 IoU가 가장 큰 사람을 선택하고,
    없으면 화면에서 가장 큰 사람을 선택한다.
    """

    if previous_bbox is not None:
        ious = np.array(
            [
                bbox_iou(previous_bbox, box)
                for box in boxes
            ],
            dtype=np.float32,
        )

        best_index = int(np.argmax(ious))

        if ious[best_index] >= TRACK_IOU_THRESHOLD:
            return best_index

    areas = (
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1])
    )

    return int(np.argmax(areas))


# ============================================================
# 모델 확인 및 로드
# ============================================================

if not YOLO_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO 모델이 없습니다: {YOLO_MODEL_PATH}"
    )

if not TSSTG_WEIGHT_PATH.exists():
    raise FileNotFoundError(
        f"TSSTG 가중치가 없습니다: "
        f"{TSSTG_WEIGHT_PATH}"
    )

print(f"YOLO loading: {YOLO_MODEL_PATH}")
pose_model = YOLO(str(YOLO_MODEL_PATH))

print(f"TSSTG loading: {TSSTG_WEIGHT_PATH}")
action_model = TSSTG(
    weight_file=str(TSSTG_WEIGHT_PATH),
    device=DEVICE,
)

print("TSSTG classes:", action_model.class_names)


# ============================================================
# RealSense 시작
# ============================================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.depth,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    rs.format.z16,
    CAMERA_FPS,
)

config.enable_stream(
    rs.stream.color,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    rs.format.bgr8,
    CAMERA_FPS,
)

pipeline.start(config)

align = rs.align(rs.stream.color)


# ============================================================
# 전역 추적 데이터
# ============================================================

skeleton_buffer: deque[np.ndarray] = deque(
    maxlen=SEQUENCE_LENGTH
)

previous_bbox: np.ndarray | None = None
missing_frames = 0

last_action_name = "Collecting"
last_action_probability = 0.0
last_probabilities = np.zeros(
    len(action_model.class_names),
    dtype=np.float32,
)

frame_lock = Lock()
latest_jpeg: bytes | None = None

app = Flask(__name__)


# ============================================================
# MQTT 클라이언트
# ============================================================

mqtt_client = mqtt.Client()

def _on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] 연결 성공: {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"[MQTT] 연결 실패 코드: {rc}")

mqtt_client.on_connect = _on_mqtt_connect

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"[MQTT] 초기 연결 실패: {e}")


# ============================================================
# 낙상 알림 함수
# ============================================================

def notify_fall(action_name: str, confidence: float) -> None:
    global _last_fall_time

    now = time.time()
    if now - _last_fall_time < FALL_COOLDOWN:
        return
    _last_fall_time = now

    event_type = (
        "fall_detected" if action_name == "Fall Down" else "lying_down"
    )

    payload = {
        "elder_id":   ELDER_ID,
        "event_type": event_type,
        "action":     action_name,
        "confidence": confidence,
        "location":   "거실",
    }

    try:
        res = requests.post(
            f"{BACKEND_URL}/events/fall",
            json=payload,
            timeout=5,
        )
        print(f"[낙상 이벤트] {action_name} → {res.status_code}")
    except Exception as e:
        print(f"[낙상 이벤트] 전송 실패: {e}")


# ============================================================
# 추론 루프
# ============================================================

def inference_loop():
    global previous_bbox
    global missing_frames
    global last_action_name
    global last_action_probability
    global last_probabilities
    global latest_jpeg
    global _last_mqtt_frame

    previous_time = time.perf_counter()

    # 카메라 자동 노출 안정화
    for _ in range(20):
        pipeline.wait_for_frames()

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = (
            aligned_frames.get_depth_frame()
        )
        color_frame = (
            aligned_frames.get_color_frame()
        )

        if not depth_frame or not color_frame:
            continue

        image = np.asanyarray(
            color_frame.get_data()
        )

        result = pose_model.predict(
            source=image,
            imgsz=YOLO_IMAGE_SIZE,
            conf=PERSON_CONFIDENCE,
            device="cpu",
            verbose=False,
        )[0]

        annotated = image.copy()

        current_time = time.perf_counter()
        elapsed = max(
            current_time - previous_time,
            1e-6,
        )
        fps = 1.0 / elapsed
        previous_time = current_time

        person_detected = (
            result.boxes is not None
            and result.keypoints is not None
            and len(result.boxes) > 0
        )

        if person_detected:
            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            keypoints_xy_all = (
                result.keypoints.xy
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if result.keypoints.conf is not None:
                keypoints_conf_all = (
                    result.keypoints.conf
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            else:
                keypoints_conf_all = np.ones(
                    (
                        len(boxes),
                        17,
                    ),
                    dtype=np.float32,
                )

            person_index = choose_person(
                boxes,
                previous_bbox,
            )

            bbox = boxes[person_index]
            keypoints_xy = keypoints_xy_all[
                person_index
            ]
            keypoints_conf = keypoints_conf_all[
                person_index
            ]

            previous_bbox = bbox.copy()
            missing_frames = 0

            # TSSTG가 학습할 때 사용한 13개 신체 관절 순서
            #
            # YOLO COCO 17 keypoints:
            # 0  nose
            # 1  left_eye
            # 2  right_eye
            # 3  left_ear
            # 4  right_ear
            # 5  left_shoulder
            # 6  right_shoulder
            # 7  left_elbow
            # 8  right_elbow
            # 9  left_wrist
            # 10 right_wrist
            # 11 left_hip
            # 12 right_hip
            # 13 left_knee
            # 14 right_knee
            # 15 left_ankle
            # 16 right_ankle
            #
            # 눈과 귀 4개를 제거:
            TSSTG_COCO_INDICES = [
                0,          # nose
                5, 6,       # shoulders
                7, 8,       # elbows
                9, 10,      # wrists
                11, 12,     # hips
                13, 14,     # knees
                15, 16,     # ankles
            ]

            tsstg_keypoints_xy = keypoints_xy[TSSTG_COCO_INDICES]
            tsstg_keypoints_conf = keypoints_conf[TSSTG_COCO_INDICES]

            pose_frame = np.concatenate(
                (
                    tsstg_keypoints_xy,
                    tsstg_keypoints_conf[:, None],
                ),
                axis=1,
            ).astype(np.float32)

            # print(
            #     "DEBUG:",
            #     "YOLO keypoints =", keypoints_xy.shape,
            #     "TSSTG pose_frame =", pose_frame.shape,
            # )

            if pose_frame.shape != (13, 3):
                print(
                    f"Invalid TSSTG pose shape: {pose_frame.shape}, "
                    "expected (13, 3)"
                )
                skeleton_buffer.clear()
                previous_bbox = None
                continue
            #

            # confidence가 너무 낮은 관절은 score만 0으로 둔다.
            # 좌표를 0으로 만들면 갑작스러운 큰 motion이 생길 수 있다.
            low_confidence = (
                tsstg_keypoints_conf
                < KEYPOINT_CONFIDENCE
            )
            pose_frame[
                low_confidence,
                2,
            ] = 0.0

            skeleton_buffer.append(pose_frame)

            if len(skeleton_buffer) == SEQUENCE_LENGTH:
                points = np.array(
                    skeleton_buffer,
                    dtype=np.float32,
                )

                #
                # print("TSSTG input shape:", points.shape)

                # if points.shape != (SEQUENCE_LENGTH, 13, 3):
                #     print(
                #         f"Wrong TSSTG input: {points.shape}, "
                #         f"expected ({SEQUENCE_LENGTH}, 13, 3)"
                #     )
                #     skeleton_buffer.clear()
                #     continue
                #

                # ActionsEstLoader 원본 코드와 동일하게
                # frame.shape[:2] = (height, width)를 전달한다.
                with torch.inference_mode():
                    output = action_model.predict(
                        points.copy(),
                        image.shape[:2],
                    )

                probabilities = output[0].astype(
                    np.float32
                )

                action_index = int(
                    np.argmax(probabilities)
                )

                last_action_name = (
                    action_model.class_names[
                        action_index
                    ]
                )
                last_action_probability = float(
                    probabilities[action_index]
                )
                last_probabilities = probabilities

            x1, y1, x2, y2 = bbox.astype(int)

            color = action_color(last_action_name)

            draw_pose(
                annotated,
                keypoints_xy,
                keypoints_conf,
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            depth = get_bbox_depth(
                depth_frame,
                bbox,
            )

            depth_text = (
                f"{depth:.2f} m"
                if depth is not None
                else "invalid"
            )

            if len(skeleton_buffer) < SEQUENCE_LENGTH:
                action_text = (
                    f"Collecting "
                    f"{len(skeleton_buffer)}"
                    f"/{SEQUENCE_LENGTH}"
                )
            else:
                action_text = (
                    f"{last_action_name}: "
                    f"{last_action_probability * 100:.1f}%"
                )

            cv2.putText(
                annotated,
                action_text,
                (
                    x1,
                    max(y1 - 34, 25),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

            cv2.putText(
                annotated,
                f"Depth: {depth_text}",
                (
                    x1,
                    max(y1 - 10, 50),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
            )

            if len(skeleton_buffer) == SEQUENCE_LENGTH:
                print(
                    f"action={last_action_name}, "
                    f"prob={last_action_probability:.3f}, "
                    f"depth={depth_text}, "
                    f"fps={fps:.1f}"
                )

                if last_action_name in ("Fall Down", "Lying Down"):
                    notify_fall(last_action_name, last_action_probability)

        else:
            missing_frames += 1

            if missing_frames > MAX_MISSING_FRAMES:
                skeleton_buffer.clear()
                previous_bbox = None

                last_action_name = "Collecting"
                last_action_probability = 0.0

            cv2.putText(
                annotated,
                "NO PERSON",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            (
                f"Sequence: "
                f"{len(skeleton_buffer)}"
                f"/{SEQUENCE_LENGTH}"
            ),
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        success, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if success:
            jpeg_bytes = encoded.tobytes()
            with frame_lock:
                latest_jpeg = jpeg_bytes

            # MQTT로 프레임 publish (10fps)
            now = time.time()
            if now - _last_mqtt_frame >= MQTT_FRAME_INTERVAL:
                _last_mqtt_frame = now
                try:
                    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
                    mqtt_client.publish(MQTT_TOPIC_FRAME, b64, qos=0)

                    # 현재 액션 publish
                    import json as _json
                    action_payload = _json.dumps({
                        "action": last_action_name,
                        "confidence": round(float(last_action_probability), 3),
                    })
                    mqtt_client.publish(MQTT_TOPIC_ACTION, action_payload, qos=0)
                except Exception:
                    pass


# ============================================================
# Flask 스트림
# ============================================================

def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

        time.sleep(0.03)


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>YOLO26 Pose + TSSTG</title>
        <style>
            body {
                background: #111;
                color: #eee;
                text-align: center;
                font-family: Arial, sans-serif;
            }

            img {
                width: min(95vw, 960px);
                border: 2px solid #555;
            }
        </style>
    </head>
    <body>
        <h2>YOLO26n-pose + Pretrained TSSTG</h2>
        <p>
            Standing · Walking · Sitting · Lying Down ·
            Stand up · Sit down · Fall Down
        </p>
        <img src="/video_feed">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        ),
    )


if __name__ == "__main__":
    import threading

    inference_thread = threading.Thread(
        target=inference_loop,
        daemon=True,
    )
    inference_thread.start()

    try:
        app.run(
            host="0.0.0.0",
            port=WEB_PORT,
            threaded=True,
            use_reloader=False,
        )
    finally:
        pipeline.stop()
