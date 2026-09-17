from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "collected_pose_data" / "침대 낙상.mp4"
OUTPUT_PATH = ROOT / "outputs" / "bed_fall_motion" / "bed_fall_temporal_overlay.png"

# 실제 영상은 25 FPS이다. 낙상 전 과정이 잘 구분되는 원본 프레임 번호를 사용한다.
FRAME_INDICES = (0, 7, 12, 32)
STAGE_LABELS = ("침대 위", "이탈 시작", "낙하", "착지")
MARKER_POINTS = ((165, 108), (344, 112), (430, 226), (360, 300))
STAGE_COLORS = (
    (39, 102, 199),
    (39, 145, 207),
    (230, 126, 34),
    (196, 57, 57),
)

# 각 시점에서 사람을 감싸는 영역이다. 경계를 크게 흐리게 처리하므로
# 사각형 자국 없이 사람 자세만 원본에 가깝게 강조된다.
SUBJECT_POLYGONS = (
    ((30, 35), (250, 25), (285, 95), (235, 155), (35, 150)),
    ((195, 30), (445, 25), (470, 120), (365, 180), (195, 145)),
    ((255, 140), (475, 135), (525, 310), (405, 375), (260, 310)),
    ((55, 205), (505, 190), (530, 365), (65, 390)),
)
SUBJECT_OPACITIES = (0.62, 0.62, 0.66, 0.92)


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "malgunbd.ttf" if bold else "malgun.ttf"
    font_path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(font_path), size=size)


def read_frames() -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {VIDEO_PATH}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    for frame_index in FRAME_INDICES:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"프레임 {frame_index}을 읽지 못했습니다.")

        # 원본은 좌측 RGB 장면과 우측 스켈레톤 화면이 결합된 영상이다.
        # 시간 경과 합성에는 좌측 RGB 장면만 사용한다.
        frames.append(frame[:, : frame.shape[1] // 2])

    capture.release()
    return frames, fps


def main() -> None:
    frames, fps = read_frames()

    # 먼저 네 시점을 동일 비율로 중첩해 고정 배경과 시간 잔상을 만든다.
    blended = np.mean(np.stack(frames).astype(np.float32), axis=0)
    height, width = blended.shape[:2]

    # 사람 영역은 각 원본 프레임을 높은 불투명도로 다시 입혀 선명도를 높인다.
    # 마지막 착지 자세는 안정된 시점의 프레임을 92%로 강조한다.
    # 넓게 feathering한 마스크를 사용해 합성 경계는 보이지 않게 한다.
    for frame, polygon, opacity in zip(
        frames, SUBJECT_POLYGONS, SUBJECT_OPACITIES
    ):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=18, sigmaY=18)
        alpha = (mask.astype(np.float32) / 255.0 * opacity)[..., None]
        blended = frame.astype(np.float32) * alpha + blended * (1.0 - alpha)

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    scale = 2
    scene = Image.fromarray(blended).resize(
        (blended.shape[1] * scale, blended.shape[0] * scale),
        Image.Resampling.LANCZOS,
    )
    header_height = 104
    footer_height = 132
    canvas = Image.new(
        "RGB",
        (scene.width, header_height + scene.height + footer_height),
        "white",
    )
    canvas.paste(scene, (0, header_height))
    draw = ImageDraw.Draw(canvas, "RGBA")

    title_font = load_font(38, bold=True)
    subtitle_font = load_font(21)
    label_font = load_font(22, bold=True)
    time_font = load_font(19)

    draw.text(
        (canvas.width // 2, 18),
        "침대 낙상 동작의 시간적 경과",
        font=title_font,
        fill=(27, 38, 51, 255),
        anchor="ma",
    )
    draw.text(
        (canvas.width // 2, 68),
        "4개 시점 중첩 · 안정된 착지 자세를 가장 선명하게 강조",
        font=subtitle_font,
        fill=(91, 103, 116, 255),
        anchor="ma",
    )

    scaled_points = [
        (x * scale, header_height + y * scale) for x, y in MARKER_POINTS
    ]
    for start, end in zip(scaled_points, scaled_points[1:]):
        draw.line((start, end), fill=(255, 255, 255, 205), width=8)
        draw.line((start, end), fill=(48, 58, 69, 225), width=4)

    for number, ((x, y), color) in enumerate(
        zip(scaled_points, STAGE_COLORS), start=1
    ):
        radius = 25
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(*color, 238),
            outline=(255, 255, 255, 255),
            width=4,
        )
        draw.text(
            (x, y - 1),
            str(number),
            font=label_font,
            fill=(255, 255, 255, 255),
            anchor="mm",
        )

    footer_top = header_height + scene.height
    draw.rectangle(
        (0, footer_top, canvas.width, canvas.height),
        fill=(247, 249, 251, 255),
    )
    segment_width = canvas.width / len(FRAME_INDICES)
    for index, (frame_index, label, color) in enumerate(
        zip(FRAME_INDICES, STAGE_LABELS, STAGE_COLORS), start=1
    ):
        center_x = int((index - 0.5) * segment_width)
        circle_y = footer_top + 43
        draw.ellipse(
            (center_x - 18, circle_y - 18, center_x + 18, circle_y + 18),
            fill=(*color, 255),
        )
        draw.text(
            (center_x, circle_y - 1),
            str(index),
            font=time_font,
            fill=(255, 255, 255, 255),
            anchor="mm",
        )
        timestamp = frame_index / fps
        draw.text(
            (center_x, footer_top + 71),
            f"{timestamp:.2f}초 · {label}",
            font=label_font,
            fill=(42, 51, 61, 255),
            anchor="ma",
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH, quality=95)
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
