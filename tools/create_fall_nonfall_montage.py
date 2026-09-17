from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "collected_pose_data"
OUTPUT_PATH = ROOT / "outputs" / "dataset_montage" / "fall_nonfall_36_montage.mp4"

SEED = 20260908
OUTPUT_FPS = 25.0
CLIP_SECONDS = 4.0
REPEAT_COUNT = 2
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
COLS = 6
ROWS = 6
CELL_WIDTH = FRAME_WIDTH // COLS
HEADER_HEIGHT = 72
CELL_HEIGHT = (FRAME_HEIGHT - HEADER_HEIGHT) // ROWS
LABEL_HEIGHT = 34
CONTENT_MARGIN_X = 3
CONTENT_HEIGHT = 118
CONTENT_WIDTH = 314

FALL_COLOR = (190, 57, 72)
NON_FALL_COLOR = (23, 126, 143)
BACKGROUND_COLOR = (15, 23, 34)


@dataclass
class Clip:
    path: Path
    label: str
    number: int
    capture: cv2.VideoCapture | None = None
    fps: float = 0.0
    frame_count: int = 0
    current_index: int = -1
    current_frame: np.ndarray | None = None

    @property
    def color(self) -> tuple[int, int, int]:
        return FALL_COLOR if self.label == "낙상" else NON_FALL_COLOR

    def open(self) -> None:
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.reset()

    def reset(self) -> None:
        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_index = -1
        self.current_frame = None

    def frame_at(self, seconds: float) -> np.ndarray:
        if self.capture is None:
            raise RuntimeError("clip is not open")
        target = min(int(round(seconds * self.fps)), self.frame_count - 1)
        if target < self.current_index:
            self.reset()
        while self.current_index < target:
            ok, frame = self.capture.read()
            if not ok:
                self.reset()
                ok, frame = self.capture.read()
                if not ok:
                    raise RuntimeError(f"프레임을 읽지 못했습니다: {self.path}")
            self.current_index += 1
            self.current_frame = frame
        if self.current_frame is None:
            ok, self.current_frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"첫 프레임을 읽지 못했습니다: {self.path}")
            self.current_index = 0
        return self.current_frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "malgunbd.ttf" if bold else "malgun.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)


def choose_clips() -> list[Clip]:
    rng = random.Random(SEED)
    fall_files = sorted((DATA_ROOT / "낙상").glob("*.mp4"))
    non_fall_files = sorted((DATA_ROOT / "비낙상").glob("*.mp4"))
    if len(fall_files) < 18 or len(non_fall_files) < 18:
        raise RuntimeError(
            f"각 클래스에 18개가 필요합니다: 낙상={len(fall_files)}, "
            f"비낙상={len(non_fall_files)}"
        )

    # 환경 차이가 큰 대표 영상은 유지하고, 나머지 낙상 영상은 무작위로 고른다.
    must_include_names = {"낙상.mp4", "침대 낙상.mp4"}
    must_include = [p for p in fall_files if p.name in must_include_names]
    other_falls = [p for p in fall_files if p.name not in must_include_names]
    selected_falls = must_include + rng.sample(other_falls, 18 - len(must_include))
    selected_non_falls = rng.sample(non_fall_files, 18)
    rng.shuffle(selected_falls)
    rng.shuffle(selected_non_falls)

    fall_queue = [Clip(path, "낙상", i + 1) for i, path in enumerate(selected_falls)]
    non_fall_queue = [
        Clip(path, "비낙상", i + 1) for i, path in enumerate(selected_non_falls)
    ]

    # 각 행에 낙상 3개와 비낙상 3개가 들어가도록 하되 위치는 무작위로 섞는다.
    arranged: list[Clip] = []
    for _ in range(ROWS):
        fall_slots = set(rng.sample(range(COLS), COLS // 2))
        for column in range(COLS):
            arranged.append(
                fall_queue.pop() if column in fall_slots else non_fall_queue.pop()
            )

    # 최종 발표 화면을 위한 지정 배치 교환(행·열은 1부터 시작).
    for first, second in (((2, 1), (2, 6)), ((5, 3), (3, 6))):
        first_index = (first[0] - 1) * COLS + first[1] - 1
        second_index = (second[0] - 1) * COLS + second[1] - 1
        arranged[first_index], arranged[second_index] = (
            arranged[second_index],
            arranged[first_index],
        )
    return arranged


def create_static_canvas(clips: list[Clip]) -> np.ndarray:
    canvas = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)
    subtitle_font = font(27, bold=True)
    label_font = font(18, bold=True)

    draw.text(
        (FRAME_WIDTH // 2, HEADER_HEIGHT // 2 - 2),
        "두 환경에서 수집한 200여 개의 시계열 샘플",
        font=subtitle_font,
        fill=(232, 238, 245),
        anchor="mm",
    )

    for index, clip in enumerate(clips):
        row, column = divmod(index, COLS)
        x = column * CELL_WIDTH
        y = HEADER_HEIGHT + row * CELL_HEIGHT
        color = clip.color
        draw.rectangle(
            (x, y, x + CELL_WIDTH - 1, y + CELL_HEIGHT - 1),
            fill=(20, 29, 41),
            outline=color,
            width=3,
        )
        draw.rectangle(
            (x + 3, y + 3, x + CELL_WIDTH - 4, y + LABEL_HEIGHT),
            fill=color,
        )
        draw.text(
            (x + 14, y + 5),
            clip.label,
            font=label_font,
            fill=(255, 255, 255),
        )

    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main() -> None:
    clips = choose_clips()
    for clip in clips:
        clip.open()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        for clip in clips:
            clip.close()
        raise RuntimeError("MP4 VideoWriter를 열 수 없습니다.")

    static_canvas = create_static_canvas(clips)
    frames_per_loop = int(round(CLIP_SECONDS * OUTPUT_FPS))
    total_frames = frames_per_loop * REPEAT_COUNT

    try:
        for output_index in range(total_frames):
            loop_index = output_index % frames_per_loop
            if loop_index == 0:
                for clip in clips:
                    clip.reset()
            seconds = loop_index / OUTPUT_FPS
            canvas = static_canvas.copy()

            for index, clip in enumerate(clips):
                source = clip.frame_at(seconds)
                resized = cv2.resize(
                    source,
                    (CONTENT_WIDTH, CONTENT_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                row, column = divmod(index, COLS)
                x = column * CELL_WIDTH + CONTENT_MARGIN_X
                y = HEADER_HEIGHT + row * CELL_HEIGHT + LABEL_HEIGHT + 3
                canvas[y : y + CONTENT_HEIGHT, x : x + CONTENT_WIDTH] = resized

            progress = (loop_index + 1) / frames_per_loop
            cv2.rectangle(canvas, (0, HEADER_HEIGHT - 5), (FRAME_WIDTH, HEADER_HEIGHT - 1), (44, 55, 69), -1)
            cv2.rectangle(
                canvas,
                (0, HEADER_HEIGHT - 5),
                (int(FRAME_WIDTH * progress), HEADER_HEIGHT - 1),
                (66, 186, 150),
                -1,
            )
            writer.write(canvas)
            if (output_index + 1) % 25 == 0:
                print(f"{output_index + 1}/{total_frames}")
    finally:
        writer.release()
        for clip in clips:
            clip.close()

    excluded_falls = sorted(
        p.name
        for p in (DATA_ROOT / "낙상").glob("*.mp4")
        if p not in {clip.path for clip in clips if clip.label == "낙상"}
    )
    print(f"output={OUTPUT_PATH.resolve()}")
    print(f"excluded_fall={excluded_falls}")


if __name__ == "__main__":
    main()
