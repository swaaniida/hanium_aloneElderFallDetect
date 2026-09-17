from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


INK = "#14283D"
MUTED = "#64748B"
LINE = "#D7E0E8"
BLUE = "#2D6FA3"
PALE_BLUE = "#EAF3F8"
ORANGE = "#E8872D"
PALE_ORANGE = "#FDF1E5"
GREEN = "#3E8E63"
RED = "#D55252"


ROWS = [
    ("포즈 추론 백엔드", "Ultralytics PyTorch", "NCNN 최적화 런타임"),
    ("YOLO 포즈 처리량", "약 5–8 FPS", "약 25–30 FPS"),
    ("실효 포즈 샘플링", "약 5–8 Hz", "목표 25 Hz 도달"),
    ("30프레임 버퍼 충전", "약 3.8–6.0초", "약 1.2초"),
    ("TSSTG 동작 추론", "약 3–5 Hz", "목표 약 5 Hz"),
    ("25 Hz 수집 지속성", "유지 어려움", "유지 가능"),
    ("CPU 동작 특성", "높은 부하 / 지연 변동 큼", "상대적으로 낮고 안정적"),
    ("권장 용도", "개발 및 예비 경로", "Raspberry Pi 배포"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--stem", default="pytorch_vs_ncnn_estimated_comparison_ko")
    return parser.parse_args()


def rounded(axis, x, y, width, height, face, edge="none", lw=1.0, radius=0.012):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
    )
    axis.add_patch(patch)
    return patch


def label(axis, x, y, value, size=10, color=INK, weight="normal", ha="center"):
    axis.text(x, y, value, fontsize=size, color=color, fontweight=weight,
              ha=ha, va="center")


def create(output_dir: Path, stem: str):
    korean_font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if korean_font_path.exists():
        korean_font = font_manager.FontProperties(fname=str(korean_font_path))
        plt.rcParams["font.family"] = korean_font.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    figure = plt.figure(figsize=(14.5, 9.2), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")

    label(axis, 0.055, 0.935, "PyTorch와 NCNN 런타임 비교", 24, INK,
          "bold", "left")
    label(axis, 0.055, 0.895,
          "이전 Raspberry Pi 분석을 바탕으로 재구성한 기억 기반 추정 범위",
          12.0, MUTED, "normal", "left")
    axis.plot([0.055, 0.945], [0.865, 0.865], color=LINE, linewidth=1.0)

    left = 0.055
    metric_w = 0.28
    value_w = 0.30
    gap = 0.012
    pytorch_x = left + metric_w + gap
    ncnn_x = pytorch_x + value_w + gap
    header_y = 0.775
    header_h = 0.073

    rounded(axis, left, header_y, metric_w, header_h, "#F4F6F8", LINE, 1.0)
    rounded(axis, pytorch_x, header_y, value_w, header_h, PALE_BLUE, BLUE, 1.5)
    rounded(axis, ncnn_x, header_y, value_w, header_h, PALE_ORANGE, ORANGE, 1.5)
    label(axis, left + metric_w / 2, header_y + header_h / 2, "지표", 14, MUTED,
          "bold")
    label(axis, pytorch_x + value_w / 2, header_y + header_h / 2, "PyTorch", 16,
          BLUE, "bold")
    label(axis, ncnn_x + value_w / 2, header_y + header_h / 2, "NCNN", 16,
          ORANGE, "bold")

    row_h = 0.070
    row_gap = 0.006
    start_y = header_y - row_gap - row_h
    for index, (metric, pytorch, ncnn) in enumerate(ROWS):
        y = start_y - index * (row_h + row_gap)
        base = "#FAFBFC" if index % 2 == 0 else "white"
        rounded(axis, left, y, metric_w, row_h, base, LINE, 0.7, 0.008)
        rounded(axis, pytorch_x, y, value_w, row_h,
                PALE_BLUE if index in {1, 2, 3, 4, 5} else base,
                LINE, 0.7, 0.008)
        rounded(axis, ncnn_x, y, value_w, row_h,
                PALE_ORANGE if index in {1, 2, 3, 4, 5} else base,
                LINE, 0.7, 0.008)
        label(axis, left + 0.018, y + row_h / 2, metric, 11.5, INK, "bold", "left")
        pytorch_color = RED if index == 5 else BLUE if index in {1, 2, 3, 4} else INK
        ncnn_color = GREEN if index == 5 else ORANGE if index in {1, 2, 3, 4} else INK
        label(axis, pytorch_x + value_w / 2, y + row_h / 2, pytorch, 11.5,
              pytorch_color, "bold" if index in {1, 2, 3, 4, 5} else "normal")
        label(axis, ncnn_x + value_w / 2, y + row_h / 2, ncnn, 11.5,
              ncnn_color, "bold" if index in {1, 2, 3, 4, 5} else "normal")

    rounded(axis, 0.195, 0.045, 0.610, 0.065, "#EDF6F0", GREEN, 1.1, 0.014)
    label(axis, 0.500, 0.077,
          "NCNN은 포즈 처리량이 약 4배 높고 25 Hz 수집 목표를 충족했다.",
          12.5, GREEN, "bold")
    label(axis, 0.945, 0.022, "기억 기반 추정치 — 논문 게재 전 재측정 필요",
          9.3, MUTED, "normal", "right")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 220
        figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def main():
    args = parse_args()
    for path in create(args.output_dir, args.stem):
        print(path.resolve())


if __name__ == "__main__":
    main()
