from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from visualize_architecture import (  # noqa: E402
    BLUE,
    GRAY,
    GREEN,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_GREEN,
    LIGHT_ORANGE,
    LIGHT_TEAL,
    LINE,
    NAVY,
    ORANGE,
    RED,
    TEAL,
    arrow,
    box,
    draw_skeleton,
    label,
)


PURPLE = "#6F5AA8"
LIGHT_PURPLE = "#F0ECF8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-style end-to-end fall detection pipeline"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/figures")
    )
    parser.add_argument("--stem", default="fall_detection_end_to_end_pipeline")
    return parser.parse_args()


def section(axis, x, y, width, height, title, number, face, edge):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.25,
        zorder=0,
    )
    axis.add_patch(patch)
    label(axis, x + 0.012, y + height - 0.022, f"{number}  {title}", ha="left",
          size=10.5, weight="bold", color=edge)


def rate_tag(axis, x, y, text, color=GRAY):
    label(
        axis,
        x,
        y,
        text,
        size=6.8,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": LINE,
            "linewidth": 0.6,
        },
        zorder=8,
    )


def create_figure(output_dir: Path, stem: str) -> list[Path]:
    figure = plt.figure(figsize=(19, 11), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    label(
        axis,
        0.035,
        0.957,
        "End-to-End Real-Time Fall Detection and Notification Pipeline",
        ha="left",
        size=20,
        weight="bold",
        color=NAVY,
    )
    label(
        axis,
        0.035,
        0.925,
        "RealSense sensing → YOLO pose extraction → 25 Hz skeleton sequence → TSSTG action recognition → local/cloud services",
        ha="left",
        size=9.4,
        color=GRAY,
    )
    axis.plot([0.035, 0.965], [0.900, 0.900], color=NAVY, linewidth=1.2)

    # Main perception pipeline.
    section(axis, 0.03, 0.665, 0.94, 0.205, "Sensing and pose perception", "01",
            "#F7FAFC", BLUE)
    main_y = 0.715
    main_h = 0.095
    stages = [
        (0.055, 0.105, "RealSense input", "RGB + depth\n640×480 @ 30 fps", LIGHT_BLUE, BLUE),
        (0.195, 0.105, "Frame alignment", "depth aligned to RGB\nBGR image", LIGHT_GRAY, GRAY),
        (0.335, 0.115, "YOLO26n-Pose", "PyTorch or NCNN\nimgsz=320, conf≥0.35", LIGHT_TEAL, TEAL),
        (0.485, 0.115, "Target selection", "previous-box IoU≥0.20\nelse largest person", LIGHT_GRAY, GRAY),
        (0.635, 0.120, "Pose conversion", "COCO 17 → TSSTG 13\n(x, y, confidence)", LIGHT_BLUE, BLUE),
        (0.790, 0.120, "Quality + sampler", "low confidence: score=0\nfixed 25 Hz scheduler", LIGHT_ORANGE, ORANGE),
    ]
    for x, width, title, subtitle, face, edge in stages:
        box(axis, x, main_y, width, main_h, title, subtitle, face=face, edge=edge,
            title_color=edge, linewidth=1.4, title_size=9.0, subtitle_size=7.2)
    for left, right in zip(stages[:-1], stages[1:]):
        arrow(axis, (left[0] + left[1], main_y + main_h / 2),
              (right[0], main_y + main_h / 2), color=GRAY, width=1.5)
    rate_tag(axis, 0.177, 0.762, "30 fps")
    rate_tag(axis, 0.317, 0.762, "RGB")
    rate_tag(axis, 0.467, 0.762, "boxes + 17 joints")
    rate_tag(axis, 0.617, 0.762, "one tracked person")
    rate_tag(axis, 0.772, 0.762, "13×3 pose")

    draw_skeleton(axis, 0.938, 0.760, scale=0.74)
    label(axis, 0.936, 0.690, "pose sample", size=7.0, color=BLUE, weight="bold")
    arrow(axis, (0.910, main_y + main_h / 2), (0.925, 0.762), color=ORANGE, width=1.5)

    label(axis, 0.055, 0.687,
          "Identity continuity: if a person is missing for >10 frames, clear the skeleton buffer and restart collection.",
          ha="left", size=7.2, color=GRAY)

    # Temporal model, including internal architecture.
    section(axis, 0.03, 0.355, 0.94, 0.275, "Temporal action recognition", "02",
            "#FBFAFD", PURPLE)
    box(axis, 0.055, 0.435, 0.125, 0.110, "Skeleton deque", "30 × 13 × 3\nrolling window ≈1.2 s",
        face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE, linewidth=1.7)
    axis.plot([0.850, 0.850, 0.117], [0.715, 0.646, 0.646], color=ORANGE,
              linewidth=1.6, zorder=3)
    arrow(axis, (0.117, 0.646), (0.117, 0.545), color=ORANGE, width=1.6)
    rate_tag(axis, 0.500, 0.646, "25 Hz pose-sample bus", ORANGE)

    box(axis, 0.215, 0.435, 0.130, 0.110, "Async inference queue", "maxsize=1\nlatest complete window",
        face=LIGHT_GRAY, edge=GRAY, title_color=GRAY)
    arrow(axis, (0.180, 0.490), (0.215, 0.490), color=GRAY, width=1.7)
    rate_tag(axis, 0.198, 0.518, "5 Hz trigger")

    # TSSTG expanded region.
    ts_x, ts_y, ts_w, ts_h = 0.385, 0.390, 0.480, 0.205
    panel = FancyBboxPatch(
        (ts_x, ts_y), ts_w, ts_h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        facecolor="white", edgecolor=PURPLE, linewidth=1.8, zorder=1,
    )
    axis.add_patch(panel)
    label(axis, ts_x + 0.015, ts_y + ts_h - 0.020,
          "TSSTG — Two-Stream Spatial–Temporal Graph Convolutional Network",
          ha="left", size=9.1, weight="bold", color=PURPLE)

    box(axis, 0.405, 0.455, 0.105, 0.080, "Normalize + neck", "scale xy to [−1,1]\n13 → 14 graph nodes",
        face=LIGHT_PURPLE, edge=PURPLE, title_color=PURPLE, title_size=8.0,
        subtitle_size=6.6)
    arrow(axis, (0.345, 0.490), (0.405, 0.495), color=PURPLE, width=1.8)
    box(axis, 0.540, 0.488, 0.105, 0.058, "Point stream", "3×30×14",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, title_size=8.0,
        subtitle_size=6.8)
    box(axis, 0.540, 0.415, 0.105, 0.062, "Motion stream", "2×29×14  (Δxy)",
        face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE, title_size=8.0,
        subtitle_size=6.8)
    arrow(axis, (0.510, 0.495), (0.540, 0.517), color=BLUE, width=1.4)
    arrow(axis, (0.510, 0.480), (0.540, 0.446), color=ORANGE, width=1.4)
    box(axis, 0.675, 0.488, 0.105, 0.058, "10 ST-GCN blocks", "64 → 128 → 256",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, title_size=7.7,
        subtitle_size=6.6)
    box(axis, 0.675, 0.415, 0.105, 0.062, "10 ST-GCN blocks", "64 → 128 → 256",
        face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE, title_size=7.7,
        subtitle_size=6.6)
    arrow(axis, (0.645, 0.517), (0.675, 0.517), color=BLUE, width=1.4)
    arrow(axis, (0.645, 0.446), (0.675, 0.446), color=ORANGE, width=1.4)
    box(axis, 0.805, 0.455, 0.045, 0.080, "GAP", "256\neach",
        face=LIGHT_GREEN, edge=GREEN, title_color=GREEN, title_size=7.6,
        subtitle_size=6.5)
    arrow(axis, (0.780, 0.517), (0.805, 0.510), color=BLUE, width=1.3)
    arrow(axis, (0.780, 0.446), (0.805, 0.480), color=ORANGE, width=1.3)

    box(axis, 0.895, 0.435, 0.055, 0.110, "Fusion + head", "concat 512\nFC 512→7\nsigmoid",
        face=LIGHT_PURPLE, edge=PURPLE, title_color=PURPLE, title_size=7.6,
        subtitle_size=6.5, linewidth=1.7)
    arrow(axis, (0.850, 0.495), (0.895, 0.495), color=PURPLE, width=1.8)
    # Decision and outputs.
    section(axis, 0.03, 0.065, 0.94, 0.255, "Decision, visualization, and external services", "03",
            "#F9FBF9", GREEN)
    box(axis, 0.055, 0.150, 0.130, 0.100, "Action result", "7-class argmax\nname + confidence",
        face=LIGHT_PURPLE, edge=PURPLE, title_color=PURPLE, linewidth=1.7)
    axis.plot([0.922, 0.922, 0.120], [0.435, 0.338, 0.338], color=PURPLE,
              linewidth=1.7, zorder=3)
    arrow(axis, (0.120, 0.338), (0.120, 0.250), color=PURPLE, width=1.7)
    rate_tag(axis, 0.520, 0.338, "asynchronous action-result bus ≈5 Hz", PURPLE)

    box(axis, 0.225, 0.185, 0.145, 0.082, "Event rule", "Fall Down or Lying Down\n10 s cooldown",
        face="#FBE9E9", edge=RED, title_color=RED, linewidth=1.7)
    arrow(axis, (0.185, 0.205), (0.225, 0.226), color=RED, width=1.6)
    box(axis, 0.410, 0.185, 0.145, 0.082, "Event queue worker", "maxsize=4 · non-blocking\nJSON event payload",
        face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE)
    arrow(axis, (0.370, 0.226), (0.410, 0.226), color=ORANGE, width=1.6)
    box(axis, 0.595, 0.185, 0.155, 0.082, "Backend notification", "HTTP POST /events/fall\nelder, action, confidence",
        face=LIGHT_GREEN, edge=GREEN, title_color=GREEN, linewidth=1.7)
    arrow(axis, (0.555, 0.226), (0.595, 0.226), color=GREEN, width=1.6)

    box(axis, 0.225, 0.085, 0.145, 0.070, "Annotated JPEG", "pose · bbox · depth · action",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, title_size=8.3,
        subtitle_size=6.7)
    arrow(axis, (0.185, 0.180), (0.225, 0.120), color=BLUE, width=1.5)
    box(axis, 0.410, 0.085, 0.145, 0.070, "Local Flask stream", "MJPEG /video_feed · port 5000",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, title_size=8.3,
        subtitle_size=6.7)
    arrow(axis, (0.370, 0.120), (0.410, 0.120), color=BLUE, width=1.5)
    box(axis, 0.595, 0.085, 0.155, 0.070, "MQTT publisher", "frame ≈10 fps + action JSON",
        face=LIGHT_TEAL, edge=TEAL, title_color=TEAL, title_size=8.3,
        subtitle_size=6.7)
    mqtt_branch = FancyArrowPatch(
        (0.370, 0.105),
        (0.595, 0.105),
        connectionstyle="arc3,rad=-0.34",
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.5,
        color=TEAL,
        zorder=4,
    )
    axis.add_patch(mqtt_branch)

    box(axis, 0.800, 0.120, 0.145, 0.130, "Remote monitoring", "MQTT broker / backend\nframe topic · action topic\nfall event API",
        face=LIGHT_GREEN, edge=GREEN, title_color=GREEN, linewidth=1.8)
    arrow(axis, (0.750, 0.226), (0.800, 0.205), color=GREEN, width=1.6)
    arrow(axis, (0.750, 0.120), (0.800, 0.165), color=TEAL, width=1.6)

    label(axis, 0.055, 0.282,
          "Depth and bounding boxes support tracking/visualization only; TSSTG receives normalized skeleton coordinates and confidence scores.",
          ha="left", size=6.8, color=GRAY)
    label(axis, 0.965, 0.028,
          "Implementation-aligned system diagram · apps/realtime_fall_detection.py",
          ha="right", size=7.0, color=GRAY)

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


def main() -> None:
    args = parse_args()
    for path in create_figure(args.output_dir, args.stem):
        print(path.resolve())


if __name__ == "__main__":
    main()
