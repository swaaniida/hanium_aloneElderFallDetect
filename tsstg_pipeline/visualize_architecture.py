from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402


NAVY = "#183153"
BLUE = "#2F6B9A"
LIGHT_BLUE = "#E8F1F7"
TEAL = "#168AAD"
LIGHT_TEAL = "#E3F4F6"
ORANGE = "#E67E22"
LIGHT_ORANGE = "#FDF0E3"
GREEN = "#3A8D5D"
LIGHT_GREEN = "#E7F3EB"
GRAY = "#5D6875"
LIGHT_GRAY = "#F3F5F7"
LINE = "#B9C2CC"
RED = "#C94C4C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-style TSSTG architecture figure"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/figures")
    )
    parser.add_argument("--stem", default="tsstg_fall_detection_architecture")
    return parser.parse_args()


def box(
    axis,
    x,
    y,
    width,
    height,
    title,
    subtitle="",
    *,
    face=LIGHT_GRAY,
    edge=LINE,
    title_color=NAVY,
    linewidth=1.4,
    linestyle="-",
    title_size=10.0,
    subtitle_size=8.0,
    zorder=2,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    axis.add_patch(patch)
    title_y = y + height * (0.60 if subtitle else 0.50)
    axis.text(
        x + width / 2,
        title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=zorder + 1,
    )
    if subtitle:
        axis.text(
            x + width / 2,
            y + height * 0.29,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=GRAY,
            linespacing=1.25,
            zorder=zorder + 1,
        )
    return patch


def arrow(axis, start, end, *, color=GRAY, width=1.5, style="-|>", zorder=4):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=width,
        color=color,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def label(axis, x, y, text, *, color=GRAY, size=8, weight="normal", **kwargs):
    axis.text(
        x,
        y,
        text,
        ha=kwargs.pop("ha", "center"),
        va=kwargs.pop("va", "center"),
        fontsize=size,
        color=color,
        fontweight=weight,
        **kwargs,
    )


def draw_skeleton(axis, cx, cy, scale=1.0, motion=False):
    points = {
        0: (0.00, 0.44),
        13: (0.00, 0.28),
        1: (-0.18, 0.22),
        2: (0.18, 0.22),
        3: (-0.29, 0.03),
        4: (0.29, 0.03),
        5: (-0.36, -0.16),
        6: (0.36, -0.16),
        7: (-0.13, -0.13),
        8: (0.13, -0.13),
        9: (-0.16, -0.38),
        10: (0.16, -0.38),
        11: (-0.18, -0.61),
        12: (0.18, -0.61),
    }
    edges = [
        (0, 13), (13, 1), (13, 2), (1, 3), (3, 5), (2, 4), (4, 6),
        (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12),
    ]
    line_color = ORANGE if motion else BLUE
    node_color = "#F5A65B" if motion else TEAL
    transformed = {
        key: (cx + x * 0.075 * scale, cy + y * 0.105 * scale)
        for key, (x, y) in points.items()
    }
    for start, end in edges:
        x1, y1 = transformed[start]
        x2, y2 = transformed[end]
        axis.plot([x1, x2], [y1, y2], color=line_color, linewidth=1.7, zorder=4)
    for node, (x, y) in transformed.items():
        radius = 0.0046 if node != 13 else 0.0058
        axis.add_patch(
            Circle((x, y), radius * scale, facecolor=node_color, edgecolor="white",
                   linewidth=0.8, zorder=5)
        )
    if motion:
        for node in (0, 5, 6, 11, 12):
            x, y = transformed[node]
            arrow(axis, (x - 0.008, y + 0.006), (x + 0.012, y + 0.002),
                  color=ORANGE, width=1.0, zorder=3)


def draw_encoder(axis, x, y, width, height, stream_color, light_color, input_dim):
    box(
        axis,
        x,
        y,
        width,
        height,
        "Shared ST-GCN encoder design",
        "10 residual spatial-temporal graph blocks",
        face="white",
        edge=stream_color,
        title_color=stream_color,
        linewidth=1.7,
        title_size=10.5,
    )
    inner_y = y + height * 0.19
    inner_h = height * 0.43
    gap = width * 0.014
    specs = [
        ("Data BN", f"{input_dim}×14"),
        ("B1", f"{input_dim}→64\ns=1"),
        ("B2–B4", "64→64\ns=1"),
        ("B5", "64→128\ns=2"),
        ("B6–B7", "128→128\ns=1"),
        ("B8", "128→256\ns=2"),
        ("B9–B10", "256→256\ns=1"),
        ("Global avg.\npooling", "T,V → 1"),
    ]
    weights = [0.85, 0.82, 1.02, 0.92, 1.02, 0.92, 1.05, 1.05]
    usable = width * 0.94 - gap * (len(specs) - 1)
    unit = usable / sum(weights)
    current_x = x + width * 0.03
    for index, ((title, subtitle), weight) in enumerate(zip(specs, weights)):
        item_w = unit * weight
        is_pool = index == len(specs) - 1
        box(
            axis,
            current_x,
            inner_y,
            item_w,
            inner_h,
            title,
            subtitle,
            face=LIGHT_GREEN if is_pool else light_color,
            edge=GREEN if is_pool else stream_color,
            title_color=GREEN if is_pool else stream_color,
            linewidth=1.0,
            title_size=7.6,
            subtitle_size=6.6,
        )
        if index < len(specs) - 1:
            arrow(
                axis,
                (current_x + item_w, inner_y + inner_h / 2),
                (current_x + item_w + gap, inner_y + inner_h / 2),
                color=stream_color,
                width=1.0,
            )
        current_x += item_w + gap
    label(
        axis,
        x + width / 2,
        y + height * 0.085,
        "Each Bk: Spatial GCN(A ⊙ Mₖ) → BN/ReLU → Temporal Conv(9×1) → BN + residual → ReLU",
        color=GRAY,
        size=7.3,
    )


def create_figure(output_dir: Path, stem: str) -> list[Path]:
    figure = plt.figure(figsize=(18, 10.2), facecolor="white")
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    label(
        axis,
        0.04,
        0.955,
        "Two-Stream Spatial–Temporal Graph Convolutional Network for Fall Detection",
        ha="left",
        size=19,
        weight="bold",
        color=NAVY,
    )
    label(
        axis,
        0.04,
        0.920,
        "Code-aligned architecture of the original 7-class TSSTG and the binary fine-tuned head",
        ha="left",
        size=9.5,
        color=GRAY,
    )
    axis.plot([0.04, 0.96], [0.895, 0.895], color=NAVY, linewidth=1.2)

    # (a) Front-end and preprocessing.
    label(axis, 0.04, 0.865, "(a) Pose sequence construction", ha="left", size=11,
          weight="bold", color=NAVY)
    box(axis, 0.04, 0.745, 0.105, 0.085, "RGB frame stream", "RealSense / camera",
        face=LIGHT_GRAY, edge=LINE)
    box(axis, 0.175, 0.745, 0.120, 0.085, "Pose estimator", "YOLO26n-Pose in this system",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE)
    box(axis, 0.325, 0.745, 0.125, 0.085, "Joint selection", "COCO 17 → 13 joints\n(x, y, confidence)",
        face=LIGHT_TEAL, edge=TEAL, title_color=TEAL)
    box(axis, 0.480, 0.745, 0.132, 0.085, "Temporal buffer", "30 frames @ 25 Hz\n≈ 1.2 s window",
        face=LIGHT_GRAY, edge=LINE)
    box(axis, 0.642, 0.745, 0.145, 0.085, "Pose normalization", "image-size normalization\nper-frame scaling to [−1, 1]",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE)
    box(axis, 0.817, 0.745, 0.143, 0.085, "Graph construction", "add neck joint → V=14\nspatial partition K=3",
        face=LIGHT_GREEN, edge=GREEN, title_color=GREEN)
    centers = [0.145, 0.295, 0.450, 0.612, 0.787]
    starts = [0.175, 0.325, 0.480, 0.642, 0.817]
    for end_x, start_x in zip(centers, starts):
        arrow(axis, (end_x, 0.7875), (start_x, 0.7875), color=GRAY)

    # Tensor split and skeleton icons.
    draw_skeleton(axis, 0.105, 0.625, scale=1.05, motion=False)
    label(axis, 0.105, 0.535, "14-node skeleton graph", size=8.3, color=BLUE, weight="bold")
    label(axis, 0.105, 0.512, "BBox is not a TSSTG input", size=7.2, color=GRAY)

    box(axis, 0.175, 0.585, 0.165, 0.082, "Point tensor  P", "N × 3 × 30 × 14\nchannels: x, y, confidence",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, linewidth=1.7)
    box(axis, 0.175, 0.455, 0.165, 0.082, "Motion tensor  ΔP", "N × 2 × 29 × 14\nΔPₜ = Pₜ₊₁ˣʸ − Pₜˣʸ",
        face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE, linewidth=1.7)
    arrow(axis, (0.890, 0.745), (0.258, 0.667), color=GREEN, width=1.4)
    arrow(axis, (0.258, 0.585), (0.258, 0.537), color=ORANGE, width=1.4)
    draw_skeleton(axis, 0.105, 0.445, scale=0.88, motion=True)

    # (b) Two-stream encoders.
    label(axis, 0.375, 0.700, "(b) Two-stream ST-GCN feature extraction", ha="left",
          size=11, weight="bold", color=NAVY)
    draw_encoder(axis, 0.375, 0.555, 0.585, 0.125, BLUE, LIGHT_BLUE, "3")
    draw_encoder(axis, 0.375, 0.405, 0.585, 0.125, ORANGE, LIGHT_ORANGE, "2")
    arrow(axis, (0.340, 0.626), (0.375, 0.626), color=BLUE, width=2.0)
    arrow(axis, (0.340, 0.496), (0.375, 0.468), color=ORANGE, width=2.0)
    label(axis, 0.352, 0.645, "P", color=BLUE, size=9, weight="bold")
    label(axis, 0.352, 0.476, "ΔP", color=ORANGE, size=9, weight="bold")

    # (c) Fusion and heads.
    label(axis, 0.04, 0.365, "(c) Feature fusion and action decision", ha="left", size=11,
          weight="bold", color=NAVY)
    box(axis, 0.175, 0.240, 0.150, 0.090, "Point embedding", "hₚ ∈ ℝ²⁵⁶",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE)
    box(axis, 0.175, 0.105, 0.150, 0.090, "Motion embedding", "hₘ ∈ ℝ²⁵⁶",
        face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE)
    box(axis, 0.385, 0.172, 0.145, 0.095, "Late fusion", "h = [hₚ ∥ hₘ] ∈ ℝ⁵¹²",
        face=LIGHT_GREEN, edge=GREEN, title_color=GREEN, linewidth=1.7)
    arrow(axis, (0.325, 0.285), (0.385, 0.235), color=BLUE, width=1.8)
    arrow(axis, (0.325, 0.150), (0.385, 0.202), color=ORANGE, width=1.8)

    box(axis, 0.585, 0.220, 0.155, 0.110, "Original classifier", "Linear 512→7 + sigmoid\n7 action scores",
        face=LIGHT_BLUE, edge=BLUE, title_color=BLUE, linewidth=1.7)
    arrow(axis, (0.530, 0.220), (0.585, 0.270), color=BLUE, width=1.8)
    box(axis, 0.785, 0.220, 0.175, 0.110, "Original decision", "Fall Down → FALL\nall other actions → NON_FALL",
        face="#FBE9E9", edge=RED, title_color=RED, linewidth=1.7)
    arrow(axis, (0.740, 0.275), (0.785, 0.275), color=RED, width=1.8)

    box(axis, 0.585, 0.075, 0.155, 0.095, "Fine-tuned binary head", "Linear 512→2 + sigmoid",
        face=LIGHT_ORANGE, edge=ORANGE, title_color=ORANGE, linewidth=1.7,
        linestyle="--")
    arrow(axis, (0.530, 0.205), (0.585, 0.125), color=ORANGE, width=1.8)
    box(axis, 0.785, 0.075, 0.175, 0.095, "Binary output", "NON_FALL  |  FALL",
        face=LIGHT_GREEN, edge=GREEN, title_color=GREEN, linewidth=1.7,
        linestyle="--")
    arrow(axis, (0.740, 0.122), (0.785, 0.122), color=GREEN, width=1.8)

    label(
        axis,
        0.662,
        0.195,
        "Standing · Walking · Sitting · Lying Down · Stand up · Sit down · Fall Down",
        size=6.9,
        color=GRAY,
    )
    label(axis, 0.04, 0.035,
          "A: fixed 14-joint adjacency  |  Mₖ: learnable edge-importance mask per block  |  s: temporal stride  |  N: batch size",
          ha="left", size=7.6, color=GRAY)
    label(axis, 0.96, 0.035, "Architecture derived from the project implementation",
          ha="right", size=7.2, color=GRAY)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        save_args = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            save_args["dpi"] = 220
        figure.savefig(path, **save_args)
        outputs.append(path)
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    for output in create_figure(args.output_dir, args.stem):
        print(output.resolve())


if __name__ == "__main__":
    main()
