from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Arc,
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


INK = "#14283D"
MUTED = "#64748B"
LINE = "#CBD5E1"
BLUE = "#2D6FA3"
TEAL = "#1597A8"
ORANGE = "#E8872D"
PURPLE = "#6E5AA8"
GREEN = "#3E8E63"
RED = "#D55252"
PALE_BLUE = "#EAF3F8"
PALE_TEAL = "#E8F7F7"
PALE_ORANGE = "#FDF1E5"
PALE_PURPLE = "#F1EEFA"
PALE_GREEN = "#EAF5EE"
PALE_RED = "#FBECEC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/figures/pipeline_iterations")
    )
    return parser.parse_args()


def rounded(axis, x, y, width, height, face, edge="none", radius=0.02, lw=1.2,
            linestyle="-", zorder=1):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=linestyle,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def text(axis, x, y, value, size=9, color=INK, weight="normal", ha="center",
         va="center", **kwargs):
    axis.text(x, y, value, fontsize=size, color=color, fontweight=weight,
              ha=ha, va=va, **kwargs)


def flow_arrow(axis, x1, x2, y=0.535, color=INK):
    arrow = FancyArrowPatch(
        (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=15,
        color=color, linewidth=1.8, shrinkA=1, shrinkB=1, zorder=8,
    )
    axis.add_patch(arrow)


def stage_number(axis, x, number, color):
    axis.scatter([x], [0.815], s=430, facecolor=color, edgecolor="white",
                 linewidth=1.5, zorder=10)
    text(axis, x, 0.815, f"{number:02d}", size=7.3, color="white", weight="bold",
         zorder=11)


SKELETON_EDGES = [
    (0, 1), (1, 2), (1, 3), (1, 4), (3, 5), (4, 6),
    (2, 7), (2, 8), (7, 9), (8, 10), (9, 11), (10, 12),
]


def skeleton_points(pose="standing"):
    standing = [
        (0.00, 0.44), (0.00, 0.29), (0.00, 0.03),
        (-0.17, 0.22), (0.17, 0.22), (-0.27, 0.03), (0.27, 0.03),
        (-0.11, -0.04), (0.11, -0.04), (-0.13, -0.29), (0.13, -0.29),
        (-0.16, -0.52), (0.16, -0.52),
    ]
    if pose == "standing":
        return standing
    if pose == "falling":
        result = []
        for x, y in standing:
            result.append((0.80 * x + 0.62 * y, -0.25 * x + 0.53 * y - 0.10))
        return result
    if pose == "lying":
        result = []
        for x, y in standing:
            result.append((1.20 * y, -0.25 * x - 0.18))
        return result
    raise ValueError(pose)


def draw_skeleton(axis, cx, cy, scale, pose="standing", color=TEAL, alpha=1.0,
                  linewidth=2.2, zorder=5):
    # x coordinates are corrected for the wide 18:7.2 figure aspect ratio so
    # human proportions remain natural in the exported physical figure.
    points = [(cx + x * scale * 0.40, cy + y * scale) for x, y in skeleton_points(pose)]
    for first, second in SKELETON_EDGES:
        x1, y1 = points[first]
        x2, y2 = points[second]
        axis.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                  linewidth=linewidth, solid_capstyle="round", zorder=zorder)
    axis.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        s=(scale * 88) ** 2,
        facecolor=color,
        edgecolor="white",
        linewidth=0.7,
        alpha=alpha,
        zorder=zorder + 1,
    )


def camera_icon(axis, cx, cy, scale=1.0, grayscale=False):
    blue = "#6B7280" if grayscale else BLUE
    pale = "#F3F4F6" if grayscale else PALE_BLUE
    rounded(axis, cx - 0.052 * scale, cy - 0.045 * scale, 0.104 * scale,
            0.090 * scale, pale, blue, radius=0.014, lw=1.8, zorder=3)
    axis.add_patch(Rectangle((cx - 0.031 * scale, cy + 0.045 * scale),
                             0.038 * scale, 0.018 * scale, facecolor=blue,
                             edgecolor="none", zorder=4))
    axis.scatter([cx], [cy], s=(47 * scale) ** 2, facecolor="white",
                 edgecolor=blue, linewidth=2.2, zorder=5)
    axis.scatter([cx], [cy], s=(28 * scale) ** 2, facecolor=blue,
                 edgecolor="white", linewidth=1.0, zorder=6)
    axis.scatter([cx - 0.035 * scale], [cy + 0.024 * scale],
                 s=(8 * scale) ** 2,
                 facecolor=ORANGE if not grayscale else blue,
                 edgecolor="none", zorder=6)
    for index in range(3):
        axis.add_patch(Arc((cx + 0.055 * scale, cy),
                           (0.030 + 0.016 * index) * scale,
                           (0.055 + 0.025 * index) * scale,
                           theta1=-50, theta2=50,
                           color=TEAL if not grayscale else blue,
                           linewidth=1.2, alpha=0.75, zorder=2))


def pose_frame_icon(axis, cx, cy, scale=1.0, grayscale=False):
    color = "#6B7280" if grayscale else TEAL
    pale = "#F3F4F6" if grayscale else PALE_TEAL
    rounded(axis, cx - 0.064 * scale, cy - 0.090 * scale, 0.128 * scale,
            0.180 * scale, pale, color, radius=0.012, lw=1.6, zorder=2)
    axis.add_patch(Rectangle((cx - 0.046 * scale, cy - 0.072 * scale),
                             0.092 * scale, 0.144 * scale,
                             facecolor="white", edgecolor=LINE, linewidth=0.8,
                             zorder=3))
    axis.add_patch(Rectangle((cx - 0.034 * scale, cy - 0.061 * scale),
                             0.068 * scale, 0.122 * scale, fill=False,
                             edgecolor=BLUE if not grayscale else color,
                             linewidth=1.0, linestyle="--", zorder=4))
    draw_skeleton(axis, cx, cy + 0.002 * scale, 0.090 * scale,
                  color=color, linewidth=1.5, zorder=5)
    rounded(axis, cx - 0.051 * scale, cy + 0.064 * scale, 0.047 * scale,
            0.017 * scale, color, color, radius=0.005, zorder=7)
    text(axis, cx - 0.027 * scale, cy + 0.073 * scale, "person", size=4.8,
         color="white", weight="bold")


def sequence_icon(axis, cx, cy, scale=1.0, grayscale=False):
    colors = ["#9CA3AF"] * 4 if grayscale else [BLUE, TEAL, ORANGE, RED]
    poses = ["standing", "standing", "falling", "lying"]
    offsets = [-0.052, -0.017, 0.018, 0.053]
    for index, (offset, pose, color) in enumerate(zip(offsets, poses, colors)):
        x = cx + offset * scale
        axis.plot([x, x], [cy - 0.100 * scale, cy + 0.102 * scale],
                  color=LINE, linewidth=0.7, zorder=1)
        draw_skeleton(axis, x, cy + (0.015 if pose == "standing" else 0) * scale,
                      0.070 * scale, pose=pose, color=color,
                      alpha=0.48 + index * 0.17, linewidth=1.45, zorder=3 + index)
    axis.plot([cx - 0.066 * scale, cx + 0.068 * scale],
              [cy - 0.112 * scale, cy - 0.112 * scale], color=INK if not grayscale else "#6B7280",
              linewidth=1.3, zorder=4)
    arrow = FancyArrowPatch((cx + 0.050 * scale, cy - 0.112 * scale),
                            (cx + 0.074 * scale, cy - 0.112 * scale),
                            arrowstyle="-|>", mutation_scale=10,
                            color=INK if not grayscale else "#6B7280", linewidth=1.3,
                            zorder=5)
    axis.add_patch(arrow)
    text(axis, cx, cy + 0.128 * scale, "30 frames", size=7.5,
         color=ORANGE if not grayscale else "#6B7280", weight="bold")


def network_layers(axis, x, cy, color, scale=1.0):
    layers = [2, 3, 3, 2]
    positions = []
    for layer_index, count in enumerate(layers):
        layer_x = x + layer_index * 0.022 * scale
        ys = [cy + (i - (count - 1) / 2) * 0.026 * scale for i in range(count)]
        positions.append([(layer_x, y) for y in ys])
    for current, following in zip(positions[:-1], positions[1:]):
        for x1, y1 in current:
            for x2, y2 in following:
                axis.plot([x1, x2], [y1, y2], color=color, alpha=0.25,
                          linewidth=0.65, zorder=3)
    for layer in positions:
        for px, py in layer:
            axis.add_patch(Circle((px, py), 0.0048 * scale, facecolor=color,
                                  edgecolor="white", linewidth=0.5, zorder=4))


def tsstg_icon(axis, cx, cy, scale=1.0, grayscale=False):
    blue = "#6B7280" if grayscale else BLUE
    orange = "#9CA3AF" if grayscale else ORANGE
    purple = "#4B5563" if grayscale else PURPLE
    rounded(axis, cx - 0.100 * scale, cy - 0.118 * scale, 0.200 * scale,
            0.236 * scale, "white", purple, radius=0.015, lw=1.7, zorder=1)
    text(axis, cx, cy + 0.091 * scale, "TWO-STREAM TSSTG", size=7.2,
         color=purple, weight="bold")
    rounded(axis, cx - 0.083 * scale, cy + 0.014 * scale, 0.166 * scale,
            0.055 * scale, PALE_BLUE if not grayscale else "#F3F4F6", blue,
            radius=0.010, lw=1.0, zorder=2)
    rounded(axis, cx - 0.083 * scale, cy - 0.070 * scale, 0.166 * scale,
            0.055 * scale, PALE_ORANGE if not grayscale else "#F3F4F6", orange,
            radius=0.010, lw=1.0, zorder=2)
    text(axis, cx - 0.066 * scale, cy + 0.041 * scale, "P", size=7.0,
         color=blue, weight="bold")
    text(axis, cx - 0.066 * scale, cy - 0.043 * scale, "ΔP", size=6.4,
         color=orange, weight="bold")
    network_layers(axis, cx - 0.038 * scale, cy + 0.041 * scale, blue, scale=scale)
    network_layers(axis, cx - 0.038 * scale, cy - 0.043 * scale, orange, scale=scale)
    axis.add_patch(Circle((cx + 0.074 * scale, cy), 0.014 * scale,
                          facecolor=purple, edgecolor="white", linewidth=1.0,
                          zorder=6))
    arrow = FancyArrowPatch((cx + 0.052 * scale, cy + 0.041 * scale),
                            (cx + 0.069 * scale, cy + 0.010 * scale),
                            arrowstyle="-|>", mutation_scale=8, color=blue,
                            linewidth=1.0, zorder=5)
    axis.add_patch(arrow)
    arrow = FancyArrowPatch((cx + 0.052 * scale, cy - 0.043 * scale),
                            (cx + 0.069 * scale, cy - 0.010 * scale),
                            arrowstyle="-|>", mutation_scale=8, color=orange,
                            linewidth=1.0, zorder=5)
    axis.add_patch(arrow)


def decision_icon(axis, cx, cy, scale=1.0, grayscale=False):
    green = "#9CA3AF" if grayscale else GREEN
    red = "#4B5563" if grayscale else RED
    rounded(axis, cx - 0.064 * scale, cy + 0.012 * scale, 0.128 * scale,
            0.071 * scale, PALE_GREEN if not grayscale else "#F3F4F6", green,
            radius=0.012, lw=1.1, zorder=2)
    rounded(axis, cx - 0.064 * scale, cy - 0.078 * scale, 0.128 * scale,
            0.071 * scale, PALE_RED if not grayscale else "#E5E7EB", red,
            radius=0.012, lw=1.8, zorder=2)
    draw_skeleton(axis, cx - 0.037 * scale, cy + 0.048 * scale, 0.050 * scale,
                  pose="standing", color=green, linewidth=1.0, zorder=4)
    draw_skeleton(axis, cx - 0.035 * scale, cy - 0.042 * scale, 0.047 * scale,
                  pose="lying", color=red, linewidth=1.0, zorder=4)
    text(axis, cx + 0.017 * scale, cy + 0.048 * scale, "NON-FALL", size=6.4,
         color=green, weight="bold")
    text(axis, cx + 0.017 * scale, cy - 0.043 * scale, "FALL", size=7.0,
         color=red, weight="bold")
    axis.scatter([cx + 0.048 * scale], [cy - 0.043 * scale],
                 s=(11 * scale) ** 2, facecolor=red, edgecolor="white",
                 linewidth=0.8, zorder=7)
    text(axis, cx + 0.048 * scale, cy - 0.043 * scale, "!", size=6.0,
         color="white", weight="bold", zorder=8)


def alert_icon(axis, cx, cy, scale=1.0, grayscale=False):
    color = "#6B7280" if grayscale else RED
    cloud = "#9CA3AF" if grayscale else BLUE
    rounded(axis, cx - 0.045 * scale, cy - 0.082 * scale, 0.090 * scale,
            0.164 * scale, "white", INK if not grayscale else color,
            radius=0.014, lw=1.7, zorder=2)
    rounded(axis, cx - 0.034 * scale, cy - 0.055 * scale, 0.068 * scale,
            0.108 * scale, PALE_RED if not grayscale else "#F3F4F6", "none",
            radius=0.008, zorder=3)
    axis.scatter([cx], [cy + 0.010 * scale], s=(35 * scale) ** 2,
                 facecolor=color, edgecolor="white", linewidth=1.2, zorder=5)
    text(axis, cx, cy + 0.010 * scale, "!", size=14 * scale, color="white",
         weight="bold", zorder=7)
    axis.add_patch(Circle((cx, cy - 0.068 * scale), 0.004 * scale,
                          facecolor=INK if not grayscale else color,
                          edgecolor="none", zorder=5))
    for index in range(2):
        axis.add_patch(Arc((cx + 0.049 * scale, cy + 0.040 * scale),
                           (0.026 + index * 0.018) * scale,
                           (0.050 + index * 0.025) * scale,
                           theta1=-55, theta2=55, color=cloud,
                           linewidth=1.4, zorder=4))


def draw_stage_card(axis, cx, width, color, pale, grayscale=False):
    face = "#F8FAFC" if grayscale else pale
    edge = "#D1D5DB" if grayscale else color
    rounded(axis, cx - width / 2, 0.365, width, 0.405, face, edge,
            radius=0.020, lw=1.1, zorder=0)


def render_wireframe(path: Path):
    fig = plt.figure(figsize=(18, 7), facecolor="white")
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
    text(axis, 0.04, 0.925, "Pipeline figure — layout wireframe", size=18,
         weight="bold", ha="left")
    centers = [0.09, 0.245, 0.405, 0.595, 0.775, 0.915]
    widths = [0.115, 0.125, 0.135, 0.185, 0.120, 0.100]
    names = ["Camera", "Pose", "Sequence", "Two-stream TSSTG", "Decision", "Alert"]
    for index, (cx, width, name) in enumerate(zip(centers, widths, names), 1):
        rounded(axis, cx - width / 2, 0.36, width, 0.40, "#F8FAFC", "#9CA3AF",
                radius=0.018, lw=1.2)
        stage_number(axis, cx, index, "#6B7280")
        rounded(axis, cx - width * 0.34, 0.47, width * 0.68, 0.18,
                "#E5E7EB", "#9CA3AF", radius=0.012)
        text(axis, cx, 0.295, name, size=9.5, weight="bold")
        if index < len(centers):
            flow_arrow(axis, cx + width / 2 + 0.008,
                       centers[index] - widths[index] / 2 - 0.008,
                       color="#6B7280")
    text(axis, 0.5, 0.12,
         "Single left-to-right story · six visual stages · implementation details moved to caption",
         size=9, color=MUTED)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_illustrated(path: Path, final=False):
    fig = plt.figure(figsize=(18, 7.2), facecolor="white")
    axis = fig.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")

    title = "Real-Time Skeleton-Based Fall Detection Pipeline"
    subtitle = (
        "From RGB-D sensing to a 30-frame pose sequence, two-stream action recognition, and fall notification"
    )
    text(axis, 0.04, 0.940, title, size=19 if final else 18, weight="bold",
         ha="left", color=INK)
    text(axis, 0.04, 0.902, subtitle, size=8.7, ha="left", color=MUTED)
    axis.plot([0.04, 0.96], [0.872, 0.872], color=LINE, linewidth=0.9)

    centers = [0.085, 0.235, 0.390, 0.585, 0.775, 0.915]
    widths = [0.105, 0.115, 0.125, 0.185, 0.115, 0.090]
    colors = [BLUE, TEAL, ORANGE, PURPLE, RED, BLUE]
    pales = [PALE_BLUE, PALE_TEAL, PALE_ORANGE, PALE_PURPLE, PALE_RED, PALE_BLUE]
    titles = ["RGB-D Camera", "Pose Estimation", "Pose Sequence",
              "Two-Stream TSSTG", "Fall Decision", "Notification"]
    details = [
        "640×480 · 30 fps",
        "YOLO26n-Pose\n17 → 13 joints",
        "30 frames · 25 Hz\n≈ 1.2 s",
        "Point P  +  Motion ΔP\nspatial-temporal features",
        "FALL / NON-FALL",
        "local + remote alert",
    ]

    for cx, width, color, pale in zip(centers, widths, colors, pales):
        draw_stage_card(axis, cx, width, color, pale)
    for index, (cx, color) in enumerate(zip(centers, colors), 1):
        stage_number(axis, cx, index, color)

    camera_icon(axis, centers[0], 0.565, 1.0)
    pose_frame_icon(axis, centers[1], 0.565, 1.0)
    sequence_icon(axis, centers[2], 0.565, 1.0)
    tsstg_icon(axis, centers[3], 0.565, 0.95)
    decision_icon(axis, centers[4], 0.565, 1.0)
    alert_icon(axis, centers[5], 0.565, 0.95)

    for index in range(len(centers) - 1):
        flow_arrow(
            axis,
            centers[index] + widths[index] / 2 + 0.010,
            centers[index + 1] - widths[index + 1] / 2 - 0.010,
            y=0.565,
            color=INK,
        )

    for cx, title_text, detail, color in zip(centers, titles, details, colors):
        text(axis, cx, 0.318, title_text, size=9.7 if final else 9.3,
             color=INK, weight="bold")
        text(axis, cx, 0.270, detail, size=7.1, color=MUTED, linespacing=1.35)

    if final:
        rounded(axis, 0.224, 0.135, 0.545, 0.050, "#F8FAFC", LINE,
                radius=0.012, lw=0.8, zorder=0)
        text(axis, 0.496, 0.160,
             "Pose only: (x, y, confidence)   •   BBox/depth are not TSSTG inputs   •   2D skeleton dynamics drive the decision",
             size=7.4, color=MUTED)
    else:
        text(axis, 0.5, 0.155,
             "Main visual variables: 30-frame temporal context · point stream · motion stream · binary fall output",
             size=8.0, color=MUTED)

    suffix = path.suffix.lower()
    kwargs = {"bbox_inches": "tight", "facecolor": "white"}
    if suffix == ".png":
        kwargs["dpi"] = 220 if final else 180
    fig.savefig(path, **kwargs)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draft1 = args.output_dir / "pipeline_v1_wireframe.png"
    draft2 = args.output_dir / "pipeline_v2_illustrated.png"
    final_png = args.output_dir / "fall_detection_pipeline_final.png"
    final_svg = args.output_dir / "fall_detection_pipeline_final.svg"
    final_pdf = args.output_dir / "fall_detection_pipeline_final.pdf"
    render_wireframe(draft1)
    render_illustrated(draft2, final=False)
    render_illustrated(final_png, final=True)
    render_illustrated(final_svg, final=True)
    render_illustrated(final_pdf, final=True)
    print(draft1.resolve())
    print(draft2.resolve())
    print(final_png.resolve())
    print(final_svg.resolve())
    print(final_pdf.resolve())


if __name__ == "__main__":
    main()
