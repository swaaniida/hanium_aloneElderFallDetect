from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "report_figures" / "tsstg_performance_improvement_compact_ko.png"

WIDTH, HEIGHT = 1200, 650
BACKGROUND = (255, 255, 255)
TEXT = (41, 47, 56)
MUTED = (99, 108, 120)
GRID = (220, 224, 230)
RAW = (155, 165, 179)
FINETUNED = (135, 84, 229)
FINETUNED_DARK = (103, 58, 190)

METRICS = ("정확도", "낙상 재현율", "낙상 F1")
RAW_VALUES = (68, 35, 52)
FINETUNED_VALUES = (85, 85, 85)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "malgunbd.ttf" if bold else "malgun.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text(
        (WIDTH // 2, 30),
        "TSSTG 미세조정 성능 비교",
        font=font(44, bold=True),
        fill=TEXT,
        anchor="ma",
    )
    draw.text(
        (WIDTH // 2, 88),
        "기존 모델 vs 두 환경 통합 모델 · 두 위치 Test 40",
        font=font(23),
        fill=MUTED,
        anchor="ma",
    )

    legend_y = 130
    draw.rounded_rectangle((380, legend_y, 410, legend_y + 22), radius=5, fill=RAW)
    draw.text((423, legend_y + 11), "기존 TSSTG", font=font(22, bold=True), fill=TEXT, anchor="lm")
    draw.rounded_rectangle((660, legend_y, 690, legend_y + 22), radius=5, fill=FINETUNED)
    draw.text((703, legend_y + 11), "전체 결합 모델", font=font(22, bold=True), fill=TEXT, anchor="lm")

    plot_left, plot_right = 105, 1155
    plot_top, plot_bottom = 190, 540
    plot_height = plot_bottom - plot_top

    for value in range(0, 101, 20):
        y = plot_bottom - int(plot_height * value / 100)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        draw.text(
            (plot_left - 18, y),
            str(value),
            font=font(19),
            fill=MUTED,
            anchor="rm",
        )

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(175, 181, 190), width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(175, 181, 190), width=2)
    centers = (285, 620, 955)
    bar_width = 92
    gap = 14
    for index, (center, metric, raw_value, tuned_value) in enumerate(
        zip(centers, METRICS, RAW_VALUES, FINETUNED_VALUES)
    ):
        raw_left = center - gap // 2 - bar_width
        tuned_left = center + gap // 2
        raw_top = plot_bottom - int(plot_height * raw_value / 100)
        tuned_top = plot_bottom - int(plot_height * tuned_value / 100)

        draw.rounded_rectangle(
            (raw_left, raw_top, raw_left + bar_width, plot_bottom),
            radius=6,
            fill=RAW,
        )
        draw.rounded_rectangle(
            (tuned_left, tuned_top, tuned_left + bar_width, plot_bottom),
            radius=6,
            fill=FINETUNED,
        )
        draw.text(
            (raw_left + bar_width // 2, raw_top - 14),
            f"{raw_value}%",
            font=font(25, bold=True),
            fill=(92, 101, 114),
            anchor="ms",
        )
        draw.text(
            (tuned_left + bar_width // 2, tuned_top - 14),
            f"{tuned_value}%",
            font=font(25, bold=True),
            fill=FINETUNED_DARK,
            anchor="ms",
        )
        draw.text(
            (center, plot_bottom + 25),
            metric,
            font=font(27, bold=True),
            fill=TEXT,
            anchor="ma",
        )

        if index == 1:
            pill_box = (center + 85, tuned_top - 63, center + 225, tuned_top - 19)
            draw.rounded_rectangle(pill_box, radius=20, fill=(241, 234, 255), outline=FINETUNED, width=2)
            draw.text(
                ((pill_box[0] + pill_box[2]) // 2, (pill_box[1] + pill_box[3]) // 2 - 1),
                "+50%p",
                font=font(22, bold=True),
                fill=FINETUNED_DARK,
                anchor="mm",
            )

    draw.rounded_rectangle((305, 600, 895, 638), radius=19, fill=(244, 240, 252))
    draw.text(
        (600, 618),
        "낙상 미탐지 13건 → 3건",
        font=font(24, bold=True),
        fill=FINETUNED_DARK,
        anchor="mm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
