from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "report_figures" / "tsstg_performance_improvement_ko.png"

WIDTH = 1600
HEIGHT = 1200
OUTPUT_SIZE = (1200, 900)

NAVY = (25, 40, 61)
TEXT = (35, 45, 59)
MUTED = (99, 112, 128)
OLD = (151, 160, 173)
OLD_LIGHT = (235, 238, 242)
NEW = (19, 143, 131)
NEW_DARK = (12, 111, 102)
NEW_LIGHT = (226, 247, 243)
ACCENT = (49, 103, 210)
WHITE = (255, 255, 255)
BACKGROUND = (248, 250, 252)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "malgunbd.ttf" if bold else "malgun.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    radius: int = 28,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def metric_card(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    title: str,
    explanation: str,
    old_value: int,
    new_value: int,
) -> None:
    left, right = 72, WIDTH - 72
    top, bottom = y, y + 300
    rounded_card(draw, (left, top, right, bottom), fill=WHITE, outline=(222, 228, 235))

    title_x = left + 42
    draw.text((title_x, top + 35), title, font=font(42, bold=True), fill=TEXT)
    draw.text((title_x, top + 93), explanation, font=font(26), fill=MUTED)

    value_y = top + 183
    old_x = 720
    new_x = 1215
    draw.text((old_x, value_y), f"{old_value}%", font=font(92, bold=True), fill=OLD, anchor="mm")
    draw.text((new_x, value_y), f"{new_value}%", font=font(92, bold=True), fill=NEW_DARK, anchor="mm")

    arrow_start = 858
    arrow_end = 1048
    arrow_y = value_y
    draw.line((arrow_start, arrow_y, arrow_end, arrow_y), fill=ACCENT, width=12)
    draw.polygon(
        ((arrow_end, arrow_y), (arrow_end - 34, arrow_y - 25), (arrow_end - 34, arrow_y + 25)),
        fill=ACCENT,
    )

    gain = new_value - old_value
    pill = (1090, top + 32, right - 35, top + 94)
    rounded_card(draw, pill, fill=NEW_LIGHT, radius=30, width=0)
    draw.text(
        ((pill[0] + pill[2]) // 2, (pill[1] + pill[3]) // 2 - 2),
        f"+{gain}%p 향상",
        font=font(29, bold=True),
        fill=NEW_DARK,
        anchor="mm",
    )

    bar_y = bottom - 39
    bar_left, bar_right = 560, right - 40
    bar_width = bar_right - bar_left
    draw.rounded_rectangle(
        (bar_left, bar_y, bar_right, bar_y + 18), radius=9, fill=OLD_LIGHT
    )
    old_end = bar_left + int(bar_width * old_value / 100)
    new_end = bar_left + int(bar_width * new_value / 100)
    draw.rounded_rectangle(
        (bar_left, bar_y, old_end, bar_y + 18), radius=9, fill=OLD
    )
    draw.rounded_rectangle(
        (bar_left, bar_y, new_end, bar_y + 18), radius=9, fill=NEW
    )


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text(
        (WIDTH // 2, 58),
        "TSSTG 미세조정 성능 개선",
        font=font(68, bold=True),
        fill=NAVY,
        anchor="ma",
    )
    draw.text(
        (WIDTH // 2, 148),
        "강의실·가정 환경 수집 데이터 200건  |  독립 테스트 20건",
        font=font(31),
        fill=MUTED,
        anchor="ma",
    )

    legend_y = 218
    draw.ellipse((445, legend_y, 469, legend_y + 24), fill=OLD)
    draw.text((482, legend_y + 12), "기존 TSSTG", font=font(27, bold=True), fill=TEXT, anchor="lm")
    draw.ellipse((825, legend_y, 849, legend_y + 24), fill=NEW)
    draw.text((862, legend_y + 12), "두 환경 미세조정 모델", font=font(27, bold=True), fill=TEXT, anchor="lm")

    metric_card(
        draw,
        y=275,
        title="낙상 재현율",
        explanation="실제 낙상을 낙상으로 찾아낸 비율",
        old_value=40,
        new_value=80,
    )
    metric_card(
        draw,
        y=600,
        title="정확도",
        explanation="전체 테스트를 올바르게 분류한 비율",
        old_value=70,
        new_value=80,
    )

    bottom_top = 935
    rounded_card(
        draw,
        (72, bottom_top, WIDTH - 72, HEIGHT - 55),
        fill=NAVY,
        radius=30,
        width=0,
    )
    draw.text(
        (160, bottom_top + 47),
        "낙상 미탐지",
        font=font(39, bold=True),
        fill=(207, 218, 230),
    )
    draw.text(
        (650, bottom_top + 103),
        "6건",
        font=font(94, bold=True),
        fill=(208, 216, 226),
        anchor="mm",
    )
    draw.line((790, bottom_top + 103, 990, bottom_top + 103), fill=(89, 211, 190), width=13)
    draw.polygon(
        ((990, bottom_top + 103), (950, bottom_top + 74), (950, bottom_top + 132)),
        fill=(89, 211, 190),
    )
    draw.text(
        (1160, bottom_top + 103),
        "2건",
        font=font(94, bold=True),
        fill=(89, 225, 200),
        anchor="mm",
    )
    draw.text(
        (1440, bottom_top + 104),
        "67% 감소",
        font=font(36, bold=True),
        fill=(89, 225, 200),
        anchor="rm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
