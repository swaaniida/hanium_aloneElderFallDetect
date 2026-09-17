from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def slide_number(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 0


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: inspect_pptx_text.py PRESENTATION.pptx")

    path = Path(sys.argv[1])
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            (
                name
                for name in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ),
            key=slide_number,
        )
        for slide_name in slides:
            root = ET.fromstring(archive.read(slide_name))
            chunks = [
                node.text.strip()
                for node in root.iter(f"{{{DRAWING_NS}}}t")
                if node.text and node.text.strip()
            ]
            print(f"\n=== SLIDE {slide_number(slide_name)} ===")
            print("\n".join(chunks))


if __name__ == "__main__":
    main()
