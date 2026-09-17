from __future__ import annotations

import re
import struct
import sys
import zlib
from pathlib import Path

import olefile


PARA_TEXT_TAG = 67


def section_number(name: str) -> int:
    match = re.search(r"Section(\d+)$", name)
    return int(match.group(1)) if match else 0


def iter_records(data: bytes):
    offset = 0
    while offset + 4 <= len(data):
        header = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        tag = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:
            if offset + 4 > len(data):
                break
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        if offset + size > len(data):
            break
        yield tag, level, data[offset : offset + size]
        offset += size


def clean_paragraph(raw: bytes) -> str:
    text = raw.decode("utf-16le", errors="ignore")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_paragraphs(path: Path) -> list[str]:
    with olefile.OleFileIO(path) as ole:
        header = ole.openstream("FileHeader").read()
        compressed = bool(struct.unpack_from("<I", header, 36)[0] & 1)
        streams = [
            "/".join(parts)
            for parts in ole.listdir(streams=True, storages=False)
            if parts and parts[0] == "BodyText" and parts[-1].startswith("Section")
        ]
        paragraphs: list[str] = []
        for stream in sorted(streams, key=section_number):
            data = ole.openstream(stream).read()
            if compressed:
                data = zlib.decompress(data, -15)
            for tag, _, payload in iter_records(data):
                if tag == PARA_TEXT_TAG:
                    text = clean_paragraph(payload)
                    if text:
                        paragraphs.append(text)
        return paragraphs


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: inspect_hwp_paragraphs.py REPORT.hwp [KEYWORD ...]")
    path = Path(sys.argv[1])
    keywords = [word.casefold() for word in sys.argv[2:]]
    paragraphs = extract_paragraphs(path)
    for index, paragraph in enumerate(paragraphs, start=1):
        if not keywords or any(word in paragraph.casefold() for word in keywords):
            print(f"[{index}] {paragraph}")


if __name__ == "__main__":
    main()
