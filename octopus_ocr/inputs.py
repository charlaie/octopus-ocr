from __future__ import annotations

from pathlib import Path
from typing import Iterable

from octopus_ocr.video import VIDEO_SUFFIXES


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUPPORTED_INPUT_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES


def expand_input_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(child for child in path.iterdir() if _is_supported_file(child))
        elif _is_supported_file(path):
            expanded.append(path)
    return sorted(_dedupe_paths(expanded), key=lambda item: (item.name.casefold(), str(item).casefold()))


def _is_supported_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        key = path.resolve(strict=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique
