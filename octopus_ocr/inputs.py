from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable
from unicodedata import normalize

from octopus_ocr.video import (
    DEFAULT_VIDEO_SAMPLE_FPS,
    VIDEO_SUFFIXES,
    VideoExtractionResult,
    extract_video_keyframes,
    is_video_path,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUPPORTED_INPUT_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES
# Leave room for annotations, row indices, field names, and extensions on filesystems with 255-byte names.
OUTPUT_LABEL_MAX_BYTES = 200


@dataclass(frozen=True)
class PreparedVideo:
    source_path: Path
    extraction: VideoExtractionResult
    seconds: float


@dataclass(frozen=True)
class PreparedInputs:
    image_paths: list[Path]
    videos: list[PreparedVideo]


def prepare_image_paths(
    input_paths: Iterable[Path],
    frames_dir: Path,
    *,
    sample_fps: float = DEFAULT_VIDEO_SAMPLE_FPS,
    on_video_start: Callable[[Path], None] | None = None,
) -> PreparedInputs:
    paths = list(input_paths)
    video_labels = source_output_labels((path for path in paths if is_video_path(path)), normalize_stem=_safe_stem)
    image_paths: list[Path] = []
    videos: list[PreparedVideo] = []
    for path in paths:
        if not is_video_path(path):
            image_paths.append(path)
            continue
        if on_video_start is not None:
            on_video_start(path)
        started = perf_counter()
        extraction = extract_video_keyframes(path, frames_dir / video_labels[path], sample_fps=sample_fps)
        videos.append(PreparedVideo(source_path=path, extraction=extraction, seconds=perf_counter() - started))
        image_paths.extend(extraction.frames)
    return PreparedInputs(image_paths=image_paths, videos=videos)


def source_output_labels(
    paths: Iterable[Path],
    *,
    normalize_stem: Callable[[str], str] | None = None,
) -> dict[Path, str]:
    paths = list(paths)
    stems = {
        path.resolve(): _truncate_label(normalize_stem(path.stem) if normalize_stem else path.stem)
        for path in paths
    }
    counts = Counter(_filename_key(stem) for stem in stems.values())
    used_names = {stem for stem, count in counts.items() if count == 1}
    labels: dict[Path, str] = {}
    for source, stem in stems.items():
        name = stem
        if counts[_filename_key(stem)] > 1:
            digest = sha256(str(source).encode("utf-8")).hexdigest()[:12]
            digest_suffix = f"_{digest}"
            name = _truncate_label(stem, OUTPUT_LABEL_MAX_BYTES - len(digest_suffix)) + digest_suffix
            suffix = 2
            while _filename_key(name) in used_names:
                digest_suffix = f"_{digest}_{suffix}"
                name = _truncate_label(stem, OUTPUT_LABEL_MAX_BYTES - len(digest_suffix)) + digest_suffix
                suffix += 1
            used_names.add(_filename_key(name))
        labels[source] = name
    return {path: labels[path.resolve()] for path in paths}


def _filename_key(name: str) -> str:
    return normalize("NFC", name).casefold()


def _truncate_label(label: str, max_bytes: int = OUTPUT_LABEL_MAX_BYTES) -> str:
    # The byte boundary can cut a UTF-8 character; discard only that trailing fragment.
    return label.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _safe_stem(stem: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return cleaned.strip("_") or "video"


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
