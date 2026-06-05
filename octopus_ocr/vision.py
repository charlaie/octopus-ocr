from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from octopus_ocr.models import BBox, Category, Direction

REFERENCE_WIDTH = 1170
REFERENCE_HEIGHT = 2532


@dataclass(frozen=True)
class RowRegion:
    row_bbox: BBox
    icon_bbox: BBox
    category: Category
    payee_bbox: BBox
    datetime_bbox: BBox
    amount_bbox: BBox
    amount_direction: Direction | None = None


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_pil_crop(image_rgb: np.ndarray, bbox: BBox, *, scale: int = 2) -> Image.Image:
    h, w = image_rgb.shape[:2]
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(w, bbox.x + bbox.width)
    y2 = min(h, bbox.y + bbox.height)
    crop = image_rgb[y1:y2, x1:x2]
    pil = Image.fromarray(crop)
    if scale != 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.LANCZOS)
    return pil


def detect_rows(image_rgb: np.ndarray) -> list[RowRegion]:
    height, width = image_rgb.shape[:2]
    sx = width / REFERENCE_WIDTH
    sy = height / REFERENCE_HEIGHT

    icon_x_min = int(55 * sx)
    icon_x_max = int(175 * sx)
    content_top = int(245 * sy)
    content_bottom = height - int(95 * sy)

    icons = _detect_icon_blobs(image_rgb, icon_x_min, icon_x_max, content_top, content_bottom)
    rows: list[RowRegion] = []
    for index, (cx, cy, radius, category) in enumerate(icons):
        prev_cy = icons[index - 1][1] if index > 0 else None
        next_cy = icons[index + 1][1] if index < len(icons) - 1 else None
        row_top = (
            int((prev_cy + cy) / 2)
            if prev_cy is not None
            else max(content_top, int(cy - max(95 * sy, radius * 1.7)))
        )
        row_bottom = (
            int((cy + next_cy) / 2)
            if next_cy is not None
            else min(content_bottom, int(cy + max(135 * sy, radius * 2.1)))
        )
        if row_top < content_top - 20 or row_bottom > height - int(70 * sy) or row_bottom - row_top < int(170 * sy):
            continue

        row_bbox = BBox(x=int(40 * sx), y=row_top, width=int(1060 * sx), height=row_bottom - row_top)
        payee_bbox, datetime_bbox = _detect_text_field_bboxes(image_rgb, row_bbox, sx, sy, cy)
        if _is_top_clipped_first_row(prev_cy, row_top, content_top, payee_bbox, sy):
            continue
        amount_bbox = BBox(
            x=int(850 * sx),
            y=max(row_top, int(cy - 45 * sy)),
            width=int(215 * sx),
            height=min(int(80 * sy), row_bottom - max(row_top, int(cy - 45 * sy))),
        )
        rows.append(
            RowRegion(
                row_bbox=row_bbox,
                icon_bbox=BBox(
                    x=int(cx - radius),
                    y=int(cy - radius),
                    width=int(radius * 2),
                    height=int(radius * 2),
                ),
                category=category,
                payee_bbox=payee_bbox,
                datetime_bbox=datetime_bbox,
                amount_bbox=amount_bbox,
                amount_direction=detect_amount_direction(image_rgb, amount_bbox),
            )
        )
    return rows


def _is_top_clipped_first_row(
    prev_cy: int | None,
    row_top: int,
    content_top: int,
    payee_bbox: BBox,
    sy: float,
) -> bool:
    if prev_cy is not None:
        return False
    if row_top > content_top + int(4 * sy):
        return False
    return payee_bbox.y - row_top < int(30 * sy)


def detect_amount_direction(image_rgb: np.ndarray, bbox: BBox) -> Direction | None:
    h, w = image_rgb.shape[:2]
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(w, bbox.x + bbox.width)
    y2 = min(h, bbox.y + bbox.height)
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    red_mask_low = cv2.inRange(hsv, np.array([0, 80, 100]), np.array([10, 255, 255]))
    red_mask_high = cv2.inRange(hsv, np.array([170, 80, 100]), np.array([179, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask_low, red_mask_high)
    green_mask = cv2.inRange(hsv, np.array([35, 70, 80]), np.array([95, 255, 255]))
    red_count = int(np.count_nonzero(red_mask))
    green_count = int(np.count_nonzero(green_mask))
    if red_count < 20 and green_count < 20:
        return None
    return "outflow" if red_count >= green_count else "inflow"


def _detect_text_field_bboxes(
    image_rgb: np.ndarray,
    row_bbox: BBox,
    sx: float,
    sy: float,
    cy: int,
) -> tuple[BBox, BBox]:
    text_x = int(205 * sx)
    payee_width = int(650 * sx)
    date_width = int(430 * sx)
    y_padding = int(10 * sy)
    line_runs = _detect_text_line_runs(
        image_rgb,
        BBox(
            x=text_x,
            y=row_bbox.y,
            width=payee_width,
            height=row_bbox.height,
        ),
    )
    if len(line_runs) >= 2:
        date_top, date_bottom = line_runs[-1]
        payee_top = line_runs[0][0]
        payee_bottom = line_runs[-2][1]
        return (
            BBox(
                x=text_x,
                y=max(row_bbox.y, payee_top - y_padding),
                width=payee_width,
                height=min(row_bbox.y + row_bbox.height, payee_bottom + y_padding)
                - max(row_bbox.y, payee_top - y_padding),
            ),
            BBox(
                x=text_x,
                y=max(row_bbox.y, date_top - y_padding),
                width=date_width,
                height=min(row_bbox.y + row_bbox.height, date_bottom + y_padding)
                - max(row_bbox.y, date_top - y_padding),
            ),
        )

    return (
        BBox(
            x=text_x,
            y=max(0, int(cy - 58 * sy)),
            width=payee_width,
            height=int(72 * sy),
        ),
        BBox(
            x=text_x,
            y=int(cy + 14 * sy),
            width=date_width,
            height=int(52 * sy),
        ),
    )


def _detect_text_line_runs(image_rgb: np.ndarray, bbox: BBox) -> list[tuple[int, int]]:
    h, w = image_rgb.shape[:2]
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(w, bbox.x + bbox.width)
    y2 = min(h, bbox.y + bbox.height)
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mask = (gray < 190).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    projection = np.count_nonzero(mask, axis=1)
    active = projection > max(8, int((x2 - x1) * 0.015))

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for offset, is_active in enumerate(active):
        if is_active and start is None:
            start = offset
        elif not is_active and start is not None:
            runs.append((start, offset))
            start = None
    if start is not None:
        runs.append((start, len(active)))

    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if end - start < 8:
            continue
        absolute = (y1 + start, y1 + end)
        if merged and absolute[0] - merged[-1][1] < 12:
            merged[-1] = (merged[-1][0], absolute[1])
        else:
            merged.append(absolute)
    return merged


def _detect_icon_blobs(
    image_rgb: np.ndarray,
    icon_x_min: int,
    icon_x_max: int,
    content_top: int,
    content_bottom: int,
) -> list[tuple[int, int, float, Category]]:
    roi = image_rgb[content_top:content_bottom, icon_x_min:icon_x_max]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    masks = {
        "transport": cv2.inRange(hsv, np.array([92, 80, 120]), np.array([112, 255, 255])),
        "eat and drink": cv2.inRange(hsv, np.array([16, 90, 130]), np.array([32, 255, 255])),
        "living and others": cv2.inRange(hsv, np.array([55, 65, 120]), np.array([85, 255, 255])),
    }

    detections: list[tuple[int, int, float, Category]] = []
    kernel = np.ones((5, 5), np.uint8)
    for category, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            if area < 2500:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if not 36 <= radius <= 64:
                continue
            cx = int(icon_x_min + x)
            cy = int(content_top + y)
            detections.append((cx, cy, max(radius, 34.0), category))  # type: ignore[arg-type]

    detections.extend(
        _detect_top_up_icons(
            hsv,
            image_rgb.shape[1] / REFERENCE_WIDTH,
            image_rgb.shape[0] / REFERENCE_HEIGHT,
            icon_x_min,
            content_top,
        )
    )
    detections.extend(
        _detect_fare_subsidy_icons(
            hsv,
            image_rgb.shape[1] / REFERENCE_WIDTH,
            image_rgb.shape[0] / REFERENCE_HEIGHT,
            icon_x_min,
            content_top,
        )
    )

    detections.sort(key=lambda item: item[1])
    merged: list[tuple[int, int, float, Category]] = []
    for detection in detections:
        if merged and abs(merged[-1][1] - detection[1]) < 60:
            existing = merged[-1]
            if existing[3] == "top-up" and detection[3] != "top-up":
                merged[-1] = detection
            elif detection[3] == "top-up":
                continue
            else:
                merged[-1] = existing if existing[2] >= detection[2] else detection
        else:
            merged.append(detection)
    return merged


def _detect_top_up_icons(
    hsv: np.ndarray,
    sx: float,
    sy: float,
    icon_x_min: int,
    content_top: int,
) -> list[tuple[int, int, float, Category]]:
    top_up_mask = cv2.inRange(hsv, np.array([5, 45, 90]), np.array([28, 255, 255]))
    top_up_mask = cv2.morphologyEx(top_up_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(top_up_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[tuple[int, int, float, Category]] = []
    min_width = 16 * sx
    max_width = 88 * sx
    min_height = 10 * sy
    max_height = 68 * sy
    min_area = 70 * sx * sy
    max_area = 1800 * sx * sy
    radius = 48.0 * max(sx, sy)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if not (min_width <= width <= max_width and min_height <= height <= max_height):
            continue
        if not min_area <= area <= max_area:
            continue
        detections.append(
            (
                int(icon_x_min + x + width / 2),
                int(content_top + y + height / 2),
                radius,
                "top-up",
            )
        )
    return detections


def _detect_fare_subsidy_icons(
    hsv: np.ndarray,
    sx: float,
    sy: float,
    icon_x_min: int,
    content_top: int,
) -> list[tuple[int, int, float, Category]]:
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 80, 95]))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[tuple[int, int, float, Category]] = []
    min_width = 26 * sx
    max_width = 50 * sx
    min_height = 26 * sy
    max_height = 50 * sy
    min_area = 120 * sx * sy
    max_area = 700 * sx * sy
    radius = 48.0 * max(sx, sy)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if not (min_width <= width <= max_width and min_height <= height <= max_height):
            continue
        if not min_area <= area <= max_area:
            continue
        fill_ratio = area / (width * height)
        if not 0.12 <= fill_ratio <= 0.35:
            continue
        detections.append(
            (
                int(icon_x_min + x + width / 2),
                int(content_top + y + height / 2),
                radius,
                "fare subsidy",
            )
        )
    return detections


def annotate_rows(image_rgb: np.ndarray, rows: list[RowRegion]) -> Image.Image:
    bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    colors = {
        "transport": (255, 180, 0),
        "eat and drink": (0, 170, 255),
        "living and others": (0, 190, 0),
        "top-up": (0, 128, 255),
        "fare subsidy": (180, 0, 180),
        "unknown": (180, 180, 180),
    }
    for row in rows:
        color = colors.get(row.category, (180, 180, 180))
        for bbox in [row.row_bbox, row.payee_bbox, row.datetime_bbox, row.amount_bbox]:
            cv2.rectangle(bgr, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), color, 2)
        cv2.putText(
            bgr,
            row.category,
            (row.row_bbox.x, max(20, row.row_bbox.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
