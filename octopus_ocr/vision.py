from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from octopus_ocr.models import BBox, Category

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
    for cx, cy, radius, category in icons:
        row_top = int(cy - max(78 * sy, radius * 1.45))
        row_bottom = int(cy + max(122 * sy, radius * 1.75))
        if row_top < content_top - 20 or row_bottom > height - int(70 * sy):
            continue

        payee_bbox = BBox(
            x=int(205 * sx),
            y=max(0, int(cy - 58 * sy)),
            width=int(600 * sx),
            height=int(72 * sy),
        )
        datetime_bbox = BBox(
            x=int(205 * sx),
            y=int(cy + 14 * sy),
            width=int(430 * sx),
            height=int(52 * sy),
        )
        amount_bbox = BBox(
            x=int(850 * sx),
            y=int(cy - 45 * sy),
            width=int(215 * sx),
            height=int(80 * sy),
        )
        rows.append(
            RowRegion(
                row_bbox=BBox(x=int(40 * sx), y=row_top, width=int(1060 * sx), height=row_bottom - row_top),
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
            )
        )
    return rows


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
    # Top-up/AAVS icons are not circular single-color badges; detect the strong orange logo separately.
    top_up_mask = cv2.inRange(hsv, np.array([7, 70, 120]), np.array([20, 255, 255]))
    masks["top-up"] = top_up_mask

    detections: list[tuple[int, int, float, Category]] = []
    kernel = np.ones((5, 5), np.uint8)
    for category, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            if category != "top-up" and area < 2500:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if category != "top-up" and not 36 <= radius <= 64:
                continue
            if category == "top-up" and area < 180:
                continue
            cx = int(icon_x_min + x)
            cy = int(content_top + y)
            detections.append((cx, cy, max(radius, 34.0), category))  # type: ignore[arg-type]

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


def annotate_rows(image_rgb: np.ndarray, rows: list[RowRegion]) -> Image.Image:
    bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    colors = {
        "transport": (255, 180, 0),
        "eat and drink": (0, 170, 255),
        "living and others": (0, 190, 0),
        "top-up": (0, 128, 255),
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
