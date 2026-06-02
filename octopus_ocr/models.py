from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


Category = Literal[
    "transport", "eat and drink", "living and others", "top-up", "unknown"
]
Direction = Literal["outflow", "inflow"]


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class OcrField(BaseModel):
    text: str = ""
    confidence: float | None = None
    bbox: BBox


class TransactionCandidate(BaseModel):
    source_image: str
    row_index: int
    row_bbox: BBox
    icon_bbox: BBox
    category: Category
    payee: OcrField
    datetime_field: OcrField
    amount_field: OcrField
    parsed_datetime: datetime | None = None
    parsed_payee: str | None = None
    parsed_amount: Decimal | None = None
    direction: Direction | None = None
    warnings: list[str] = Field(default_factory=list)


class TransactionRecord(BaseModel):
    id: str
    fitid: str
    source_images: list[str]
    row_refs: list[str]
    duplicate_group: str
    dedupe_status: Literal["kept", "duplicate", "review"]
    datetime: datetime
    payee: str
    amount: Decimal
    direction: Direction
    category: Category
    confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)
    raw_candidates: list[TransactionCandidate] = Field(default_factory=list)


class PipelineResult(BaseModel):
    generated_at: datetime
    input_images: list[str]
    candidates: list[TransactionCandidate]
    transactions: list[TransactionRecord]
