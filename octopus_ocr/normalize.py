from __future__ import annotations

import hashlib
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation

from dateutil import parser

from octopus_ocr.models import Category, Direction

HONG_KONG_TZ = "Asia/Hong_Kong"


def normalize_payee(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -\t")


def parse_amount(text: str) -> tuple[Decimal | None, Direction | None]:
    cleaned = text.strip()
    cleaned = cleaned.replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = cleaned.replace(",", "").replace("HK$", "").replace("$", "")
    match = re.search(r"([+-]?\s*\d+(?:\.\d+)?)", cleaned)
    if not match:
        return None, None

    value_text = match.group(1).replace(" ", "")
    sign = ""
    if value_text[:1] in {"+", "-"}:
        sign = value_text[0]
        value_text = value_text[1:]
    if "." not in value_text and len(value_text) >= 2:
        value_text = f"{value_text[:-1]}.{value_text[-1]}"
    value_text = f"{sign}{value_text}"
    try:
        amount = Decimal(value_text).quantize(Decimal("0.1"))
    except InvalidOperation:
        return None, None

    direction: Direction = "inflow" if amount >= 0 else "outflow"
    return amount, direction


def parse_datetime(text: str) -> datetime | None:
    cleaned = text.strip()
    cleaned = cleaned.replace(".", "-").replace("/", "-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    match = re.search(r"(\d{4}-\d{1,2}-\d{1,2})\s*(\d{1,2}:\d{2})", cleaned)
    if not match:
        return None
    try:
        return parser.parse(f"{match.group(1)} {match.group(2)}")
    except (ValueError, OverflowError):
        return None


def canonical_key(payee: str, when: datetime, amount: Decimal, category: Category, direction: Direction) -> str:
    normalized_payee = normalize_payee(payee).casefold()
    minute = when.replace(second=0, microsecond=0).isoformat(timespec="minutes")
    return "|".join([minute, normalized_payee, str(amount), category, direction])


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"
