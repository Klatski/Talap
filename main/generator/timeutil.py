"""Время: внутри генератора все моменты — aware datetime в UTC, наружу — ISO-8601 со смещением."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc
_OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")


def offset_tz(minutes: int) -> timezone:
    return timezone(timedelta(minutes=minutes))


def format_offset(minutes: int) -> str:
    sign = "+" if minutes >= 0 else "-"
    m = abs(minutes)
    return f"{sign}{m // 60:02d}:{m % 60:02d}"


def parse_offset(text: str) -> int:
    match = _OFFSET_RE.match(text)
    if not match:
        raise ValueError(f"bad UTC offset: {text!r}")
    sign = 1 if match.group(1) == "+" else -1
    return sign * (int(match.group(2)) * 60 + int(match.group(3)))


def to_iso(instant: datetime, offset_minutes: int) -> str:
    """Момент времени в ISO-8601 с секундами и явным смещением (+00:00, не Z)."""
    if instant.tzinfo is None:
        raise ValueError("naive datetime")
    return instant.astimezone(offset_tz(offset_minutes)).isoformat(timespec="seconds")


def parse_iso(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError(f"timestamp without offset: {text!r}")
    return value


def local_date(instant: datetime, offset_minutes: int) -> date:
    return instant.astimezone(offset_tz(offset_minutes)).date()
