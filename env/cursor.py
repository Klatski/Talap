"""Стабильные курсоры пагинации.

Курсор = "<table>.<offset>.<tag>", tag = первые 12 hex sha256("cursor|<episode_id>|<table>|<offset>").
* Детерминирован: тот же эпизод и позиция -> тот же курсор в любом процессе.
* Стабилен: порядок строк таблицы внутри эпизода фиксирован (порядок файла).
* Привязан к таблице и эпизоду: курсор ledger не подходит к invoices, курсор прошлого
  эпизода не подходит после reset.
* Ничего не раскрывает, кроме позиции, которую агент и так знает.
"""

from __future__ import annotations

import hashlib
import re

_CURSOR_RE = re.compile(r"^(?P<table>[a-z]+)\.(?P<offset>0|[1-9][0-9]{0,6})\.(?P<tag>[0-9a-f]{12})$")


class CursorError(ValueError):
    pass


def _tag(episode_id: str, table: str, offset: int) -> str:
    return hashlib.sha256(f"cursor|{episode_id}|{table}|{offset}".encode("utf-8")).hexdigest()[:12]


def encode_cursor(episode_id: str, table: str, offset: int) -> str:
    return f"{table}.{offset}.{_tag(episode_id, table, offset)}"


def decode_cursor(token: object, episode_id: str, table: str, total: int) -> int:
    """Курсор -> смещение. None -> 0. Любая несостыковка -> CursorError."""
    if token is None:
        return 0
    if not isinstance(token, str) or len(token) > 64:
        raise CursorError("type")
    m = _CURSOR_RE.match(token)
    if not m or m.group("table") != table:
        raise CursorError("format")
    offset = int(m.group("offset"))
    if offset > total or m.group("tag") != _tag(episode_id, table, offset):
        raise CursorError("tag")
    return offset
