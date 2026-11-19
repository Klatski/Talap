"""Разбор JSON-текста от агента без исключений «вне каталога».

Python 3.11 отказывается превращать в int строки длиннее sys.get_int_max_str_digits() (4300) цифр и
бросает ValueError прямо из json.loads. Для агента это корректный JSON, поэтому такое число
превращается в маркер UNREPRESENTABLE. Маркер не сериализуется в JSON, и дальше действие обрабатывается
по общему правилу: обычное действие -> invalid_action (шаг расходуется), submit -> неверная структура (-1).
NaN/Infinity отвергаются (ValueError), как и в протоколе.
"""

from __future__ import annotations

import json
from typing import Any


class Unrepresentable:
    """Целое, слишком длинное для int(); json.dumps на нём падает с TypeError."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unrepresentable integer>"


UNREPRESENTABLE = Unrepresentable()


def _parse_int(text: str) -> Any:
    try:
        return int(text)
    except ValueError:
        return UNREPRESENTABLE


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")


def loads_text(text: str) -> Any:
    """json.loads с маркером для сверхдлинных целых и запретом NaN/Infinity."""
    return json.loads(text, parse_int=_parse_int, parse_constant=_reject_constant)
