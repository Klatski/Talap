"""Разбор и классификация действий агента — одна функция и для ядра среды, и для клиента.

* Python 3.11 отказывается превращать в int строки длиннее sys.get_int_max_str_digits() (4300) цифр
  и бросает ValueError прямо из json.loads. NaN/Infinity — не JSON. Такие значения превращаются в
  маркер UNREPRESENTABLE: он не сериализуется, и действие классифицируется как «не JSON».
* classify_action() решает судьбу действия до его выполнения. EnvCore применяет её к тому, что
  получил (в процессе или по проводу), AuditEnv-клиент — до отправки. Поэтому изолированная и
  внутрипроцессная среды дают одинаковый итог на любом вводе.
* Пороги размера: обычное действие — MAX_ACTION_BYTES; submit — MAX_SUBMIT_BYTES, что больше
  худшего ВАЛИДНОГО ответа (30 выводов, ID и объяснения из четырёхбайтовых символов, суммы по 4300
  цифр — около 0.96 МБ в экранированном JSON; см. tests/env/test_env_hostile_input.py), поэтому
  порог никогда не превращает валидный ответ в -1 и награда не зависит от текста explanation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .constants import MAX_ACTION_BYTES, MAX_PARSE_BYTES, MAX_SUBMIT_BYTES


class Unrepresentable:
    """Значение из JSON-текста, которое нельзя представить: целое длиннее 4300 цифр, NaN, Infinity."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unrepresentable number>"


UNREPRESENTABLE = Unrepresentable()

# Строку больше MAX_PARSE_BYTES среда не разбирает; submit узнаётся по началу: {"action": "submit", …
_SUBMIT_PREFIX = re.compile(r'\s*\{\s*"action"\s*:\s*"submit"\s*[,}]')

# Причины, по которым действие нельзя выполнить как есть.
INVALID, NOT_JSON, TOO_LARGE = "invalid", "not_json", "too_large"
REASONS = (INVALID, NOT_JSON, TOO_LARGE)


def _parse_int(text: str) -> Any:
    try:
        return int(text)
    except ValueError:
        return UNREPRESENTABLE


def _parse_constant(name: str) -> Any:
    return UNREPRESENTABLE


def loads_text(text: str) -> Any:
    """json.loads, который не бросает ничего, кроме ValueError/RecursionError на синтаксисе."""
    return json.loads(text, parse_int=_parse_int, parse_constant=_parse_constant)


def is_json_value(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def classify_action(action: Any) -> tuple[dict | None, tuple[bool, str] | None]:
    """(объект, None) — действие можно выполнять; (None, (это_submit, причина)) — нельзя.

    Строка: больше MAX_PARSE_BYTES (UTF-8) не разбирается; не JSON или не объект -> invalid.
    Объект: не сериализуется (NaN, сверхдлинное целое, смешанные типы ключей, set, bytes, глубина)
    -> not_json; больше порога (submit — MAX_SUBMIT_BYTES, прочие — MAX_ACTION_BYTES) -> too_large.
    """
    if isinstance(action, str):
        if len(action.encode("utf-8", "replace")) > MAX_PARSE_BYTES:
            return None, (bool(_SUBMIT_PREFIX.match(action)), TOO_LARGE)
        try:
            action = loads_text(action)
        except (ValueError, RecursionError):
            return None, (False, INVALID)
    if not isinstance(action, dict):
        return None, (False, INVALID)
    is_submit = action.get("action") == "submit"
    try:
        size = len(json.dumps(action, allow_nan=False, sort_keys=True).encode("utf-8"))
    except (TypeError, ValueError, RecursionError):
        return None, (is_submit, NOT_JSON)
    if size > (MAX_SUBMIT_BYTES if is_submit else MAX_ACTION_BYTES):
        return None, (is_submit, TOO_LARGE)
    return action, None
