"""Разбор и классификация действий агента — одна функция и для ядра среды, и для клиента.

* Python 3.11 отказывается превращать в int строки длиннее sys.get_int_max_str_digits() (4300) цифр
  и бросает ValueError прямо из json.loads. NaN/Infinity — не JSON. Такие значения превращаются в
  маркер UNREPRESENTABLE: он не сериализуется, и действие классифицируется как «не JSON».
* classify_action() решает судьбу действия до его выполнения. EnvCore применяет её к тому, что
  получил (в процессе или по проводу), AuditEnv-клиент — до отправки. Поэтому изолированная и
  внутрипроцессная среды дают одинаковый итог на любом вводе.
* Пороги размера: обычное действие — MAX_ACTION_BYTES; submit — MAX_SUBMIT_BYTES, но размер submit
  считается БЕЗ текста валидных explanation (строка до MAX_EXPLANATION_CHARS code points заменяется
  пустой, submit_size_basis). Поэтому текст объяснения никогда не решает, дойдёт ли ответ до оценщика:
  награда от него не зависит (требование ТЗ). Остальное содержимое ответа (ID, поля сверх формата)
  структурой не ограничено (правила жюри: лишний или длинный ID — FP, а не -1), и больше 1 МБ без
  объяснений — транспортный предел среды, как 100 шагов и 60 с: такой submit — неверная структура.
  Реальный ответ (30 выводов по 10 существующих ID) — десятки КБ. Сообщение по проводу не больше
  1 МБ + 30 x 1000 x 12 байт объяснений (< MAX_PARSE_BYTES < предела строки протокола 4 МБ).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .constants import MAX_ACTION_BYTES, MAX_EXPLANATION_CHARS, MAX_PARSE_BYTES, MAX_SUBMIT_BYTES


class Unrepresentable:
    """Значение из JSON-текста, которое нельзя представить: целое длиннее 4300 цифр, NaN, Infinity."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unrepresentable number>"


UNREPRESENTABLE = Unrepresentable()

# Строку больше MAX_PARSE_BYTES среда не разбирает; submit узнаётся по первому ключу верхнего уровня
# ({"action": "submit", … или {"name": "submit", …) либо по последнему (так пишет json.dumps(sort_keys=True)
# формат жюри: {"args": {…}, "name": "submit"}). Ключ, вложенный глубже, конец строки не даёт: там два "}".
_SUBMIT_PREFIX = re.compile(r'\s*\{\s*"(?:action|name)"\s*:\s*"submit"\s*[,}]')
_SUBMIT_SUFFIX = re.compile(r'[{,]\s*"(?:action|name)"\s*:\s*"submit"\s*\}\s*$')


def looks_like_submit(text: str) -> bool:
    """Неразобранная строка-действие — submit (по первому или последнему ключу верхнего уровня)."""
    return bool(_SUBMIT_PREFIX.match(text) or _SUBMIT_SUFFIX.search(text[-512:]))

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


def action_parts(action: Any) -> tuple[Any, Any]:
    """(имя, аргументы) действия в любом из двух форматов — для показа и разбора траекторий, без проверки."""
    if not isinstance(action, dict):
        return None, None
    if "action" not in action and "name" in action:
        return action.get("name"), action.get("args") or {}
    return action.get("action"), {k: v for k, v in action.items() if k != "action"}


def is_submit_action(action: dict) -> bool:
    """submit в любом из двух форматов: {"action": "submit", ...} или {"name": "submit", "args": ...}."""
    if "action" in action:
        return action.get("action") == "submit"
    return action.get("name") == "submit"


def submit_size_basis(action: dict) -> dict:
    """То, по чему меряется размер submit: копия действия, где explanation каждого вывода, если это строка не
    длиннее MAX_EXPLANATION_CHARS, заменён пустой строкой. Более длинный explanation — неверная структура
    всё равно (-1), поэтому он учитывается целиком. Исходное действие не меняется."""
    if "action" in action:
        holder = action
    else:
        holder = action.get("args")
        if not isinstance(holder, dict):
            return action
    payload = holder.get("findings")
    if not isinstance(payload, dict) or not isinstance(payload.get("findings"), list):
        return action
    findings = [{**f, "explanation": ""} if isinstance(f, dict) and isinstance(f.get("explanation"), str)
                and len(f["explanation"]) <= MAX_EXPLANATION_CHARS else f for f in payload["findings"]]
    payload = {**payload, "findings": findings}
    if holder is action:
        return {**action, "findings": payload}
    return {**action, "args": {**holder, "findings": payload}}


def classify_action(action: Any) -> tuple[dict | None, tuple[bool, str] | None]:
    """(объект, None) — действие можно выполнять; (None, (это_submit, причина)) — нельзя.

    Строка: больше MAX_PARSE_BYTES (UTF-8) не разбирается; не JSON или не объект -> invalid.
    Объект: не сериализуется (NaN, сверхдлинное целое, смешанные типы ключей, set, bytes, глубина)
    -> not_json; больше порога (submit — MAX_SUBMIT_BYTES без текста валидных explanation, прочие —
    MAX_ACTION_BYTES) -> too_large.
    """
    if isinstance(action, str):
        if len(action.encode("utf-8", "replace")) > MAX_PARSE_BYTES:
            return None, (looks_like_submit(action), TOO_LARGE)
        try:
            action = loads_text(action)
        except (ValueError, RecursionError):
            return None, (False, INVALID)
    if not isinstance(action, dict):
        return None, (False, INVALID)
    is_submit = is_submit_action(action)
    try:
        json.dumps(action, allow_nan=False, sort_keys=True)  # всё действие должно быть JSON
        basis = submit_size_basis(action) if is_submit else action
        size = len(json.dumps(basis, allow_nan=False, sort_keys=True).encode("utf-8"))
    except (TypeError, ValueError, RecursionError):
        return None, (is_submit, NOT_JSON)
    if size > (MAX_SUBMIT_BYTES if is_submit else MAX_ACTION_BYTES):
        return None, (is_submit, TOO_LARGE)
    return action, None
