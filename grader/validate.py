"""Строгая валидация submit (findings.json) — зеркало schemas/findings.schema.json.

Любое нарушение = неверная структура = награда -1 и запись в журнале оценщика:
не объект, лишние/отсутствующие поля, неверные типы (bool и float — не integer),
отрицательные суммы, неизвестный type, валюта не из трёх заглавных латинских букв,
> 30 выводов, affected_ids вне 1..10, > 10 evidence_ids, explanation > 1000 символов,
строки ID вне 1..64 символов, чужой episode_id.
Сообщения об ошибках описывают только структуру ответа, ничего об эталоне.
"""

from __future__ import annotations

import re
from typing import Any

ISSUE_TYPES = ("duplicate_invoice", "unapplied_payment", "fx_rate_error", "cutoff_error")
TOP_FIELDS = frozenset({"episode_id", "findings"})
FINDING_FIELDS = frozenset({"type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"})
MAX_FINDINGS = 30
MAX_AFFECTED = 10
MAX_EVIDENCE = 10
MAX_EXPLANATION = 1000
MAX_ID_LEN = 64
MAX_ERRORS = 20
_CURRENCY = re.compile(r"[A-Z]{3}")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _id_list(value: Any, path: str, lo: int, hi: int, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append(f"{path}: must be an array")
        return
    if len(value) < lo:
        errors.append(f"{path}: needs at least {lo} item(s)")
    if len(value) > hi:
        errors.append(f"{path}: at most {hi} items")
    for i, item in enumerate(value):
        if not isinstance(item, str) or not 1 <= len(item) <= MAX_ID_LEN:
            errors.append(f"{path}[{i}]: must be a string of 1..{MAX_ID_LEN} characters")


def _finding(f: Any, path: str, errors: list[str]) -> None:
    if not isinstance(f, dict):
        errors.append(f"{path}: must be an object")
        return
    keys = set(f)
    for k in sorted(FINDING_FIELDS - keys):
        errors.append(f"{path}: missing field {k}")
    if keys - FINDING_FIELDS:
        errors.append(f"{path}: unexpected field(s)")
    if "type" in f and (not isinstance(f["type"], str) or f["type"] not in ISSUE_TYPES):
        errors.append(f"{path}.type: must be one of {', '.join(ISSUE_TYPES)}")
    if "affected_ids" in f:
        _id_list(f["affected_ids"], f"{path}.affected_ids", 1, MAX_AFFECTED, errors)
    if "claimed_amount_minor" in f:
        v = f["claimed_amount_minor"]
        if not _is_int(v):
            errors.append(f"{path}.claimed_amount_minor: must be an integer (minor units)")
        elif v < 0:
            errors.append(f"{path}.claimed_amount_minor: must be >= 0")
    if "currency" in f and (not isinstance(f["currency"], str) or not _CURRENCY.fullmatch(f["currency"])):
        errors.append(f"{path}.currency: must be three uppercase letters")
    if "evidence_ids" in f:
        _id_list(f["evidence_ids"], f"{path}.evidence_ids", 0, MAX_EVIDENCE, errors)
    if "explanation" in f:
        v = f["explanation"]
        if not isinstance(v, str):
            errors.append(f"{path}.explanation: must be a string")
        elif len(v) > MAX_EXPLANATION:
            errors.append(f"{path}.explanation: at most {MAX_EXPLANATION} characters")


def validate_submission(submission: Any, episode_id: str | None = None) -> list[str]:
    """Пустой список = структура верна. episode_id — ожидаемый ID эпизода (из эталона)."""
    errors: list[str] = []
    if not isinstance(submission, dict):
        return ["submission: must be an object {episode_id, findings}"]
    keys = set(submission)
    for k in sorted(TOP_FIELDS - keys):
        errors.append(f"submission: missing field {k}")
    if keys - TOP_FIELDS:
        errors.append("submission: unexpected field(s)")
    if "episode_id" in submission:
        ep = submission["episode_id"]
        if not isinstance(ep, str) or not 1 <= len(ep) <= MAX_ID_LEN:
            errors.append(f"episode_id: must be a string of 1..{MAX_ID_LEN} characters")
        elif episode_id is not None and ep != episode_id:
            errors.append("episode_id: does not match the current episode")
    if "findings" in submission:
        findings = submission["findings"]
        if not isinstance(findings, list):
            errors.append("findings: must be an array")
        else:
            if len(findings) > MAX_FINDINGS:
                errors.append(f"findings: at most {MAX_FINDINGS} items")
            for i, f in enumerate(findings[: MAX_FINDINGS + 1]):
                _finding(f, f"findings[{i}]", errors)
                if len(errors) >= MAX_ERRORS:
                    break
    return errors[:MAX_ERRORS]
