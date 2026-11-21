"""Валидация СТРУКТУРЫ submit (findings.json) — зеркало schemas/findings.schema.json.

По «Определениям проверок» жюри («Точные соглашения matcher», п. 6–8) структурная ошибка — это
неверный тип значения, отсутствующее обязательное поле и превышение лимита кейса. Тогда неверен весь
submit: эпизод завершён, награда -1, ошибки пишутся в журнал оценщика:
  * не объект; нет episode_id или findings; episode_id не строка или чужой (не из manifest эпизода);
  * findings не массив или больше 30 выводов;
  * вывод не объект; нет одного из полей type, affected_ids, claimed_amount_minor, currency, evidence_ids,
    explanation;
  * type, currency, explanation — не строки; affected_ids, evidence_ids — не массивы строк;
  * claimed_amount_minor — не JSON-целое (bool, дробное число и строка не принимаются, пример M09);
  * больше 10 evidence_ids (считается исходная длина списка, до удаления повторов — пример M05);
  * explanation длиннее 1000 символов (Unicode code points после разбора JSON).
Структурно правильный вывод с НЕВЕРНЫМ содержимым — неизвестный type, валюта другого регистра или с
пробелом, отрицательная сумма, пустые или лишние affected_ids, несуществующие ID доказательств — не
ошибка структуры, а несопоставимый вывод: FP в matcher (примеры M02, M04, M06, M07). Поля сверх
перечисленных не входят в формат и игнорируются: на сопоставление и награду они не влияют.
Сообщения об ошибках описывают только структуру ответа: без эха значений и без сведений об эталоне.
"""

from __future__ import annotations

from typing import Any

ISSUE_TYPES = ("duplicate_invoice", "unapplied_payment", "fx_rate_error", "cutoff_error")
TOP_FIELDS = ("episode_id", "findings")
FINDING_FIELDS = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")
MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_EXPLANATION = 1000
MAX_ERRORS = 20
MALFORMED_REASONS = {
    "wrapper": "must be an object with the argument findings",
    "not_json": "is not JSON-serializable (e.g. NaN or an integer too long to parse)",
    "too_large": "is larger than the 1 MB submission limit",
}


def malformed_message(reason: str) -> str:
    """Сообщение для submit, который среда не смогла передать как есть (фиксированный набор текстов)."""
    return f"submit action: {MALFORMED_REASONS.get(reason, MALFORMED_REASONS['wrapper'])}"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _string_list(value: Any, path: str, errors: list[str], max_items: int | None = None) -> None:
    if not isinstance(value, list):
        errors.append(f"{path}: must be an array of strings")
        return
    if max_items is not None and len(value) > max_items:
        errors.append(f"{path}: at most {max_items} items")
    if any(not isinstance(item, str) for item in value):
        errors.append(f"{path}: every item must be a string")


def _finding(f: Any, path: str, errors: list[str]) -> None:
    if not isinstance(f, dict):
        errors.append(f"{path}: must be an object")
        return
    for k in FINDING_FIELDS:
        if k not in f:
            errors.append(f"{path}: missing field {k}")
    if "type" in f and not isinstance(f["type"], str):
        errors.append(f"{path}.type: must be a string")
    if "affected_ids" in f:
        _string_list(f["affected_ids"], f"{path}.affected_ids", errors)
    if "claimed_amount_minor" in f and not _is_int(f["claimed_amount_minor"]):
        errors.append(f"{path}.claimed_amount_minor: must be a JSON integer (minor units)")
    if "currency" in f and not isinstance(f["currency"], str):
        errors.append(f"{path}.currency: must be a string")
    if "evidence_ids" in f:
        _string_list(f["evidence_ids"], f"{path}.evidence_ids", errors, MAX_EVIDENCE)
    if "explanation" in f:
        v = f["explanation"]
        if not isinstance(v, str):
            errors.append(f"{path}.explanation: must be a string")
        elif len(v) > MAX_EXPLANATION:
            errors.append(f"{path}.explanation: at most {MAX_EXPLANATION} characters")


def validate_submission(submission: Any, episode_id: str | None = None) -> list[str]:
    """Пустой список = структура верна. episode_id — ожидаемый ID эпизода (из manifest)."""
    errors: list[str] = []
    if not isinstance(submission, dict):
        return ["submission: must be an object {episode_id, findings}"]
    for k in TOP_FIELDS:
        if k not in submission:
            errors.append(f"submission: missing field {k}")
    if "episode_id" in submission:
        ep = submission["episode_id"]
        if not isinstance(ep, str):
            errors.append("episode_id: must be a string")
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
