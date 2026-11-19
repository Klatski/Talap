"""Константы контракта среды (зона B): лимиты, таблицы, действия, каталог ошибок.

Все тексты ошибок фиксированы: в них нет пользовательского ввода, путей, ID эталона
или подсказок о правильности. Это часть защиты от утечек.
"""

from __future__ import annotations

ENV_VERSION = "1.0.0"
CONTRACT_VERSION = "1.0"

MAX_STEPS = 100            # включая submit
PAGE_LIMIT = 50            # максимум строк на страницу list_records
TIME_LIMIT_S = 60.0        # на эпизод, от reset
MAX_ACTION_BYTES = 262_144  # сериализованное действие; валидный submit занимает < 100 КБ
MAX_ID_CHARS = 128

MODES = ("train", "eval")

# table -> (файл, поле-ключ, поле документа или None)
TABLES: dict[str, dict] = {
    "ledger": {"file": "ledger.csv", "id_field": "row_id", "doc_field": None},
    "invoices": {"file": "invoices.jsonl", "id_field": "invoice_id", "doc_field": "doc_id"},
    "payments": {"file": "payments.jsonl", "id_field": "payment_id", "doc_field": "doc_id"},
    "fx": {"file": "fx.csv", "id_field": "fx_id", "doc_field": None},
}
TABLE_ORDER = ("ledger", "invoices", "payments", "fx")

LEDGER_COLUMNS = (
    "row_id", "timestamp", "account", "debit_minor", "credit_minor",
    "currency", "entity_id", "invoice_id", "payment_id",
)
FX_COLUMNS = ("date", "source_currency", "target_currency", "rate")
PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")

# action -> (обязательные аргументы, необязательные аргументы)
ACTIONS: dict[str, tuple[frozenset, frozenset]] = {
    "list_records": (frozenset({"table"}), frozenset({"cursor", "limit"})),
    "get_record": (frozenset({"table", "id"}), frozenset()),
    "get_document": (frozenset({"doc_id"}), frozenset()),
    "calculator": (frozenset({"expression"}), frozenset()),
    "submit": (frozenset({"findings"}), frozenset()),
}

ERRORS: dict[str, str] = {
    "invalid_action": "Action must be a JSON object with a string field 'action'.",
    "action_too_large": "Action is too large.",
    "unknown_action": "Unknown action. Allowed: list_records, get_record, get_document, calculator, submit.",
    "missing_argument": "A required argument is missing for this action.",
    "unexpected_argument": "This action got an argument it does not accept.",
    "invalid_argument": "An argument has a wrong type or value.",
    "unknown_table": "Unknown table. Allowed: ledger, invoices, payments, fx.",
    "invalid_limit": "limit must be an integer >= 1 (values above 50 are reduced to 50).",
    "invalid_cursor": "Cursor is not valid for this table in this episode.",
    "not_found": "No record with this id in the requested table.",
    "document_not_found": "No document with this doc_id.",
    "calculator_error": "Expression rejected by the calculator.",
    "time_limit": "Time limit exceeded: the episode is truncated with an empty final answer.",
    "episode_over": "The episode is over. Call reset() to start a new one.",
    "not_reset": "No active episode. Call reset() first.",
}

# Детали калькулятора — тоже фиксированный набор.
CALC_DETAILS = {
    "not_a_string": "expression must be a string",
    "empty": "empty expression",
    "too_long": "expression is longer than 256 characters",
    "bad_char": "only digits, '.', '+', '-', '*', '/', '(', ')', ',', spaces and the functions abs, round_half_up, round_half_even are allowed",
    "too_many_tokens": "expression has more than 128 tokens",
    "number_too_long": "a number has more than 40 digits",
    "bad_number": "malformed number",
    "syntax": "syntax error",
    "unknown_function": "unknown function",
    "bad_arity": "wrong number of function arguments",
    "bad_ndigits": "ndigits must be an integer between -12 and 12",
    "too_deep": "expression is nested more than 32 levels",
    "division_by_zero": "division by zero",
    "too_large": "intermediate or final value is too large (>= 1e30)",
    "invalid_operation": "invalid arithmetic operation",
}

END_REASONS = ("submit", "step_limit", "time_limit", "abandoned")
