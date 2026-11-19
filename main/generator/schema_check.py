"""Минимальный валидатор JSON Schema (подмножество draft 2020-12) без внешних зависимостей.

Поддерживает: type, enum, const, required, properties, patternProperties,
additionalProperties, items, minItems, maxItems, minLength, maxLength, pattern,
minimum, maximum, anyOf, $ref на "#/$defs/...". Этого хватает для схем в schemas/.
bool не считается integer/number.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def load_schema(name: str) -> dict:
    with open(SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as fh:
        return json.load(fh)


def validate(value: Any, schema: dict, root: dict | None = None, path: str = "$") -> list[str]:
    root = root or schema
    errors: list[str] = []
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/$defs/"):
            return [f"{path}: unsupported $ref {ref}"]
        return validate(value, root["$defs"][ref[len("#/$defs/") :]], root, path)
    if "anyOf" in schema:
        if not any(not validate(value, sub, root, path) for sub in schema["anyOf"]):
            errors.append(f"{path}: does not match any allowed shape")
        return errors
    if "const" in schema and (value != schema["const"] or type(value) is not type(schema["const"])):
        errors.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")
    expected = schema.get("type")
    if expected is not None:
        kinds = expected if isinstance(expected, list) else [expected]
        if not any(_TYPES[k](value) for k in kinds):
            return errors + [f"{path}: expected {expected}"]
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']}")
    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                errors += validate(item, schema["items"], root, f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing {key}")
        for key, item in value.items():
            sub_path = f"{path}.{key}"
            matched = False
            if key in props:
                matched = True
                errors += validate(item, props[key], root, sub_path)
            for pattern, sub in patterns.items():
                if re.search(pattern, key):
                    matched = True
                    errors += validate(item, sub, root, sub_path)
            if not matched:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    errors.append(f"{sub_path}: unexpected field")
                elif isinstance(extra, dict):
                    errors += validate(item, extra, root, sub_path)
    return errors


def validate_episode_texts(public: dict[str, str], private: dict[str, str] | None = None) -> list[str]:
    """Проверить все файлы эпизода по схемам из schemas/. Пустой список = всё валидно."""
    from .fileio import parse_fx, parse_jsonl, parse_ledger

    errors: list[str] = []
    errors += validate(json.loads(public["manifest.json"]), load_schema("manifest"), path="manifest")
    errors += validate(json.loads(public["policy.json"]), load_schema("policy"), path="policy")
    for name, schema_name, rows in (
        ("ledger", "ledger_row", parse_ledger(public["ledger.csv"])),
        ("invoices", "invoice", parse_jsonl(public["invoices.jsonl"])),
        ("payments", "payment", parse_jsonl(public["payments.jsonl"])),
        ("fx", "fx_row", parse_fx(public["fx.csv"])),
    ):
        schema = load_schema(schema_name)
        for i, row in enumerate(rows):
            errors += validate(row, schema, path=f"{name}[{i}]")
    if private:
        errors += validate(json.loads(private["truth.json"]), load_schema("truth"), path="truth")
        errors += validate(json.loads(private["provenance.json"]), load_schema("provenance"), path="provenance")
    return errors
