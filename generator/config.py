"""Конфигурация генератора: значения по умолчанию, строгая валидация, дайджест, сэмплирование.

Конфиг — это JSON-объект. Любое поле можно опустить (будет значение по умолчанию).
Числа ошибок задаются целым числом или диапазоном [lo, hi] (сэмплируется по seed).
Необязательный ключ variants заказывает вариант ошибки типа (VARIANTS): строка — все ошибки
типа этого варианта, список — i-я ошибка типа берёт variants[i % len] (при числе ошибок >= длины
списка каждый вариант встретится). Без ключа вариант выбирается случайно. Ключа нет в
DEFAULT_CONFIG, поэтому конфиги без него сохраняют прежний дайджест (и прежние эпизоды).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .money import CURRENCY_EXPONENTS
from .timeutil import parse_offset

CONFIG_VERSION = "1.0"

ERROR_TYPES = ("duplicate_invoice", "unapplied_payment", "fx_rate_error", "cutoff_error")

# Сценарий взаимодействующих ошибок -> какие записи эталона он порождает (по одной на тип).
SCENARIO_TYPES: dict[str, tuple[str, ...]] = {
    # Частичный платёж в иностранной валюте на границе периода: курс взят по дате,
    # записанной в документе (не в часовом поясе компании) + платёж разнесён не полностью.
    "fx_partial_boundary_payment": ("fx_rate_error", "unapplied_payment"),
    # Счёт в иностранной валюте из соседнего периода, который по записи в документе
    # выглядит как текущий: проведён не в том периоде и по курсу не той даты.
    "cutoff_fx_boundary_invoice": ("cutoff_error", "fx_rate_error"),
}

# Варианты ошибки каждого типа (provenance.injections[].variant), которые можно заказать в config.variants.
VARIANTS: dict[str, tuple[str, ...]] = {
    "duplicate_invoice": ("reentered_document",),
    # полный пропуск разнесения на счёт / частичное разнесение (остаток на 2300)
    "unapplied_payment": ("missing_allocation", "partial_allocation"),
    "fx_rate_error": ("adjacent_date", "other_rule", "due_date", "document_offset_date", "utc_date"),
    # проводка документа соседнего периода датирована внутри периода: у самой границы / в любой день у неё
    "cutoff_error": ("shifted_into_period", "entered_in_period"),
}

DISTRACTOR_TYPES = (
    "recurring_invoices",
    "partial_shipments",
    "split_payments",
    "partial_payments",
    "batch_payments",
    "overpayment_refunds",
    "on_account_payments",
    "boundary_documents",
    "posting_lag",
)

SUPPORTED_BASE = ("KZT", "USD", "EUR", "GBP", "JPY")
SUPPORTED_FOREIGN = ("USD", "EUR", "GBP", "CNY", "JPY", "KWD", "RUB", "TRY", "AED", "KZT")
COMPANY_OFFSETS = (-300, 0, 60, 180, 240, 300, 330, 360, 540)
ROUNDING_CHOICES = ("half_up", "half_even")
RATE_RULES = ("document_date", "previous_day")

SIZE_PROFILES: dict[str, dict[str, tuple[int, int]]] = {
    "small": {"invoices_current": (16, 24), "customers": (5, 9), "ledger_rows": (200, 320)},
    "medium": {"invoices_current": (28, 40), "customers": (8, 14), "ledger_rows": (360, 700)},
    "large": {"invoices_current": (42, 52), "customers": (12, 18), "ledger_rows": (800, 1800)},
}

MAX_ISSUES = 12

OPTIONAL_CONFIG_KEYS = ("variants",)  # не входят в DEFAULT_CONFIG: без них дайджест прежний

DEFAULT_CONFIG: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "name": "default",
    "size": "medium",
    "base_currency": None,
    "foreign_currencies": None,
    "timezone": None,
    "period": None,
    "rounding_mode": None,
    "fx_rate_date_rule": None,
    "errors": {t: 0 for t in ERROR_TYPES},
    "scenarios": {s: 0 for s in SCENARIO_TYPES},
    "distractors": {d: True for d in DISTRACTOR_TYPES},
}

_PERIOD_RE = re.compile(r"^(\d{4})-(\d{2})$")


class ConfigError(ValueError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _count_bounds(name: str, value: Any) -> tuple[int, int]:
    if _is_int(value):
        lo = hi = value
    elif (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_int(v) for v in value)
    ):
        lo, hi = value
    else:
        raise ConfigError(f"{name}: expected int or [lo, hi], got {value!r}")
    if lo < 0 or hi < lo:
        raise ConfigError(f"{name}: invalid range {value!r}")
    return lo, hi


def _normalize_variants(value: Any) -> dict[str, list[str]]:
    """{тип: вариант | [варианты]} -> {тип: [варианты]}; null и пустые значения снимаются."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("variants must be an object {error type: variant or list of variants}")
    bad = set(value) - set(VARIANTS)
    if bad:
        raise ConfigError(f"unknown variants types: {sorted(bad)}")
    out: dict[str, list[str]] = {}
    for t in ERROR_TYPES:
        v = value.get(t)
        if v is None:
            continue
        names = [v] if isinstance(v, str) else v
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise ConfigError(f"variants.{t} must be a variant name or a non-empty list of names")
        unknown = [n for n in names if n not in VARIANTS[t]]
        if unknown:
            raise ConfigError(f"variants.{t}: unknown variant {unknown[0]!r}; one of {list(VARIANTS[t])}")
        if len(set(names)) != len(names):
            raise ConfigError(f"variants.{t}: repeated variant")
        out[t] = list(names)
    return out


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Слить с умолчаниями и строго проверить. Неизвестные ключи — ошибка."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a JSON object")
    unknown = set(raw) - set(DEFAULT_CONFIG) - set(OPTIONAL_CONFIG_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in raw.items():
        if key == "variants":
            variants = _normalize_variants(value)
            if variants:
                cfg["variants"] = variants
        elif key in ("errors", "scenarios", "distractors"):
            if not isinstance(value, dict):
                raise ConfigError(f"{key} must be an object")
            bad = set(value) - set(cfg[key])
            if bad:
                raise ConfigError(f"unknown {key}: {sorted(bad)}")
            cfg[key].update(copy.deepcopy(value))
        else:
            cfg[key] = copy.deepcopy(value)

    if cfg["config_version"] != CONFIG_VERSION:
        raise ConfigError(f"config_version must be {CONFIG_VERSION!r}")
    if not isinstance(cfg["name"], str) or not 1 <= len(cfg["name"]) <= 64:
        raise ConfigError("name must be a string of 1..64 chars")
    if cfg["size"] not in SIZE_PROFILES:
        raise ConfigError(f"size must be one of {sorted(SIZE_PROFILES)}")
    base = cfg["base_currency"]
    if base is not None and base not in SUPPORTED_BASE:
        raise ConfigError(f"base_currency must be one of {SUPPORTED_BASE} or null")
    foreign = cfg["foreign_currencies"]
    if foreign is not None:
        if _is_int(foreign):
            if not 0 <= foreign <= 3:
                raise ConfigError("foreign_currencies count must be 0..3")
        elif isinstance(foreign, list):
            if len(foreign) > 3 or len(set(foreign)) != len(foreign):
                raise ConfigError("foreign_currencies: up to 3 unique codes")
            for code in foreign:
                if code not in SUPPORTED_FOREIGN:
                    raise ConfigError(f"unsupported foreign currency {code!r}")
                if base is not None and code == base:
                    raise ConfigError("foreign_currencies must not contain base_currency")
        else:
            raise ConfigError("foreign_currencies must be null, int or list")
    tz = cfg["timezone"]
    if tz is not None:
        try:
            minutes = parse_offset(tz)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"timezone must look like +05:00: {exc}") from None
        if minutes % 15 or not -660 <= minutes <= 780:
            raise ConfigError("timezone must be a multiple of 15 min within -11:00..+13:00")
    period = cfg["period"]
    if period is not None:
        match = _PERIOD_RE.match(period) if isinstance(period, str) else None
        if not match or not 1 <= int(match.group(2)) <= 12 or not 2020 <= int(match.group(1)) <= 2035:
            raise ConfigError("period must be 'YYYY-MM' (2020..2035)")
    if cfg["rounding_mode"] is not None and cfg["rounding_mode"] not in ROUNDING_CHOICES:
        raise ConfigError(f"rounding_mode must be one of {ROUNDING_CHOICES} or null")
    if cfg["fx_rate_date_rule"] is not None and cfg["fx_rate_date_rule"] not in RATE_RULES:
        raise ConfigError(f"fx_rate_date_rule must be one of {RATE_RULES} or null")

    total_hi = 0
    for t in ERROR_TYPES:
        total_hi += _count_bounds(f"errors.{t}", cfg["errors"][t])[1]
    for s, produced in SCENARIO_TYPES.items():
        total_hi += _count_bounds(f"scenarios.{s}", cfg["scenarios"][s])[1] * len(produced)
    if total_hi > MAX_ISSUES:
        raise ConfigError(f"at most {MAX_ISSUES} issues per episode (scenario = 2 issues)")
    for d in DISTRACTOR_TYPES:
        if not isinstance(cfg["distractors"][d], bool):
            raise ConfigError(f"distractors.{d} must be bool")
    if isinstance(foreign, list) and not foreign:
        needs = _count_bounds("x", cfg["errors"]["fx_rate_error"])[1] + sum(
            _count_bounds("x", cfg["scenarios"][s])[1] for s in SCENARIO_TYPES
        )
        if needs:
            raise ConfigError("fx errors / scenarios need at least one foreign currency")
    return cfg


CONFIG_KEYS = tuple(DEFAULT_CONFIG) + OPTIONAL_CONFIG_KEYS  # все ключи верхнего уровня конфига


def config_digest(cfg: dict[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return normalize_config(json.load(fh))


@dataclass
class Params:
    """Конкретные параметры эпизода после сэмплирования конфига."""

    base: str
    foreign: list[str]
    company_offset: int
    year: int
    month: int
    rounding: str
    rate_rule: str
    n_invoices_cur: int
    n_customers: int
    target_rows: int
    errors: dict[str, int]
    scenarios: dict[str, int]
    distractors: dict[str, bool] = field(default_factory=dict)
    variants: dict[str, list[str]] = field(default_factory=dict)  # config.variants (пусто — случайный вариант)

    def variant_for(self, error_type: str, index: int) -> str | None:
        names = self.variants.get(error_type)
        return names[index % len(names)] if names else None

    def to_json(self) -> dict[str, Any]:
        extra = {"variants": {t: list(v) for t, v in self.variants.items()}} if self.variants else {}
        return {
            "base_currency": self.base,
            "foreign_currencies": list(self.foreign),
            "company_offset_minutes": self.company_offset,
            "period": f"{self.year:04d}-{self.month:02d}",
            "rounding_mode": self.rounding,
            "fx_rate_date_rule": self.rate_rule,
            "invoices_current": self.n_invoices_cur,
            "customers": self.n_customers,
            "target_ledger_rows": self.target_rows,
            "errors": dict(self.errors),
            "scenarios": dict(self.scenarios),
            "distractors": dict(self.distractors),
            **extra,
        }


def _sample(rng: random.Random, value: Any) -> int:
    lo, hi = _count_bounds("count", value)
    return lo if lo == hi else rng.randint(lo, hi)


def resolve_params(cfg: dict[str, Any], rng: random.Random) -> Params:
    profile = SIZE_PROFILES[cfg["size"]]
    errors = {t: _sample(rng, cfg["errors"][t]) for t in ERROR_TYPES}
    scenarios = {s: _sample(rng, cfg["scenarios"][s]) for s in SCENARIO_TYPES}

    base = cfg["base_currency"] or rng.choice(["KZT", "KZT", "USD", "EUR", "GBP", "JPY"])
    needs_foreign = errors["fx_rate_error"] > 0 or any(scenarios.values())
    pool = [c for c in SUPPORTED_FOREIGN if c != base and c in CURRENCY_EXPONENTS]
    spec = cfg["foreign_currencies"]
    if isinstance(spec, list):
        foreign = sorted(c for c in spec if c != base)
    else:
        k = spec if _is_int(spec) else rng.choice([0, 1, 1, 2, 2, 3])
        if needs_foreign and k == 0:
            k = 1
        foreign = sorted(rng.sample(pool, k))
    if needs_foreign and not foreign:
        raise ConfigError("fx errors / scenarios need at least one foreign currency")

    offset = parse_offset(cfg["timezone"]) if cfg["timezone"] else rng.choice(COMPANY_OFFSETS)
    if cfg["period"]:
        year, month = (int(x) for x in cfg["period"].split("-"))
    else:
        year, month = rng.choice([2025, 2026]), rng.randint(1, 12)
    rounding = cfg["rounding_mode"] or rng.choice(ROUNDING_CHOICES)
    rate_rule = cfg["fx_rate_date_rule"] or rng.choice(RATE_RULES)

    n_cur = rng.randint(*profile["invoices_current"])
    n_cur = max(n_cur, errors["duplicate_invoice"] + errors["fx_rate_error"] + errors["unapplied_payment"] + 8)
    n_customers = rng.randint(*profile["customers"])
    target_rows = rng.randint(*profile["ledger_rows"])
    return Params(
        base=base,
        foreign=foreign,
        company_offset=offset,
        year=year,
        month=month,
        rounding=rounding,
        rate_rule=rate_rule,
        n_invoices_cur=n_cur,
        n_customers=n_customers,
        target_rows=target_rows,
        errors=errors,
        scenarios=scenarios,
        distractors=dict(cfg["distractors"]),
        variants={t: list(v) for t, v in (cfg.get("variants") or {}).items()},
    )
