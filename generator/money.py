"""Деньги и курсы: только int (минимальные единицы) и Decimal. Никаких float."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext

# Показатель разрядности (число знаков после запятой) по ISO 4217.
CURRENCY_EXPONENTS: dict[str, int] = {
    "AED": 2,
    "CNY": 2,
    "EUR": 2,
    "GBP": 2,
    "JPY": 0,
    "KWD": 3,
    "KZT": 2,
    "RUB": 2,
    "TRY": 2,
    "USD": 2,
}

# Условная стоимость одной основной единицы валюты в USD. Используется ТОЛЬКО как
# стартовая точка синтетического (вымышленного) ряда курсов.
REFERENCE_USD_VALUE: dict[str, Decimal] = {
    "AED": Decimal("0.2723"),
    "CNY": Decimal("0.1385"),
    "EUR": Decimal("1.0850"),
    "GBP": Decimal("1.2700"),
    "JPY": Decimal("0.00665"),
    "KWD": Decimal("3.2500"),
    "KZT": Decimal("0.00205"),
    "RUB": Decimal("0.0110"),
    "TRY": Decimal("0.0305"),
    "USD": Decimal("1"),
}

# half_up = половина округляется от нуля (Decimal.ROUND_HALF_UP), half_even = банковское.
ROUNDING_MODES = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN}


def convert_minor(
    amount_minor: int,
    source: str,
    target: str,
    rate: Decimal | str,
    mode: str,
    exponents: dict[str, int] | None = None,
) -> int:
    """Перевод суммы из минимальных единиц source в минимальные единицы target.

    target_minor = round(amount_minor * rate * 10^(exp(target) - exp(source)))
    rate — сколько единиц target за 1 единицу source. Округление — по mode.
    """
    if isinstance(amount_minor, bool) or not isinstance(amount_minor, int):
        raise TypeError("amount_minor must be int")
    if source == target:
        return amount_minor
    exps = exponents or CURRENCY_EXPONENTS
    with localcontext() as ctx:
        ctx.prec = 60
        value = Decimal(amount_minor) * Decimal(rate) * (Decimal(10) ** (exps[target] - exps[source]))
        return int(value.quantize(Decimal(1), rounding=ROUNDING_MODES[mode]))


def format_rate(value: Decimal) -> str:
    """Курс как десятичная строка без экспоненты: 4 знака для >=1, 6 для >=0.01, иначе 8."""
    if value >= 1:
        quantum = Decimal("0.0001")
    elif value >= Decimal("0.01"):
        quantum = Decimal("0.000001")
    else:
        quantum = Decimal("0.00000001")
    return format(value.quantize(quantum), "f")
