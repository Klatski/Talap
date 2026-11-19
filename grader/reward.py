from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


DEFAULT_STEP_PENALTY = Decimal("0.002")


def compute_f1(tp: int, fp: int, fn: int) -> tuple[Decimal, Decimal, Decimal]:
    precision = Decimal(tp) / Decimal(tp + fp) if (tp + fp) > 0 else Decimal(0)
    recall = Decimal(tp) / Decimal(tp + fn) if (tp + fn) > 0 else Decimal(0)

    if tp == 0 and fp == 0 and fn == 0:
        f1 = Decimal(1)
    elif (2 * tp + fp + fn) == 0:
        f1 = Decimal(0)
    else:
        f1 = Decimal(2 * tp) / Decimal(2 * tp + fp + fn)

    return precision, recall, f1


def compute_terminal_reward(
    f1: Decimal,
    inspection_steps: int,
    step_penalty: Decimal = DEFAULT_STEP_PENALTY,
) -> Decimal:
    raw = Decimal(2) * f1 - Decimal(1) - step_penalty * Decimal(inspection_steps)
    clamped = max(Decimal("-1"), min(Decimal("1"), raw))
    return clamped.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)