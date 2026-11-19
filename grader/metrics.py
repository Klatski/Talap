"""Метрики и награда — точно по ТЗ (раздел 05). Арифметика в Fraction, наружу — float.

P = TP/(TP+FP), R = TP/(TP+FN); при нулевом знаменателе P или R = 0.
F1 = 2TP/(2TP+FP+FN); пустой эталон и пустой ответ -> F1 = 1; пустой эталон и есть выводы -> F1 = 0
(это частный случай той же формулы: 2*0/(0+FP+0) = 0).
Общая награда: clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1); неверная структура -> -1.
Экспериментальная награда (показывается ОТДЕЛЬНО, общую не заменяет): как общая, но усечённый
эпизод (лимит шагов/времени, брошенный эпизод) получает -1 — закрывает лазейку «тянуть до
тайм-аута на чистой книге» (пустой итоговый ответ там даёт F1 = 1).
"""

from __future__ import annotations

from fractions import Fraction

STEP_COST = Fraction(2, 1000)


def prf(tp: int, fp: int, fn: int) -> tuple[Fraction, Fraction, Fraction]:
    precision = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
    recall = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
    denom = 2 * tp + fp + fn
    f1 = Fraction(2 * tp, denom) if denom else Fraction(1)
    return precision, recall, f1


def clip(x: Fraction, lo: int = -1, hi: int = 1) -> Fraction:
    return max(Fraction(lo), min(Fraction(hi), x))


def shared_reward(f1: Fraction, inspection_steps: int, valid: bool = True) -> float:
    if not valid:
        return -1.0
    return float(clip(2 * f1 - 1 - STEP_COST * inspection_steps))


def experimental_reward(f1: Fraction, inspection_steps: int, valid: bool, truncated: bool) -> float:
    if not valid or truncated:
        return -1.0
    return shared_reward(f1, inspection_steps, True)
