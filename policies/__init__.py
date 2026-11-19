"""Базовые политики-адаптеры (интерфейс — policies/base.py, docs/contract.md раздел 9).

    random     — случайный исследователь с фиксированным seed          (зона B)
    reconcile  — простой алгоритм сверки, детектор утечек               (зона B)
    flag_all   — атака «пометить всё»                                   (зона B)
    oracle     — эталонный ответ из truth, только на стороне оценщика   (зона A)
    auditor    — полный скриптовый аудитор по контракту (демо, валидатор данных) (зона B)
"""

from __future__ import annotations

from typing import Any

BASELINES = ("random", "reconcile", "flag_all", "oracle")
ALL_POLICIES = BASELINES + ("auditor",)


def make_policy(name: str, *, seed: int = 0, truth: dict | None = None) -> Any:
    if name == "random":
        from .random_explorer import RandomExplorer

        return RandomExplorer(seed=seed)
    if name == "reconcile":
        from .reconcile import ReconcilePolicy

        return ReconcilePolicy()
    if name == "flag_all":
        from .flag_all import FlagAllPolicy

        return FlagAllPolicy()
    if name == "auditor":
        from .auditor import AuditorPolicy

        return AuditorPolicy()
    if name == "oracle":
        if truth is None:
            raise ValueError("oracle needs the reference (grader-side only)")
        from .oracle import OraclePolicy

        return OraclePolicy(truth)
    raise ValueError(f"unknown policy {name!r}; known: {', '.join(ALL_POLICIES)}")
