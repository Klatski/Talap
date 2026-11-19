"""Общие помощники тестов генератора (импортируются как модуль из этой папки)."""

from __future__ import annotations

import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.generate import generate_episode  # noqa: E402

ALL_TYPES = {"duplicate_invoice": 1, "unapplied_payment": 1, "fx_rate_error": 1, "cutoff_error": 1}
BOTH_SCENARIOS = {"fx_partial_boundary_payment": 1, "cutoff_fx_boundary_invoice": 1}

CONFIGS = {
    "clean": {"name": "t_clean", "size": "small"},
    "all_types": {"name": "t_all", "size": "small", "errors": ALL_TYPES},
    "scenarios": {"name": "t_scen", "size": "medium", "scenarios": BOTH_SCENARIOS, "errors": {"duplicate_invoice": 1}},
    "heavy": {
        "name": "t_heavy",
        "size": "large",
        "errors": {"duplicate_invoice": 2, "unapplied_payment": 2, "fx_rate_error": 2, "cutoff_error": 2},
        "scenarios": BOTH_SCENARIOS,
    },
    "base_only": {"name": "t_base", "size": "small", "foreign_currencies": 0, "errors": {"duplicate_invoice": 1, "unapplied_payment": 1, "cutoff_error": 1}},
}


@functools.lru_cache(maxsize=256)
def episode(config_key: str, seed: int):
    return generate_episode(seed, CONFIGS[config_key])
