"""Oracle-политика (зона A): отправляет верные выводы с доказательствами из truth.json.

Работает ТОЛЬКО на стороне оценщика (ей нужен закрытый эталон). Проверяет, что
максимум награды достижим и эталон корректен: на каждом эпизоде oracle обязан
получать F1 = 1. Агенту эта политика недоступна.

Интерфейс политики (docs/contract.md, раздел 9; policies/base.py):
    policy.reset(observation, info) -> None
    policy.act(observation) -> action (dict)
Действие submit: {"action": "submit", "findings": <findings.json>}.
В прогонах обвязка создаёт её из эталона на стороне оценщика: policies.make_policy("oracle", truth=...)
(env/runner.py); её траектории в общий trajectories.jsonl не пишутся.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPLANATIONS = {
    "duplicate_invoice": "Two invoices with the same customer, timestamp and order lines are both posted.",
    "unapplied_payment": "Payment references the invoice, but part of it stayed on 2300 instead of settling 1200.",
    "fx_rate_error": "Ledger amount differs from conversion at the policy rate date.",
    "cutoff_error": "Document date in company timezone is outside the period, but it is posted in this ledger.",
}


def oracle_findings(truth: dict[str, Any]) -> dict[str, Any]:
    """findings.json из эталона: первый (лексикографически) допустимый набор доказательств."""
    findings = []
    for issue in truth["issues"]:
        findings.append(
            {
                "type": issue["type"],
                "affected_ids": list(issue["affected_ids"]),
                "claimed_amount_minor": issue["expected_amount_minor"],
                "currency": issue["currency"],
                "evidence_ids": list(issue["evidence_sets"][0]),
                "explanation": EXPLANATIONS[issue["type"]],
            }
        )
    return {"episode_id": truth["episode_id"], "findings": findings}


def load_truth(episode_dir: str | Path) -> dict[str, Any]:
    with open(Path(episode_dir) / "private" / "truth.json", encoding="utf-8") as fh:
        return json.load(fh)


class OraclePolicy:
    """Сразу отправляет эталонный ответ: 1 шаг (submit), inspection_steps = 0."""

    name = "oracle"

    def __init__(self, truth: dict[str, Any]) -> None:
        self._findings = oracle_findings(truth)

    @classmethod
    def from_episode_dir(cls, episode_dir: str | Path) -> "OraclePolicy":
        return cls(load_truth(episode_dir))

    def reset(self, observation: Any = None, info: Any = None) -> None:
        return None

    def act(self, observation: Any = None) -> dict[str, Any]:
        return {"action": "submit", "findings": json.loads(json.dumps(self._findings))}
