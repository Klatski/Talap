"""score_submission — единая точка оценки (её же вызывают тесты A через grader.score_submission).

    score_submission(truth, submission, inspection_steps) -> {
        "valid", "errors", "tp", "fp", "fn", "precision", "recall", "f1",
        "inspection_steps", "reward", "reward_experimental", "diagnostics"}

`diagnostics` (разбор по выводам, пропуски по типам, FP на отвлекающих случаях) — только для
журнала оценщика, политике не передаётся никогда.
Для неверной структуры: valid=False, tp/fp/fn=None, P=R=F1=0, reward=-1.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .matcher import match
from .metrics import experimental_reward, prf, shared_reward
from .validate import validate_submission


def _distractor_index(provenance: dict | None) -> dict[str, list[str]]:
    """ID документа -> виды отвлекающих случаев, в которых он участвует."""
    index: dict[str, list[str]] = {}
    for d in (provenance or {}).get("distractors", []) or []:
        kind = d.get("kind", "?")
        for key, value in d.items():
            if key == "kind":
                continue
            values = value if isinstance(value, list) else [value]
            for v in values:
                if isinstance(v, str) and (v.startswith("INV-") or v.startswith("PAY-")):
                    index.setdefault(v, [])
                    if kind not in index[v]:
                        index[v].append(kind)
    return index


def score_submission(
    truth: dict,
    submission: Any,
    inspection_steps: int,
    *,
    provenance: dict | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    if isinstance(inspection_steps, bool) or not isinstance(inspection_steps, int) or inspection_steps < 0:
        raise ValueError("inspection_steps must be a non-negative integer")
    issues = truth["issues"]
    errors = validate_submission(submission, truth["episode_id"])
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "inspection_steps": inspection_steps,
            "truncated": truncated,
            "reward": -1.0,
            "reward_experimental": -1.0,
            "diagnostics": {"findings": 0, "issues": len(issues)},
        }
    findings = submission["findings"]
    result = match(findings, issues)
    precision, recall, f1 = prf(result.tp, result.fp, result.fn)
    distractors = _distractor_index(provenance)
    fp_reasons: Counter = Counter()
    fp_by_type: Counter = Counter()
    fp_on_distractor: Counter = Counter()
    for st in result.finding_status:
        if st["status"] != "fp":
            continue
        fp_reasons[st["reason"]] += 1
        fp_by_type[st["type"]] += 1
        kinds = sorted({k for a in findings[st["finding"]]["affected_ids"] for k in distractors.get(a, [])})
        for k in kinds:
            fp_on_distractor[k] += 1
    fn_by_type = Counter(m["type"] for m in result.missed)
    tp_by_type = Counter(issues[i]["type"] for _, i in result.pairs)
    diagnostics = {
        "findings": len(findings),
        "issues": len(issues),
        "tp_by_type": dict(sorted(tp_by_type.items())),
        "fp_by_type": dict(sorted(fp_by_type.items())),
        "fn_by_type": dict(sorted(fn_by_type.items())),
        "fp_reasons": dict(sorted(fp_reasons.items())),
        "fp_on_distractor": dict(sorted(fp_on_distractor.items())),
        "fn_near_miss": sum(1 for m in result.missed if m["reason"] == "near_miss"),
        "per_finding": result.finding_status,
        "missed": [{**m, "issue_id": issues[m["issue_index"]]["issue_id"]} for m in result.missed],
    }
    return {
        "valid": True,
        "errors": [],
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "inspection_steps": inspection_steps,
        "truncated": truncated,
        "reward": shared_reward(f1, inspection_steps, True),
        "reward_experimental": experimental_reward(f1, inspection_steps, True, truncated),
        "diagnostics": diagnostics,
    }


def empty_answer(episode_id: str) -> dict:
    """Итоговый ответ при усечении (лимит шагов/времени): пустой список выводов."""
    return {"episode_id": episode_id, "findings": []}
