from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from grader.reward import compute_f1, compute_terminal_reward


@dataclass(frozen=True)
class TruthIssue:
    issue_id: str
    issue_type: str
    affected_ids: tuple[str, ...]
    expected_amount_minor: int
    currency: str
    tolerance_minor: int = 0
    valid_evidence_sets: tuple[frozenset[str], ...] = ()


def is_finding_match(finding: dict[str, Any], truth: TruthIssue) -> bool:
    if finding["type"] != truth.issue_type:
        return False
    if finding["affected_ids"] != truth.affected_ids:
        return False
    if finding["currency"] != truth.currency:
        return False
    if abs(finding["claimed_amount_minor"] - truth.expected_amount_minor) > truth.tolerance_minor:
        return False
    if truth.valid_evidence_sets:
        ev = finding["evidence_ids"]
        if not any(valid_set.issubset(ev) for valid_set in truth.valid_evidence_sets):
            return False
    return True


def _max_bipartite_matching(adj: list[list[int]], right_count: int) -> int:
    match_left: list[int] = [-1] * right_count

    def dfs(u: int, visited: set[int]) -> bool:
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if match_left[v] == -1 or dfs(match_left[v], visited):
                match_left[v] = u
                return True
        return False

    matched = 0
    for u in range(len(adj)):
        if dfs(u, set()):
            matched += 1
    return matched


def evaluate_findings(
    normalized_findings: list[dict[str, Any]],
    truth_issues: list[TruthIssue],
    inspection_steps: int,
    step_penalty: Decimal = Decimal("0.002"),
) -> dict[str, Any]:
    adj: list[list[int]] = [[] for _ in range(len(normalized_findings))]
    for i, f in enumerate(normalized_findings):
        for j, t in enumerate(truth_issues):
            if is_finding_match(f, t):
                adj[i].append(j)

    tp = _max_bipartite_matching(adj, len(truth_issues))
    fp = max(0, len(normalized_findings) - tp)
    fn = max(0, len(truth_issues) - tp)

    precision, recall, f1 = compute_f1(tp=tp, fp=fp, fn=fn)
    reward = compute_terminal_reward(f1=f1, inspection_steps=inspection_steps, step_penalty=step_penalty)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "inspection_steps": inspection_steps,
        "reward": float(reward),
    }