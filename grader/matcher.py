"""Сопоставление выводов агента с эталоном (публичный код нормализации, допусков и выбора).

Нормализация вывода:
  * affected_ids -> множество (порядок и повторы не важны), evidence_ids -> множество;
  * строки сравниваются точно (без смены регистра и обрезки пробелов);
  * type, currency — точное совпадение; сумма — |claimed - expected| <= amount_tolerance_minor.
Ребро «вывод f — ошибка i» есть, только если выполнены ВСЕ условия TP:
  type, множество affected_ids, валюта, сумма в допуске и evidence_ids ⊇ хотя бы одного
  допустимого набора доказательств целиком.
Паросочетание — максимальное взаимно однозначное (алгоритм Куна по дополняющим путям),
а не жадное: TP = размер максимального паросочетания. Выбор при неоднозначности
детерминирован: выводы и ошибки обходятся в каноническом порядке (по отсортированным полям),
поэтому результат не зависит от порядка выводов в ответе.
Несопоставленные выводы — FP (в том числе повторы), несопоставленные ошибки — FN.
explanation в сопоставлении не участвует.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormFinding:
    index: int
    type: str
    affected: frozenset
    amount: int
    currency: str
    evidence: frozenset

    def sort_key(self) -> tuple:
        return (self.type, sorted(self.affected), self.amount, self.currency, sorted(self.evidence), self.index)


@dataclass(frozen=True)
class NormIssue:
    index: int
    issue_id: str
    type: str
    affected: frozenset
    amount: int
    currency: str
    tolerance: int
    evidence_sets: tuple

    def sort_key(self) -> tuple:
        return (self.type, sorted(self.affected), self.amount, self.currency, self.issue_id)


def normalize_finding(f: dict, index: int) -> NormFinding:
    return NormFinding(
        index=index,
        type=f["type"],
        affected=frozenset(f["affected_ids"]),
        amount=f["claimed_amount_minor"],
        currency=f["currency"],
        evidence=frozenset(f["evidence_ids"]),
    )


def normalize_issue(issue: dict, index: int) -> NormIssue:
    return NormIssue(
        index=index,
        issue_id=issue["issue_id"],
        type=issue["type"],
        affected=frozenset(issue["affected_ids"]),
        amount=issue["expected_amount_minor"],
        currency=issue["currency"],
        tolerance=issue.get("amount_tolerance_minor", 0),
        evidence_sets=tuple(frozenset(s) for s in issue["evidence_sets"]),
    )


def failed_conditions(f: NormFinding, i: NormIssue) -> list[str]:
    """Какие условия TP не выполнены (пустой список = ребро есть)."""
    failed = []
    if f.type != i.type:
        failed.append("type")
    if f.affected != i.affected:
        failed.append("affected_ids")
    if f.currency != i.currency:
        failed.append("currency")
    if abs(f.amount - i.amount) > i.tolerance:
        failed.append("amount")
    if not any(s <= f.evidence for s in i.evidence_sets):
        failed.append("evidence")
    return failed


@dataclass
class MatchResult:
    pairs: list[tuple[int, int]]           # (индекс вывода в ответе, индекс ошибки в эталоне)
    tp: int
    fp: int
    fn: int
    finding_status: list[dict] = field(default_factory=list)  # диагностика по выводам (только журнал оценщика)
    missed: list[dict] = field(default_factory=list)          # диагностика по пропущенным ошибкам


def _max_matching(adj: dict[int, list[int]], left: list[int]) -> dict[int, int]:
    """Алгоритм Куна. adj: вывод -> ошибки (в каноническом порядке). Возвращает ошибка -> вывод."""
    owner: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in adj.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in owner or augment(owner[v], seen):
                owner[v] = u
                return True
        return False

    for u in left:
        augment(u, set())
    return owner


def match(findings: list[dict], issues: list[dict]) -> MatchResult:
    """findings — уже валидированные выводы; issues — truth['issues']."""
    nf = [normalize_finding(f, k) for k, f in enumerate(findings)]
    ni = [normalize_issue(x, k) for k, x in enumerate(issues)]
    f_order = sorted(nf, key=NormFinding.sort_key)
    i_order = sorted(ni, key=NormIssue.sort_key)
    adj = {f.index: [i.index for i in i_order if not failed_conditions(f, i)] for f in f_order}
    owner = _max_matching(adj, [f.index for f in f_order])
    matched_f = {u: v for v, u in owner.items()}
    pairs = sorted((u, v) for v, u in owner.items())

    status = []
    for f in nf:
        if f.index in matched_f:
            status.append({"finding": f.index, "status": "tp", "issue_index": matched_f[f.index]})
            continue
        if adj[f.index]:
            reason, near = "duplicate", []  # мог бы совпасть, но ошибка уже закрыта другим выводом
        else:
            near = [i for i in ni if i.affected == f.affected or (i.type == f.type and i.affected & f.affected)]
            if not near:
                reason = "no_such_issue"
            else:
                best = min(near, key=lambda i: (len(failed_conditions(f, i)), i.sort_key()))
                reason = "wrong_" + "+".join(failed_conditions(f, best))
        status.append({"finding": f.index, "status": "fp", "reason": reason, "type": f.type})
    missed = []
    for i in ni:
        if i.index in owner:
            continue
        near = any(f.type == i.type and f.affected == i.affected for f in nf)
        missed.append({"issue_index": i.index, "type": i.type, "reason": "near_miss" if near else "not_reported"})
    tp = len(pairs)
    return MatchResult(pairs=pairs, tp=tp, fp=len(nf) - tp, fn=len(ni) - tp, finding_status=status, missed=missed)
