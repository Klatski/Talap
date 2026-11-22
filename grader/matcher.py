"""Сопоставление выводов агента с эталоном — точные соглашения matcher из «Определений проверок» жюри.

Нормализация (только после проверки структуры и исходных размеров списков, grader/validate.py):
  * ID сравниваются как строки точно: с учётом регистра и пробелов, без снятия префиксов и поиска похожих;
  * affected_ids -> множество: порядок и повторы внутри списка не меняют множество (M01); для TP оно
    должно точно совпасть с каноническим набором ошибки (лишний существующий ID -> FP, M02);
  * evidence_ids -> множество (ключ курса в записи жюри <date>:<source>:<target> — синоним fx_id
    FX:<source>:<target>:<date>; к fx_id приводятся ОБЕ стороны: доказательства вывода и наборы эталона,
    поэтому truth фикстуры с ключом в записи жюри сопоставляется так же, как сгенерированный);
    TP требует полного включения хотя бы одного допустимого набора;
    лишние СУЩЕСТВУЮЩИЕ доказательства допустимы (M03); если указан хотя бы один ID, которого нет в
    открытых данных эпизода, вывод не сопоставляется и даёт FP (M04). Существующие = доступные через
    инструменты: row_id, invoice_id, payment_id, fx_id и doc_id (env/records.py);
  * type — точное совпадение с одним из четырёх типов; неизвестная строка просто несопоставима;
  * currency — точное совпадение строки: без конвертации, смены регистра и обрезки пробелов (M06);
  * сумма — JSON-целое, |claimed - expected| <= τ; τ = amount_tolerance_minor ошибки в truth (нет поля
    или null — 0). Генератор пишет опубликованные значения env/constants.py::AMOUNT_TOLERANCE_MINOR:
    0 для duplicate_invoice, unapplied_payment, cutoff_error и 1 для fx_rate_error (округление пересчёта
    по курсу), docs/contract.md, раздел 4 (M07, M08).
Ребро «вывод — ошибка» есть только при выполнении всех условий TP. TP — мощность МАКСИМАЛЬНОГО
паросочетания (не жадного, M10), FP = выводов - TP, FN = ошибок - TP; повторный вывод остаётся
отдельной вершиной и после закрытия ошибки даёт FP (M12). При нескольких максимальных паросочетаниях
выбирается лексикографически наименьшая отсортированная последовательность пар (issue_id, индекс
вывода): issue_id сравниваются по Unicode code points (M11). Порядок выводов меняет только то, какой из
одинаковых выводов сопоставлен, но не TP/FP/FN и награду. explanation в выборе рёбер не участвует.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AbstractSet

from env.records import canonical_record_id

from .validate import ISSUE_TYPES


@dataclass(frozen=True)
class NormFinding:
    index: int
    type: str
    affected: frozenset
    amount: int
    currency: str
    evidence: frozenset


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
        evidence=frozenset(canonical_record_id(e) for e in f["evidence_ids"]),  # ключ курса жюри -> fx_id
    )


def issue_label(issue: dict, index: int) -> str:
    """issue_id ошибки эталона; у фикстуры без него (или с null) — номер по порядку в truth.json."""
    issue_id = issue.get("issue_id")
    return str(issue_id) if issue_id is not None else f"#{index:06d}"


def normalize_issue(issue: dict, index: int) -> NormIssue:
    return NormIssue(
        index=index,
        issue_id=issue_label(issue, index),
        type=issue["type"],
        affected=frozenset(issue["affected_ids"]),
        amount=issue["expected_amount_minor"],
        currency=issue["currency"],
        tolerance=issue.get("amount_tolerance_minor") or 0,  # нет поля или null — τ = 0 (правила жюри)
        # тот же canonical_record_id, что у вывода: ключ курса жюри в truth фикстуры -> fx_id
        evidence_sets=tuple(frozenset(canonical_record_id(e) for e in s) for s in issue["evidence_sets"]),
    )


def failed_conditions(f: NormFinding, i: NormIssue) -> list[str]:
    """Какие условия TP для пары не выполнены (пустой список = ребро есть, если доказательства существуют)."""
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


def _max_matching_size(edges: dict[int, list[int]], issues: list[int], blocked: AbstractSet[int]) -> int:
    """Алгоритм Куна: мощность максимального паросочетания ошибок `issues` с выводами вне `blocked`."""
    owner: dict[int, int] = {}

    def augment(i: int, seen: set[int]) -> bool:
        for f in edges[i]:
            if f in blocked or f in seen:
                continue
            seen.add(f)
            if f not in owner or augment(owner[f], seen):
                owner[f] = i
                return True
        return False

    return sum(1 for i in issues if augment(i, set()))


def _lexmin_max_matching(edges: dict[int, list[int]], order: list[int]) -> dict[int, int]:
    """Максимальное паросочетание с лексикографически наименьшей последовательностью пар (issue_id, вывод).

    order — ошибки по возрастанию issue_id; edges[i] — выводы по возрастанию индекса. Ошибка включается,
    если с уже выбранными парами ещё достижима максимальная мощность, и получает наименьший такой вывод.
    """
    target = _max_matching_size(edges, order, frozenset())
    chosen: dict[int, int] = {}
    used: set[int] = set()
    for pos, i in enumerate(order):
        rest = order[pos + 1:]
        for f in edges[i]:
            if f in used:
                continue
            if len(chosen) + 1 + _max_matching_size(edges, rest, used | {f}) == target:
                chosen[i] = f
                used.add(f)
                break
    return chosen


def match(findings: list[dict], issues: list[dict], known_ids: AbstractSet[str] | None = None) -> MatchResult:
    """findings — уже валидированные выводы; issues — truth['issues'];
    known_ids — все ID открытых данных эпизода (None = проверка существования доказательств выключена)."""
    nf = [normalize_finding(f, k) for k, f in enumerate(findings)]
    ni = [normalize_issue(x, k) for k, x in enumerate(issues)]
    unknown_evidence = {f.index for f in nf if known_ids is not None and not f.evidence <= known_ids}
    edges = {i.index: [f.index for f in nf if f.index not in unknown_evidence and not failed_conditions(f, i)] for i in ni}
    order = [i.index for i in sorted(ni, key=lambda i: (i.issue_id, i.index))]
    chosen = _lexmin_max_matching(edges, order)
    matched_f = {f: i for i, f in chosen.items()}
    pairs = sorted((f, i) for i, f in chosen.items())

    status = []
    for f in nf:
        if f.index in matched_f:
            status.append({"finding": f.index, "status": "tp", "issue_index": matched_f[f.index]})
            continue
        if f.index in unknown_evidence:
            reason = "unknown_evidence"  # ID доказательства, которого нет в открытых данных (M04)
        elif any(f.index in edges[i.index] for i in ni):
            reason = "duplicate"  # мог бы совпасть, но ошибка уже закрыта другим выводом
        elif f.type not in ISSUE_TYPES:
            reason = "unknown_type"
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
        if i.index in chosen:
            continue
        near = any(f.type == i.type and f.affected == i.affected for f in nf)
        missed.append({"issue_index": i.index, "type": i.type, "reason": "near_miss" if near else "not_reported"})
    tp = len(pairs)
    return MatchResult(pairs=pairs, tp=tp, fp=len(nf) - tp, fn=len(ni) - tp, finding_status=status, missed=missed)
