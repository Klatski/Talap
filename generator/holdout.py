"""Имитация закрытой проверки жюри (раздел 10 чек-листа).

* `sample_config(rng, kind)` — новое сочетание параметров, которого нет в train/dev:
  размер, базовая и иностранные валюты, произвольный часовой пояс с шагом 15 минут,
  любой месяц 2021–2034, правило даты курса, округление, набор отвлекающих случаев, ошибки.
* `build_holdout(salt, n=20)` — n скрытых эпизодов (5 чистых, 5 с взаимодействующими
  ошибками, остальные смешанные), seed выводятся из соли и не пересекаются с открытыми.
* `verify_episode(public, private)` — всё, что жюри проверяет «по записям»: схема,
  эталон == независимый вывод, достаточность наборов доказательств, чистота контрфакта,
  размеры, и парная проверка: перестановка + смена ID сохраняют эталон и ответ oracle.

  python -m generator holdout --salt <секрет> --out <dir>   # собрать и проверить, записать на диск
  python -m generator stress --n 200                         # то же в памяти, на случайных конфигах

Всё в выходной папке — закрытое (seed, конфиги, truth): только для оценщика.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from .audit import compare_with_truth, parse_public
from .config import (
    COMPANY_OFFSETS,
    DISTRACTOR_TYPES,
    ERROR_TYPES,
    MAX_ISSUES,
    RATE_RULES,
    ROUNDING_CHOICES,
    SCENARIO_TYPES,
    SIZE_PROFILES,
    SUPPORTED_BASE,
    config_digest,
    normalize_config,
)
from .fileio import json_text, write_episode_dir
from .generate import COUNTERFACTUAL_DIR, generate_episode
from .money import CURRENCY_EXPONENTS
from .schema_check import validate_episode_texts
from .timeutil import format_offset
from .transform import remap_findings, transform_texts

KINDS = ("clean", "interacting", "mixed")


def holdout_composition(n: int) -> list[str]:
    """Как у жюри: на 20 эпизодов 5 чистых и 5 с взаимодействующими ошибками."""
    clean = max(1, round(n / 4))
    interacting = max(1, round(n / 4))
    return ["clean"] * clean + ["interacting"] * interacting + ["mixed"] * max(0, n - clean - interacting)


def sample_config(rng: random.Random, kind: str, name: str) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    base = rng.choice(SUPPORTED_BASE)
    foreign_pool = sorted(c for c in CURRENCY_EXPONENTS if c != base)
    budget = MAX_ISSUES
    scenarios = {s: 0 for s in SCENARIO_TYPES}
    errors = {t: 0 for t in ERROR_TYPES}
    if kind == "interacting":
        chosen = rng.sample(sorted(SCENARIO_TYPES), rng.choice([1, 2]))
        for s in chosen:
            scenarios[s] = rng.choice([1, 1, 2])
            budget -= 2 * scenarios[s]
    if kind != "clean":
        if kind == "mixed" and rng.random() < 0.3:
            s = rng.choice(sorted(SCENARIO_TYPES))
            scenarios[s] = 1
            budget -= 2
        types = list(ERROR_TYPES)
        rng.shuffle(types)
        for t in types:
            k = min(rng.choice([0, 1, 1, 2, 3]), budget)
            errors[t] = k
            budget -= k
        if kind == "mixed" and not any(errors.values()) and not any(scenarios.values()):
            errors[rng.choice(ERROR_TYPES)] = 1
    needs_fx = errors["fx_rate_error"] or any(scenarios.values())
    n_foreign = rng.choice([1, 2, 3]) if needs_fx else rng.choice([0, 1, 2, 3])
    if rng.random() < 0.5:
        foreign: Any = rng.sample(foreign_pool, n_foreign)
    else:
        foreign = n_foreign
    if rng.random() < 0.5:
        offset = rng.choice(COMPANY_OFFSETS)
    else:
        offset = rng.randrange(-44, 53) * 15  # любое смещение -11:00..+13:00 с шагом 15 минут
    cfg = {
        "config_version": "1.0",
        "name": name,
        "size": rng.choice(sorted(SIZE_PROFILES)),
        "base_currency": base,
        "foreign_currencies": foreign,
        "timezone": format_offset(offset),
        "period": f"{rng.randint(2021, 2034):04d}-{rng.randint(1, 12):02d}",
        "rounding_mode": rng.choice(ROUNDING_CHOICES),
        "fx_rate_date_rule": rng.choice(RATE_RULES),
        "errors": errors,
        "scenarios": scenarios,
        "distractors": {d: rng.random() < 0.8 for d in DISTRACTOR_TYPES},
    }
    return normalize_config(cfg)


def _clean_public(private: dict[str, str]) -> dict[str, str]:
    prefix = COUNTERFACTUAL_DIR + "/"
    return {k[len(prefix):]: v for k, v in private.items() if k.startswith(prefix)}


def _canonical(findings: dict) -> list:
    return sorted(
        (f["type"], tuple(sorted(set(f["affected_ids"]))), f["claimed_amount_minor"], f["currency"])
        for f in findings["findings"]
    )


def verify_episode(public: dict[str, str], private: dict[str, str], pair_seed: int | str = 1) -> list[str]:
    """Пустой список = эпизод корректен и выдерживает парную проверку."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from policies.oracle import oracle_findings

    problems: list[str] = []
    truth = json.loads(private["truth.json"])
    manifest = json.loads(public["manifest.json"])
    counts = manifest["counts"]
    if not 200 <= counts["ledger_rows"] <= 2000:
        problems.append(f"ledger_rows {counts['ledger_rows']} outside 200..2000")
    if not 20 <= counts["documents"] <= 200:
        problems.append(f"documents {counts['documents']} outside 20..200")
    if len(truth["issues"]) > MAX_ISSUES:
        problems.append(f"{len(truth['issues'])} issues > {MAX_ISSUES}")
    core_private = {k: v for k, v in private.items() if "/" not in k}
    problems += [f"schema: {e}" for e in validate_episode_texts(public, core_private)]
    problems += [f"truth: {p}" for p in compare_with_truth(parse_public(public), truth)]
    clean = _clean_public(private)
    if clean:
        problems += [
            f"counterfactual: {p}"
            for p in compare_with_truth(parse_public(clean), {"episode_id": truth["episode_id"], "issues": []})
        ]
    # парная проверка: перестановка строк + смена всех ID
    new_public, new_private, mapping = transform_texts(public, core_private, pair_seed)
    new_truth = json.loads(new_private["truth.json"])
    problems += [f"pair truth: {p}" for p in compare_with_truth(parse_public(new_public), new_truth)]
    remapped = remap_findings(oracle_findings(truth), mapping)
    if _canonical(remapped) != _canonical(oracle_findings(new_truth)):
        problems.append("pair: remapped oracle answer differs from oracle on transformed episode")
    by_key = {(i["type"], tuple(sorted(i["affected_ids"]))): i for i in new_truth["issues"]}
    for f in remapped["findings"]:
        issue = by_key.get((f["type"], tuple(sorted(set(f["affected_ids"])))))
        if issue is None or not any(set(s) <= set(f["evidence_ids"]) for s in issue["evidence_sets"]):
            problems.append(f"pair: remapped evidence of {f['type']} is not a full evidence set")
    old_ids = set(mapping)
    leaked = [i for i in old_ids if i in "".join(new_public.values())]
    if leaked:
        problems.append(f"pair: {len(leaked)} old ids survive the transform, e.g. {leaked[0]}")
    if remapped["episode_id"] != new_truth["episode_id"]:
        problems.append("pair: episode_id not remapped")
    return problems


def _seed_from(salt: str, i: int) -> int:
    return int.from_bytes(hashlib.sha256(f"holdout|{salt}|{i}".encode()).digest()[:8], "big")


def build_holdout(salt: str, n: int = 20, out: str | Path | None = None) -> dict[str, Any]:
    """Собрать и проверить n скрытых эпизодов; при out — записать эпизоды и их пары на диск."""
    if not salt:
        raise ValueError("salt is required: closed episodes must not be reproducible from public code alone")
    rng = random.Random(f"holdout-configs|{salt}")
    report: dict[str, Any] = {"n": n, "episodes": [], "failures": 0}
    out_dir = Path(out) if out else None
    for i, kind in enumerate(holdout_composition(n)):
        cfg = sample_config(rng, kind, f"holdout_{kind}_{i:02d}")
        seed = _seed_from(salt, i)
        t0 = time.perf_counter()
        ep = generate_episode(seed, cfg, salt=salt)
        gen_s = time.perf_counter() - t0
        problems = verify_episode(ep.public, ep.private, pair_seed=f"{salt}|pair|{i}")
        has_interaction = any(issue["scenario"] for issue in ep.truth["issues"])
        if kind == "clean" and ep.truth["issues"]:
            problems.append("clean episode has issues")
        if kind == "interacting" and not has_interaction:
            problems.append("interacting episode has no scenario issues")
        entry = {
            "episode_id": ep.episode_id,
            "kind": kind,
            "seed": seed,
            "config": cfg,
            "config_digest": config_digest(cfg),
            "issues": len(ep.truth["issues"]),
            "issue_types": sorted({x["type"] for x in ep.truth["issues"]}),
            "scenarios": sorted({x["scenario"] for x in ep.truth["issues"] if x["scenario"]}),
            "ledger_rows": json.loads(ep.public["manifest.json"])["counts"]["ledger_rows"],
            "documents": json.loads(ep.public["manifest.json"])["counts"]["documents"],
            "generation_seconds": round(gen_s, 3),
            "problems": problems,
        }
        report["failures"] += bool(problems)
        report["episodes"].append(entry)
        if out_dir is not None:
            write_episode_dir(out_dir / "episodes" / ep.episode_id, ep.public, ep.private)
            core_private = {k: v for k, v in ep.private.items() if "/" not in k}
            pub2, priv2, _ = transform_texts(ep.public, core_private, f"{salt}|pair|{i}")
            pair_id = json.loads(pub2["manifest.json"])["episode_id"]
            write_episode_dir(out_dir / "pairs" / pair_id, pub2, priv2)
            entry["pair_episode_id"] = pair_id
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "holdout_index.json", "w", encoding="utf-8", newline="") as fh:
            fh.write(json_text(report))
    return report


def stress(n: int, salt: str = "stress") -> dict[str, Any]:
    """n случайных конфигов (все виды) в памяти: сколько эпизодов не прошли проверки, худшее время."""
    rng = random.Random(f"stress-configs|{salt}")
    failures, worst, total = [], 0.0, 0.0
    for i in range(n):
        kind = KINDS[i % len(KINDS)]
        cfg = sample_config(rng, kind, f"stress_{i}")
        seed = _seed_from(salt, i)
        t0 = time.perf_counter()
        try:
            ep = generate_episode(seed, cfg, salt=salt)
            problems = verify_episode(ep.public, ep.private, pair_seed=i)
        except Exception as exc:  # noqa: BLE001 — отчёт о любой поломке генератора
            problems = [f"{type(exc).__name__}: {exc}"]
        dt = time.perf_counter() - t0
        worst, total = max(worst, dt), total + dt
        if problems:
            failures.append({"i": i, "seed": seed, "config": cfg, "problems": problems[:5]})
    return {"episodes": n, "failures": len(failures), "worst_seconds": round(worst, 3),
            "mean_seconds": round(total / max(1, n), 3), "failed": failures[:10]}
