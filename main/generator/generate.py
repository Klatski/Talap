"""Генерация эпизода: чистая функция (seed, config, version[, salt]) -> файлы.

Каждая стадия получает свой локальный random.Random, засеянный строкой
"<версия>|<seed>|<дайджест конфига>|<соль>|<попытка>|<стадия>". Глобальный random,
время, uuid4 и hash() не используются, поэтому результат побайтово воспроизводим
в разных процессах при любом PYTHONHASHSEED.

salt — необязательная секретная соль для закрытых прогонов: без неё по открытому коду
и episode_id можно перебрать маленькое пространство seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import compare_with_truth, parse_public
from .config import config_digest, normalize_config, resolve_params
from .fileio import json_text, write_episode_dir
from .history import GenerationShortfall, build_history
from .inject import finalize_issues, inject_all
from .invariants import check_after, check_clean
from .render import render
from .version import GENERATOR_VERSION

MAX_ATTEMPTS = 40
# Та же история с теми же ID, но без внесённых ошибок: доказательство «равенства до внесения»
# и точного объявленного изменения после (diff чистой и испорченной версий).
COUNTERFACTUAL_DIR = "counterfactual"
LEDGER_ROWS_RANGE = (200, 2000)
DOCUMENTS_RANGE = (20, 200)
MAX_ISSUES = 12


class GenerationError(RuntimeError):
    pass


class SelfCheckError(AssertionError):
    """Эталон не выводится из открытых данных — это баг генератора, а не сложность."""


@dataclass
class Episode:
    episode_id: str
    public: dict[str, str]
    private: dict[str, str]
    truth: dict
    provenance: dict
    attempt: int


def _bounds_ok(rendered) -> str | None:
    import json

    counts = json.loads(rendered.public["manifest.json"])["counts"]
    if not LEDGER_ROWS_RANGE[0] <= counts["ledger_rows"] <= LEDGER_ROWS_RANGE[1]:
        return f"ledger_rows={counts['ledger_rows']}"
    if not DOCUMENTS_RANGE[0] <= counts["documents"] <= DOCUMENTS_RANGE[1]:
        return f"documents={counts['documents']}"
    if len(rendered.truth["issues"]) > MAX_ISSUES:
        return "too many issues"
    return None


def generate_episode(seed: int | str, config: dict[str, Any], salt: str = "") -> Episode:
    cfg = normalize_config(config)
    digest = config_digest(cfg)
    last_reason = ""
    for attempt in range(MAX_ATTEMPTS):
        material = f"{GENERATOR_VERSION}|{seed}|{digest}|{salt}|{attempt}"

        def stream(stage: str) -> random.Random:
            return random.Random(f"{material}|{stage}")

        id_material = f"{material}|ids"
        order_material = f"{material}|order"
        try:
            params = resolve_params(cfg, stream("params"))
            h = build_history(params, stream("history"))
            h.build_doc_entries()
            h.pad_expenses(stream("expenses"))
            checks_before = check_clean(h)
            clean = render(h, [], id_material, order_material, {})
            issues = inject_all(h, stream("inject"))
            h.build_doc_entries()
            finalize_issues(h, issues)
            checks_after = check_after(h)
        except GenerationShortfall as exc:
            last_reason = str(exc)
            continue
        meta = {
            "seed": str(seed),
            "salted": bool(salt),
            "attempt": attempt,
            "config_name": cfg["name"],
            "config_digest": digest,
            "config": cfg,
            "params": params.to_json(),
            "checks_before_injection": checks_before,
            "checks_after_injection": checks_after,
        }
        rendered = render(h, issues, id_material, order_material, meta)
        reason = _bounds_ok(rendered)
        if reason:
            last_reason = reason
            continue
        problems = compare_with_truth(parse_public(rendered.public), rendered.truth)
        clean_truth = {"episode_id": rendered.episode_id, "issues": []}
        problems += [f"counterfactual: {p}" for p in compare_with_truth(parse_public(clean.public), clean_truth)]
        if problems:
            raise SelfCheckError(f"seed={seed} config={cfg['name']}: " + "; ".join(problems[:10]))
        rendered.provenance["self_check"] = {
            "audit_matches_truth": True,
            "evidence_sets_sufficient": True,
            "counterfactual_clean": True,
        }
        private = {
            "truth.json": json_text(rendered.truth),
            "provenance.json": json_text(rendered.provenance),
        }
        for name, text in clean.public.items():
            private[f"{COUNTERFACTUAL_DIR}/{name}"] = text
        return Episode(rendered.episode_id, rendered.public, private, rendered.truth, rendered.provenance, attempt)
    raise GenerationError(f"could not generate episode for seed={seed}: {last_reason}")


def write_episode(episode: Episode, out_dir: str | Path) -> Path:
    return write_episode_dir(out_dir, episode.public, episode.private)
