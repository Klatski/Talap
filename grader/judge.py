"""Закрытый вызов оценщика для жюри: score_episode(episode_dir, findings, inspection_steps) -> метрики.

Тот же путь, что у submit в среде: структура (grader/validate.py) -> matcher (правила M01–M12,
grader/matcher.py) -> награда (grader/metrics.py). Каталог эпизода — любой (раскладка public/ + private/
или плоская, grader/episodes.py). Результат:
  валидный ответ   -> {"tp": int, "fp": int, "fn": int, "precision": float, "recall": float, "f1": float, "return": float};
  неверная структура -> {"return": -1.0, остальные поля None}; ошибки структуры — в журнале оценщика.
Журнал оценщика (JSON-lines, дописывается): аргумент log_path, иначе переменная окружения AUDIT_GRADER_LOG,
иначе reports/grader_log_judge.jsonl в корне репозитория. Политике этот вызов недоступен: его делает жюри.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from .episodes import ROOT, load_episode_dir
from .scoring import score_submission
from .version import GRADER_VERSION

METRIC_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1")
DEFAULT_LOG = ROOT / "reports" / "grader_log_judge.jsonl"


def grader_log_path(log_path: str | os.PathLike | None = None) -> Path:
    return Path(log_path or os.environ.get("AUDIT_GRADER_LOG") or DEFAULT_LOG)


def append_log(record: dict, log_path: str | os.PathLike | None = None) -> None:
    """Дописать запись в журнал оценщика; недоступный для записи журнал не роняет оценку (сообщение в stderr)."""
    path = grader_log_path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError as exc:
        print(f"grader log is not writable ({type(exc).__name__}); the score is still returned", file=sys.stderr)


def score_episode(episode_dir: str, findings: Any, inspection_steps: int, *,
                  log_path: str | os.PathLike | None = None) -> dict:
    """Оценить ответ `findings` (объект findings.json) на эпизоде `episode_dir` с `inspection_steps` шагами исследования."""
    t0 = time.perf_counter()
    ep = load_episode_dir(str(episode_dir) if isinstance(episode_dir, os.PathLike) else episode_dir)
    result = score_submission(ep.truth, findings, inspection_steps, provenance=ep.provenance, public_ids=ep.public_ids)
    if result["valid"]:
        out = {k: result[k] for k in METRIC_KEYS}
    else:
        out = {k: None for k in METRIC_KEYS}
    out["return"] = float(result["reward"])
    from env.constants import ENV_VERSION

    append_log({
        "kind": "score_episode",
        "time_unix": round(time.time(), 3),
        "grader_version": GRADER_VERSION,
        "env_version": ENV_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "episode_id": ep.episode_id,
        "episode_dir": ep.source.get("episode_dir"),
        "inspection_steps": inspection_steps,
        "valid": result["valid"],
        "errors": result["errors"],
        **{k: result[k] for k in METRIC_KEYS},
        "reward": result["reward"],
        "issues_total": len(ep.truth["issues"]),
        "diagnostics": result["diagnostics"],
        "elapsed_s": round(time.perf_counter() - t0, 4),
    }, log_path)
    return out
