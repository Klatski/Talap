"""Закрытые вызовы жюри на стороне команды: score_episode и generate_episode (judge_adapter.py запускает их
отдельным процессом `python -E -s -m grader.judge score|generate` с рабочим каталогом = корень репозитория).

score_episode(episode_dir, findings, inspection_steps) — тот же путь, что у submit в среде: структура
(grader/validate.py) -> matcher (правила M01–M12, grader/matcher.py) -> награда (grader/metrics.py). Каталог
эпизода — любой (раскладка public/ + private/ или плоская, BOM и CRLF допустимы, grader/episodes.py). Результат:
  валидный ответ     -> {"tp": int, "fp": int, "fn": int, "precision": float, "recall": float, "f1": float, "return": float};
  неверная структура -> {"return": -1.0, остальные поля None}; ошибки структуры — в журнале оценщика.
findings приходит через JSON (адаптер делает json.dumps/json.loads): кортежи — те же массивы, что через step;
не-JSON значение (NaN, set, сверхдлинное целое) — неверная структура (malformed="not_json"), как в среде.

generate_episode(seed, config, output_dir) — обёртка над генератором команды (generator/). config — конфиг
генератора (docs/generator.md) или обёртка {"team_config": ...}; ключи, которых генератор не знает, не мешают:
они игнорируются и записываются в журнал оценщика (ignored_config_keys).

Журнал оценщика (JSON-lines, дописывается): аргумент log_path, иначе переменная окружения AUDIT_GRADER_LOG,
иначе env.constants.default_grader_log() — вне репозитория, каталог 0700, файл 0600. Политике эти вызовы
недоступны: их делает жюри.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from env.constants import ENV_VERSION, default_grader_log

from .episodes import EpisodeLoadError, load_episode_dir, resolve_generator_config
from .scoring import score_submission
from .session import append_private_log
from .version import GRADER_VERSION

METRIC_KEYS = ("tp", "fp", "fn", "precision", "recall", "f1")
# Ключи reset жюри, которые в config генерации ожидаемы (не генератора, но и не «лишние»)
PROTOCOL_KEYS = frozenset({"mode", "episode_dir", "team_config", "max_steps", "time_limit_s", "config_version"})


def grader_log_path(log_path: str | os.PathLike | None = None) -> Path:
    return Path(log_path or os.environ.get("AUDIT_GRADER_LOG") or default_grader_log())


def append_log(record: dict, log_path: str | os.PathLike | None = None) -> None:
    """Дописать запись в журнал оценщика; недоступный для записи журнал не роняет оценку (сообщение в stderr)."""
    text = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        append_private_log(grader_log_path(log_path), text)
    except OSError as exc:
        print(f"grader log is not writable ({type(exc).__name__}); the result is still returned", file=sys.stderr)


def _environment() -> dict:
    return {"grader_version": GRADER_VERSION, "env_version": ENV_VERSION, "python": platform.python_version(),
            "platform": platform.platform(terse=True)}


def score_episode(episode_dir: str, findings: Any, inspection_steps: int, *, malformed: str | None = None,
                  log_path: str | os.PathLike | None = None) -> dict:
    """Оценить ответ `findings` (объект findings.json) на эпизоде `episode_dir` с `inspection_steps` шагами исследования.
    malformed — ответ не удалось передать как JSON (адаптер): неверная структура, как у такого submit в среде."""
    t0 = time.perf_counter()
    ep = load_episode_dir(str(episode_dir) if isinstance(episode_dir, os.PathLike) else episode_dir)
    result = score_submission(ep.truth, findings, inspection_steps, provenance=ep.provenance, public_ids=ep.public_ids,
                              malformed_reason=malformed)
    if result["valid"]:
        out = {k: result[k] for k in METRIC_KEYS}
    else:
        out = {k: None for k in METRIC_KEYS}
    out["return"] = float(result["reward"])
    append_log({
        "kind": "score_episode",
        "time_unix": round(time.time(), 3),
        **_environment(),
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


def generate_episode(seed: Any, config: Any, output_dir: str, *, log_path: str | os.PathLike | None = None) -> dict:
    """Сгенерировать эпизод генератором команды в output_dir (public/ + private/). Возвращает {"episode_id"}."""
    from generator.generate import generate_episode as _generate
    from generator.generate import write_episode

    if not isinstance(config, dict):
        raise TypeError("config must be an object (dict)")
    if seed is None or isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise TypeError("seed must be an integer (or a string)")
    ignored: list[str] = []
    if "team_config" in config:  # обёртка протокола жюри: генератору — только team_config
        team = config["team_config"]
        ignored += sorted(str(k) for k in config if k not in PROTOCOL_KEYS)
    else:
        team = config
    cfg = resolve_generator_config(team if team is not None else {}, ignored,
                                   prefix="team_config." if "team_config" in config else "")
    episode = _generate(seed, cfg, salt=os.environ.get("AUDIT_GRADER_SALT", ""))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_episode(episode, str(out_dir))
    if ignored:
        print(f"generate_episode: ignored config keys {sorted(set(ignored))}", file=sys.stderr)
    append_log({"kind": "generate_episode", "time_unix": round(time.time(), 3), **_environment(), "seed": seed,
                "config": config, "ignored_config_keys": sorted(set(ignored)), "episode_id": episode.episode_id,
                "output_dir": str(out_dir.resolve())}, log_path)
    return {"episode_id": episode.episode_id}


# ---------------------------------------------------------------------- процесс для judge_adapter.py
_ERROR_TYPES = {EpisodeLoadError: "ValueError", TypeError: "TypeError", ValueError: "ValueError",
                FileNotFoundError: "FileNotFoundError", PermissionError: "PermissionError", OSError: "OSError"}


def main(argv: list[str] | None = None) -> int:
    """Запрос — один JSON-объект в stdin, ответ — одна строка JSON в stdout: {"ok": true, "result": ...} или
    {"ok": false, "error": {"type", "message"}} (адаптер поднимает исключение этого типа у жюри)."""
    args = sys.argv[1:] if argv is None else argv
    stdout = sys.stdout
    sys.stdout = sys.stderr  # случайный print не испортит ответ
    try:
        from env.jsonsafe import loads_text

        request = loads_text(sys.stdin.buffer.read().decode("utf-8"))
        if not args or args[0] not in ("score", "generate") or not isinstance(request, dict):
            raise ValueError("usage: python -m grader.judge score|generate < request.json")
        if args[0] == "score":
            result = score_episode(request["episode_dir"], request.get("findings"), request["inspection_steps"],
                                   malformed=request.get("malformed"))
        else:
            result = generate_episode(request["seed"], request["config"], request["output_dir"])
        reply = {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001 — любая ошибка уходит жюри типом и текстом
        kind = next((name for cls, name in _ERROR_TYPES.items() if isinstance(exc, cls)), "RuntimeError")
        if isinstance(exc, KeyError):
            kind, message = "ValueError", f"missing request field {exc}"
        else:
            message = str(exc)[:1000]
        reply = {"ok": False, "error": {"type": kind, "message": message}}
    stdout.write(json.dumps(reply, ensure_ascii=True, allow_nan=False) + "\n")
    stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
