"""Адаптер жюри (единый способ подключения из «Определений проверок FinTech»): три точки входа.

    from judge_adapter import generate_episode, make_env, score_episode

Рабочая директория — корень репозитория (в контейнере /app); только stdlib Python 3.11, без сборки.
Все пути, seed, config, truth, provenance и score_episode остаются у процесса жюри.

generate_episode(seed, config, output_dir) -> None
    Обёртка над генератором команды (generator/), не отдельный генератор. Пишет
        output_dir/public/{manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv}
        output_dir/private/{truth.json, provenance.json, counterfactual/*}
    config — конфиг генератора команды (docs/generator.md): {} — чистая компания; {"preset": "<имя>",
    ...переопределения} — пресет из generator/configs/; полный объект {"size", "errors": {тип: число},
    "scenarios", "distractors", ...}; обёртка {"team_config": <любое из этого>} тоже принимается, ключи
    reset (mode, episode_dir, max_steps, time_limit_s) игнорируются. Пример одной ошибки:
    {"errors": {"duplicate_invoice": 1}}. Одинаковые seed+config (+ соль AUDIT_GRADER_SALT, по умолчанию
    пустая) -> побайтово одинаковые файлы — тот же эпизод, что reset({"mode": "train", "team_config": config}).
    Существующие файлы эпизода в output_dir перезаписываются, остальное не трогается.

make_env() -> среда с reset(seed, config) -> (observation, info) и step(action) -> (observation, reward,
    terminated, truncated, info). Среда и оценщик — отдельные процессы (env/client.py): в процессе жюри нет
    эталона. reset понимает протокол жюри:
        {"mode": "closed", "episode_dir": "<каталог>", "team_config": {...}} — ровно этот эпизод (любое имя
            каталога, без генерации и смены ID); повторный reset начинает его заново; reward политике = 0;
        {"mode": "train", "team_config": {...}} — эпизод генерируется по seed и team_config;
            терминальная награда видна; можно и {"mode": "train", "episode_dir": ...}.
    step принимает {"name": <действие>, "args": {...}}: list_records{table, cursor, limit},
    get_record{table, id}, get_document{doc_id}, calculator{expression}, submit{findings}. Таблицы: ledger,
    invoices, payments, fx (id записи fx — "FX:<source>:<target>:<date>"). Результаты: list_records ->
    {"records": [...], "next_cursor": null | str}; get_record / get_document -> объект; calculator ->
    десятичная строка; submit -> {"accepted": true}.

score_episode(episode_dir, findings, inspection_steps) -> {"tp", "fp", "fn", "precision", "recall", "f1", "return"}
    Закрытый вызов оценщика (grader/judge.py): невалидный submit -> return = -1, остальные поля None.

Журнал оценщика: переменная окружения AUDIT_GRADER_LOG, иначе reports/grader_log_judge.jsonl. Туда пишутся
окружение, версии, seed, config и результат каждого эпизода (среда — при закрытии env.close()) и каждого
вызова score_episode.

Соответствие четырёх типов ошибок кейса (имена совпадают с внутренними, перевод не нужен):
    дубль счёта -> duplicate_invoice; пропущенное/частичное распределение платежа -> unapplied_payment;
    неверная дата или курс конвертации -> fx_rate_error; ошибка отнесения к периоду -> cutoff_error.
Политика команды запускается в отдельном ограниченном процессе: load_policy(путь) ниже (env/sandbox.py).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["generate_episode", "make_env", "score_episode", "load_policy"]
RESET_KEYS = frozenset({"mode", "episode_dir", "max_steps", "time_limit_s"})


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    from generator.generate import generate_episode as _generate
    from generator.generate import write_episode
    from grader.episodes import resolve_generator_config

    if not isinstance(config, dict):
        raise TypeError("config must be an object (dict)")
    team = config.get("team_config") if "team_config" in config else config
    if isinstance(team, dict):
        team = {k: v for k, v in team.items() if k not in RESET_KEYS}
    cfg = resolve_generator_config(team if team is not None else {})
    episode = _generate(seed, cfg, salt=os.environ.get("AUDIT_GRADER_SALT", ""))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_episode(episode, output_dir)


def make_env():
    from env.client import AuditEnv
    from grader.judge import grader_log_path

    return AuditEnv(grader_log=grader_log_path())


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    from grader.judge import score_episode as _score

    return _score(episode_dir, findings, inspection_steps)


def load_policy(path: str, **kwargs: Any):
    """Политика из файла (make_policy(**kwargs)) в отдельном ограниченном процессе: reset(obs, info), act(obs), close()."""
    from env.sandbox import SandboxedPolicy

    return SandboxedPolicy.from_file(path, kwargs=kwargs)
