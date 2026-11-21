"""Адаптер жюри (единый способ подключения из «Определений проверок FinTech»): три точки входа.

    from judge_adapter import generate_episode, make_env, score_episode

Рабочая директория — корень репозитория (в контейнере /app); только stdlib Python 3.11, без сборки.
Все пути, seed, config, truth, provenance и score_episode остаются у процесса жюри.

Самодостаточность: модуль не меняет sys.path и не импортирует в процесс жюри пакеты env/, grader/,
generator/, policies/ под их именами — у жюри могут быть свои пакеты с такими именами (например, policies
организаторов), и они остаются их собственными. generate_episode и score_episode выполняются отдельным
процессом `python -E -s -m grader.judge ...` с рабочим каталогом = корень репозитория; make_env и load_policy
берут клиентскую часть env/ (client, sandbox, protocol, jsonsafe, constants) под частным именем пакета
`_talaps_team_env`, а сама среда и оценщик — тоже отдельные процессы (`python -E -s -m env.server`).

generate_episode(seed, config, output_dir) -> None
    Обёртка над генератором команды (generator/), не отдельный генератор. Пишет
        output_dir/public/{manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv}
        output_dir/private/{truth.json, provenance.json, counterfactual/*}
    config — конфиг генератора команды (docs/generator.md): {} — чистая компания; {"preset": "<имя>",
    ...переопределения} — пресет из generator/configs/; полный объект {"size", "errors": {тип: число},
    "scenarios", "distractors", ...}; обёртка {"team_config": <любое из этого>} тоже принимается. Ключи,
    которых генератор не знает (и ключи reset: mode, episode_dir, ...), игнорируются и пишутся в журнал
    оценщика; ключ, похожий на известный ({"Errors": ...}, {"error": ...}, "team_confg"), — ValueError, а не
    молча чистая компания. Пример одной ошибки: {"errors": {"duplicate_invoice": 1}}. Одинаковые seed+config (+ соль
    AUDIT_GRADER_SALT, по умолчанию пустая) -> побайтово одинаковые файлы — тот же эпизод, что
    reset({"mode": "train", "team_config": config}). Файлы эпизода в output_dir перезаписываются.

make_env() -> среда с reset(seed, config) -> (observation, info) и step(action) -> (observation, reward,
    terminated, truncated, info). Среда и оценщик — отдельные процессы: в процессе жюри нет эталона. reset:
        {"mode": "closed", "episode_dir": "<каталог>", "team_config": {...}} — ровно этот эпизод (любое имя
            каталога, без генерации и смены ID; truth.json в каталоге не обязателен); повторный reset начинает
            его заново; reward политике = 0;
        {"mode": "train", "team_config": {...}} — эпизод генерируется по seed и team_config;
            терминальная награда видна; можно и {"mode": "train", "episode_dir": ...}.
    Лишние ключи config не мешают (игнорируются, пишутся в журнал оценщика); похожий на известный ключ
    ("Mode", "episode-dir", "team_confg") — ValueError. Лимит 60 с — от начала reset.
    step принимает {"name": <действие>, "args": {...}}: list_records{table, cursor, limit (null = 50)},
    get_record{table, id}, get_document{doc_id}, calculator{expression}, submit{findings}. Таблицы: ledger,
    invoices, payments, fx (id записи fx — "FX:<source>:<target>:<date>"; "<date>:<source>:<target>" — то же).
    Результаты: list_records -> {"records": [...], "next_cursor": null | str}; get_record / get_document ->
    объект; calculator -> десятичная строка; submit -> {"accepted": true}, а при неверной структуре ответа —
    result null и ограниченная ошибка {"code": "invalid_submission", "message", "detail"}.
    env.grader_log — путь журнала оценщика. Брошенная без close() среда останавливается сама (не дольше 2 с).

score_episode(episode_dir, findings, inspection_steps) -> {"tp", "fp", "fn", "precision", "recall", "f1", "return"}
    Закрытый вызов оценщика (grader/judge.py): невалидный submit -> return = -1, остальные поля None.
    findings проходит через json.dumps/json.loads (кортежи = массивы, как через step); не-JSON значение —
    неверная структура.

Журнал оценщика: grader_log_path() — переменная окружения AUDIT_GRADER_LOG (относительный путь — от
рабочего каталога жюри), иначе <временная папка системы>/talaps-grader-log-<uid>/grader_log_judge.jsonl
на Linux и macOS, <временная папка>/talaps-grader-log/grader_log_judge.jsonl на Windows (вне репозитория,
каталог 0700, файл 0600). Туда пишутся окружение, версии, sha256 исходников команды (code_version:
python -m grader.codeversion), seed, config и результат каждого эпизода среды (сразу по его завершении),
каждого generate_episode и score_episode.

Соответствие четырёх типов ошибок кейса (имена совпадают с внутренними, перевод не нужен):
    дубль счёта -> duplicate_invoice; пропущенное/частичное распределение платежа -> unapplied_payment;
    неверная дата или курс конвертации -> fx_rate_error; ошибка отнесения к периоду -> cutoff_error.
Политика команды запускается в отдельном ограниченном процессе: load_policy(путь) ниже (env/sandbox.py).
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import operator
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

__all__ = ["generate_episode", "make_env", "score_episode", "load_policy", "grader_log_path"]
_PACKAGE = "_talaps_team_env"  # env/ команды под частным именем: пакеты процесса жюри не затрагиваются
_JUDGE_TIMEOUT_S = 300.0
_ERRORS = {"ValueError": ValueError, "TypeError": TypeError, "FileNotFoundError": FileNotFoundError,
           "PermissionError": PermissionError, "OSError": OSError}


def _team_env(module: str):
    """Модуль env/<module>.py команды как `_talaps_team_env.<module>`: без sys.path и без env/__init__.py.
    Импорты внутри этих модулей относительные, поэтому они не трогают имена env/grader/policies жюри."""
    package = sys.modules.get(_PACKAGE)
    if package is None:
        package = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(_PACKAGE, None, is_package=True))
        package.__path__ = [str(ROOT / "env")]
        sys.modules[_PACKAGE] = package
    return importlib.import_module(f"{_PACKAGE}.{module}")


def _abs_path(value: Any) -> Any:
    """Путь жюри — относительно ЕГО рабочего каталога; процесс команды работает из корня репозитория."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    return os.path.abspath(value) if isinstance(value, str) and value.strip() else value


def _json_scalar(value: Any) -> Any:
    """int-подобное значение (например, numpy.int64) -> int; прочее как есть, проверку делает команда."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return operator.index(value)
    except TypeError:
        return repr(value)


def _judge_process(command: str, request: dict) -> Any:
    """`python -E -s -m grader.judge <command>`: запрос JSON в stdin, ответ — последняя строка stdout."""
    argv = [sys.executable or "python3", "-E", "-s", "-m", "grader.judge", command]
    data = json.dumps(request, ensure_ascii=True, allow_nan=False, sort_keys=True).encode("ascii")
    env = {**os.environ, "AUDIT_GRADER_LOG": grader_log_path()}  # абсолютный: процесс работает из корня репозитория
    proc = subprocess.run(argv, input=data, stdout=subprocess.PIPE, stderr=None, cwd=str(ROOT), env=env,
                          timeout=_JUDGE_TIMEOUT_S, check=False)
    lines = [line for line in proc.stdout.decode("utf-8", "replace").splitlines() if line.strip()]
    try:
        reply = json.loads(lines[-1])
    except (IndexError, ValueError):
        raise RuntimeError(f"grader process failed (exit code {proc.returncode})") from None
    if not isinstance(reply, dict) or not reply.get("ok"):
        error = (reply.get("error") if isinstance(reply, dict) else None) or {}
        raise _ERRORS.get(error.get("type"), RuntimeError)(error.get("message", "grader process failed"))
    return reply.get("result")


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    if not isinstance(config, dict):
        raise TypeError("config must be an object (dict)")
    try:
        config = json.loads(json.dumps(config, allow_nan=False))
    except (TypeError, ValueError, RecursionError):
        raise TypeError("config must be a JSON object") from None
    _judge_process("generate", {"seed": _json_scalar(seed), "config": config, "output_dir": _abs_path(output_dir)})


def grader_log_path() -> str:
    """Абсолютный путь журнала оценщика, куда пишут make_env, generate_episode и score_episode."""
    return os.path.abspath(os.environ.get("AUDIT_GRADER_LOG") or _team_env("constants").default_grader_log())


def make_env():
    return _team_env("client").AuditEnv(grader_log=grader_log_path())


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    request: dict[str, Any] = {"episode_dir": _abs_path(episode_dir), "inspection_steps": _json_scalar(inspection_steps)}
    try:  # тот же JSON, что прошёл бы через step: кортежи -> массивы
        request["findings"] = json.loads(json.dumps(findings, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError, RecursionError):  # NaN, set, bytes, сверхдлинное целое: неверная структура
        request["findings"], request["malformed"] = None, "not_json"
    return _judge_process("score", request)


def load_policy(path: str, **kwargs: Any):
    """Политика из файла (make_policy(**kwargs)) в отдельном ограниченном процессе: reset(obs, info), act(obs), close()."""
    return _team_env("sandbox").SandboxedPolicy.from_file(path, kwargs=kwargs)
