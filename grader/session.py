"""Сессия оценщика: держит эталон текущего эпизода, принимает ровно один итоговый ответ,
считает награду и ведёт журнал. Работает в отдельном процессе (grader.service);
в тестах может использоваться в процессе (env.local) — это НЕ изоляция, только отладка.

Что уходит наружу (в процесс среды):
  load   -> только шесть открытых файлов эпизода и mode (никакого truth/provenance/seed/путей);
  finish -> train: {"reward", "info": агрегированные метрики}; eval/closed: {"reward": 0.0, "info": {}}.
Полная диагностика (разбор по выводам) — только в журнале оценщика: запись эпизода (окружение, версии,
seed, config, метрики, награда) дописывается в файл сразу по его завершении, поэтому журнал не теряется,
даже если процесс жюри потом прервут (награды eval/closed не видны никому, кроме оценщика).
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .codeversion import code_version
from .episodes import EpisodeLoadError, EpisodeStore, LoadedEpisode
from .scoring import check_structure_only, empty_answer, score_submission
from .version import GRADER_VERSION

try:  # версия среды — для журнала (пакет env лежит рядом; оценщик его не требует)
    from env.constants import ENV_VERSION
except ImportError:  # pragma: no cover
    ENV_VERSION = None

END_REASONS = ("submit", "step_limit", "time_limit", "abandoned")
MODES = ("train", "eval", "closed")  # closed — имя жюри для eval: награда и диагностика остаются у оценщика
INFO_KEYS = ("valid", "errors", "tp", "fp", "fn", "precision", "recall", "f1", "inspection_steps")
ENVIRONMENT = {"python": platform.python_version(), "implementation": platform.python_implementation(),
               "platform": platform.platform(terse=True)}


class GraderError(RuntimeError):
    pass


def _open_append(path: Path) -> int:
    """Дескриптор, запись в который всегда идёт в конец файла атомарно — и когда в один журнал пишут
    несколько процессов оценщика (параллельные среды жюри). POSIX: O_APPEND, файл 0600. Windows: O_APPEND там
    эмулирует библиотека C (сдвиг в конец, потом запись) — между процессами строки затирают друг друга;
    поэтому файл открывается с правом только на дописывание (FILE_APPEND_DATA), и конец выбирает ядро."""
    if os.name != "nt":
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    import _winapi
    import msvcrt

    file_append_data, share_all, open_always, normal = 0x0004, 0x0007, 4, 0x80
    handle = _winapi.CreateFile(str(path), file_append_data, share_all, 0, open_always, normal, 0)
    try:
        return msvcrt.open_osfhandle(handle, os.O_BINARY)
    except OSError:
        _winapi.CloseHandle(handle)
        raise


def append_private_log(path: Path, text: str) -> None:
    """Дописать текст в журнал оценщика одной записью в конец (см. _open_append): каталог создаётся с правами
    0700, файл — 0600 (POSIX; на Windows права наследуются от профиля пользователя). Журнал лежит вне
    каталогов эпизодов и кода политики."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = memoryview(text.encode("utf-8"))
    fd = _open_append(path)
    try:
        while data:  # без буфера io: строка не делится на несколько записей вперемешку с другим процессом
            data = data[os.write(fd, data):]
    finally:
        os.close(fd)


class GraderSession:
    def __init__(self, episodes_root: str | Path | None = None, salt: str = "", log_path: str | Path | None = None) -> None:
        self.store = EpisodeStore(episodes_root, salt)
        self.log_path = Path(log_path) if log_path else None
        self.records: list[dict] = []
        self._unwritten: list[dict] = []  # записи, которые не удалось дописать в журнал (повтор при close)
        self._current: LoadedEpisode | None = None
        self._mode = "train"
        self._finished = True
        self._run_index = -1
        self.code_version = code_version()  # заранее: хэш исходников не должен попадать на путь завершения эпизода

    # ------------------------------------------------------------------ протокол
    def load(self, seed: Any, config: dict, mode: str) -> dict:
        if mode not in MODES:
            raise EpisodeLoadError("mode must be train, eval or closed")
        episode = self.store.load(seed, config, require_truth=(mode == "train"))  # train отдаёт награду политике
        self._current = episode
        self._config = {**config, "mode": mode} if isinstance(config, dict) else {"mode": mode}
        self._seed = seed
        self._mode = mode
        self._finished = False
        self._run_index += 1
        return {"public": dict(episode.public), "mode": mode}

    def finish(self, payload: Any, inspection_steps: int, steps_used: int, end_reason: str, elapsed_s: float | None = None,
               *, malformed: str | None = None) -> dict:
        """malformed — причина от среды, если submit нельзя было передать как есть (wrapper/not_json/too_large)."""
        if self._current is None or self._finished:
            raise GraderError("no episode awaiting a final answer")
        if end_reason not in END_REASONS:
            raise GraderError("unknown end reason")
        ep = self._current
        self._finished = True
        truncated = end_reason != "submit"
        submission = payload if end_reason == "submit" else empty_answer(ep.episode_id)
        if ep.truth is None:  # eval/closed без truth.json: только структура, метрики — у жюри
            result = check_structure_only(submission, ep.episode_id, inspection_steps, truncated=truncated,
                                          malformed_reason=malformed if end_reason == "submit" else None)
        else:
            result = score_submission(ep.truth, submission, inspection_steps, provenance=ep.provenance,
                                      truncated=truncated, malformed_reason=malformed if end_reason == "submit" else None,
                                      public_ids=ep.public_ids)
        record = {
            "run_index": self._run_index,
            "grader_version": GRADER_VERSION,
            "env_version": ENV_VERSION,
            "environment": ENVIRONMENT,  # машинозависимое: в снимках check.py не сравнивается
            "code_version": self.code_version,  # sha256 исходников команды (grader/codeversion.py); в снимках тоже нет
            "seed": self._seed,
            "config": self._config,
            "episode_id": ep.episode_id,
            "source": ep.source,
            "mode": self._mode,
            "end_reason": end_reason,
            "terminated": not truncated,
            "truncated": truncated,
            "steps_used": steps_used,
            "elapsed_s": elapsed_s,
            "issues_total": len(ep.truth["issues"]) if ep.truth is not None else None,
            **{k: result[k] for k in ("valid", "errors", "tp", "fp", "fn", "precision", "recall", "f1",
                                      "inspection_steps", "reward", "reward_experimental")},
            "warnings": result.get("warnings", []),  # напр. чужой episode_id: на оценку не влияет
            "diagnostics": result["diagnostics"],
        }
        self.records.append(record)
        self._append_to_log(record)
        # Структура ответа (не правильность) сообщается среде во всех режимах: на неверный submit политика
        # получает ограниченную ошибку invalid_submission вместо {"accepted": true} (правила жюри это разрешают).
        submission_status = {"valid": bool(result["valid"]), "error": (result["errors"] or [None])[0]}
        if self._mode != "train":
            return {"reward": 0.0, "info": {}, "submission": submission_status}
        info = {k: result[k] for k in INFO_KEYS}
        info["truncated_reason"] = end_reason if truncated else None
        info["reward_experimental"] = result["reward_experimental"]
        return {"reward": result["reward"], "info": info, "submission": submission_status}

    def _append_to_log(self, record: dict) -> None:
        if self.log_path is not None:
            self._unwritten.append(record)
            self._flush_log()

    def _flush_log(self) -> None:
        """Дописать накопленные записи одной операцией. Недоступный для записи журнал не роняет эпизод:
        записи остаются в self.records и в очереди (повтор при следующем эпизоде и при close()),
        в stderr — короткое сообщение без путей и содержимого."""
        if not self._unwritten or self.log_path is None:
            return
        text = "".join(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n" for rec in self._unwritten)
        try:
            append_private_log(self.log_path, text)
        except OSError as exc:
            print(f"grader log is not writable ({type(exc).__name__}); records are kept in memory", file=sys.stderr)
            return
        self._unwritten.clear()

    def close(self) -> None:
        """Закрыть сессию (идемпотентно): дописать то, что не удалось записать раньше.
        Незавершённый эпизод до этого закрывает среда."""
        self._flush_log()
        self._closed = True
