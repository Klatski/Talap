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
import platform
import sys
from pathlib import Path
from typing import Any

from .episodes import EpisodeLoadError, EpisodeStore, LoadedEpisode
from .scoring import empty_answer, score_submission
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

    # ------------------------------------------------------------------ протокол
    def load(self, seed: Any, config: dict, mode: str) -> dict:
        if mode not in MODES:
            raise EpisodeLoadError("mode must be train, eval or closed")
        episode = self.store.load(seed, config)
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
        result = score_submission(ep.truth, submission, inspection_steps, provenance=ep.provenance, truncated=truncated,
                                  malformed_reason=malformed if end_reason == "submit" else None,
                                  public_ids=ep.public_ids)
        record = {
            "run_index": self._run_index,
            "grader_version": GRADER_VERSION,
            "env_version": ENV_VERSION,
            "environment": ENVIRONMENT,  # машинозависимое: в снимках check.py не сравнивается
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
            "issues_total": len(ep.truth["issues"]),
            **{k: result[k] for k in ("valid", "errors", "tp", "fp", "fn", "precision", "recall", "f1",
                                      "inspection_steps", "reward", "reward_experimental")},
            "diagnostics": result["diagnostics"],
        }
        self.records.append(record)
        self._append_to_log(record)
        if self._mode != "train":
            return {"reward": 0.0, "info": {}}
        info = {k: result[k] for k in INFO_KEYS}
        info["truncated_reason"] = end_reason if truncated else None
        info["reward_experimental"] = result["reward_experimental"]
        return {"reward": result["reward"], "info": info}

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
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8", newline="") as fh:
                fh.write(text)
                fh.flush()
        except OSError as exc:
            print(f"grader log is not writable ({type(exc).__name__}); records are kept in memory", file=sys.stderr)
            return
        self._unwritten.clear()

    def close(self) -> None:
        """Закрыть сессию (идемпотентно): дописать то, что не удалось записать раньше.
        Незавершённый эпизод до этого закрывает среда."""
        self._flush_log()
        self._closed = True
