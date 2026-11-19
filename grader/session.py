"""Сессия оценщика: держит эталон текущего эпизода, принимает ровно один итоговый ответ,
считает награду и ведёт журнал. Работает в отдельном процессе (grader.service);
в тестах может использоваться в процессе (env.local) — это НЕ изоляция, только отладка.

Что уходит наружу (в процесс среды):
  load   -> только шесть открытых файлов эпизода и mode (никакого truth/provenance/seed/путей);
  finish -> train: {"reward", "info": агрегированные метрики}; eval: {"reward": 0.0, "info": {}}.
Полная диагностика (разбор по выводам) — только в журнале оценщика, который пишется
в файл при закрытии сессии (до конца прогона награды eval не видны никому).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .episodes import EpisodeLoadError, EpisodeStore, LoadedEpisode
from .scoring import empty_answer, score_submission
from .version import GRADER_VERSION

END_REASONS = ("submit", "step_limit", "time_limit", "abandoned")
INFO_KEYS = ("valid", "errors", "tp", "fp", "fn", "precision", "recall", "f1", "inspection_steps")


class GraderError(RuntimeError):
    pass


class GraderSession:
    def __init__(self, episodes_root: str | Path | None = None, salt: str = "", log_path: str | Path | None = None) -> None:
        self.store = EpisodeStore(episodes_root, salt)
        self.log_path = Path(log_path) if log_path else None
        self.records: list[dict] = []
        self._current: LoadedEpisode | None = None
        self._mode = "train"
        self._finished = True
        self._run_index = -1

    # ------------------------------------------------------------------ протокол
    def load(self, seed: Any, config: dict, mode: str) -> dict:
        if mode not in ("train", "eval"):
            raise EpisodeLoadError("mode must be train or eval")
        episode = self.store.load(seed, config)
        self._current = episode
        self._mode = mode
        self._finished = False
        self._run_index += 1
        return {"public": dict(episode.public), "mode": mode}

    def finish(self, payload: Any, inspection_steps: int, steps_used: int, end_reason: str, elapsed_s: float | None = None) -> dict:
        if self._current is None or self._finished:
            raise GraderError("no episode awaiting a final answer")
        if end_reason not in END_REASONS:
            raise GraderError("unknown end reason")
        ep = self._current
        self._finished = True
        truncated = end_reason != "submit"
        submission = payload if end_reason == "submit" else empty_answer(ep.episode_id)
        result = score_submission(ep.truth, submission, inspection_steps, provenance=ep.provenance, truncated=truncated)
        record = {
            "run_index": self._run_index,
            "grader_version": GRADER_VERSION,
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
        if self._mode == "eval":
            return {"reward": 0.0, "info": {}}
        info = {k: result[k] for k in INFO_KEYS}
        info["truncated_reason"] = end_reason if truncated else None
        info["reward_experimental"] = result["reward_experimental"]
        return {"reward": result["reward"], "info": info}

    def close(self) -> None:
        """Записать журнал (идемпотентно). Незавершённый эпизод сюда не попадает: его закрывает среда."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w", encoding="utf-8", newline="") as fh:
                for rec in self.records:
                    fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
