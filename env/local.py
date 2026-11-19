"""LocalAuditEnv — та же среда и тот же оценщик в ОДНОМ процессе.

Без изоляции (эталон лежит в памяти этого процесса), поэтому только для модульных тестов,
отладки и быстрых экспериментов. Официальный режим — env.AuditEnv (отдельные процессы).
Контракт reset/step идентичен AuditEnv.
"""

from __future__ import annotations

import os
from typing import Any

from .service import EnvService, InProcessGrader


class LocalAuditEnv:
    def __init__(self, episodes_root: str | os.PathLike | None = None, grader_log: str | os.PathLike | None = None,
                 salt: str = "") -> None:
        self._grader = InProcessGrader(episodes_root, salt, grader_log)
        self._service = EnvService(self._grader)

    @property
    def grader_records(self) -> list[dict]:
        """Журнал оценщика (для тестов). В изолированном режиме доступен только после close()."""
        return self._grader.session.records

    def reset(self, seed: Any = None, config: dict | None = None) -> tuple[dict, dict]:
        return self._service.reset(seed, config)

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        return self._service.step(action)

    def close(self) -> None:
        self._service.close()

    def __enter__(self) -> "LocalAuditEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
