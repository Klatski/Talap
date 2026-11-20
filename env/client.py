"""AuditEnv — официальный изолированный клиент среды (Gymnasium-подобный, JSON-совместимый).

    from env import AuditEnv
    with AuditEnv(grader_log="reports/grader_log.jsonl") as env:
        obs, info = env.reset(seed=3, config={"split": "train", "mode": "eval"})
        obs, reward, terminated, truncated, info = env.step({"action": "list_records", "table": "ledger"})

Топология процессов:
    агент (этот процесс: политика + AuditEnv)
        └─ python -m env.server       — открытые записи, действия, лимиты, таймер
             └─ python -m grader.service — эталон, награда, журнал оценщика
В процессе агента нет ни эталона, ни пути к нему, ни канала к оценщику.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .jsonsafe import classify_action
from .protocol import Channel, RemoteError

ROOT = Path(__file__).resolve().parent.parent


class EnvError(RuntimeError):
    pass



def wire_params(action: Any) -> dict:
    """Параметры step для сервера. Классификация та же, что в EnvCore (env/jsonsafe.classify_action):
    выполнимое действие уходит как объект, а невыполнимое — только флагом unsendable {submit, reason},
    поэтому итог совпадает с LocalAuditEnv, а гигантские или не-JSON значения не идут по проводу."""
    obj, bad = classify_action(action)
    if bad is None:
        return {"action": obj}
    is_submit, reason = bad
    return {"action": None, "unsendable": {"submit": is_submit, "reason": reason}}

class AuditEnv:
    def __init__(
        self,
        episodes_root: str | os.PathLike | None = None,
        grader_log: str | os.PathLike | None = None,
        salt: str | None = None,
        response_timeout_s: float = 120.0,
    ) -> None:
        argv = [sys.executable, "-m", "env.server"]
        if episodes_root:
            argv += ["--episodes-root", str(Path(episodes_root).resolve())]
        if grader_log:
            argv += ["--grader-log", str(Path(grader_log).resolve())]
        env = dict(os.environ)
        if salt is not None:
            env["AUDIT_GRADER_SALT"] = salt
        self._timeout = response_timeout_s
        self._channel = Channel(argv, cwd=str(ROOT), env=env)
        self._closed = False

    def reset(self, seed: Any = None, config: dict | None = None) -> tuple[dict, dict]:
        try:
            json.dumps({"seed": seed, "config": config}, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError, RecursionError):
            raise EnvError("reset failed: seed and config must be JSON values (no NaN, string keys only)") from None
        try:
            out = self._channel.request("reset", {"seed": seed, "config": config}, self._timeout)
        except RemoteError as exc:
            raise EnvError(f"reset failed: {exc.public_message}") from None
        return out["observation"], out["info"]

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        try:
            out = self._channel.request("step", wire_params(action), self._timeout)
        except RemoteError as exc:
            raise EnvError(f"step failed: {exc.public_message}") from None
        return out["observation"], out["reward"], out["terminated"], out["truncated"], out["info"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._channel.request("close", {}, timeout=60.0)
        except Exception:  # noqa: BLE001
            pass
        self._channel.close()

    def __enter__(self) -> "AuditEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
