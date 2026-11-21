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
import weakref
from pathlib import Path
from typing import Any

from .jsonsafe import classify_action
from .protocol import Channel, RemoteError, stop_process

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
        # -E -s: процессы среды и оценщика не берут PYTHONPATH/user site обвязки (у жюри могут быть свои пакеты
        # env/grader/policies) — только код этого репозитория (cwd = корень).
        argv = [sys.executable, "-E", "-s", "-m", "env.server"]
        if episodes_root:
            argv += ["--episodes-root", str(Path(episodes_root).resolve())]
        self.grader_log = str(Path(grader_log).resolve()) if grader_log else None
        if self.grader_log:
            argv += ["--grader-log", self.grader_log]
        env = dict(os.environ)
        if salt is not None:
            env["AUDIT_GRADER_SALT"] = salt
        self._timeout = response_timeout_s
        self._channel = Channel(argv, cwd=str(ROOT), env=env)
        self._closed = False
        # Брошенная без close() среда останавливает свой процесс (EOF -> эпизод закрыт как abandoned, журнал
        # дописан; не дольше 2 с, затем kill). При выходе интерпретатора всех останавливает env/protocol.py.
        self._finalizer = weakref.finalize(self, stop_process, self._channel.proc)
        self._finalizer.atexit = False

    def reset(self, seed: Any = None, config: dict | None = None) -> tuple[dict, dict]:
        if isinstance(config, dict) and isinstance(config.get("episode_dir"), (str, os.PathLike)):
            # путь жюри — относительно ЕГО рабочего каталога; процесс оценщика работает из корня репозитория
            raw = os.fspath(config["episode_dir"])
            config = {**config, "episode_dir": os.path.abspath(raw) if raw.strip() else raw}
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
        self._finalizer.detach()
        try:
            self._channel.request("close", {}, timeout=30.0)
        except Exception:  # noqa: BLE001
            pass
        self._channel.close(timeout=5.0)

    def __enter__(self) -> "AuditEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
