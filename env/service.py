"""EnvService: reset/step поверх EnvCore + транспорт к оценщику + сторожевой таймер.

Транспорты оценщика:
  SubprocessGrader — отдельный процесс `python -m grader.service` (официальный режим);
  InProcessGrader  — тот же GraderSession в этом процессе (только тесты/отладка, без изоляции).
Ключи config, которые разбирает среда: mode (train|eval), max_steps (1..100), time_limit_s (0..60].
Остальные ключи уходят оценщику (выбор эпизода); их ошибки не раскрывают путей и эталона.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .constants import MAX_STEPS, MODES, TIME_LIMIT_S
from .core import EnvCore
from .protocol import Channel, ProtocolError, RemoteError
from .jsonsafe import is_json_value
from .records import EpisodeDataError, load_public

ROOT = Path(__file__).resolve().parent.parent
ENV_KEYS = ("mode", "max_steps", "time_limit_s")


class ResetError(ValueError):
    """Ошибка reset (неверный config или эпизод не найден). Сообщение безопасно для показа."""

    code = "reset_failed"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.public_message = message


# ---------------------------------------------------------------------- транспорты
class InProcessGrader:
    """НЕ изолирован: эталон в том же процессе. Только для модульных тестов и отладки."""

    def __init__(self, episodes_root: str | Path | None = None, salt: str = "", log_path: str | Path | None = None) -> None:
        from grader.session import GraderSession

        self.session = GraderSession(episodes_root, salt, log_path)

    def load(self, seed: Any, config: dict, mode: str) -> dict:
        from grader.episodes import EpisodeLoadError

        try:
            return json.loads(json.dumps(self.session.load(seed, config, mode)))
        except EpisodeLoadError as exc:
            raise ResetError(str(exc)) from None

    def finish(self, payload: Any, inspection_steps: int, steps_used: int, end_reason: str, elapsed_s: float,
               malformed: str | None = None) -> dict:
        payload = json.loads(json.dumps(payload))  # как через границу процесса
        return json.loads(json.dumps(self.session.finish(payload, inspection_steps, steps_used, end_reason, elapsed_s,
                                                         malformed=malformed)))

    def close(self) -> None:
        self.session.close()


class SubprocessGrader:
    def __init__(self, episodes_root: str | Path | None = None, log_path: str | Path | None = None, salt: str | None = None) -> None:
        argv = [sys.executable, "-m", "grader.service"]
        if episodes_root:
            argv += ["--episodes-root", str(episodes_root)]
        if log_path:
            argv += ["--log", str(log_path)]
        env = dict(os.environ)
        if salt is not None:
            env["AUDIT_GRADER_SALT"] = salt
        self.channel = Channel(argv, cwd=str(ROOT), env=env)
        self._lock = threading.Lock()

    def load(self, seed: Any, config: dict, mode: str) -> dict:
        with self._lock:
            try:
                return self.channel.request("load", {"seed": seed, "config": config, "mode": mode})
            except RemoteError as exc:
                raise ResetError(exc.public_message) from None

    def finish(self, payload: Any, inspection_steps: int, steps_used: int, end_reason: str, elapsed_s: float,
               malformed: str | None = None) -> dict:
        with self._lock:
            return self.channel.request("finish", {
                "payload": payload, "inspection_steps": inspection_steps, "steps_used": steps_used,
                "end_reason": end_reason, "elapsed_s": elapsed_s, "malformed": malformed,
            })

    def close(self) -> None:
        with self._lock:
            try:
                self.channel.request("close", {}, timeout=30.0)
            except (ProtocolError, RemoteError):
                pass
            self.channel.close()


# ---------------------------------------------------------------------- сервис
def split_config(config: Any) -> tuple[dict, dict]:
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ResetError("config must be an object or null")
    env_part = {k: config[k] for k in ENV_KEYS if k in config}
    grader_part = {k: v for k, v in config.items() if k not in ENV_KEYS}
    mode = env_part.get("mode", "train")
    if mode not in MODES:
        raise ResetError("mode must be 'train' or 'eval'")
    max_steps = env_part.get("max_steps", MAX_STEPS)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEPS:
        raise ResetError(f"max_steps must be an integer 1..{MAX_STEPS}")
    limit = env_part.get("time_limit_s", TIME_LIMIT_S)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 < limit <= TIME_LIMIT_S:
        raise ResetError(f"time_limit_s must be a number in (0, {TIME_LIMIT_S:g}]")
    return {"mode": mode, "max_steps": max_steps, "time_limit_s": float(limit)}, grader_part


class EnvService:
    def __init__(self, grader) -> None:
        self.grader = grader
        self.core = EnvCore(self._on_finish)
        self._timer: threading.Timer | None = None
        self._started = 0.0
        self._closed = False

    def _on_finish(self, payload: Any, inspection_steps: int, steps_used: int, end_reason: str,
                   malformed: str | None = None) -> dict:
        elapsed = round(time.monotonic() - self._started, 3)
        self._cancel_timer()
        return self.grader.finish(payload, inspection_steps, steps_used, end_reason, elapsed, malformed)

    # ------------------------------------------------------------------ сторож
    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()

    def _arm_timer(self, delay: float) -> None:
        timer = threading.Timer(max(0.0, delay) + 0.01, self._watchdog)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _watchdog(self) -> None:
        with self.core.lock:
            if self._closed or not self.core.active:
                return
            if not self.core.expire_if_due():
                deadline = self.core.deadline
                if deadline is not None:
                    self._arm_timer(deadline - time.monotonic())

    # ------------------------------------------------------------------ API
    def reset(self, seed: Any = None, config: Any = None) -> tuple[dict, dict]:
        if self._closed:
            raise ResetError("environment is closed")
        if not is_json_value({"seed": seed, "config": config}):  # до abandon: текущий эпизод не трогаем
            raise ResetError("seed and config must be JSON values")
        env_cfg, grader_cfg = split_config(config)
        with self.core.lock:
            self._cancel_timer()
            self.core.abandon()
            loaded = self.grader.load(seed, grader_cfg, env_cfg["mode"])
            try:
                data = load_public(loaded["public"])
            except (EpisodeDataError, KeyError, TypeError) as exc:
                raise ResetError("episode public data does not follow the contract") from exc
            self._started = time.monotonic()
            obs, info = self.core.reset(data, env_cfg["mode"], env_cfg["max_steps"], env_cfg["time_limit_s"])
            self._arm_timer(env_cfg["time_limit_s"])
            return obs, info

    def step(self, action: Any, unsendable: Any = None) -> tuple[dict, float, bool, bool, dict]:
        return self.core.step(action, unsendable)

    def close(self) -> None:
        if self._closed:
            return
        with self.core.lock:
            self._cancel_timer()
            self.core.abandon()
            self._closed = True
        self.grader.close()
