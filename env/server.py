"""Процесс среды: python -m env.server [--episodes-root DIR] [--grader-log FILE]

Читает запросы агента из stdin (JSON-lines), отвечает в stdout. Держит только открытые
записи текущего эпизода. Эталон живёт в дочернем процессе оценщика (grader.service),
к которому у агента нет канала. Соль для генерируемых эпизодов — AUDIT_GRADER_SALT
(переменная окружения обвязки; среда её не читает, а лишь наследует оценщику).

Методы: reset {seed, config} -> {observation, info}
        step {action} -> {observation, reward, terminated, truncated, info}
        close -> закрыть эпизод и оценщик (журнал оценщика записывается в этот момент)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.protocol import StopServer, serve  # noqa: E402
from env.service import EnvService, SubprocessGrader  # noqa: E402


class _Fail(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m env.server")
    parser.add_argument("--episodes-root", default=None)
    parser.add_argument("--grader-log", default=None)
    args = parser.parse_args(argv)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    service = EnvService(SubprocessGrader(args.episodes_root, args.grader_log))

    def handler(method: str, params: dict):
        if method == "reset":
            obs, info = service.reset(params.get("seed"), params.get("config"))
            return {"observation": obs, "info": info}
        if method == "step":
            if "action" not in params or set(params) - {"action", "unsendable"}:
                raise _Fail("bad_params", "step needs an action")
            obs, reward, terminated, truncated, info = service.step(params["action"], params.get("unsendable"))
            return {"observation": obs, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}
        if method == "close":
            service.close()
            raise StopServer({"closed": True})
        raise _Fail("unknown_method", "unknown method")

    serve(handler, stdin, stdout, on_eof=service.close)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
