"""Процесс оценщика: python -m grader.service [--episodes-root DIR] [--log FILE]

Запускается процессом среды (env/server.py); агент к нему доступа не имеет.
Соль для генерируемых эпизодов — из переменной окружения AUDIT_GRADER_SALT (задаёт обвязка).
Методы JSON-lines: load {seed, config, mode} -> {public, mode};
                   finish {payload, inspection_steps, steps_used, end_reason, elapsed_s} -> {reward, info};
                   close -> журнал записан, процесс завершается.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.protocol import StopServer, serve  # noqa: E402
from grader.episodes import EpisodeLoadError  # noqa: E402
from grader.session import GraderError, GraderSession  # noqa: E402


class _Fail(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grader.service")
    parser.add_argument("--episodes-root", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # случайный print не испортит протокол
    session = GraderSession(args.episodes_root, os.environ.get("AUDIT_GRADER_SALT", ""), args.log)

    def handler(method: str, params: dict):
        try:
            if method == "load":
                return session.load(params.get("seed"), params.get("config") or {}, params.get("mode", "train"))
            if method == "finish":
                return session.finish(
                    params.get("payload"), params["inspection_steps"], params["steps_used"],
                    params["end_reason"], params.get("elapsed_s"), malformed=params.get("malformed"),
                )
            if method == "close":
                session.close()
                raise StopServer({"closed": True})
        except EpisodeLoadError as exc:
            raise _Fail("load_failed", str(exc)) from None
        except GraderError as exc:
            raise _Fail("grader_error", str(exc)) from None
        except (KeyError, TypeError, ValueError):
            raise _Fail("bad_params", "invalid parameters") from None
        raise _Fail("unknown_method", "unknown method")

    serve(handler, stdin, stdout, on_eof=session.close)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
