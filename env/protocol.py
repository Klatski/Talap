"""JSON-lines поверх stdin/stdout: один JSON-объект на строку, ASCII, sort_keys.

Запрос:  {"id": n, "method": "...", "params": {...}}
Ответ:   {"id": n, "ok": true, "result": ...} | {"id": n, "ok": false, "error": {"code", "message"}}
Используется на обеих границах: агент <-> среда и среда <-> оценщик.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from typing import Any, BinaryIO

from .jsonsafe import loads_text

MAX_LINE_BYTES = 4 * 1024 * 1024  # ответ list_records на 50 строк — десятки КБ; эпизод целиком (load) — до ~3 МБ


class ProtocolError(RuntimeError):
    pass


def dumps(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n"


def loads(line: bytes) -> Any:
    """Сверхдлинные целые -> маркер (env/jsonsafe.py), NaN/Infinity -> ValueError (bad_request)."""
    return loads_text(line.decode("utf-8"))


def read_line(stream: BinaryIO, limit: int = MAX_LINE_BYTES) -> tuple[bytes | None, bool]:
    """(строка, too_large). None — конец потока. Слишком длинная строка вычитывается до конца и отбрасывается."""
    line = stream.readline(limit + 1)
    if not line:
        return None, False
    if len(line) > limit and not line.endswith(b"\n"):
        while True:
            chunk = stream.readline(65536)
            if not chunk or chunk.endswith(b"\n"):
                break
        return b"", True
    return line, False


def write_message(stream: BinaryIO, obj: Any) -> None:
    stream.write(dumps(obj))
    stream.flush()


def serve(handler, stdin: BinaryIO, stdout: BinaryIO, on_eof=None) -> None:
    """Цикл сервера: handler(method, params) -> result; исключение с .code -> ответ-ошибка."""
    while True:
        line, too_large = read_line(stdin)
        if line is None:
            break
        if too_large:
            write_message(stdout, {"id": None, "ok": False, "error": {"code": "request_too_large", "message": "request line is too large"}})
            continue
        if not line.strip():
            continue
        try:
            req = loads(line)
            if not isinstance(req, dict) or not isinstance(req.get("method"), str):
                raise ValueError("bad envelope")
            params = req.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("bad params")
        except (ValueError, UnicodeDecodeError, RecursionError):
            write_message(stdout, {"id": None, "ok": False, "error": {"code": "bad_request", "message": "request is not a valid JSON envelope"}})
            continue
        rid = req.get("id")
        try:
            result = handler(req["method"], params)
            reply = {"id": rid, "ok": True, "result": result}
        except StopServer as stop:
            write_message(stdout, {"id": rid, "ok": True, "result": stop.result})
            break
        except Exception as exc:  # noqa: BLE001 — любой сбой превращается в ограниченную ошибку
            code = getattr(exc, "code", "internal_error")
            message = getattr(exc, "public_message", None) or "request failed"
            reply = {"id": rid, "ok": False, "error": {"code": code, "message": message}}
        try:
            write_message(stdout, reply)
        except (TypeError, ValueError):
            write_message(stdout, {"id": rid, "ok": False, "error": {"code": "internal_error", "message": "response is not JSON"}})
    if on_eof is not None:
        on_eof()


class StopServer(Exception):
    def __init__(self, result: Any = None) -> None:
        super().__init__("stop")
        self.result = result


class RemoteError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.public_message = message


class Channel:
    """Клиентская сторона: дочерний процесс + поток-читатель + ожидание ответа с тайм-аутом."""

    def __init__(self, argv: list[str], cwd: str, env: dict | None = None) -> None:
        self.proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=0,
        )
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        stream = self.proc.stdout
        while True:
            line, too_large = read_line(stream)
            if line is None:
                self._queue.put(None)
                return
            if too_large or not line.strip():
                continue
            try:
                self._queue.put(loads(line))
            except ValueError:
                continue

    def request(self, method: str, params: dict, timeout: float = 120.0) -> Any:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            try:
                write_message(self.proc.stdin, {"id": rid, "method": method, "params": params})
            except (BrokenPipeError, OSError) as exc:
                raise ProtocolError("peer process is not running") from exc
            while True:
                try:
                    msg = self._queue.get(timeout=timeout)
                except queue.Empty as exc:
                    raise ProtocolError(f"no response to {method} within {timeout} s") from exc
                if msg is None:
                    raise ProtocolError("peer process exited")
                if msg.get("id") != rid:
                    continue
                if msg.get("ok"):
                    return msg.get("result")
                err = msg.get("error") or {}
                raise RemoteError(str(err.get("code", "error")), str(err.get("message", "")))

    def close(self, timeout: float = 10.0) -> None:
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._reader.join(timeout=timeout)
        try:
            if self.proc.stdout:
                self.proc.stdout.close()
        except OSError:
            pass
