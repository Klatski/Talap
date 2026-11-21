"""Процесс песочницы политики. Запускает env/sandbox.py; напрямую не импортируется.

    python -I -S -B -X utf8 env/sandbox_child.py

Только stdlib: проектные пакеты (env, grader, generator) этому процессу недоступны и не нужны.
Протокол — тот же JSON-lines, что env/protocol.py: {"id", "method", "params"} -> {"id", "ok", "result" | "error"}.
  load  {sources: {модуль: [это_пакет, текст]}, entry: {module, factory, args, kwargs}, preimport: [...]}
        -> {isolation: {...}} — код политики приходит текстом от родителя, файлов политика не открывает;
  new   {}                          -> {}     новый экземпляр политики (свежий на каждый эпизод);
  reset {observation, info}         -> {}
  act   {observation}               -> {action_json: str} | {unsendable: true, submit: bool}
  close {}                          -> процесс завершается.
Порядок запуска (до первой строки кода политики):
  1. заранее импортировать модули stdlib, нужные типичной политике (потом файлы открыть будет нельзя);
  2. Linux: seccomp-фильтр (необратим, наследуется потоками и детьми): open/openat/exec/socket/ptrace/
     kill/unlink/rename/... -> EPERM — граница ядра, её не обойти и подменой байт-кода;
  3. аудит-хук без состояния (необратим): open, listdir/scandir, процессы, сокеты, ctypes, mmap, _winapi,
     субинтерпретаторы, байт-код (code.__new__, marshal), импорт проектных и опасных модулей -> PermissionError;
     запрещённые модули выгружаются из sys.modules, а первый искатель в sys.meta_path отказывает им и при
     importlib.import_module;
  4. загрузить политику из памяти и обслуживать запросы.
"""

import json
import os
import sys

PREIMPORT = (
    "__future__", "abc", "array", "base64", "binascii", "bisect", "calendar", "cmath", "codecs", "collections",
    "collections.abc", "contextlib", "copy", "csv", "dataclasses", "datetime", "_strptime", "decimal", "difflib",
    "enum", "fractions", "functools", "hashlib", "heapq", "hmac", "importlib", "importlib.abc", "importlib.machinery",
    "importlib.util", "io", "itertools", "json", "keyword", "linecache", "math", "numbers", "operator", "pprint",
    "random", "re", "secrets", "statistics", "string", "struct", "textwrap", "time", "traceback", "types", "typing",
    "unicodedata", "warnings", "weakref", "zlib",
)

# Импорт этих модулей (верхний уровень имени) запрещён: проектные пакеты и всё, что даёт процессы, сеть,
# машинный код, память других процессов, субинтерпретаторы без хуков или интерпретатор Tcl.
BLOCKED_IMPORTS = frozenset({
    "grader", "generator", "env", "tests", "tools", "check", "judge_adapter",
    "ctypes", "_ctypes", "subprocess", "_posixsubprocess", "socket", "_socket", "ssl", "_ssl", "select", "selectors",
    "asyncio", "multiprocessing", "_multiprocessing", "concurrent", "mmap", "sqlite3", "_sqlite3", "dbm", "_dbm",
    "_gdbm", "shelve", "winreg", "_winapi", "msvcrt", "_overlapped", "_testcapi", "_testinternalcapi",
    "_testmultiphase", "_testbuffer", "_xxsubinterpreters", "_xxinterpchannels", "_interpreters",
    "_interpchannels", "_interpqueues", "pty", "tty", "termios", "fcntl", "resource", "urllib", "http", "ftplib",
    "smtplib", "poplib", "imaplib", "nntplib", "telnetlib", "webbrowser", "shutil", "tempfile", "glob", "zipimport",
    "_tkinter", "tkinter", "turtle", "idlelib", "ensurepip", "venv", "pip", "runpy", "pdb", "faulthandler",
})
BLOCKED_SUBMODULES = frozenset({"policies.oracle"})  # oracle — только на стороне жюри (ему нужен эталон)

DENY_EVENTS = frozenset({
    "open", "builtins.input", "builtins.input/result", "code.__new__", "sys.addaudithook", "setopencodehook",
    "cpython.PyInterpreterState_New", "signal.pthread_kill", "marshal.load", "marshal.loads",
})
DENY_PREFIXES = (
    "os.", "subprocess.", "pty.", "socket.", "ssl.", "urllib.", "http.", "ftplib.", "smtplib.", "poplib.",
    "imaplib.", "nntplib.", "telnetlib.", "webbrowser.", "ctypes.", "_winapi.", "msvcrt.", "winreg.", "mmap.",
    "sqlite3.", "dbm.", "shutil.", "tempfile.", "glob.", "fcntl.", "resource.", "syslog.", "zipimport.",
    "ensurepip.", "_posixsubprocess",
)

# seccomp: номера системных вызовов, которые после блокировки возвращают EPERM (всё прочее разрешено).
# Плюс newfstatat разрешён только как fstat открытого дескриптора (флаг AT_EMPTY_PATH): stat/access по пути
# запрещены, чтобы политика не узнала даже размер или наличие файлов эпизода.
_SYSCALLS_X86_64 = (
    4, 6, 21, 269, 439, 332,                        # stat, lstat, access, faccessat, faccessat2, statx
    2, 257, 437, 85, 303, 304,                      # open, openat, openat2, creat, name_to_handle_at, open_by_handle_at
    59, 322, 134,                                   # execve, execveat, uselib
    41, 53, 42, 49, 50, 43, 288,                    # socket, socketpair, connect, bind, listen, accept, accept4
    101, 310, 311, 312,                             # ptrace, process_vm_readv/writev, kcmp
    62, 200, 234, 129, 297, 424, 434, 438,          # kill, tkill, tgkill, rt_(tg)sigqueueinfo, pidfd_*
    87, 263, 82, 264, 316, 83, 258, 84, 86, 265, 88, 266,  # unlink*, rename*, mkdir*, rmdir, link*, symlink*
    90, 268, 452, 92, 94, 260, 76, 133, 259, 132, 235, 280, 261,  # chmod*, chown*, truncate, mknod*, utime*
    188, 189, 190, 197, 198, 199, 89, 267, 254, 301, 163,  # xattr setters, readlink*, inotify/fanotify marks, acct
    161, 165, 166, 155, 167, 168, 272, 308, 321, 298, 323, 425, 426, 427,  # chroot, mount..., unshare, setns, bpf, ...
    250, 248, 249, 175, 313, 176, 246, 320, 169, 300, 428, 429, 430, 431, 432, 433, 442,
)
_SYSCALLS_AARCH64 = (
    48, 439, 291,                                   # faccessat, faccessat2, statx
    56, 437, 264, 265,
    221, 281,
    198, 199, 203, 200, 201, 202, 242,
    117, 270, 271, 272,
    129, 130, 131, 138, 240, 424, 434, 438,
    35, 38, 276, 34, 37, 36,
    53, 452, 54, 45, 33, 88,
    5, 6, 7, 14, 15, 16, 78, 27, 263, 89,
    51, 40, 39, 41, 224, 225, 97, 268, 280, 241, 282, 425, 426, 427,
    219, 217, 218, 105, 273, 106, 104, 294, 142, 262, 428, 429, 430, 431, 432, 433, 442,
)


def _preimport(names) -> None:
    import importlib

    for name in names:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — нет модуля на этой платформе: политика просто не сможет его взять
            pass


def _install_seccomp() -> str:
    """Seccomp-фильтр запрета (Linux, x86_64/aarch64). Возвращает 'active' или причину, почему его нет."""
    if not sys.platform.startswith("linux"):
        return "not linux"
    machine = os.uname().machine
    if machine == "x86_64":
        arch, numbers, x32, fstatat = 0xC000003E, _SYSCALLS_X86_64, True, 262
    elif machine in ("aarch64", "arm64"):
        arch, numbers, x32, fstatat = 0xC00000B7, _SYSCALLS_AARCH64, False, 79
    else:
        return f"unsupported architecture {machine}"
    try:
        import ctypes
    except ImportError:
        return "ctypes is not available"
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int

        class SockFilter(ctypes.Structure):
            _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]

        class SockFprog(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

        ld_abs, jeq, jge, jset, ret = 0x20, 0x15, 0x35, 0x45, 0x06
        allow, errno_eperm, kill = 0x7FFF0000, 0x00050000 | 1, 0x80000000
        prog = [(ld_abs, 0, 0, 4), (jeq, 1, 0, arch), (ret, 0, 0, kill), (ld_abs, 0, 0, 0)]
        if x32:
            prog += [(jge, 0, 1, 0x40000000), (ret, 0, 0, errno_eperm)]
        for nr in sorted(set(numbers)):
            prog += [(jeq, 0, 1, nr), (ret, 0, 0, errno_eperm)]
        # newfstatat(dirfd, path, buf, flags): флаги — args[3] (смещение 40 в seccomp_data); без AT_EMPTY_PATH -> EPERM
        prog += [(jeq, 0, 3, fstatat), (ld_abs, 0, 0, 40), (jset, 1, 0, 0x1000), (ret, 0, 0, errno_eperm)]
        prog.append((ret, 0, 0, allow))
        table = (SockFilter * len(prog))(*[SockFilter(*ins) for ins in prog])
        fprog = SockFprog(len(prog), table)
        if prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            return f"PR_SET_NO_NEW_PRIVS failed (errno {ctypes.get_errno()})"
        if prctl(22, 2, ctypes.addressof(fprog), 0, 0) != 0:  # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
            return f"seccomp filter rejected (errno {ctypes.get_errno()})"
        return "active"
    except Exception as exc:  # noqa: BLE001 — нет seccomp: остаётся аудит-хук, причина сообщается родителю
        return f"seccomp unavailable ({type(exc).__name__})"
    finally:
        for name in [m for m in sys.modules if m == "ctypes" or m.startswith("ctypes.") or m == "_ctypes"]:
            del sys.modules[name]


def _blocked_module(name: str) -> bool:
    return name.split(".", 1)[0] in BLOCKED_IMPORTS or any(name == b or name.startswith(b + ".") for b in BLOCKED_SUBMODULES)


def _audit_hook(event: str, args: tuple) -> None:
    """Без состояния: решение зависит только от события, поэтому политике нечего «переключить»."""
    if event in DENY_EVENTS or event.startswith(DENY_PREFIXES):
        raise PermissionError(f"sandbox: {event} is not allowed for a policy")
    if event == "import" and args and isinstance(args[0], str) and _blocked_module(args[0]):
        raise PermissionError(f"sandbox: import of {args[0]} is not allowed for a policy")


class _BlockedFinder:
    """Первый в sys.meta_path: запрещённые модули не грузятся и через importlib (событие import есть не везде)."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if _blocked_module(fullname):
            raise PermissionError(f"sandbox: import of {fullname} is not allowed for a policy")
        return None


def _drop_blocked_modules() -> None:
    """Запрещённые модули, оказавшиеся загруженными при старте, убираются из sys.modules."""
    for name in [m for m in sys.modules if _blocked_module(m)]:
        del sys.modules[name]


def _install_memory_importer(sources: dict) -> None:
    import importlib.abc
    import importlib.util

    class MemoryLoader(importlib.abc.Loader):
        def __init__(self, name: str, text: str) -> None:
            self.name, self.text = name, text

        def create_module(self, spec):
            return None

        def exec_module(self, module) -> None:
            exec(compile(self.text, f"<sandbox {self.name}>", "exec"), module.__dict__)  # noqa: S102 — код политики

    class MemoryFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in sources:
                return None
            is_package, text = sources[fullname]
            spec = importlib.util.spec_from_loader(fullname, MemoryLoader(fullname, text), is_package=bool(is_package))
            if is_package:
                spec.submodule_search_locations = []
            return spec

    sys.meta_path.insert(0, MemoryFinder())
    sys.meta_path.insert(0, _BlockedFinder())


def _reply(stream, rid, ok: bool, payload) -> None:
    msg = {"id": rid, "ok": ok, ("result" if ok else "error"): payload}
    stream.write(json.dumps(msg, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n")
    stream.flush()


def _encode_action(action) -> dict:
    try:
        return {"action_json": json.dumps(action, ensure_ascii=True, allow_nan=True)}
    except (TypeError, ValueError, RecursionError):
        is_submit = isinstance(action, dict) and (
            action.get("action") == "submit" if "action" in action else action.get("name") == "submit")
        return {"unsendable": True, "submit": bool(is_submit)}


def main() -> int:
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # print() политики не испортит протокол
    sys.set_int_max_str_digits(0)  # сверхдлинные целые в действии передаются как есть, решает среда
    state = {"policy": None, "factory": None}
    loaded = False
    while True:
        line = stdin.readline()
        if not line:
            return 0
        if not line.strip():
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            method, params = req.get("method"), req.get("params") or {}
            if method == "load" and not loaded:
                _preimport(PREIMPORT + tuple(params.get("preimport") or ()))
                seccomp = _install_seccomp()
                _drop_blocked_modules()
                sys.addaudithook(_audit_hook)
                _install_memory_importer({k: tuple(v) for k, v in params["sources"].items()})
                entry = params["entry"]
                module = sys.modules["importlib"].import_module(entry["module"])
                factory = getattr(module, entry.get("factory") or "make_policy")
                state["factory"] = lambda: factory(*entry.get("args", []), **entry.get("kwargs", {}))
                state["policy"] = state["factory"]()
                loaded = True
                _reply(stdout, rid, True, {"isolation": {
                    "process": "separate", "audit_hook": True, "seccomp": seccomp, "platform": sys.platform,
                    "python": sys.version.split()[0], "files_open": False}})
            elif not loaded:
                _reply(stdout, rid, False, {"code": "not_loaded", "message": "load the policy first"})
            elif method == "new":
                state["policy"] = state["factory"]()
                _reply(stdout, rid, True, {})
            elif method == "reset":
                state["policy"].reset(params.get("observation"), params.get("info"))
                _reply(stdout, rid, True, {})
            elif method == "act":
                _reply(stdout, rid, True, _encode_action(state["policy"].act(params.get("observation"))))
            elif method == "close":
                _reply(stdout, rid, True, {"closed": True})
                return 0
            else:
                _reply(stdout, rid, False, {"code": "unknown_method", "message": "unknown method"})
        except Exception as exc:  # noqa: BLE001 — сбой политики не роняет протокол: родитель решает, что делать
            _reply(stdout, rid, False, {"code": "policy_error", "message": f"{type(exc).__name__}: {exc}"[:500]})


if __name__ == "__main__":
    sys.exit(main())
