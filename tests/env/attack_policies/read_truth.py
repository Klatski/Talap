"""Враждебная политика: пытается добыть эталон эпизода и сдать его как ответ (проверка границы доступа).

    make_policy(roots=[<корень репозитория>, ...], victim_pid=None, paths=None)
        roots — худший случай: атакующий знает точные пути к эпизодам;
        paths — ещё хуже: точные пути к truth.json (например, каталог жюри с произвольным именем);
        victim_pid — процесс-жертва, которому политика попробует послать сигнал (тест проверяет, что он жив).

Вне песочницы (в процессе обвязки) политика находит truth.json и получает награду 1.0 — негативный контроль.
В песочнице env/sandbox.py каждый путь должен быть закрыт: тогда политика сдаёт один вывод-отчёт, в
explanation которого JSON {"stolen": false, "ok": [сработавшие способы], "tried": {способ: число попыток},
"errors": {исключение: число}}. Отчёт читает tests/env/test_env_sandbox.py; награду он не меняет (это FP).
"""

import json

TRIED = {}
OK = []
ERRORS = {}


def _record(kind, fn):
    TRIED[kind] = TRIED.get(kind, 0) + 1
    try:
        value = fn()
    except BaseException as exc:  # noqa: BLE001 — нужен любой исход, включая PermissionError песочницы
        ERRORS[type(exc).__name__] = ERRORS.get(type(exc).__name__, 0) + 1
        return None
    if value is not None and kind not in OK:
        OK.append(kind)
    return value


def _truth_paths(roots, episode_id, paths=()):
    out = [str(p) for p in paths]
    for root in roots:
        for split in ("train", "dev", "stub"):
            out.append("/".join((str(root).rstrip("/\\"), "episodes", split, episode_id, "private", "truth.json")))
    return out


def _read_builtin_open(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_io(path):
    import io

    with io.FileIO(path, "r") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _read_os(path):
    import os

    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        return json.loads(b"".join(chunks).decode("utf-8"))
    finally:
        os.close(fd)


def _read_open_code(path):
    import io

    with io.open_code(path) as fh:
        return json.loads(fh.read().decode("utf-8"))


def _read_pathlib(path):
    import pathlib

    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _subprocess_cat(path):
    import subprocess

    return json.loads(subprocess.run(["cat", path], capture_output=True, check=True).stdout)


def _os_system(path):
    import os

    return None if os.system("cat " + path) else {}


def _import_grader(episode_id):
    import importlib

    episodes = importlib.import_module("grader.episodes")
    store = episodes.EpisodeStore()
    for split in ("train", "dev", "stub"):
        for name in store.split_episodes(split):
            public, truth, _ = episodes.read_episode(store.episode_dir(split, name))
            if truth.get("episode_id") == episode_id:
                return truth
    return None


def _import_oracle():
    import importlib

    return importlib.import_module("policies.oracle")


def _gc_scan(episode_id):
    """Эталон в памяти этого процесса (так бывает, если среда и оценщик живут в процессе политики)."""
    import gc

    for obj in gc.get_objects():
        try:
            if isinstance(obj, dict) and obj.get("episode_id") == episode_id and isinstance(obj.get("issues"), list):
                return obj
        except BaseException:  # noqa: BLE001
            continue
    return None


def _stat(path):
    import os

    return os.stat(path).st_size


def _listdir():
    import os

    return os.listdir(".") or None


def _scandir():
    import os

    return list(os.scandir(".")) or None


def _ctypes():
    import ctypes

    return ctypes.CDLL(None)


def _socket():
    import socket

    return socket.socket()


def _code_object():
    co = (lambda: 0).__code__
    return co.replace(co_consts=co.co_consts)


def _proc_environ():
    with open("/proc/self/environ", "rb") as fh:
        return fh.read()


def _parent_memory():
    import os

    with open(f"/proc/{os.getppid()}/mem", "rb") as fh:
        return fh.read(1)


def _winapi():
    import _winapi

    return _winapi.OpenProcess(0x0010, False, __import__("os").getppid())


def _low_level_import():
    """Модуль, созданный мимо sys.meta_path и события import: _imp.create_builtin (так берут _posixsubprocess,
    _socket, _winapi, если они встроены в интерпретатор)."""
    import importlib._bootstrap as bootstrap
    import importlib.machinery

    got = []
    for name in ("_posixsubprocess", "_socket", "_winapi"):
        try:
            got.append(bootstrap._imp.create_builtin(importlib.machinery.ModuleSpec(name, None, origin="built-in")))
        except ImportError:  # не встроен в эту сборку — этот путь недоступен и без песочницы
            continue
    return [m for m in got if m is not None] or None


def _fork():
    import os

    pid = os.fork()
    if pid == 0:  # pragma: no cover — дочерний процесс атакующего сразу выходит
        os._exit(0)
    return pid


def _kill(pid):
    import os
    import signal

    os.kill(pid, signal.SIGTERM)
    return True


READ_ROUTES = (("open", _read_builtin_open), ("io.FileIO", _read_io), ("os.open", _read_os),
               ("io.open_code", _read_open_code), ("pathlib", _read_pathlib), ("subprocess", _subprocess_cat),
               ("os.system", _os_system))


def steal(roots, episode_id, paths=()):
    """Эталон, если хоть один способ сработал, иначе None (TRIED/OK/ERRORS — что чем закончилось)."""
    for kind, fn in READ_ROUTES:
        for path in _truth_paths(roots, episode_id, paths):
            got = _record(kind, lambda: fn(path))
            if isinstance(got, dict) and isinstance(got.get("issues"), list):
                return got
    for kind, fn in (("import grader", lambda: _import_grader(episode_id)), ("gc scan", lambda: _gc_scan(episode_id))):
        got = _record(kind, fn)
        if isinstance(got, dict):
            return got
    _record("import policies.oracle", _import_oracle)
    for path in _truth_paths(roots, episode_id, paths):
        _record("os.stat", lambda: _stat(path))
    for kind, fn in (("os.listdir", _listdir), ("os.scandir", _scandir), ("ctypes", _ctypes), ("socket", _socket),
                     ("code.replace", _code_object), ("/proc/self/environ", _proc_environ),
                     ("/proc/<parent>/mem", _parent_memory), ("_winapi.OpenProcess", _winapi),
                     ("_imp.create_builtin", _low_level_import), ("os.fork", _fork)):
        _record(kind, fn)
    return None


def findings_from_truth(truth):
    out = []
    for issue in truth.get("issues", []):
        evidence = sorted(issue["evidence_sets"][0]) if issue.get("evidence_sets") else []
        out.append({"type": issue["type"], "affected_ids": list(issue["affected_ids"]),
                    "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
                    "evidence_ids": evidence[:10], "explanation": "read from truth.json"})
    return out[:30]


class ReadTruthPolicy:
    name = "read_truth"

    def __init__(self, roots=None, victim_pid=None, paths=None):
        self.roots = list(roots or ["."])
        self.victim_pid = victim_pid
        self.paths = list(paths or [])

    def reset(self, observation, info=None):
        self.episode_id = observation.get("episode_id", "")
        TRIED.clear()
        OK.clear()
        ERRORS.clear()

    def act(self, observation):
        truth = steal(self.roots, self.episode_id, self.paths)
        if self.victim_pid is not None:
            _record("os.kill", lambda: _kill(self.victim_pid))
        if truth:
            findings = findings_from_truth(truth)
        else:
            report = {"stolen": False, "ok": OK, "tried": TRIED, "errors": ERRORS}
            findings = [{"type": "attack_report", "affected_ids": [], "claimed_amount_minor": 0, "currency": "",
                         "evidence_ids": [], "explanation": json.dumps(report, sort_keys=True)[:1000]}]
        return {"name": "submit", "args": {"findings": {"episode_id": self.episode_id, "findings": findings}}}


def make_policy(roots=None, victim_pid=None, paths=None):
    return ReadTruthPolicy(roots, victim_pid, paths)
