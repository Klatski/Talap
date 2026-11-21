"""Песочница политики: политика работает в ОТДЕЛЬНОМ ОГРАНИЧЕННОМ процессе (граница доступа жюри).

    from env.sandbox import SandboxedPolicy
    policy = SandboxedPolicy.builtin("reconcile")            # базовая политика из policies/ (кроме oracle)
    policy = SandboxedPolicy.from_file("my_policy.py")       # любая политика: файл с make_policy() / классом
    policy.reset(obs, info); action = policy.act(obs); policy.close()

Политика получает только наблюдения и возвращает действия; всё остальное — у обвязки (этого процесса):
reset, пути, seed, config, среда и оценщик. Дочерний процесс (env/sandbox_child.py):
  * запускается как `python -I -S -B` с пустым окружением во временном каталоге (там только копия
    env/sandbox_child.py), поэтому не знает даже путь к репозиторию;
  * получает КОД политики текстом по каналу и грузит его из памяти — к файлам проекта и эпизодов не обращается;
  * до первой строки кода политики ставит необратимые ограничения: аудит-хук (open, listdir, процессы, сеть,
    ctypes, mmap, субинтерпретаторы, байт-код, импорт grader/generator/env -> PermissionError) и на Linux
    seccomp-фильтр ядра (open/openat/exec/socket/ptrace/kill/unlink/... -> EPERM);
  * сообщает уровень изоляции (`policy.isolation`): seccomp "active" на Linux, иначе причина.
Ответ политики ждётся не дольше act(obs, timeout_s=...) (обвязка передаёт остаток 60 с эпизода; без аргумента —
до 60 + POLICY_GRACE_S с от последнего reset() политики): зависший или
упавший процесс политики убивается (PolicyError), следующий reset() поднимает новый. Эпизод при этом завершает
сама среда по своему таймеру (truncated, пустые findings) — обвязка не выбирает действия за политику.
Oracle в песочницу не попадает: ему нужен эталон, он строится на стороне жюри/оценщика (policies/oracle.py).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .constants import POLICY_GRACE_S, TIME_LIMIT_S
from .jsonsafe import loads_text
from .protocol import Channel, ProtocolError, RemoteError

ROOT = Path(__file__).resolve().parent.parent
CHILD = Path(__file__).resolve().parent / "sandbox_child.py"
POLICIES_DIR = ROOT / "policies"
NOT_IN_SANDBOX = frozenset({"oracle.py"})  # политика с доступом к эталону — только на стороне жюри


class PolicyError(RuntimeError):
    """Политика упала или нарушила протокол песочницы (текст — от процесса политики, без данных среды)."""


def package_sources(package_dir: Path, package: str, exclude: frozenset = frozenset()) -> dict[str, list]:
    """{модуль: [это_пакет, текст]} для всех .py пакета (рекурсивно)."""
    sources: dict[str, list] = {}
    for path in sorted(package_dir.rglob("*.py")):
        if path.name in exclude or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(package_dir).with_suffix("")
        parts = [package, *rel.parts]
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        with open(path, encoding="utf-8") as fh:
            sources[".".join(parts)] = [is_package, fh.read()]
    return sources


def _interpreter() -> str:
    """Базовый интерпретатор (не лаунчер venv): песочнице нужен только stdlib, site-packages ей не нужны."""
    base = getattr(sys, "_base_executable", None)
    return base if base and os.path.isfile(base) else sys.executable


def _child_env() -> dict[str, str]:
    if sys.platform == "win32":
        return {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR") if k in os.environ}
    return {"LANG": "C.UTF-8"}


def _unsendable_action(is_submit: bool) -> dict:
    """Действие, которое нельзя передать как JSON: среде уходит такое же непредставимое значение (NaN),
    поэтому итог тот же, что без песочницы (submit -> неверная структура, иначе invalid_action)."""
    if is_submit:
        return {"action": "submit", "findings": float("nan")}
    return {"action": "list_records", "table": float("nan")}


class SandboxedPolicy:
    """Политика в отдельном ограниченном процессе. Интерфейс как у policies/base.py: reset(obs, info), act(obs).

    fresh_per_episode=True — новый экземпляр политики на каждый reset (так работают базовые политики в
    раннере); False — один экземпляр на весь прогон (обучаемый агент копит опыт между эпизодами).
    """

    def __init__(self, sources: dict[str, list], entry: dict, *, name: str = "policy", fresh_per_episode: bool = True,
                 timeout_s: float = 120.0, preimport: tuple[str, ...] = ()) -> None:
        self.name = name
        self.fresh_per_episode = fresh_per_episode
        self._timeout = timeout_s
        self._load = {"sources": sources, "entry": entry, "preimport": list(preimport)}
        self._closed = False
        self._channel = None
        self._workdir = None
        self.restarts = 0
        self._deadline: float | None = None
        self._spawn()

    def _spawn(self) -> None:
        self._episodes = 0
        self._workdir = tempfile.mkdtemp(prefix="policy-sandbox-")
        # Копия скрипта во временном каталоге: процесс политики не узнаёт даже путь к репозиторию и эпизодам.
        shutil.copyfile(CHILD, os.path.join(self._workdir, CHILD.name))
        self._channel = Channel([_interpreter(), "-I", "-S", "-B", "-X", "utf8", CHILD.name], cwd=self._workdir,
                                env=_child_env())
        try:
            out = self._call("load", self._load)
        except BaseException:
            self.close()
            raise
        self.isolation: dict = out["isolation"]

    # ------------------------------------------------------------------ фабрики
    @classmethod
    def builtin(cls, name: str, seed: int = 0, **kw) -> "SandboxedPolicy":
        """Базовая политика из policies/ (random, reconcile, flag_all, auditor). oracle — нельзя: ей нужен эталон."""
        if name == "oracle":
            raise ValueError("oracle needs the reference: it runs on the jury/grader side, not in the sandbox")
        sources = package_sources(POLICIES_DIR, "policies", NOT_IN_SANDBOX)
        entry = {"module": "policies", "factory": "make_policy", "args": [name], "kwargs": {"seed": seed}}
        return cls(sources, entry, name=name, **kw)

    @classmethod
    def from_file(cls, path: str | os.PathLike, factory: str = "make_policy", kwargs: dict | None = None,
                  name: str | None = None, **kw) -> "SandboxedPolicy":
        """Политика из одного файла: функция `factory(**kwargs)` (по умолчанию make_policy) или класс с этим именем.
        Пакет политики — каталог с __init__.py: путь к каталогу, модуль грузится как пакет `user_policy`.
        Рядом доступна библиотека `policies` (base.py и базовые политики, без oracle): `from policies.base import …`."""
        path = Path(path)
        sources = package_sources(POLICIES_DIR, "policies", NOT_IN_SANDBOX)
        if path.is_dir():
            sources.update(package_sources(path, "user_policy"))
        else:
            with open(path, encoding="utf-8") as fh:
                sources["user_policy"] = [False, fh.read()]
        entry = {"module": "user_policy", "factory": factory, "args": [], "kwargs": dict(kwargs or {})}
        return cls(sources, entry, name=name or path.stem, **kw)

    # ------------------------------------------------------------------ интерфейс политики
    def reset(self, observation: dict, info: dict | None = None, timeout_s: float | None = None) -> None:
        if self._closed:
            raise PolicyError("sandbox is closed")
        if self._channel is None:  # прошлый процесс завис или упал и убит — новый эпизод в новом процессе
            self.restarts += 1
            self._spawn()
        self._deadline = time.monotonic() + TIME_LIMIT_S + POLICY_GRACE_S  # лимит эпизода включает время политики
        if timeout_s is None:
            timeout_s = self._deadline - time.monotonic()
        if self.fresh_per_episode and self._episodes:
            self._call("new", {}, timeout_s)
        self._episodes += 1
        self._call("reset", {"observation": observation, "info": info or {}}, timeout_s)

    def act(self, observation: dict, timeout_s: float | None = None) -> Any:
        """timeout_s — сколько ждать ответа (обвязка передаёт остаток времени эпизода); без него — до конца
        60 с эпизода (+POLICY_GRACE_S) от последнего reset(), но не дольше timeout_s песочницы."""
        if timeout_s is None and self._deadline is not None:
            timeout_s = self._deadline - time.monotonic()
        out = self._call("act", {"observation": observation}, timeout_s)
        if out.get("unsendable"):
            return _unsendable_action(bool(out.get("submit")))
        return loads_text(out["action_json"])

    def close(self) -> None:
        self._closed = True
        self._kill()

    def _kill(self, hard: bool = False) -> None:
        """hard — процесс завис или сломал протокол: убить сразу, не дожидаясь вежливого close."""
        channel, self._channel = getattr(self, "_channel", None), None
        if channel is not None:
            if hard:
                try:
                    channel.proc.kill()
                except OSError:
                    pass
            else:
                try:
                    channel.request("close", {}, timeout=10.0)
                except (ProtocolError, RemoteError, OSError):
                    pass
            channel.close()
        workdir = getattr(self, "_workdir", None)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
            self._workdir = None

    def __enter__(self) -> "SandboxedPolicy":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ внутреннее
    def _call(self, method: str, params: dict, timeout_s: float | None = None) -> dict:
        if self._channel is None:
            raise PolicyError("sandbox is closed" if self._closed else "sandbox process was stopped; call reset()")
        wait = self._timeout if timeout_s is None else max(0.05, min(self._timeout, float(timeout_s)))
        try:
            out = self._channel.request(method, params, timeout=wait)
        except RemoteError as exc:  # исключение в коде политики: процесс жив, протокол цел
            raise PolicyError(f"policy {self.name}: {exc.public_message}") from None
        except ProtocolError as exc:  # завис (нет ответа) или умер: процесс убивается, reset() поднимет новый
            self._kill(hard=True)
            raise PolicyError(f"policy {self.name}: sandbox process failed ({exc})") from None
        return out if isinstance(out, dict) else {}
