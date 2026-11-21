"""Версия кода для журнала оценщика: sha256 по исходникам команды, посчитанный при запуске процесса.

Журнал оценщика не коммитится и побайтово не сравнивается, поэтому хэш туда пишется без замкнутого круга
(в закоммиченные снимки reports/ и траектории он не попадает: check.py считает поле `code_version` машинным).

Что входит (CODE_PARTS): все *.py в env/, grader/, generator/, policies/, tests/, tools/; check.py,
judge_adapter.py, requirements.txt; schemas/*.json и все файлы generator/configs/. Файлы жюри, положенные
рядом (своя политика в корне, каталоги эпизодов), хэш не меняют. Байты берутся с LF вместо CRLF, чтобы
checkout на Windows с autocrlf давал тот же хэш, что и на Linux. Хэш = sha256 по строкам
"<путь>\\0<sha256 файла>\\n" в порядке путей. git_commit — HEAD из .git, если каталог есть (в контейнере
его нет: .dockerignore), иначе null.

    python -m grader.codeversion        # то же значение, что в журнале: {"sha256", "files", "git_commit"}
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_DIRS = ("env", "grader", "generator", "policies", "tests", "tools")
CODE_FILES = ("check.py", "judge_adapter.py", "requirements.txt")
DATA_GLOBS = ("schemas/*.json", "generator/configs/*")
_SHA_RE = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")


def code_files(root: Path = ROOT) -> list[str]:
    """Пути (относительно корня, через /) всех файлов, входящих в хэш кода, в порядке сортировки."""
    found: set[str] = set()
    for d in CODE_DIRS:
        for p in (root / d).rglob("*.py"):
            if "__pycache__" not in p.parts and p.is_file():
                found.add(p.relative_to(root).as_posix())
    for name in CODE_FILES:
        if (root / name).is_file():
            found.add(name)
    for pattern in DATA_GLOBS:
        for p in root.glob(pattern):
            if p.is_file():
                found.add(p.relative_to(root).as_posix())
    return sorted(found)


def code_digest(root: Path = ROOT) -> tuple[str, int]:
    h = hashlib.sha256()
    files = code_files(root)
    for rel in files:
        data = (root / rel).read_bytes().replace(b"\r\n", b"\n")
        h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256(data).hexdigest().encode("ascii") + b"\n")
    return h.hexdigest(), len(files)


def git_commit(root: Path = ROOT) -> str | None:
    """HEAD без запуска git: .git/HEAD -> ref -> refs/... или packed-refs. Нет .git или непонятный формат -> None."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if _SHA_RE.fullmatch(head):
            return head
        if not head.startswith("ref: "):
            return None
        ref = head[5:].strip()
        loose = git / ref
        if loose.is_file():
            sha = loose.read_text(encoding="utf-8").strip()
            return sha if _SHA_RE.fullmatch(sha) else None
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref and _SHA_RE.fullmatch(parts[0]):
                    return parts[0]
    except (OSError, UnicodeDecodeError):
        return None
    return None


@functools.lru_cache(maxsize=None)
def code_version() -> dict:
    """Считается один раз на процесс (оценщик, grader.judge): {"sha256", "files", "git_commit"}."""
    try:
        digest, count = code_digest(ROOT)
    except OSError:  # pragma: no cover — нечитаемый файл: версия неизвестна, журнал всё равно пишется
        digest, count = None, 0
    return {"sha256": digest, "files": count, "git_commit": git_commit(ROOT)}


if __name__ == "__main__":
    print(json.dumps(code_version(), sort_keys=True))
