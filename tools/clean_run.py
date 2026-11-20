"""Прогон единой команды проверки «с нуля», не в разогретой рабочей копии (ловит «работает только у меня»).

    python tools/clean_run.py                    # клон origin (то, что увидит жюри), HEAD ветки по умолчанию
    python tools/clean_run.py --source .         # клон локального репозитория (только закоммиченное)
    python tools/clean_run.py --ref 4cadd96 --keep

Что делает:
  1. свежий git clone во временный каталог с пробелом в пути;
  2. новый venv без pip и пакетов (интерпретатор — тот, которым запущен скрипт; check.py сам проверит 3.11);
  3. урезанное окружение: только PATH/системные переменные, свой пустой TEMP, без PYTHONPATH и т.п.;
  4. sitecustomize в venv (его загружает каждый процесс venv, включая процессы среды и оценщика):
     2 CPU (affinity), сеть запрещена audit-хуком (любая попытка = FAIL), пик памяти каждого процесса;
  5. python check.py (полный, с имитацией жюри), затем итог: время, процессы, сеть, память (оценка сверху
     одновременного объёма), не изменил ли прогон отслеживаемые файлы, не осталось ли мусора в TEMP.
Лимит ТЗ 4 ГБ RAM проверяется оценкой сверху; жёсткие лимиты --cpus 2 --memory 4g --network none даёт
контейнер (CI: .github/workflows/check.yml).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SITECUSTOMIZE = r'''# clean_run harness (not part of the project)
import atexit, os, sys, time
LOG = {log!r}

def _log(line):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{{time.time():.3f}} {{os.getpid()}} {{line}}\n")

_aff = "n/a"
try:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
        _aff = "ok"
    elif sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        _k = ctypes.WinDLL("kernel32")
        _k.GetCurrentProcess.restype = wintypes.HANDLE
        _k.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        _aff = "ok" if _k.SetProcessAffinityMask(_k.GetCurrentProcess(), 0b11) else "failed"
except Exception as exc:
    _aff = "failed:" + type(exc).__name__
_log("START affinity=" + _aff)

_BLOCKED = {{"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
             "socket.sendto", "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect"}}

def _hook(event, args):
    if event in _BLOCKED:
        _log("NETWORK " + event)
        raise PermissionError("network is disabled in the clean run")

sys.addaudithook(_hook)

def _peak_mb():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
                "peak_ws", "ws", "qpp_peak", "qpp", "qnp_peak", "qnp", "pagefile", "pagefile_peak")]
        k, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.peak_ws / 2**20
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 1024

atexit.register(lambda: _log(f"PEAK {{_peak_mb():.1f}}"))
'''


def _default_source() -> str:
    try:
        url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        url = ""
    return url or str(ROOT)


def _scrubbed_env(tmp: Path) -> dict[str, str]:
    if sys.platform == "win32":
        keep = {k: os.environ[k] for k in ("SYSTEMROOT", "WINDIR", "COMSPEC", "NUMBER_OF_PROCESSORS") if k in os.environ}
        system = os.environ.get("SYSTEMROOT", r"C:\Windows")
        return {**keep, "PATH": f"{system}\\System32;{system}", "TEMP": str(tmp), "TMP": str(tmp)}
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(tmp), "TMPDIR": str(tmp), "LANG": "C.UTF-8"}


def _analyse(log: Path, run_end: float) -> dict:
    procs: dict[str, dict] = {}
    network = []
    for line in log.read_text(encoding="utf-8").splitlines() if log.is_file() else []:
        t, pid, rest = line.split(" ", 2)
        p = procs.setdefault(pid, {"start": float(t), "end": run_end, "peak": None, "affinity": None})
        if rest.startswith("START"):
            p["start"], p["affinity"] = float(t), rest.split("=", 1)[1]
        elif rest.startswith("PEAK"):
            p["end"], p["peak"] = float(t), float(rest.split()[1])
        elif rest.startswith("NETWORK"):
            network.append(rest)
    events = sorted({p["start"] for p in procs.values()})
    concurrent = max((sum(p["peak"] or 0 for p in procs.values() if p["start"] <= t <= p["end"]) for t in events), default=0)
    most = max((sum(1 for p in procs.values() if p["start"] <= t <= p["end"]) for t in events), default=0)
    return {"processes": len(procs), "affinity_failed": sum(1 for p in procs.values() if p["affinity"] != "ok"),
            "no_peak": sum(1 for p in procs.values() if p["peak"] is None), "network": network,
            "max_process_mb": max((p["peak"] or 0 for p in procs.values()), default=0),
            "concurrent_mb_upper": concurrent, "max_alive": most}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/clean_run.py")
    parser.add_argument("--source", default=None, help="git URL or path to clone (default: origin, else this repo)")
    parser.add_argument("--ref", default=None, help="commit/branch to check out (default: the clone's HEAD)")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    parser.add_argument("--quick", action="store_true", help="pass --quick to check.py")
    args = parser.parse_args(argv)
    source = args.source or _default_source()
    base = Path(tempfile.mkdtemp(prefix="clean run "))
    repo, venv, tmp, log = base / "repo", base / "venv", base / "tmp", base / "procs.log"
    tmp.mkdir()
    try:
        print(f"clone {source.split('@')[-1]} -> {repo}", flush=True)
        subprocess.run(["git", "clone", "-q", source, str(repo)], check=True, timeout=600)
        if args.ref:
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", args.ref], check=True, timeout=120)
        head = subprocess.run(["git", "-C", str(repo), "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=300)
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        purelib = subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        Path(purelib, "sitecustomize.py").write_text(SITECUSTOMIZE.format(log=str(log)), encoding="utf-8")
        print(f"commit {head}; venv {py}; network blocked, 2 CPUs, clean TEMP", flush=True)
        t0 = time.time()
        cmd = [str(py), "check.py"] + (["--quick"] if args.quick else [])
        rc = subprocess.run(cmd, cwd=str(repo), env=_scrubbed_env(tmp), timeout=3600).returncode
        wall = time.time() - t0
        stats = _analyse(log, time.time())
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout.split("\n")
        dirty = [d for d in dirty if d.strip()]
        leftovers = sorted(p.name for p in tmp.iterdir())
        print("\n=== CLEAN RUN")
        print(f"check.py exit code {rc}, wall {wall:.0f} s, commit {head}")
        print(f"processes {stats['processes']} (max {stats['max_alive']} alive at once), 2-CPU affinity failed in "
              f"{stats['affinity_failed']}, without exit record {stats['no_peak']}")
        print(f"network attempts: {len(stats['network'])} {stats['network'][:3]}")
        print(f"peak memory: largest process {stats['max_process_mb']:.0f} MB; at once <= {stats['concurrent_mb_upper']:.0f} MB "
              f"(upper bound; limit 4096 MB)")
        print(f"tracked files changed by the run: {len(dirty)} {dirty[:5]}")
        print(f"left in TEMP: {len(leftovers)} {leftovers[:5]}")
        ok = rc == 0 and not stats["network"] and not dirty and stats["concurrent_mb_upper"] < 4096
        print(("CLEAN RUN PASSED" if ok else "CLEAN RUN FAILED") + ("" if not leftovers else " (warning: TEMP not empty)"))
        return 0 if ok else 1
    finally:
        if args.keep:
            print(f"kept: {base}")
        else:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
