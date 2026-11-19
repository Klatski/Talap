"""Единая команда проверки: python check.py   (в контейнере: docker run --network none audit-env)

Этапы (каждый печатает PASS/FAIL, код возврата 0 только если всё прошло):
  1. окружение: Python 3.11, нет сторонних зависимостей;
  2. тесты: tests/generator, tests/env, tests/grader (unittest);
  3. данные: check-splits (побайтовая воспроизводимость 21 открытого эпизода), схемы и
     эталон == независимый вывод A по открытым записям;
  4. базовые политики (random, reconcile, flag_all, oracle + демо-аудитор) в режиме eval через
     изолированные процессы на train+dev -> reports/baselines.{md,json}; ожидания: oracle F1=1 и
     награда 1 на каждом эпизоде, «пометить всё» <= -0.9, простая сверка F1 < 0.5;
  5. траектории: пересборка trajectories/trajectories.jsonl побайтово + сканер утечек A;
  6. имитация закрытой проверки: 20 новых эпизодов с солью (5 чистых, 5 взаимодействующих) + 20
     пар «перестановка + смена ID»; все политики через изолированную среду; oracle = 1 везде,
     награды пар совпадают -> reports/holdout.md.
Флаги: --quick (без этапа 6), --skip-tests, --update (перезаписать траектории и отчёты).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

TRAJ_POLICIES = ("random", "reconcile", "flag_all", "auditor")
EVAL_POLICIES = ("random", "reconcile", "flag_all", "oracle", "auditor")


class Stage:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, float, str]] = []

    def run(self, name: str, fn) -> bool:
        print(f"\n=== {name}", flush=True)
        t0 = time.perf_counter()
        try:
            ok, note = fn()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            ok, note = False, f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0
        self.results.append((name, ok, dt, note))
        print(f"--- {'PASS' if ok else 'FAIL'} {name} ({dt:.1f} s) {note}", flush=True)
        return ok


def stage_environment():
    v = sys.version_info
    note = f"Python {platform.python_version()} on {platform.system()}"
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    third_party = [ln for ln in req.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return (v.major, v.minor) == (3, 11) and not third_party, note + ("; deps: none" if not third_party else f"; deps: {third_party}")


def stage_tests():
    ok_all, notes = True, []
    for folder, pattern in (("tests/generator", "test_gen_*.py"), ("tests/env", "test_env_*.py"), ("tests/grader", "test_grader_*.py")):
        stream = io.StringIO()
        suite = unittest.defaultTestLoader.discover(str(ROOT / folder), pattern=pattern, top_level_dir=str(ROOT / folder))
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
        ok = result.wasSuccessful()
        ok_all &= ok
        notes.append(f"{folder}: {result.testsRun} run, {len(result.failures)} failed, {len(result.errors)} errors, {len(result.skipped)} skipped")
        if not ok:
            print(stream.getvalue()[-6000:])
        sys.path[:] = [p for p in sys.path if not p.endswith(folder.replace("/", os.sep))]
    print("\n".join(notes))
    return ok_all, "; ".join(n.split(":")[0] + ":" + n.split(":")[1].split(",")[0] for n in notes)


def stage_data():
    from generator.audit import compare_with_truth, parse_public
    from generator.fileio import read_episode_dir
    from generator.schema_check import validate_episode_texts
    from generator.splits import check_splits

    problems = check_splits()
    count = 0
    for d in sorted((ROOT / "episodes").glob("*/ep-*")):
        public, private = read_episode_dir(d)
        problems += [f"{d.name}: {e}" for e in validate_episode_texts(public, private)]
        problems += [f"{d.name}: {p}" for p in compare_with_truth(parse_public(public), json.loads(private["truth.json"]))]
        count += 1
    for p in problems[:20]:
        print("  ", p)
    return not problems, f"{count} episodes byte-reproducible, schema-valid, reference derivable"


def _run(policies, splits, mode, root=None, trajectories=None, salt=None):
    from env.runner import episode_plan, run_policy

    plan = episode_plan(splits, root)
    results = []
    fh = open(trajectories, "w", encoding="utf-8", newline="") if trajectories else None
    try:
        for name in policies:
            results += run_policy(name, plan, mode, episodes_root=root, trajectory_fh=fh, salt=salt)
    finally:
        if fh:
            fh.close()
    return results


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def stage_baselines(update: bool):
    from env.constants import ENV_VERSION
    from env.runner import markdown_report, summarize
    from grader.version import GRADER_VERSION

    results = _run(EVAL_POLICIES, ("train", "dev"), "eval")
    table = summarize(results)
    meta = {"env_version": ENV_VERSION, "grader_version": GRADER_VERSION, "mode": "eval", "splits": ["train", "dev"],
            "episodes": len(results) // len(EVAL_POLICIES), "isolated": True, "policy_seed": 0}
    md = markdown_report(table, "Baseline diagnostics (train + dev, eval mode)", meta)
    print(md)
    problems = []
    for r in results:
        g = r["grader"]
        if r["policy"] == "oracle" and (g["f1"] != 1.0 or g["reward"] != 1.0):
            problems.append(f"oracle not perfect on {r['episode_id']}")
        if r["policy"] == "auditor" and g["f1"] != 1.0:
            problems.append(f"auditor F1 {g['f1']} on {r['episode_id']}")
        if r["seconds"] > 60:
            problems.append(f"{r['policy']} {r['episode_id']} took {r['seconds']} s")
    if table["flag_all"]["return"] > -0.9:
        problems.append("flag_all is not punished")
    if table["reconcile"]["f1"] >= 0.5:
        problems.append("naive reconcile finds errors without evidence analysis (leak?)")
    if update or not (ROOT / "reports" / "baselines.md").is_file():
        _write(ROOT / "reports" / "baselines.md", md)
        per_episode = [{**{k: v for k, v in r.items() if k not in ("final_info", "grader")},
                        "grader": {k: v for k, v in r["grader"].items() if k != "diagnostics"},
                        "diagnostics": {k: v for k, v in r["grader"]["diagnostics"].items() if k not in ("per_finding", "missed")}}
                       for r in results]
        _write(ROOT / "reports" / "baselines.json", json.dumps({"meta": meta, "summary": table, "episodes": per_episode},
                                                              indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    for p in problems:
        print("  ", p)
    s = table
    return not problems, (f"oracle F1 {s['oracle']['f1']:.2f}/ret {s['oracle']['return']:.2f}; auditor ret {s['auditor']['return']:.3f}; "
                          f"reconcile F1 {s['reconcile']['f1']:.3f}; flag_all ret {s['flag_all']['return']:.2f}; random ret {s['random']['return']:.3f}")


def stage_trajectories(update: bool):
    from generator.leakscan import scan_trajectories

    target = ROOT / "trajectories" / "trajectories.jsonl"
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "trajectories.jsonl"
        _run(TRAJ_POLICIES, ("train",), "train", trajectories=fresh)
        data = fresh.read_bytes()
        same = target.is_file() and target.read_bytes() == data
        if update or not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            same = True
        leaks = scan_trajectories(fresh, ROOT / "episodes")
    lines = data.count(b"\n")
    required = {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}
    first = json.loads(data.split(b"\n", 1)[0])
    for p in leaks[:20]:
        print("  LEAK", p)
    if not same:
        print("  trajectories.jsonl differs from a fresh run (run: python check.py --update)")
    ok = same and not leaks and required <= set(first)
    return ok, f"{lines} lines, byte-identical to repo: {same}, leak scan: {'OK' if not leaks else len(leaks)}"


def stage_holdout(update: bool, salt: str):
    from env.runner import markdown_report, summarize
    from generator.holdout import build_holdout

    with tempfile.TemporaryDirectory(prefix="holdout-") as tmp:
        root = Path(tmp)
        report = build_holdout(salt, 20, root)
        kinds = {e["episode_id"]: e["kind"] for e in report["episodes"]}
        pair_of = {e["episode_id"]: e["pair_episode_id"] for e in report["episodes"]}
        results = {}
        for split in ("episodes", "pairs"):
            results[split] = _run(EVAL_POLICIES, (split,), "eval", root=root)
    problems = [f"generator self-check failed: {e['episode_id']}" for e in report["episodes"] if e["problems"]]
    by = {(r["policy"], r["episode_id"]): r for split in results for r in results[split]}
    for r in results["episodes"] + results["pairs"]:
        if r["policy"] == "oracle" and r["grader"]["reward"] != 1.0:
            problems.append(f"oracle reward {r['grader']['reward']} on {r['episode_id']}")
    for ep, pair in pair_of.items():
        for policy in ("oracle", "auditor", "reconcile", "flag_all"):
            a, b = by[(policy, ep)]["grader"]["reward"], by[(policy, pair)]["grader"]["reward"]
            if a != b:
                problems.append(f"pair reward differs for {policy}: {a} vs {b}")
    table = summarize(results["episodes"])
    pairs_table = summarize(results["pairs"])
    meta = {"env_version": "", "grader_version": "", "mode": "eval", "splits": ["holdout (salted, new configs)"],
            "episodes": 20, "isolated": True}
    from env.constants import ENV_VERSION
    from grader.version import GRADER_VERSION

    meta.update(env_version=ENV_VERSION, grader_version=GRADER_VERSION)
    md = markdown_report(table, "Closed-check emulation: 20 salted episodes (5 clean, 5 interacting, 10 mixed)", meta)
    md += "\n## Paired episodes (rows permuted, all IDs renamed)\n\n"
    md += "| policy | return (original) | return (pair) | equal per pair |\n|---|---|---|---|\n"
    for policy in EVAL_POLICIES:
        eq = sum(1 for ep, pair in pair_of.items() if by[(policy, ep)]["grader"]["reward"] == by[(policy, pair)]["grader"]["reward"])
        md += f"| {policy} | {table[policy]['return']:.4f} | {pairs_table[policy]['return']:.4f} | {eq}/20 |\n"
    md += f"\nEpisode kinds: {sum(1 for k in kinds.values() if k == 'clean')} clean, " \
          f"{sum(1 for k in kinds.values() if k == 'interacting')} interacting, {sum(1 for k in kinds.values() if k == 'mixed')} mixed. " \
          "Salt and seeds are not published; this is an emulation of the jury's closed run, not the real one.\n"
    print(md)
    if update or not (ROOT / "reports" / "holdout.md").is_file():
        _write(ROOT / "reports" / "holdout.md", md)
    for p in problems[:20]:
        print("  ", p)
    return not problems, f"40 episodes, oracle=1 everywhere: {not any('oracle' in p for p in problems)}, pairs equal: {not any('pair' in p for p in problems)}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python check.py")
    parser.add_argument("--quick", action="store_true", help="skip the closed-check emulation")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--update", action="store_true", help="rewrite trajectories and reports from this run")
    parser.add_argument("--salt", default=os.environ.get("CHECK_SALT", "check-emulation-salt"))
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    st = Stage()
    st.run("1. environment", stage_environment)
    if not args.skip_tests:
        st.run("2. unit tests (generator, env, grader)", stage_tests)
    st.run("3. episodes: reproducibility, schemas, reference", stage_data)
    st.run("4. baseline policies through isolated env (eval)", lambda: stage_baselines(args.update))
    st.run("5. trajectories: reproducible + leak scan", lambda: stage_trajectories(args.update))
    if not args.quick:
        st.run("6. closed-check emulation (holdout + pairs)", lambda: stage_holdout(args.update, args.salt))
    print("\n=== SUMMARY")
    for name, ok, dt, note in st.results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {dt:6.1f} s  {note}")
    ok = all(r[1] for r in st.results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.perf_counter() - t0:.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
