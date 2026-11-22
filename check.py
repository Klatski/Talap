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
     награды пар совпадают -> reports/holdout.{md,json};
  7. карточка жюри: раздел 10 кейса в коде (tests/jury/scorecard.py) на закрытых эпизодах этапа 6 —
     20 проверок контракта, 20 эпизодов, 20 проверок награды, 20 парных проверок -> reports/jury_scorecard.md.
Снимки (golden): результаты этапов 4 и 6 по каждому эпизоду (награда, P/R/F1, TP/FP/FN, шаги, причины
FP; без времени) и карточка этапа 7 сравниваются с закоммиченными reports/*; расхождение = FAIL.
Флаги: --quick (без этапов 6–7), --skip-tests, --update (перезаписать траектории и отчёты-снимки),
--salt (своя соль имитации; снимки этапов 6–7 сравниваются только для соли по умолчанию).
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
DEFAULT_SALT = "check-emulation-salt"
VOLATILE = frozenset({"seconds", "elapsed_s", "environment", "code_version"})  # время, машина, хэш кода — не часть снимка
UPDATE_HINT = "if the change is intended: python check.py --update and commit reports/"


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
    for folder, pattern in (("tests/generator", "test_gen_*.py"), ("tests/env", "test_env_*.py"), ("tests/grader", "test_grader_*.py"),
                            ("tests/jury", "test_jury_*.py")):
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


def _stored(results: list[dict]) -> list[dict]:
    """Результаты прогона в том виде, в каком они лежат в reports/*.json."""
    return [{**{k: v for k, v in r.items() if k not in ("final_info", "grader", "sandbox")},
             "grader": {k: v for k, v in r["grader"].items() if k != "diagnostics"},
             "diagnostics": {k: v for k, v in r["grader"]["diagnostics"].items() if k not in ("per_finding", "missed")}}
            for r in results]


def _stable(obj):
    if isinstance(obj, dict):
        return {k: _stable(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_stable(v) for v in obj]
    return obj


def _golden_diff(path: Path, meta: dict, episodes: list[dict]) -> list[str]:
    """Сравнить прогон со снимком (без времени); [] — совпадает."""
    if not path.is_file():
        return [f"{path.relative_to(ROOT).as_posix()} is missing ({UPDATE_HINT})"]
    old = json.loads(path.read_text(encoding="utf-8"))
    out = []
    if _stable(old.get("meta")) != _stable(meta):
        out.append(f"meta differs: {old.get('meta')} vs {meta}")
    before = {(e["policy"], e["episode_id"]): _stable(e) for e in old.get("episodes", [])}
    now = {(e["policy"], e["episode_id"]): _stable(e) for e in episodes}
    out += [f"{p} {ep}: in the snapshot, not in this run" for p, ep in sorted(before.keys() - now.keys())]
    out += [f"{p} {ep}: new in this run" for p, ep in sorted(now.keys() - before.keys())]
    for key in sorted(before.keys() & now.keys()):
        a, b = before[key], now[key]
        if a != b:
            flat_a = {**a.get("grader", {}), **a.get("diagnostics", {}), **{k: v for k, v in a.items() if not isinstance(v, dict)}}
            flat_b = {**b.get("grader", {}), **b.get("diagnostics", {}), **{k: v for k, v in b.items() if not isinstance(v, dict)}}
            changed = sorted(k for k in flat_a.keys() | flat_b.keys() if flat_a.get(k) != flat_b.get(k))
            out.append(f"{key[0]} {key[1]}: " + ", ".join(f"{k} {flat_a.get(k)!r}->{flat_b.get(k)!r}" for k in changed[:4]))
    return out


def stage_baselines(update: bool):
    from env.constants import ENV_VERSION
    from env.runner import markdown_report, summarize
    from grader.version import GRADER_VERSION

    results = _run(EVAL_POLICIES, ("train", "dev"), "eval")
    table = summarize(results)
    meta = {"env_version": ENV_VERSION, "grader_version": GRADER_VERSION, "mode": "eval", "splits": ["train", "dev"],
            "episodes": len(results) // len(EVAL_POLICIES), "isolated": True, "policy_seed": 0,
            "policy_sandbox": "separate restricted process (env/sandbox.py); oracle grader-side"}
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
    per_episode = _stored(results)
    snapshot = ROOT / "reports" / "baselines.json"
    if update or not snapshot.is_file():
        _write(ROOT / "reports" / "baselines.md", md)
        _write(snapshot, json.dumps({"meta": meta, "summary": table, "episodes": per_episode},
                                    indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        golden = "snapshot written"
    else:
        diff = _golden_diff(snapshot, meta, per_episode)
        problems += [f"golden: {d}" for d in diff[:15]] + ([f"golden: {len(diff)} runs differ from reports/baselines.json ({UPDATE_HINT})"] if diff else [])
        golden = "= snapshot" if not diff else f"{len(diff)} runs differ from snapshot"
    for p in problems:
        print("  ", p)
    s = table
    return not problems, (f"oracle F1 {s['oracle']['f1']:.2f}/ret {s['oracle']['return']:.2f}; auditor ret {s['auditor']['return']:.3f}; "
                          f"reconcile F1 {s['reconcile']['f1']:.3f}; flag_all ret {s['flag_all']['return']:.2f}; random ret {s['random']['return']:.3f}; "
                          f"{len(results)} runs {golden}")


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


def stage_holdout(update: bool, salt: str, root: Path, state: dict):
    """root — общий временный каталог этапов 6–7; state получает root и награды для этапа 7."""
    from env.runner import markdown_report, summarize
    from generator.holdout import build_holdout

    report = build_holdout(salt, 20, root)
    kinds = {e["episode_id"]: e["kind"] for e in report["episodes"]}
    pair_of = {e["episode_id"]: e["pair_episode_id"] for e in report["episodes"]}
    results = {}
    for split in ("episodes", "pairs"):
        results[split] = _run(EVAL_POLICIES, (split,), "eval", root=root)
    state["root"] = root
    state["rewards"] = {(r["policy"], r["episode_id"]): r["grader"]["reward"] for split in results for r in results[split]}
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
            "episodes": 20, "isolated": True, "policy_sandbox": "separate restricted process (env/sandbox.py); oracle grader-side"}
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
    per_episode = _stored(results["episodes"] + results["pairs"])
    snapshot = ROOT / "reports" / "holdout.json"
    meta = {**meta, "salt": "default" if salt == DEFAULT_SALT else "custom"}
    if salt != DEFAULT_SALT:
        golden = "custom salt: snapshot not compared"
    elif update or not snapshot.is_file():
        _write(ROOT / "reports" / "holdout.md", md)
        _write(snapshot, json.dumps({"meta": meta, "summary": {"episodes": table, "pairs": pairs_table}, "episodes": per_episode},
                                    indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        golden = "snapshot written"
    else:
        diff = _golden_diff(snapshot, meta, per_episode)
        problems += [f"golden: {d}" for d in diff[:15]] + ([f"golden: {len(diff)} runs differ from reports/holdout.json ({UPDATE_HINT})"] if diff else [])
        golden = "= snapshot" if not diff else f"{len(diff)} runs differ from snapshot"
    for p in problems[:20]:
        print("  ", p)
    return not problems, (f"40 episodes, oracle=1 everywhere: {not any('oracle' in p for p in problems)}, "
                          f"pairs equal: {not any('pair' in p for p in problems)}; {len(per_episode)} runs {golden}")


def stage_jury(update: bool, salt: str, state: dict):
    import importlib.util

    if "root" not in state:
        return False, "closed episodes were not built (stage 6 failed)"
    spec = importlib.util.spec_from_file_location("jury_scorecard", ROOT / "tests" / "jury" / "scorecard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    card = mod.run_scorecard(state["root"], salt, state.get("rewards"))
    md = mod.markdown(card)
    problems = [f"{c.name}: {c.note}" for block in card["blocks"].values() for c in block if not c.ok]
    for key, title, weight in mod.BLOCKS:
        s = card["summary"][key]
        print(f"  {key:9s} {s['passed']:2d}/{s['of']}  {s['points']:6.2f} / {weight}")
    print(f"  engineering (self-check) {card['engineering_points']}/10;  total {card['total']:.2f}/100  ({card['seconds']} s)")
    path = ROOT / "reports" / "jury_scorecard.md"
    # снимок — четыре блока проверок; раздел самопроверки инженерии зависит от текста документации и не сравнивается
    checks_only = lambda text: text.split(mod.ENGINEERING_HEADING, 1)[0]  # noqa: E731
    if salt != DEFAULT_SALT:
        golden = "custom salt: snapshot not compared"
    elif update or not path.is_file():
        _write(path, md)
        golden = "written to reports/jury_scorecard.md"
    elif checks_only(path.read_text(encoding="utf-8")) != checks_only(md):
        problems.append(f"golden: reports/jury_scorecard.md differs from this run ({UPDATE_HINT})")
        golden = "differs from snapshot"
    else:
        golden = "= snapshot"
    for p in problems[:20]:
        print("  ", p)
    s = card["summary"]
    return not problems, (f"contract {s['contract']['passed']}/20, data {s['data']['passed']}/20, reward {s['reward']['passed']}/20, "
                          f"pairs {s['pairs']['passed']}/20 -> {card['total']:.2f}/100 (emulation); {golden}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python check.py")
    parser.add_argument("--quick", action="store_true", help="skip the closed-check emulation and the jury scorecard")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--update", action="store_true", help="rewrite trajectories and report snapshots from this run")
    parser.add_argument("--salt", default=os.environ.get("CHECK_SALT", DEFAULT_SALT))
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    st = Stage()
    st.run("1. environment", stage_environment)
    if not args.skip_tests:
        st.run("2. unit tests (generator, env, grader, jury)", stage_tests)
    st.run("3. episodes: reproducibility, schemas, reference", stage_data)
    st.run("4. baseline policies through isolated env (eval)", lambda: stage_baselines(args.update))
    st.run("5. trajectories: reproducible + leak scan", lambda: stage_trajectories(args.update))
    if not args.quick:
        with tempfile.TemporaryDirectory(prefix="holdout-") as tmp:
            state: dict = {}
            st.run("6. closed-check emulation (holdout + pairs)", lambda: stage_holdout(args.update, args.salt, Path(tmp), state))
            st.run("7. jury scorecard (case section 10, 4 x 20 checks)", lambda: stage_jury(args.update, args.salt, state))
    print("\n=== SUMMARY")
    for name, ok, dt, note in st.results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:55s} {dt:6.1f} s  {note}")
    ok = all(r[1] for r in st.results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} in {time.perf_counter() - t0:.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
