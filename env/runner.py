"""Прогон политик через среду: траектории, журнал оценщика, таблица диагностики.

    python -m env.runner --policies random,reconcile,flag_all,oracle,auditor --splits train,dev --mode eval \
        --report-json reports/baselines.json --report-md reports/baselines.md
    python -m env.runner --policies random,reconcile,flag_all,auditor --splits train --mode train \
        --trajectories trajectories/trajectories.jsonl
    python -m env.runner --policy-file my_policy.py --splits dev --mode closed   # своя политика в песочнице

Каждая политика получает свою пару процессов среда+оценщик (AuditEnv) и свой процесс-песочницу
(env/sandbox.py): политика видит только наблюдения и не может открыть ни одного файла, импортировать
grader/generator или запустить процесс. Награды режима eval/closed политика не видит: раннер берёт их
из журнала оценщика ПОСЛЕ close() — «до конца прогона награда и диагностика остаются у оценщика».
Oracle создаётся здесь же из эталона (обвязка на стороне оценщика, как у жюри), поэтому его траектории
в общий trajectories.jsonl не пишутся.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.constants import ENV_VERSION, POLICY_GRACE_S  # noqa: E402
from grader.version import GRADER_VERSION  # noqa: E402

GRADER_SIDE_POLICIES = {"oracle"}


def _dump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def episode_plan(splits: Iterable[str], episodes_root: Path | None = None) -> list[tuple[str, int, str]]:
    from grader.episodes import EpisodeStore

    store = EpisodeStore(episodes_root)
    plan = []
    for split in splits:
        for i, name in enumerate(store.split_episodes(split)):
            plan.append((split, i, name))
    return plan


def load_truth(split: str, name: str, episodes_root: Path | None) -> dict:
    from grader.episodes import EpisodeStore, read_episode

    store = EpisodeStore(episodes_root)
    return read_episode(store.episode_dir(split, name))[1]


def run_episode(env, policy, seed: int, config: dict, on_step: Callable[[dict], None] | None = None) -> dict:
    """Один эпизод. Политика в песочнице ждётся не дольше остатка лимита времени (+POLICY_GRACE_S): зависшая или
    упавшая политика больше не спрашивается, а эпизод завершает сама среда — по таймеру (truncated, пустые findings)
    или при следующем reset/close (abandoned). Действия за политику обвязка не выбирает."""
    from env.constants import TIME_LIMIT_S
    from env.sandbox import PolicyError, SandboxedPolicy

    t0 = time.perf_counter()
    limit = config.get("time_limit_s", TIME_LIMIT_S) if isinstance(config, dict) else TIME_LIMIT_S
    obs, info = env.reset(seed=seed, config=config)
    deadline = time.monotonic() + float(limit) + POLICY_GRACE_S
    episode_id = obs.get("episode_id")
    if on_step:
        on_step({"step": 0, "action": None, "observation": obs, "reward": 0.0, "terminated": False,
                 "truncated": False, "info": info, "episode_id": episode_id})
    sandboxed = isinstance(policy, SandboxedPolicy)
    total, step, terminated, truncated, last_info, policy_error = 0.0, 0, False, False, {}, None
    try:
        if sandboxed:
            policy.reset(obs, info, timeout_s=deadline - time.monotonic())
        else:
            policy.reset(obs, info)
    except PolicyError as exc:
        policy_error = str(exc)
    while not (terminated or truncated or policy_error):
        try:
            action = policy.act(obs, timeout_s=deadline - time.monotonic()) if sandboxed else policy.act(obs)
        except PolicyError as exc:
            policy_error = str(exc)
            break
        obs, reward, terminated, truncated, last_info = env.step(action)
        step += 1
        total += reward
        if on_step:
            on_step({"step": step, "action": action, "observation": obs, "reward": reward, "terminated": terminated,
                     "truncated": truncated, "info": last_info, "episode_id": episode_id})
        if step > 1000:  # среда обязана завершить эпизод раньше; страховка от зацикливания
            raise RuntimeError("environment did not end the episode")
    out = {"episode_id": episode_id, "steps": step, "return_seen_by_policy": total, "terminated": terminated,
           "truncated": truncated, "seconds": round(time.perf_counter() - t0, 4), "final_info": last_info}
    if policy_error is not None:
        out["policy_error"] = policy_error
    return out


def load_policy_file(path: str | Path, kwargs: dict | None = None) -> Any:
    """Политика из файла В ПРОЦЕССЕ обвязки — только для отладки и негативного контроля: изоляции нет."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("user_policy_in_process", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_policy(**(kwargs or {}))


def run_policy(name: str, plan: list[tuple[str, int, str]], mode: str, *, episodes_root: Path | None = None,
               policy_seed: int = 0, trajectory_fh=None, isolated: bool = True, salt: str | None = None,
               sandboxed: bool | None = None, policy_file: str | Path | None = None,
               policy_kwargs: dict | None = None) -> list[dict]:
    """sandboxed (по умолчанию = isolated): политика в отдельном ограниченном процессе env/sandbox.py;
    policy_file — своя политика из файла (make_policy(**policy_kwargs)) вместо базовой `name`.
    oracle — всегда на стороне оценщика: ему нужен эталон, в песочницу он не попадает."""
    from env import AuditEnv, LocalAuditEnv
    from env.sandbox import SandboxedPolicy
    from policies import make_policy

    sandboxed = isolated if sandboxed is None else sandboxed
    grader_side = name in GRADER_SIDE_POLICIES and policy_file is None
    with tempfile.TemporaryDirectory(prefix="grader-log-") as tmp:
        log_path = Path(tmp) / "grader_log.jsonl"
        env = (AuditEnv(episodes_root, log_path, salt=salt) if isolated
               else LocalAuditEnv(episodes_root, log_path, salt=salt or ""))
        sandbox = None
        runs = []
        try:
            if sandboxed and not grader_side:
                sandbox = (SandboxedPolicy.from_file(policy_file, kwargs=policy_kwargs, name=name) if policy_file
                           else SandboxedPolicy.builtin(name, seed=policy_seed))
            for split, index, ep_name in plan:
                if grader_side:
                    policy = make_policy(name, seed=policy_seed, truth=load_truth(split, ep_name, episodes_root))
                elif sandbox is not None:
                    policy = sandbox  # свежий экземпляр политики на каждый reset — внутри песочницы
                elif policy_file is not None:
                    policy = load_policy_file(policy_file, policy_kwargs)
                else:
                    policy = make_policy(name, seed=policy_seed)
                lines: list[dict] = []
                writer = lines.append if trajectory_fh is not None and not grader_side else None
                res = run_episode(env, policy, index, {"split": split, "index": index, "mode": mode}, writer)
                for rec in lines:
                    trajectory_fh.write(_dump({**rec, "policy": name, "split": split, "mode": mode}) + "\n")
                runs.append({**res, "policy": name, "split": split, "index": index,
                             "sandbox": dict(sandbox.isolation) if sandbox is not None else None})
        finally:
            if sandbox is not None:
                sandbox.close()
            env.close()
        grader_records = []
        if log_path.is_file():
            with open(log_path, encoding="utf-8") as fh:
                grader_records = [json.loads(line) for line in fh if line.strip()]
    if len(grader_records) != len(runs):
        raise RuntimeError(f"{name}: grader logged {len(grader_records)} episodes, runner ran {len(runs)}")
    out = []
    for run, rec in zip(runs, grader_records):
        if rec["episode_id"] != run["episode_id"]:
            raise RuntimeError("grader log is out of order")
        if mode == "train" and abs(rec["reward"] - run["return_seen_by_policy"]) > 1e-12:
            raise RuntimeError("train reward returned to the policy differs from the grader log")
        if mode != "train" and run["return_seen_by_policy"] != 0.0:
            raise RuntimeError(f"{mode} reward leaked to the policy")
        out.append({**run, "grader": rec})
    return out


def summarize(results: list[dict]) -> dict:
    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    by_policy: dict[str, list[dict]] = {}
    for r in results:
        by_policy.setdefault(r["policy"], []).append(r)
    table = {}
    for policy, rows in by_policy.items():
        g = [r["grader"] for r in rows]
        tp = sum(x["tp"] or 0 for x in g)
        fp = sum(x["fp"] or 0 for x in g)
        fn = sum((x["fn"] if x["fn"] is not None else x["issues_total"]) for x in g)
        fp_reasons, fp_distractor, fn_types, fp_types = Counter(), Counter(), Counter(), Counter()
        for x in g:
            d = x.get("diagnostics") or {}
            fp_reasons.update(d.get("fp_reasons", {}))
            fp_distractor.update(d.get("fp_on_distractor", {}))
            fn_types.update(d.get("fn_by_type", {}))
            fp_types.update(d.get("fp_by_type", {}))
        clean = [x for x in g if x["issues_total"] == 0]
        table[policy] = {
            "episodes": len(rows),
            "precision": mean([x["precision"] for x in g]),
            "recall": mean([x["recall"] for x in g]),
            "f1": mean([x["f1"] for x in g]),
            "micro_f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 1.0,
            "return": mean([x["reward"] for x in g]),
            "return_experimental": mean([x["reward_experimental"] for x in g]),
            "return_clean_books": mean([x["reward"] for x in clean]),
            "steps": mean([x["steps_used"] for x in g]),
            "inspection_steps": mean([x["inspection_steps"] for x in g]),
            "seconds": mean([r["seconds"] for r in rows]),
            "max_seconds": round(max(r["seconds"] for r in rows), 4),
            "tp": tp, "fp": fp, "fn": fn,
            "truncated": sum(1 for x in g if x["truncated"]),
            "invalid": sum(1 for x in g if not x["valid"]),
            "fp_reasons": dict(sorted(fp_reasons.items())),
            "fp_by_type": dict(sorted(fp_types.items())),
            "fp_on_distractor": dict(sorted(fp_distractor.items())),
            "fn_by_type": dict(sorted(fn_types.items())),
        }
    return table


def markdown_report(table: dict, title: str, meta: dict) -> str:
    lines = [f"# {title}", "",
             f"env {meta['env_version']}, grader {meta['grader_version']}, mode `{meta['mode']}`, "
             f"splits {', '.join(meta['splits'])}, {meta['episodes']} episodes per policy, isolated processes: {meta['isolated']}"
             + (f", policy sandbox: {meta['policy_sandbox']}." if meta.get("policy_sandbox") else "."),
             "Values are means over episodes; `micro F1` pools TP/FP/FN over all episodes. "
             "`return` is the shared reward; `return_exp` is the experimental one (truncation = -1), shown separately. "
             "By the spec P and R are 0 on a clean book with an empty answer (zero denominators) while F1 = 1, "
             "so mean P/R stay below 1 even for the oracle.", "",
             "| policy | P | R | F1 | micro F1 | return | return_exp | return on clean | steps | time, s (mean / max) | TP | FP | FN | truncated | invalid |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, s in table.items():
        lines.append(f"| {p} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {s['micro_f1']:.3f} | {s['return']:.3f} | "
                     f"{s['return_experimental']:.3f} | {s['return_clean_books']:.3f} | {s['steps']:.1f} | {s['seconds']:.3f} / {s['max_seconds']:.3f} | "
                     f"{s['tp']} | {s['fp']} | {s['fn']} | {s['truncated']} | {s['invalid']} |")
    lines += ["", "## Failure analysis (measured)", "",
              "| policy | FP reasons | FP on correct distractors | FN by type |", "|---|---|---|---|"]
    for p, s in table.items():
        fmt = lambda d: ", ".join(f"{k}: {v}" for k, v in d.items()) or "-"  # noqa: E731
        lines.append(f"| {p} | {fmt(s['fp_reasons'])} | {fmt(s['fp_on_distractor'])} | {fmt(s['fn_by_type'])} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m env.runner")
    parser.add_argument("--policies", default="random,reconcile,flag_all,oracle,auditor")
    parser.add_argument("--splits", default="train,dev")
    parser.add_argument("--mode", choices=("train", "eval", "closed"), default="eval")
    parser.add_argument("--episodes-root", default=None)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--trajectories", default=None, help="write trajectories.jsonl here")
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--policy-file", default=None, help="run this policy file (make_policy()) in the sandbox")
    parser.add_argument("--policy-kwargs", default="{}", help="JSON object passed to make_policy() of --policy-file")
    parser.add_argument("--local", action="store_true", help="in-process env+grader (debug only, no isolation)")
    parser.add_argument("--no-sandbox", action="store_true", help="policies in this process (debug only, no isolation)")
    args = parser.parse_args(argv)

    root = Path(args.episodes_root).resolve() if args.episodes_root else None
    splits = [s for s in args.splits.split(",") if s]
    plan = episode_plan(splits, root)
    traj_fh = None
    if args.trajectories:
        Path(args.trajectories).parent.mkdir(parents=True, exist_ok=True)
        traj_fh = open(args.trajectories, "w", encoding="utf-8", newline="")
    results = []
    sandboxed = not (args.local or args.no_sandbox)
    jobs = ([(Path(args.policy_file).stem, args.policy_file)] if args.policy_file
            else [(p, None) for p in args.policies.split(",") if p])
    try:
        for name, policy_file in jobs:
            results += run_policy(name, plan, args.mode, episodes_root=root, policy_seed=args.policy_seed,
                                  trajectory_fh=traj_fh, isolated=not args.local, sandboxed=sandboxed,
                                  policy_file=policy_file, policy_kwargs=json.loads(args.policy_kwargs))
    finally:
        if traj_fh:
            traj_fh.close()
    table = summarize(results)
    sandbox = next((r["sandbox"] for r in results if r.get("sandbox")), None)
    meta = {"env_version": ENV_VERSION, "grader_version": GRADER_VERSION, "mode": args.mode, "splits": splits,
            "episodes": len(plan), "isolated": not args.local, "policy_seed": args.policy_seed,
            "policy_sandbox": (f"separate process, audit hook, seccomp {sandbox['seccomp']}" if sandbox else
                               "off (oracle is grader-side)" if sandboxed else "off (debug)")}
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        per_episode = [{k: v for k, v in r.items() if k != "final_info"} for r in results]
        with open(args.report_json, "w", encoding="utf-8", newline="") as fh:
            fh.write(json.dumps({"meta": meta, "summary": table, "episodes": per_episode}, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    md = markdown_report(table, "Baseline diagnostics", meta)
    if args.report_md:
        Path(args.report_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_md, "w", encoding="utf-8", newline="") as fh:
            fh.write(md)
    print(md)
    failed = [r for r in results if r.get("policy_error")]
    for r in failed:
        print(f"policy {r['policy']} on {r['split']}#{r['index']} ({r['episode_id']}): {r['policy_error']}"
              f" -> episode ended by the environment, empty findings")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
