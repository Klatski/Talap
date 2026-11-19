"""CLI генератора.

  python -m generator generate --seed 7 --config all_types --out /tmp/ep
  python -m generator build-splits              # episodes/train, dev, stub
  python -m generator check-splits              # пересобрать и сравнить побайтово
  python -m generator audit <episode_dir>       # эталон == вывод по открытым данным
  python -m generator validate <episode_dir>    # схемы schemas/*.json
  python -m generator transform <src> <dst> --seed 1
  python -m generator oracle <episode_dir>      # findings.json от oracle
  python -m generator leakcheck --n 60          # признаки-артефакты vs ошибки
  python -m generator attacks <episode_dir> --out attacks.json   # атаки на оценщик B
  python -m generator scan-trajectories trajectories/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_generate(args) -> int:
    from .generate import generate_episode, write_episode
    from .splits import resolve_config

    cfg = resolve_config(args.config, json.loads(args.overrides) if args.overrides else None)
    episode = generate_episode(args.seed, cfg, salt=args.salt or "")
    out = Path(args.out)
    write_episode(episode, out)
    print(json.dumps({"episode_id": episode.episode_id, "dir": str(out), "issues": len(episode.truth["issues"])}))
    return 0


def _cmd_build_splits(args) -> int:
    from .splits import build_splits

    result = build_splits(args.splits, args.out, args.only.split(",") if args.only else None)
    for name, index in result.items():
        print(f"{name}: {len(index['episodes'])} episodes")
        for e in index["episodes"]:
            print(f"  {e['episode_id']} seed={e['seed']} config={e['config']} issues={e['issues']} {e['scenarios']}")
    return 0


def _cmd_check_splits(args) -> int:
    from .splits import check_splits

    problems = check_splits(args.splits, args.out)
    for p in problems:
        print("FAIL", p)
    print("check-splits:", "OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


def _cmd_audit(args) -> int:
    from .audit import audit, compare_with_truth, parse_public, summarize
    from .fileio import read_episode_dir

    public, private = read_episode_dir(args.episode_dir)
    pub = parse_public(public)
    truth = json.loads(private["truth.json"])
    problems = compare_with_truth(pub, truth)
    print(json.dumps({"derived": summarize(audit(pub)), "truth_issues": len(truth["issues"]), "problems": problems}, indent=2))
    return 1 if problems else 0


def _cmd_validate(args) -> int:
    from .fileio import read_episode_dir
    from .schema_check import validate_episode_texts

    public, private = read_episode_dir(args.episode_dir)
    errors = validate_episode_texts(public, private)
    for e in errors[:50]:
        print("FAIL", e)
    print("validate:", "OK" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


def _cmd_transform(args) -> int:
    from .transform import transform_episode

    mapping = transform_episode(args.src, args.dst, args.seed)
    print(json.dumps({"dst": args.dst, "renamed_ids": len(mapping)}))
    return 0


def _cmd_oracle(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from policies.oracle import load_truth, oracle_findings

    print(json.dumps(oracle_findings(load_truth(args.episode_dir)), indent=2, ensure_ascii=False))
    return 0


def _cmd_leakcheck(args) -> int:
    from .generate import generate_episode
    from .leakcheck import leak_report
    from .splits import resolve_config

    cfg = resolve_config(args.config)
    episodes = []
    for seed in range(args.n):
        ep = generate_episode(10_000 + seed, cfg)
        episodes.append((ep.public, ep.truth))
    print(json.dumps(leak_report(episodes), indent=2))
    return 0


def _cmd_attacks(args) -> int:
    from .attacks import attacks_for_episode_dir

    cases = attacks_for_episode_dir(args.episode_dir, args.seed)
    text = json.dumps(cases, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        print(json.dumps({"cases": len(cases), "out": args.out}))
    else:
        sys.stdout.write(text)
    return 0


def _cmd_scan(args) -> int:
    from .leakscan import scan_trajectories

    problems = scan_trajectories(args.trajectories, args.episodes)
    for p in problems[:100]:
        print("LEAK", p)
    print("scan-trajectories:", "OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m generator", description="Synthetic audit episode generator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="generate one episode")
    p.add_argument("--seed", required=True)
    p.add_argument("--config", required=True, help="preset name from generator/configs or path to JSON")
    p.add_argument("--overrides", help="JSON object merged into the config")
    p.add_argument("--salt", help="secret salt for closed evaluation")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_generate)

    for name, func in (("build-splits", _cmd_build_splits), ("check-splits", _cmd_check_splits)):
        p = sub.add_parser(name)
        p.add_argument("--splits", default=str(Path(__file__).resolve().parent / "configs" / "splits.json"))
        p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "episodes"))
        if name == "build-splits":
            p.add_argument("--only", help="comma-separated split names")
        p.set_defaults(func=func)

    for name, func in (("audit", _cmd_audit), ("validate", _cmd_validate), ("oracle", _cmd_oracle)):
        p = sub.add_parser(name)
        p.add_argument("episode_dir")
        p.set_defaults(func=func)

    p = sub.add_parser("transform", help="permute rows and rename all IDs")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--seed", required=True)
    p.set_defaults(func=_cmd_transform)

    p = sub.add_parser("leakcheck", help="artifact features vs. injected errors")
    p.add_argument("--config", default="mixed_random")
    p.add_argument("--n", type=int, default=60)
    p.set_defaults(func=_cmd_leakcheck)

    p = sub.add_parser("attacks", help="adversarial submissions with spec-derived expected tp/fp/fn/reward")
    p.add_argument("episode_dir")
    p.add_argument("--out")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_attacks)

    p = sub.add_parser("scan-trajectories", help="look for truth leaks in trajectories.jsonl")
    p.add_argument("trajectories")
    p.add_argument("--episodes", default=str(Path(__file__).resolve().parent.parent / "episodes"))
    p.set_defaults(func=_cmd_scan)

    args = parser.parse_args(argv)
    if hasattr(args, "seed") and isinstance(args.seed, str) and args.seed.lstrip("-").isdigit():
        args.seed = int(args.seed)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
