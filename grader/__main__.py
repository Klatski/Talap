"""Ручная проверка одного ответа против эталона (для людей, вне агентского контура: читает truth).

    python -m grader --truth truth.json --submission answer.json [--steps N]
    python -m grader --episode episodes/stub/ep-SUW6BEVZNR --submission answer.json --steps 3

--steps — inspection_steps, то есть все шаги кроме submit. Печатает TP/FP/FN, P/R/F1 дробями и
десятичными, раскрытую формулу награды и разбор каждого вывода (какое условие TP не выполнено).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import STEP_COST, clip, prf
from .scoring import score_submission


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m grader", description="score one submission against a truth file")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--truth", help="path to truth.json")
    src.add_argument("--episode", help="episode directory (uses <dir>/private/truth.json)")
    parser.add_argument("--submission", required=True, help="path to {episode_id, findings} JSON")
    parser.add_argument("--steps", type=int, default=0, help="inspection steps = all steps except submit (default 0)")
    args = parser.parse_args(argv)
    truth_path = Path(args.truth) if args.truth else Path(args.episode) / "private" / "truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    submission = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    r = score_submission(truth, submission, args.steps)
    if not r["valid"]:
        print("INVALID structure -> reward -1")
        for e in r["errors"]:
            print("  ", e)
        return 0
    p, rec, f1 = prf(r["tp"], r["fp"], r["fn"])
    raw = 2 * f1 - 1 - STEP_COST * args.steps
    print(f"TP {r['tp']}   FP {r['fp']}   FN {r['fn']}   (issues in truth: {len(truth['issues'])}, findings: {len(submission['findings'])})")
    print(f"P  = TP/(TP+FP)       = {p} = {float(p):.6f}")
    print(f"R  = TP/(TP+FN)       = {rec} = {float(rec):.6f}")
    print(f"F1 = 2TP/(2TP+FP+FN)  = {f1} = {float(f1):.6f}")
    print(f"reward = clip(2*F1 - 1 - 0.002*{args.steps}, -1, 1) = clip({raw}) = {clip(raw)} = {r['reward']}")
    issues = truth["issues"]
    for st in r["diagnostics"]["per_finding"]:
        f = submission["findings"][st["finding"]]
        head = f"  finding {st['finding']}: {f['type']} {sorted(set(f['affected_ids']))} {f['claimed_amount_minor']} {f['currency']}"
        if st["status"] == "tp":
            print(f"{head} -> TP ({issues[st['issue_index']]['issue_id']})")
        else:
            print(f"{head} -> FP ({st['reason']})")
    for m in r["diagnostics"]["missed"]:
        i = issues[m["issue_index"]]
        print(f"  missed {i['issue_id']}: {i['type']} {sorted(i['affected_ids'])} {i['expected_amount_minor']} {i['currency']} -> FN ({m['reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
