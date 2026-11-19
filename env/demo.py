"""Демо работы среды со скриптовой политикой (обязательное демо из ТЗ).

    python -m env.demo                         # аудитор на заглушке, изолированные процессы, режим train
    python -m env.demo --policy reconcile --split dev --index 2
    python -m env.demo --policy random --mode eval

Печатает каждый шаг: действие, краткий результат/ошибку, остаток шагов; в конце — награду и
метрики (train) или «награда у оценщика» (eval) и журнал оценщика после закрытия.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _short_action(action) -> str:
    if not isinstance(action, dict):
        return f"<malformed {type(action).__name__}>"
    name = action.get("action")
    if name == "submit":
        payload = action.get("findings")
        n = len(payload.get("findings", [])) if isinstance(payload, dict) and isinstance(payload.get("findings"), list) else "?"
        return f"submit({n} findings)"
    args = ", ".join(f"{k}={v!r}" for k, v in action.items() if k != "action")
    return f"{name}({args})"[:110]


def _short_result(obs: dict) -> str:
    if obs.get("error"):
        e = obs["error"]
        return f"ERROR {e['code']}" + (f" ({e['detail']})" if e.get("detail") else "")
    res = obs.get("result")
    if isinstance(res, dict):
        if "records" in res:
            return f"{len(res['records'])} {res['table']} rows from offset {res['offset']} of {res['total']}" + \
                   ("" if res["next_cursor"] else ", last page")
        if "record" in res:
            return f"record {json.dumps(res['record'])[:90]}"
        if "document" in res:
            return f"{res['doc_type']} {json.dumps(res['document'])[:90]}"
        if "value" in res:
            return f"= {res['value']}"
        return json.dumps(res)[:100]
    return "-"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m env.demo")
    parser.add_argument("--policy", default="auditor", choices=("auditor", "reconcile", "flag_all", "random"))
    parser.add_argument("--split", default="stub")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", default="train", choices=("train", "eval"))
    parser.add_argument("--seed", type=int, default=0, help="policy seed (random explorer)")
    args = parser.parse_args(argv)

    from env import AuditEnv
    from policies import make_policy

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "grader_log.jsonl"
        env = AuditEnv(grader_log=log)
        obs, info = env.reset(seed=args.index, config={"split": args.split, "mode": args.mode})
        m = obs["manifest"]
        print(f"reset -> episode {obs['episode_id']}: base {m['base_currency']}, tz {m['timezone']}, "
              f"period {m['period_start']} .. {m['period_end']}, {m['counts']['ledger_rows']} ledger rows, "
              f"{m['counts']['documents']} documents; rate rule {obs['policy']['fx']['rate_date_rule']}, "
              f"rounding {obs['policy']['rounding']['mode']}; budget {obs['remaining_steps']} steps; info {info}")
        policy = make_policy(args.policy, seed=args.seed)
        policy.reset(obs, info)
        step, done = 0, False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated
            print(f"{step:3d}. {_short_action(action):<112} -> {_short_result(obs)}  [left {obs['remaining_steps']}]")
        if args.mode == "train":
            print(f"\nterminated={terminated} truncated={truncated} reward={reward:.4f}")
            print("metrics:", json.dumps({k: info.get(k) for k in ("valid", "tp", "fp", "fn", "precision", "recall", "f1", "inspection_steps")}))
            if isinstance(action, dict) and action.get("action") == "submit" and isinstance(action.get("findings"), dict):
                for f in action["findings"].get("findings", [])[:12]:
                    print(f"  {f['type']:<18} {','.join(f['affected_ids']):<28} {f['claimed_amount_minor']:>14} {f['currency']}  evidence={f['evidence_ids']}")
        else:
            print(f"\nterminated={terminated} truncated={truncated}; the policy saw reward={reward} and info={info} (eval mode)")
        env.close()
        rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        print(f"grader log after close: reward={rec['reward']} f1={rec['f1']} tp={rec['tp']} fp={rec['fp']} fn={rec['fn']} "
              f"end_reason={rec['end_reason']} elapsed={rec['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
