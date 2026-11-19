from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BasePolicy
from .baselines import build_policy


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")


def _sanitize_obs(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    clean: dict[str, Any] = {}
    for key, val in obs.items():
        if key in ("truth", "provenance", "seed"):
            continue
        clean[key] = val
    return clean


def run_episode(
    env: Any,
    policy: BasePolicy,
    seed: int = 0,
    config: dict[str, Any] | None = None,
    trajectory_path: str | Path | None = None,
    max_steps: int = 100,
) -> dict[str, Any]:
    cfg = dict(config or {})
    obs, info = env.reset(seed=seed, config=cfg)
    policy.reset(obs, info)

    traj_file = Path(trajectory_path) if trajectory_path is not None else None

    terminated = False
    truncated = False
    final_reward = 0.0
    final_info: dict[str, Any] = dict(info or {})
    steps_used = 0

    while not terminated and not truncated and steps_used < max_steps:
        action = policy.act(obs, info)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        steps_used += 1
        final_reward = float(reward)
        if isinstance(step_info, dict):
            final_info = {k: v for k, v in {**final_info, **step_info}.items() if k not in ("truth", "provenance", "seed")}

        if traj_file is not None:
            _append_jsonl(
                traj_file,
                {
                    "episode_id": policy.episode_id,
                    "step": steps_used,
                    "action": action,
                    "observation": _sanitize_obs(next_obs),
                    "reward": final_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                },
            )

        obs = next_obs
        info = step_info if isinstance(step_info, dict) else info

    summary = {
        "policy": policy.name,
        "episode_id": policy.episode_id,
        "steps_used": steps_used,
        "reward": final_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if cfg.get("mode") == "train":
        for metric in ("tp", "fp", "fn", "f1"):
            if metric in final_info:
                summary[metric] = final_info[metric]

    return summary


def run_suite(
    env: Any,
    policy_names: list[str] | None = None,
    seeds: list[int] | None = None,
    mode: str = "eval",
    trajectory_path: str | Path = "trajectories/trajectories.jsonl",
    truth_provider: Any = None,
    max_steps: int = 100,
) -> list[dict[str, Any]]:
    names = policy_names or ["random", "reconcile", "all", "oracle"]
    seed_list = seeds or [0, 1, 2]
    results: list[dict[str, Any]] = []

    for pname in names:
        for s in seed_list:
            pol = build_policy(pname, seed=s, truth_provider=truth_provider)
            ep_res = run_episode(
                env=env,
                policy=pol,
                seed=s,
                config={"mode": mode},
                trajectory_path=trajectory_path,
                max_steps=max_steps,
            )
            results.append(ep_res)

    return results