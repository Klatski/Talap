"""Общие помощники тестов оценщика."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grader.episodes import EpisodeStore, read_episode  # noqa: E402

EPISODES = ROOT / "episodes"


def open_episodes():
    """[(split, name, public, truth, provenance)] всех открытых эпизодов."""
    store = EpisodeStore()
    out = []
    for split in ("train", "dev", "stub"):
        for name in store.split_episodes(split):
            public, truth, prov = read_episode(EPISODES / split / name)
            out.append((split, name, public, truth, prov))
    return out


def issue(issue_id="ISS-AAAAAAAA", type="fx_rate_error", affected=("INV-AAAAAAA",), amount=100, currency="KZT",
          tol=1, evidence=(("D-AAAAAAAA", "L-AAAAAAAA", "FX:USD:KZT:2026-03-02"),)):
    return {"issue_id": issue_id, "type": type, "affected_ids": list(affected), "expected_amount_minor": amount,
            "currency": currency, "amount_tolerance_minor": tol, "evidence_sets": [list(e) for e in evidence],
            "scenario": None, "scenario_group": None}


def truth_of(*issues, episode_id="ep-TESTTEST00"):
    return {"schema_version": "1.0", "episode_id": episode_id, "issues": list(issues)}


def finding_for(i: dict, **changes) -> dict:
    f = {"type": i["type"], "affected_ids": list(i["affected_ids"]), "claimed_amount_minor": i["expected_amount_minor"],
         "currency": i["currency"], "evidence_ids": list(i["evidence_sets"][0]), "explanation": "x"}
    f.update(changes)
    return f


def sub(findings, episode_id="ep-TESTTEST00"):
    return {"episode_id": episode_id, "findings": copy.deepcopy(findings)}
