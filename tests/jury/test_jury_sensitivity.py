"""Карточка жюри сама проверена мутациями: типичный дефект оценщика/данных роняет нужную проверку.

Без этого 20/20 ничего бы не значило (проверка, которая проходит всегда, бесполезна).
Мутации оценщика внедряются обёрткой вокруг score_submission; мутации данных — правкой копии
закрытого эпизода во временном каталоге.
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scorecard as sc  # noqa: E402
from generator.holdout import build_holdout  # noqa: E402

REAL = sc.score_submission


def ignore_currency(truth, sub, steps, **kw):
    cur = {tuple(sorted(i["affected_ids"])): i["currency"] for i in truth["issues"]}
    sub = copy.deepcopy(sub)
    for f in sub.get("findings") if isinstance(sub.get("findings"), list) else []:
        if isinstance(f, dict) and isinstance(f.get("affected_ids"), list):
            f["currency"] = cur.get(tuple(sorted(set(f["affected_ids"]))), f.get("currency"))
    return REAL(truth, sub, steps, **kw)


def dedup_findings(truth, sub, steps, **kw):
    sub = copy.deepcopy(sub)
    if isinstance(sub.get("findings"), list):
        seen, out = set(), []
        for f in sub["findings"]:
            if repr(f) not in seen:
                seen.add(repr(f))
                out.append(f)
        sub["findings"] = out
    return REAL(truth, sub, steps, **kw)


def free_steps(truth, sub, steps, **kw):
    return REAL(truth, sub, 0, **kw)


def ignore_evidence(truth, sub, steps, **kw):
    truth = copy.deepcopy(truth)
    for issue in truth["issues"]:
        issue["evidence_sets"] = [[]]
    return REAL(truth, sub, steps, **kw)


def explanation_bonus(truth, sub, steps, **kw):
    r = REAL(truth, sub, steps, **kw)
    fs = sub.get("findings") if isinstance(sub.get("findings"), list) else []
    if any(isinstance(f, dict) and len(str(f.get("explanation", ""))) > 500 for f in fs):
        r = {**r, "reward": min(1.0, r["reward"] + 0.1)}
    return r


def lenient_structure(truth, sub, steps, **kw):
    r = REAL(truth, sub, steps, **kw)
    if not r["valid"] and isinstance(sub.get("findings"), list):
        good = [f for f in sub["findings"] if isinstance(f, dict) and f.get("type") != "fraud"]
        return REAL(truth, {"episode_id": truth["episode_id"], "findings": good[:30]}, steps, **kw)
    return r


class Sensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="jury-mut-")
        cls.root = Path(cls.tmp.name)
        build_holdout("jury-sensitivity", 20, cls.root)
        cls.h = sc.Holdout(cls.root, "jury-sensitivity")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def failed_reward_checks(self, mutant) -> set[str]:
        sc.score_submission = mutant
        try:
            return {c.name for c in sc.reward_block(self.h) if not c.ok}
        finally:
            sc.score_submission = REAL

    def test_unmutated_grader_passes(self):
        self.assertEqual(self.failed_reward_checks(REAL), set())

    def test_each_grader_defect_is_caught_by_its_check(self):
        expected = {ignore_currency: "R14", dedup_findings: "R04", free_steps: "R17", ignore_evidence: "R12",
                    explanation_bonus: "R19", lenient_structure: "R18"}
        for mutant, check in expected.items():
            with self.subTest(mutant=mutant.__name__):
                self.assertIn(check, self.failed_reward_checks(mutant))

    def test_data_defects_are_caught(self):
        h = sc.Holdout(self.root, self.h.salt)
        entry = next(e for e in h.entries if e["kind"] == "interacting")
        h.entries = [entry]
        truth_path = h.episode_dir(entry) / "private" / "truth.json"
        original = truth_path.read_bytes()
        try:
            self.assertTrue(all(c.ok for c in sc.data_block(h)))
            truth = json.loads(original)
            truth["issues"] = truth["issues"][1:]  # эталон «забыл» одну ошибку
            truth_path.write_bytes(json.dumps(truth).encode("utf-8"))
            self.assertFalse(sc.data_block(h)[0].ok)
            truth = json.loads(original)
            truth["issues"][0]["evidence_sets"] = [["L-NOTPUBLIC"]]  # доказательство, которого нет в открытых данных
            truth_path.write_bytes(json.dumps(truth).encode("utf-8"))
            check = sc.data_block(h)[0]
            self.assertFalse(check.ok)
            self.assertIn("not public", check.note)
        finally:
            truth_path.write_bytes(original)

    def test_pair_defect_is_caught(self):
        h = sc.Holdout(self.root, self.h.salt)
        entry = next(e for e in h.entries if e["kind"] != "clean")
        h.entries = [entry]
        rewards = {(p, ep): 1.0 if p == "oracle" else 0.0 for p in ("oracle", "auditor", "reconcile", "flag_all")
                   for ep in (entry["episode_id"], entry["pair_episode_id"])}
        self.assertTrue(sc.pairs_block(h, rewards)[0].ok, sc.pairs_block(h, rewards)[0].note)
        self.assertFalse(sc.pairs_block(h, {**rewards, ("auditor", entry["pair_episode_id"]): 0.5})[0].ok)
        truth_path = h.episode_dir(entry, "pairs") / "private" / "truth.json"
        original = truth_path.read_bytes()
        try:
            truth = json.loads(original)
            truth["issues"][0]["expected_amount_minor"] += 1  # пара «съехала» на 1 минимальную единицу
            truth_path.write_bytes(json.dumps(truth).encode("utf-8"))
            self.assertFalse(sc.pairs_block(h, rewards)[0].ok)
        finally:
            truth_path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
