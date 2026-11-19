"""Воспроизводимость: одинаковые seed+config+версия -> побайтово одинаковые файлы (и в разных процессах)."""

import hashlib
import json
import os
import subprocess
import sys
import unittest

from gen_helpers import CONFIGS, ROOT
from generator.generate import generate_episode

SCRIPT = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from generator.generate import generate_episode
cfg = json.loads(sys.argv[2])
ep = generate_episode(int(sys.argv[3]), cfg)
h = hashlib.sha256()
for name in sorted(ep.public):
    h.update(name.encode()); h.update(ep.public[name].encode("utf-8"))
for name in sorted(ep.private):
    h.update(name.encode()); h.update(ep.private[name].encode("utf-8"))
print(ep.episode_id, h.hexdigest())
"""


def digest(ep) -> str:
    h = hashlib.sha256()
    for name in sorted(ep.public):
        h.update(name.encode())
        h.update(ep.public[name].encode("utf-8"))
    for name in sorted(ep.private):
        h.update(name.encode())
        h.update(ep.private[name].encode("utf-8"))
    return f"{ep.episode_id} {h.hexdigest()}"


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_bytes_in_process(self):
        for key in ("all_types", "scenarios"):
            a = generate_episode(17, CONFIGS[key])
            b = generate_episode(17, CONFIGS[key])
            self.assertEqual(a.public, b.public)
            self.assertEqual(a.private, b.private)

    def test_same_seed_same_bytes_across_processes_and_hash_seeds(self):
        cfg = json.dumps(CONFIGS["heavy"])
        outputs = set()
        for hash_seed in ("0", "1", "12345", "random"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            out = subprocess.run(
                [sys.executable, "-c", SCRIPT, str(ROOT), cfg, "23"],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            ).stdout.strip()
            outputs.add(out)
        outputs.add(digest(generate_episode(23, CONFIGS["heavy"])))
        self.assertEqual(len(outputs), 1, outputs)

    def test_different_seeds_differ(self):
        ids = {generate_episode(s, CONFIGS["all_types"]).episode_id for s in range(10)}
        self.assertEqual(len(ids), 10)

    def test_salt_changes_episode(self):
        a = generate_episode(5, CONFIGS["all_types"])
        b = generate_episode(5, CONFIGS["all_types"], salt="closed-eval-secret")
        self.assertNotEqual(a.public, b.public)
        self.assertTrue(json.loads(b.private["provenance.json"])["salted"])
        self.assertNotIn("closed-eval-secret", b.private["provenance.json"])

    def test_config_change_changes_episode(self):
        a = generate_episode(5, CONFIGS["all_types"])
        b = generate_episode(5, {**CONFIGS["all_types"], "name": "renamed"})
        self.assertNotEqual(a.episode_id, b.episode_id)


if __name__ == "__main__":
    unittest.main()
