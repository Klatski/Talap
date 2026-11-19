"""Общие помощники тестов среды (импортируются как модуль из этой папки)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import LocalAuditEnv  # noqa: E402
from env.constants import ERRORS  # noqa: E402
from grader.episodes import EpisodeStore, read_episode  # noqa: E402

EPISODES = ROOT / "episodes"
STUB = {"split": "stub"}


def episode_dirs(splits=("train", "dev", "stub")) -> list[Path]:
    store = EpisodeStore()
    return [EPISODES / s / name for s in splits for name in store.split_episodes(s)]


def clean_episode() -> tuple[str, int]:
    """(split, index) первого чистого эпизода train."""
    store = EpisodeStore()
    for i, name in enumerate(store.split_episodes("train")):
        if not read_episode(EPISODES / "train" / name)[1]["issues"]:
            return "train", i
    raise AssertionError("no clean train episode")


def stub_truth() -> dict:
    store = EpisodeStore()
    name = store.split_episodes("stub")[0]
    return read_episode(EPISODES / "stub" / name)[1]


def assert_obs_shape(test, obs: dict) -> None:
    test.assertEqual(set(obs), {"result", "error", "remaining_steps"})
    if obs["error"] is not None:
        test.assertIn(obs["error"]["code"], ERRORS)
        test.assertEqual(obs["error"]["message"], ERRORS[obs["error"]["code"]])


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


class TempRoot:
    """Временный корень эпизодов: {split: [episode_dir, ...]} копируются/пишутся сюда."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="episodes-")
        self.path = Path(self._tmp.name)

    def copy(self, split: str, src: Path) -> Path:
        dst = self.path / split / src.name
        shutil.copytree(src, dst)
        return dst

    def write(self, split: str, public: dict, private: dict) -> Path:
        from generator.fileio import write_episode_dir

        episode_id = json.loads(public["manifest.json"])["episode_id"]
        return write_episode_dir(self.path / split / episode_id, public, private)

    def cleanup(self) -> None:
        self._tmp.cleanup()


def local_env(**kw) -> LocalAuditEnv:
    return LocalAuditEnv(**kw)
