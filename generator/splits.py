"""Сборка открытых сплитов (train/dev/stub) по generator/configs/splits.json.

Каждый эпизод: episodes/<split>/<episode_id>/{public,private}/...
Индекс: episodes/<split>/index.json (seed, имя конфига, дайджест, версия генератора).
check_splits() пересобирает всё в памяти и сравнивает побайтово с файлами на диске.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

from .config import config_digest, normalize_config
from .fileio import PRIVATE_DIR, PUBLIC_DIR, json_text, read_episode_dir
from .generate import generate_episode, write_episode
from .version import GENERATOR_VERSION, SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_SPLITS = CONFIG_DIR / "splits.json"
DEFAULT_OUT = ROOT / "episodes"


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_config(name_or_path: str, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.json"
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return normalize_config(_merge(raw, overrides or {}))


def load_splits(path: str | Path = DEFAULT_SPLITS) -> dict[str, list[dict]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["splits"]


def build_split(name: str, items: list[dict], out_root: Path) -> dict:
    split_dir = out_root / name
    split_dir.mkdir(parents=True, exist_ok=True)
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.startswith("ep-") and (child / PUBLIC_DIR).is_dir():
            shutil.rmtree(child)
    index = {"split": name, "generator_version": GENERATOR_VERSION, "schema_version": SCHEMA_VERSION, "episodes": []}
    for item in items:
        cfg = resolve_config(item["config"], item.get("overrides"))
        episode = generate_episode(item["seed"], cfg)
        write_episode(episode, split_dir / episode.episode_id)
        index["episodes"].append(
            {
                "episode_id": episode.episode_id,
                "path": episode.episode_id,
                "seed": item["seed"],
                "config": item["config"],
                "overrides": item.get("overrides", {}),
                "config_digest": config_digest(cfg),
                "issues": len(episode.truth["issues"]),
                "issue_types": sorted({i["type"] for i in episode.truth["issues"]}),
                "scenarios": sorted({i["scenario"] for i in episode.truth["issues"] if i["scenario"]}),
            }
        )
    with open(split_dir / "index.json", "w", encoding="utf-8", newline="") as fh:
        fh.write(json_text(index))
    return index


def build_splits(splits_path: str | Path = DEFAULT_SPLITS, out_root: str | Path = DEFAULT_OUT, only: list[str] | None = None) -> dict:
    splits = load_splits(splits_path)
    out = Path(out_root)
    result = {}
    for name, items in splits.items():
        if only and name not in only:
            continue
        result[name] = build_split(name, items, out)
    return result


def check_splits(splits_path: str | Path = DEFAULT_SPLITS, out_root: str | Path = DEFAULT_OUT) -> list[str]:
    """Пересобрать каждый эпизод и сравнить побайтово с файлами в репозитории."""
    problems = []
    out = Path(out_root)
    for name, items in load_splits(splits_path).items():
        index_path = out / name / "index.json"
        if not index_path.exists():
            problems.append(f"{name}: index.json missing")
            continue
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
        if len(index["episodes"]) != len(items):
            problems.append(f"{name}: index has {len(index['episodes'])} episodes, splits.json has {len(items)}")
        for item, entry in zip(items, index["episodes"]):
            cfg = resolve_config(item["config"], item.get("overrides"))
            episode = generate_episode(item["seed"], cfg)
            if episode.episode_id != entry["episode_id"]:
                problems.append(f"{name}/{entry['episode_id']}: regenerated as {episode.episode_id}")
                continue
            ep_dir = out / name / entry["path"]
            expected = {f"{PUBLIC_DIR}/{k}": v for k, v in episode.public.items()}
            expected.update({f"{PRIVATE_DIR}/{k}": v for k, v in episode.private.items()})
            for rel, text in expected.items():
                path = ep_dir / rel
                disk = path.read_text(encoding="utf-8") if path.exists() else None
                if disk is not None:
                    with open(path, encoding="utf-8", newline="") as fh:
                        disk = fh.read()
                if disk != text:
                    problems.append(f"{name}/{entry['episode_id']}/{rel}: differs from regenerated")
    return problems


def iter_episode_dirs(out_root: str | Path = DEFAULT_OUT, splits: tuple[str, ...] = ("train", "dev", "stub")):
    out = Path(out_root)
    for name in splits:
        index_path = out / name / "index.json"
        if not index_path.exists():
            continue
        with open(index_path, encoding="utf-8") as fh:
            for entry in json.load(fh)["episodes"]:
                yield name, out / name / entry["path"]


__all__ = [
    "build_splits",
    "check_splits",
    "iter_episode_dirs",
    "resolve_config",
    "PRIVATE_DIR",
]
