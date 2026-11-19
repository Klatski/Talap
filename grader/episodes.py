"""Где оценщик берёт эпизод для reset(seed, config). Живёт только в процессе оценщика.

Корень данных (`episodes_root`) и соль (`salt`) задаёт обвязка при запуске процесса оценщика,
а не reset: политика/агент не может указать оценщику произвольную папку с «эталоном».

Ключи config, которые понимает оценщик (env-ключи mode/max_steps/time_limit_s снимает среда):
  split        — папка сплита в корне (train | dev | stub | episodes | pairs ...);
  episode_id   — конкретный эпизод сплита;  index — номер в сплите;
                 если не задано ни то ни другое — номер = seed mod N (seed=None -> 0);
  generator    — имя пресета generator/configs/<имя>.json или объект-конфиг генератора:
                 эпизод генерируется здесь же по (seed, config, salt) — truth не покидает процесс;
  config_version — необязательная метка версии конфигурации (пишется в журнал).
Порядок эпизодов сплита — из <split>/index.json, иначе отсортированные имена папок.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EPISODES_ROOT = ROOT / "episodes"
PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")
GRADER_KEYS = frozenset({"split", "episode_id", "index", "generator", "config_version"})
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_EPISODE_RE = re.compile(r"^ep-[A-Za-z0-9]{4,32}$")


class EpisodeLoadError(ValueError):
    """Сообщение безопасно для показа: без путей и закрытых данных."""


@dataclass
class LoadedEpisode:
    episode_id: str
    public: dict[str, str]
    truth: dict
    provenance: dict | None
    source: dict  # для журнала оценщика: откуда эпизод (сплит/пресет, seed) — наружу не отдаётся


def _read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def read_episode(episode_dir: Path) -> tuple[dict[str, str], dict, dict | None]:
    try:
        public = {name: _read(episode_dir / "public" / name) for name in PUBLIC_FILES}
        truth = json.loads(_read(episode_dir / "private" / "truth.json"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeLoadError("episode files are missing or unreadable") from exc
    prov_path = episode_dir / "private" / "provenance.json"
    provenance = json.loads(_read(prov_path)) if prov_path.is_file() else None
    manifest = json.loads(public["manifest.json"])
    if manifest.get("episode_id") != truth.get("episode_id"):
        raise EpisodeLoadError("episode is inconsistent (manifest and reference disagree)")
    return public, truth, provenance


class EpisodeStore:
    def __init__(self, episodes_root: str | Path | None = None, salt: str = "") -> None:
        self.root = Path(episodes_root).resolve() if episodes_root else DEFAULT_EPISODES_ROOT
        self.salt = salt or ""

    # ------------------------------------------------------------------ сплиты
    def split_episodes(self, split: str) -> list[str]:
        if not isinstance(split, str) or not _NAME_RE.match(split) or ".." in split:
            raise EpisodeLoadError("invalid split name")
        split_dir = (self.root / split).resolve()
        if self.root not in split_dir.parents or not split_dir.is_dir():
            raise EpisodeLoadError("unknown split")
        index_path = split_dir / "index.json"
        if index_path.is_file():
            index = json.loads(_read(index_path))
            names = [e.get("path") or e["episode_id"] for e in index.get("episodes", [])]
        else:
            names = sorted(p.name for p in split_dir.iterdir() if (p / "public" / "manifest.json").is_file())
        return [n for n in names if _EPISODE_RE.match(n) and (split_dir / n / "public").is_dir()]

    def episode_dir(self, split: str, name: str) -> Path:
        if name not in self.split_episodes(split):
            raise EpisodeLoadError("unknown episode in this split")
        return self.root / split / name

    # ------------------------------------------------------------------ загрузка
    def load(self, seed: Any, config: dict) -> LoadedEpisode:
        if not isinstance(config, dict):
            raise EpisodeLoadError("config must be an object")
        unknown = set(config) - GRADER_KEYS
        if unknown:
            raise EpisodeLoadError("config has unknown keys")
        if "generator" in config:
            if "split" in config or "episode_id" in config or "index" in config:
                raise EpisodeLoadError("use either generator or split, not both")
            return self._generate(seed, config["generator"], config.get("config_version"))
        split = config.get("split", "train")
        names = self.split_episodes(split)
        if not names:
            raise EpisodeLoadError("split is empty")
        if "episode_id" in config:
            name = config["episode_id"]
            if not isinstance(name, str) or name not in names:
                raise EpisodeLoadError("unknown episode in this split")
        else:
            if "index" in config:
                idx = config["index"]
                if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(names):
                    raise EpisodeLoadError("index out of range")
            else:
                if seed is None:
                    idx = 0
                elif isinstance(seed, bool) or not isinstance(seed, int):
                    raise EpisodeLoadError("seed must be an integer for split episodes")
                else:
                    idx = seed % len(names)
            name = names[idx]
        public, truth, provenance = read_episode(self.root / split / name)
        source = {"kind": "split", "split": split, "episode": name, "seed": seed,
                  "config_version": config.get("config_version")}
        return LoadedEpisode(truth["episode_id"], public, truth, provenance, source)

    def _generate(self, seed: Any, gen_config: Any, config_version: Any) -> LoadedEpisode:
        if seed is None or isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise EpisodeLoadError("seed (int or string) is required for generated episodes")
        try:
            from generator.config import ConfigError, config_digest, normalize_config
            from generator.generate import generate_episode
            from generator.splits import resolve_config
        except ImportError as exc:  # pragma: no cover
            raise EpisodeLoadError("generator is not available") from exc
        try:
            if isinstance(gen_config, str):
                if not re.fullmatch(r"[a-z0-9_]{1,64}", gen_config):
                    raise EpisodeLoadError("invalid generator preset name")
                cfg = resolve_config(gen_config)
            elif isinstance(gen_config, dict):
                cfg = normalize_config(gen_config)
            else:
                raise EpisodeLoadError("generator must be a preset name or a config object")
        except (ConfigError, OSError, json.JSONDecodeError) as exc:
            raise EpisodeLoadError("invalid generator config") from exc
        episode = generate_episode(seed, cfg, salt=self.salt)
        public = dict(episode.public)
        source = {"kind": "generated", "generator_config": cfg.get("name"), "config_digest": config_digest(cfg),
                  "seed": seed, "salted": bool(self.salt), "config_version": config_version}
        return LoadedEpisode(episode.episode_id, public, episode.truth, episode.provenance, source)
