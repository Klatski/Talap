"""Где оценщик берёт эпизод для reset(seed, config). Живёт только в процессе оценщика.

Ключи config, которые понимает оценщик (ключи среды mode/max_steps/time_limit_s снимает среда):
  episode_dir  — каталог эпизода с ЛЮБЫМ именем (протокол жюри: config={"mode": "closed",
                 "episode_dir": "<каталог>", "team_config": {...}}). Загружается ровно он, без
                 генерации и смены ID; повторный reset начинает его заново. team_config при этом не нужен;
  team_config  — без episode_dir: эпизод генерируется генератором команды по (seed, team_config, соль
                 обвязки AUDIT_GRADER_SALT) прямо здесь — truth не покидает процесс. Значение — имя пресета
                 generator/configs/<имя>.json или объект-конфиг (docs/generator.md), в том числе
                 {"preset": "<имя>", ...переопределения};
  generator    — то же, что team_config (прежнее имя ключа);
  split        — папка сплита в корне данных обвязки (train | dev | stub | episodes | pairs ...);
  episode_id   — имя каталога эпизода в сплите;  index — номер в сплите;
                 если не задано ни то ни другое — номер = seed mod N (seed=None -> 0);
  config_version — необязательная метка версии конфигурации (пишется в журнал).
Порядок эпизодов сплита — из <split>/index.json, иначе отсортированные имена каталогов (любые имена).

Раскладка каталога эпизода (обе поддерживаются):
  <dir>/public/{manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv}
  <dir>/private/{truth.json, provenance.json}                  — как пишет генератор (docs/contract.md, раздел 1);
  или плоско: <dir>/{шесть открытых файлов, truth.json, provenance.json} (truth/provenance можно и в private/).
Reset вызывает доверенная обвязка (процесс жюри или раннер), а не политика: политика работает в
песочнице (env/sandbox.py) и канала к reset не имеет. Сообщения об ошибках — без путей и закрытых данных.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EPISODES_ROOT = ROOT / "episodes"
PUBLIC_FILES = ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv")
GRADER_KEYS = frozenset({"split", "episode_id", "index", "generator", "team_config", "episode_dir", "config_version"})
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_PRESET_RE = re.compile(r"[a-z0-9_]{1,64}")


class EpisodeLoadError(ValueError):
    """Сообщение безопасно для показа: без путей и закрытых данных."""


@dataclass
class LoadedEpisode:
    episode_id: str
    public: dict[str, str]
    truth: dict
    provenance: dict | None
    source: dict  # для журнала оценщика: откуда эпизод (сплит/каталог/пресет, seed) — наружу не отдаётся
    public_ids: frozenset = frozenset()  # все ID, доступные политике через инструменты (правило M04)


def _read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def episode_layout(episode_dir: Path) -> tuple[Path, Path] | None:
    """(папка открытых файлов, папка закрытых) или None, если это не каталог эпизода."""
    if (episode_dir / "public" / "manifest.json").is_file():
        return episode_dir / "public", episode_dir / "private"
    if (episode_dir / "manifest.json").is_file():
        private = episode_dir / "private"
        return episode_dir, (private if (private / "truth.json").is_file() else episode_dir)
    return None


def public_ids(public: dict[str, str]) -> frozenset:
    """ID, которые политика может получить инструментами: row_id, invoice_id, payment_id, fx_id, doc_id.
    Разбор тот же, что у среды (env/records.py), поэтому «существует» = «доступно через get_record/get_document»."""
    from env.records import EpisodeDataError, load_public

    try:
        ep = load_public(public)
    except (EpisodeDataError, KeyError, TypeError, ValueError) as exc:
        raise EpisodeLoadError("episode public data does not follow the contract") from exc
    ids = {rid for index in ep.by_id.values() for rid in index}
    return frozenset(ids | set(ep.documents))


def read_episode(episode_dir: Path) -> tuple[dict[str, str], dict, dict | None]:
    layout = episode_layout(Path(episode_dir))
    if layout is None:
        raise EpisodeLoadError("episode files are missing or unreadable")
    public_dir, private_dir = layout
    try:
        public = {name: _read(public_dir / name) for name in PUBLIC_FILES}
        truth = json.loads(_read(private_dir / "truth.json"))
        manifest = json.loads(public["manifest.json"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeLoadError("episode files are missing or unreadable") from exc
    prov_path = private_dir / "provenance.json"
    try:
        provenance = json.loads(_read(prov_path)) if prov_path.is_file() else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeLoadError("episode files are missing or unreadable") from exc
    if not isinstance(truth, dict) or not isinstance(truth.get("issues"), list) or not isinstance(manifest, dict):
        raise EpisodeLoadError("episode reference does not follow the contract")
    if "episode_id" not in truth:  # фикстура жюри может не повторять episode_id в truth
        truth = {**truth, "episode_id": manifest.get("episode_id")}
    if manifest.get("episode_id") != truth.get("episode_id"):
        raise EpisodeLoadError("episode is inconsistent (manifest and reference disagree)")
    return public, truth, provenance


def load_episode_dir(episode_dir: Any, source: dict | None = None) -> LoadedEpisode:
    """Загрузить каталог эпизода с любым именем (episode_dir жюри; score_episode)."""
    if isinstance(episode_dir, os.PathLike):
        episode_dir = os.fspath(episode_dir)
    if not isinstance(episode_dir, str) or not episode_dir.strip():
        raise EpisodeLoadError("episode_dir must be a non-empty string")
    path = Path(episode_dir)
    try:
        path = path.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise EpisodeLoadError("episode files are missing or unreadable") from exc
    public, truth, provenance = read_episode(path)
    src = {"kind": "episode_dir", "episode_dir": str(path), **(source or {})}
    return LoadedEpisode(truth["episode_id"], public, truth, provenance, src, public_ids(public))


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
            names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        out = []
        for n in names:
            if not isinstance(n, str) or not n or n.startswith(".") or "/" in n or "\\" in n:
                continue
            d = split_dir / n
            if d.resolve().parent == split_dir and episode_layout(d) is not None:
                out.append(n)
        return out

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
        version = config.get("config_version")
        if config.get("episode_dir") is not None:
            if any(config.get(k) is not None for k in ("split", "episode_id", "index", "generator")):
                raise EpisodeLoadError("episode_dir is used alone (team_config is ignored with it)")
            return load_episode_dir(config["episode_dir"], {"seed": seed, "config_version": version})
        gen = config.get("generator")
        team = config.get("team_config")
        if "team_config" in config and team is None and all(
                config.get(k) is None for k in ("split", "episode_id", "index", "generator")):
            team = {}  # протокол жюри: без episode_dir — генератор команды; team_config: null = настройки по умолчанию
        if gen is not None and team is not None:
            raise EpisodeLoadError("use either team_config or generator, not both")
        if team is not None:
            gen = team
        if gen is not None:
            if any(config.get(k) is not None for k in ("split", "episode_id", "index")):
                raise EpisodeLoadError("use either a generator config or split, not both")
            return self._generate(seed, gen, version)
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
        source = {"kind": "split", "split": split, "episode": name, "seed": seed, "config_version": version}
        return LoadedEpisode(truth["episode_id"], public, truth, provenance, source, public_ids(public))

    def _generate(self, seed: Any, gen_config: Any, config_version: Any) -> LoadedEpisode:
        if seed is None or isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise EpisodeLoadError("seed (int or string) is required for generated episodes")
        try:
            cfg = resolve_generator_config(gen_config)
            from generator.config import config_digest
            from generator.generate import generate_episode
        except ImportError as exc:  # pragma: no cover
            raise EpisodeLoadError("generator is not available") from exc
        episode = generate_episode(seed, cfg, salt=self.salt)
        public = dict(episode.public)
        source = {"kind": "generated", "generator_config": cfg.get("name"), "config_digest": config_digest(cfg),
                  "seed": seed, "salted": bool(self.salt), "config_version": config_version}
        return LoadedEpisode(episode.episode_id, public, episode.truth, episode.provenance, source, public_ids(public))


def resolve_generator_config(gen_config: Any) -> dict:
    """Конфиг генератора команды: имя пресета, объект-конфиг или {"preset": имя, ...переопределения}."""
    try:
        from generator.config import ConfigError, normalize_config
        from generator.splits import resolve_config
    except ImportError as exc:  # pragma: no cover
        raise EpisodeLoadError("generator is not available") from exc
    try:
        if isinstance(gen_config, str):
            if not _PRESET_RE.fullmatch(gen_config):
                raise EpisodeLoadError("invalid generator preset name")
            return resolve_config(gen_config)
        if isinstance(gen_config, dict) and "preset" in gen_config:
            preset = gen_config["preset"]
            if not isinstance(preset, str) or not _PRESET_RE.fullmatch(preset):
                raise EpisodeLoadError("invalid generator preset name")
            return resolve_config(preset, {k: v for k, v in gen_config.items() if k != "preset"})
        if isinstance(gen_config, dict):
            return normalize_config(gen_config)
        raise EpisodeLoadError("generator config must be a preset name or an object")
    except (ConfigError, OSError, json.JSONDecodeError) as exc:
        raise EpisodeLoadError("invalid generator config") from exc
