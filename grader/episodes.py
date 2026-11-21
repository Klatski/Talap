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
Прочие ключи (и с episode_dir — лишние селекторы split/episode_id/index/generator) не мешают: они
игнорируются и записываются в журнал оценщика (source.ignored_config_keys). Неизвестные ключи верхнего
уровня team_config — так же (генератору уходят только его ключи, docs/generator.md). Исключение — ключ,
похожий на известный (misspelled_key: регистр, пробелы, `-`/`_`, единственное/множественное число,
одна-две опечатки): {"Errors": ...}, {"error": ...}, {"episode-dir": ...}, {"Mode": ...} — ошибка config,
иначе опечатка молча дала бы другой эпизод (чистую компанию, сплит по seed, режим train). Явно чужие
ключи (kind, label, note, case, ...) по-прежнему игнорируются.
Порядок эпизодов сплита — из <split>/index.json, иначе отсортированные имена каталогов (любые имена).

Раскладка каталога эпизода (обе поддерживаются):
  <dir>/public/{manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv}
  <dir>/private/{truth.json, provenance.json}                  — как пишет генератор (docs/contract.md, раздел 1);
  или плоско: <dir>/{шесть открытых файлов, truth.json, provenance.json} (truth/provenance можно и в private/).
Файлы читаются терпимо: UTF-8 с BOM или без, переводы строк LF или CRLF, колонки CSV в любом порядке.
episode_id эпизода — из manifest (его видит политика); truth без него или с другим значением приводится к
manifest. В режимах eval/closed truth.json может отсутствовать: среде эталон не нужен, эпизод проводится,
метрики в журнале — null (reference: "missing"); train и score_episode требуют эталон.
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
RESET_ENV_KEYS = frozenset({"mode", "max_steps", "time_limit_s"})  # разбирает среда; здесь — только для опечаток
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_PRESET_RE = re.compile(r"[a-z0-9_]{1,64}")


class EpisodeLoadError(ValueError):
    """Сообщение безопасно для показа: без путей и закрытых данных."""


def _key_stem(key: str) -> str:
    """Ключ без регистра, пробелов и разделителей, в единственном числе: "Errors " -> "error"."""
    k = "".join(ch for ch in key.casefold() if ch.isalnum())
    if len(k) > 4 and k.endswith("ies"):
        return k[:-3] + "y"
    if len(k) > 3 and k.endswith("s"):
        return k[:-1]
    return k


def _osa_distance(a: str, b: str) -> int:
    """Расстояние Дамерау–Левенштейна (оптимальное выравнивание): перестановка соседних букв — одна правка."""
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[len(b)]


def misspelled_key(key: Any, known) -> str | None:
    """Известный ключ, опечаткой которого выглядит `key`, иначе None (ключ известен или явно чужой).
    Похож: совпадает без регистра/пробелов/`-`/`_` и числа (error ~ errors, Team-Config ~ team_config) или
    отличается на одну правку (на две, если оба имени длиннее 7 букв): erors, scenarois, team_confg.
    Короткие чужие имена не задевает: kind, label, note, case, seed, episode, salt."""
    if not isinstance(key, str) or key in known:
        return None
    stem = _key_stem(key)
    if not stem:
        return None
    for name in sorted(known):
        other = _key_stem(name)
        limit = 1 if min(len(stem), len(other)) <= 7 else 2
        if stem == other or _osa_distance(stem, other) <= limit:
            return name
    return None


def reject_misspelled_keys(config: dict, known, prefix: str = "") -> None:
    """EpisodeLoadError, если какой-то ключ config похож на известный, но им не является (см. misspelled_key)."""
    for key in config:
        name = misspelled_key(key, known)
        if name is not None:
            shown = prefix + (key[:64] if isinstance(key, str) else repr(key))
            raise EpisodeLoadError(f"config key {shown!r} looks like a misspelling of {prefix + name!r}")


@dataclass
class LoadedEpisode:
    episode_id: str
    public: dict[str, str]
    truth: dict
    provenance: dict | None
    source: dict  # для журнала оценщика: откуда эпизод (сплит/каталог/пресет, seed) — наружу не отдаётся
    public_ids: frozenset = frozenset()  # все ID, доступные политике через инструменты (правило M04)


def _read(path: Path) -> str:
    with open(path, encoding="utf-8-sig", newline="") as fh:  # BOM, если есть, снимается
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


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def issue_follows_contract(issue: Any) -> bool:
    """Ошибка эталона в формате README («Интерфейс жюри», truth.json): обязательные type, affected_ids,
    expected_amount_minor, currency, evidence_sets — то, без чего matcher не может сравнивать; необязательные
    issue_id (любое значение) и amount_tolerance_minor (целое τ >= 0; нет поля или null — 0). Лишние поля
    не мешают. Нарушение — понятная ошибка загрузки при reset/score_episode, а не авария на submit."""
    tolerance = issue.get("amount_tolerance_minor") if isinstance(issue, dict) else None
    return (isinstance(issue, dict) and isinstance(issue.get("type"), str) and _strings(issue.get("affected_ids"))
            and _int(issue.get("expected_amount_minor")) and isinstance(issue.get("currency"), str)
            and isinstance(issue.get("evidence_sets"), list) and all(_strings(s) for s in issue["evidence_sets"])
            and (tolerance is None or (_int(tolerance) and tolerance >= 0)))


def read_episode(episode_dir: Path, require_truth: bool = True) -> tuple[dict[str, str], dict | None, dict | None]:
    """(открытые тексты, truth, provenance). Без truth.json и с require_truth=False -> truth None."""
    layout = episode_layout(Path(episode_dir))
    if layout is None:
        raise EpisodeLoadError("episode files are missing or unreadable")
    public_dir, private_dir = layout
    try:
        public = {name: _read(public_dir / name) for name in PUBLIC_FILES}
        manifest = json.loads(public["manifest.json"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeLoadError("episode files are missing or unreadable") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episode_id"), str):
        raise EpisodeLoadError("episode manifest does not follow the contract (episode_id)")
    truth_path = private_dir / "truth.json"
    truth = None
    if require_truth or truth_path.is_file():
        try:
            truth = json.loads(_read(truth_path))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EpisodeLoadError("episode files are missing or unreadable") from exc
        if not isinstance(truth, dict) or not isinstance(truth.get("issues"), list):
            raise EpisodeLoadError("episode reference does not follow the contract")
        if not all(issue_follows_contract(i) for i in truth["issues"]):
            raise EpisodeLoadError("episode reference does not follow the contract (truth.json issue fields)")
        # главный — episode_id из manifest: его видит политика и пишет в submit (копия жюри может не обновить truth)
        truth = {**truth, "episode_id": manifest["episode_id"]}
    prov_path = private_dir / "provenance.json"
    try:
        provenance = json.loads(_read(prov_path)) if prov_path.is_file() else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EpisodeLoadError("episode files are missing or unreadable") from exc
    return public, truth, provenance


def _episode_id(public: dict[str, str]) -> str:
    return json.loads(public["manifest.json"])["episode_id"]


def load_episode_dir(episode_dir: Any, source: dict | None = None, require_truth: bool = True) -> LoadedEpisode:
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
    public, truth, provenance = read_episode(path, require_truth)
    src = {"kind": "episode_dir", "episode_dir": str(path), **(source or {})}
    if truth is None:
        src["reference"] = "missing"
    return LoadedEpisode(_episode_id(public), public, truth, provenance, src, public_ids(public))


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
    def load(self, seed: Any, config: dict, require_truth: bool = True) -> LoadedEpisode:
        """require_truth=False (eval/closed): каталог episode_dir без truth.json тоже загружается."""
        if not isinstance(config, dict):
            raise EpisodeLoadError("config must be an object")
        reject_misspelled_keys(config, GRADER_KEYS | RESET_ENV_KEYS)  # "episode-dir", "Mode": не молча другой эпизод
        ignored = sorted(k for k in config if k not in GRADER_KEYS)  # не мешают: пишутся в журнал оценщика
        version = config.get("config_version")
        if config.get("episode_dir") is not None:  # протокол жюри: загружается ровно этот каталог
            ignored += sorted(k for k in ("split", "episode_id", "index", "generator") if config.get(k) is not None)
            return load_episode_dir(config["episode_dir"], _source_extra(seed, version, ignored), require_truth)
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
            return self._generate(seed, gen, version, ignored)
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
        source = {"kind": "split", "split": split, "episode": name, **_source_extra(seed, version, ignored)}
        return LoadedEpisode(_episode_id(public), public, truth, provenance, source, public_ids(public))

    def _generate(self, seed: Any, gen_config: Any, config_version: Any, ignored: list | None = None) -> LoadedEpisode:
        if seed is None or isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise EpisodeLoadError("seed (int or string) is required for generated episodes")
        ignored = list(ignored or [])
        try:
            cfg = resolve_generator_config(gen_config, ignored, prefix="team_config.")
            from generator.config import config_digest
            from generator.generate import generate_episode
        except ImportError as exc:  # pragma: no cover
            raise EpisodeLoadError("generator is not available") from exc
        episode = generate_episode(seed, cfg, salt=self.salt)
        public = dict(episode.public)
        source = {"kind": "generated", "generator_config": cfg.get("name"), "config_digest": config_digest(cfg),
                  "salted": bool(self.salt), **_source_extra(seed, config_version, ignored)}
        return LoadedEpisode(episode.episode_id, public, episode.truth, episode.provenance, source, public_ids(public))


def _source_extra(seed: Any, config_version: Any, ignored: list) -> dict:
    out = {"seed": seed, "config_version": config_version}
    if ignored:
        out["ignored_config_keys"] = sorted(set(ignored))
    return out


def generator_keys() -> frozenset:
    """Ключи верхнего уровня конфига генератора (generator/config.py::CONFIG_KEYS, включая необязательный
    variants) и "preset"."""
    try:
        from generator.config import CONFIG_KEYS
    except ImportError as exc:  # pragma: no cover
        raise EpisodeLoadError("generator is not available") from exc
    return frozenset(CONFIG_KEYS) | {"preset"}


def resolve_generator_config(gen_config: Any, ignored: list | None = None, prefix: str = "",
                             also_known=frozenset()) -> dict:
    """Конфиг генератора команды: имя пресета, объект-конфиг или {"preset": имя, ...переопределения}.
    Ключи верхнего уровня, которых генератор не знает (docs/generator.md), снимаются и дописываются в
    `ignored` (с префиксом prefix) — для журнала оценщика. Ключ, похожий на ключ генератора или на ключ из
    also_known ({"Errors": ...}, {"error": ...}, "team_confg"), — ошибка конфига; вложенные ключи (errors,
    scenarios, distractors) генератор проверяет строго. Иначе опечатка молча дала бы чистую компанию."""
    try:
        from generator.config import CONFIG_KEYS, ConfigError, normalize_config
        from generator.splits import resolve_config
    except ImportError as exc:  # pragma: no cover
        raise EpisodeLoadError("generator is not available") from exc
    if isinstance(gen_config, dict):
        known = set(CONFIG_KEYS) | {"preset"}
        reject_misspelled_keys(gen_config, known | set(also_known), prefix)
        dropped = sorted(str(k) for k in gen_config if k not in known)
        if ignored is not None:
            ignored.extend(prefix + k for k in dropped)
        gen_config = {k: v for k, v in gen_config.items() if k in known}
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
    except ConfigError as exc:  # текст — о конфиге самой обвязки (без путей и эталона)
        raise EpisodeLoadError(f"invalid generator config: {str(exc)[:300]}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EpisodeLoadError("invalid generator config") from exc
