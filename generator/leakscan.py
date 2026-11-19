"""Сканер утечек эталона в траекториях/наблюдениях среды — перекрёстная проверка A.

Ищет в каждой строке trajectories.jsonl (поля observation, info, action и любые другие)
признаки закрытых данных соответствующего эпизода:
  * issue_id, scenario_group, имена сценариев, тексты cause/variant из provenance;
  * ключи и слова закрытых файлов: expected_amount_minor, evidence_sets, truth, provenance,
    amount_tolerance_minor, scenario, injections, private/ (путь);
  * seed и config_digest эпизода в полях info/observation;
  * сумму эталона, которой нет нигде в открытых данных эпизода.
Пример: python -m generator scan-trajectories trajectories/trajectories.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .fileio import read_episode_dir

PRIVATE_WORDS = re.compile(
    r"expected_amount_minor|evidence_sets|amount_tolerance_minor|scenario_group|injections|"
    r"provenance|truth\.json|[\\/]private[\\/]|\"issue_id\"",
    re.I,
)


def _episode_index(episodes_root: Path) -> dict[str, Path]:
    index = {}
    for manifest in episodes_root.glob("*/*/public/manifest.json"):
        with open(manifest, encoding="utf-8") as fh:
            index[json.load(fh)["episode_id"]] = manifest.parent.parent
    return index


def _secrets(episode_dir: Path) -> dict[str, Any]:
    public, private = read_episode_dir(episode_dir)
    truth = json.loads(private["truth.json"])
    prov = json.loads(private["provenance.json"])
    public_blob = "".join(public.values())
    tokens = set()
    for issue in truth["issues"]:
        tokens.add(issue["issue_id"])
        if issue["scenario_group"]:
            tokens.add(issue["scenario_group"])
        if issue["scenario"]:
            tokens.add(issue["scenario"])
    for inj in prov.get("injections", []):
        tokens.add(inj["cause"])
        tokens.add(inj["variant"])
    tokens.add(prov["config_digest"])
    # короткие числа (остаток шагов и т.п.) дают ложные срабатывания — смотрим суммы от 4 цифр
    hidden_amounts = {
        str(i["expected_amount_minor"])
        for i in truth["issues"]
        if i["expected_amount_minor"] >= 1000 and not re.search(rf"\b{i['expected_amount_minor']}\b", public_blob)
    }
    return {"tokens": tokens, "hidden_amounts": hidden_amounts, "seed": str(prov["seed"])}


def scan_records(records: Iterable[dict], episodes_root: str | Path) -> list[str]:
    root = Path(episodes_root)
    index = _episode_index(root)
    cache: dict[str, dict] = {}
    problems = []
    for n, rec in enumerate(records, start=1):
        ep_id = rec.get("episode_id")
        blob = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        if PRIVATE_WORDS.search(blob):
            problems.append(f"line {n}: private field name or path: {PRIVATE_WORDS.search(blob).group(0)!r}")
        if ep_id not in index:
            continue
        if ep_id not in cache:
            cache[ep_id] = _secrets(index[ep_id])
        secrets = cache[ep_id]
        for token in secrets["tokens"]:
            if token and token in blob:
                problems.append(f"line {n}: private token {token[:40]!r}")
        obs_blob = json.dumps({k: rec.get(k) for k in ("observation", "info")}, ensure_ascii=False)
        for amount in secrets["hidden_amounts"]:
            if re.search(rf"\b{amount}\b", obs_blob):
                problems.append(f"line {n}: truth amount {amount} not present in public data")
        if re.search(rf"\"seed\"\s*:\s*\"?{re.escape(secrets['seed'])}\b", obs_blob):
            problems.append(f"line {n}: episode seed exposed")
    return problems


def scan_trajectories(path: str | Path, episodes_root: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return scan_records(records, episodes_root)
