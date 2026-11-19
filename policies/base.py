from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    name: str = "base"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.step_count = 0
        self.episode_id: str = ""
        self.seen_records: dict[str, list[dict[str, Any]]] = {
            "ledger": [],
            "invoices": [],
            "payments": [],
            "fx": [],
        }
        self._seen_ids: dict[str, set[str]] = {
            "ledger": set(),
            "invoices": set(),
            "payments": set(),
            "fx": set(),
        }

    def reset(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> None:
        self.step_count = 0
        manifest = obs.get("manifest") if isinstance(obs.get("manifest"), dict) else {}
        self.episode_id = str(
            manifest.get("episode_id")
            or (info or {}).get("episode_id")
            or "unknown_episode"
        )
        self.seen_records = {"ledger": [], "invoices": [], "payments": [], "fx": []}
        self._seen_ids = {"ledger": set(), "invoices": set(), "payments": set(), "fx": []}
        self.ingest_obs(obs)

    def ingest_obs(self, obs: dict[str, Any]) -> None:
        if not isinstance(obs, dict):
            return
        manifest = obs.get("manifest")
        if isinstance(manifest, dict) and manifest.get("episode_id"):
            self.episode_id = str(manifest["episode_id"])

        for table in ("ledger", "invoices", "payments", "fx"):
            rows = obs.get(table)
            if isinstance(rows, list):
                self._add_rows(table, rows)

        records = obs.get("records")
        table_hint = obs.get("table")
        if isinstance(records, list) and isinstance(table_hint, str) and table_hint in self.seen_records:
            self._add_rows(table_hint, records)

    def _add_rows(self, table: str, rows: list[Any]) -> None:
        bucket = self.seen_records[table]
        seen = self._seen_ids[table]
        for idx, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            row_id = str(
                item.get("row_id")
                or item.get("record_id")
                or item.get("invoice_id")
                or item.get("payment_id")
                or item.get("id")
                or f"{table}_{len(bucket)}_{idx}"
            )
            if row_id not in seen:
                seen.add(row_id)
                enriched = dict(item)
                enriched.setdefault("_id", row_id)
                bucket.append(enriched)

    def make_submit(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "name": "submit",
            "args": {
                "episode_id": self.episode_id,
                "findings": findings,
            },
        }

    @abstractmethod
    def act(self, obs: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError