"""Интерфейс политики (docs/contract.md, раздел 9) и помощник для скриптовых политик.

    class Policy:
        name: str
        def reset(self, observation: dict, info: dict) -> None: ...
        def act(self, observation: dict) -> dict: ...   # одно действие среды

Политика видит только наблюдения среды. ScriptedPolicy позволяет писать сценарий как
генератор: `obs = yield action`; помощник read_table() листает таблицу курсорами.
"""

from __future__ import annotations

from typing import Any, Generator, Protocol

Script = Generator[dict, dict, None]


class Policy(Protocol):
    name: str

    def reset(self, observation: dict, info: dict) -> None: ...

    def act(self, observation: dict) -> dict: ...


def submit_action(episode_id: str, findings: list[dict]) -> dict:
    return {"action": "submit", "findings": {"episode_id": episode_id, "findings": findings[:30]}}


class TableReadError(RuntimeError):
    pass


class ScriptedPolicy:
    name = "scripted"

    def __init__(self) -> None:
        self._gen: Script | None = None
        self._started = False
        self.episode_id = ""
        self.initial: dict = {}

    def reset(self, observation: dict, info: dict | None = None) -> None:
        self.initial = observation
        self.episode_id = observation.get("episode_id", "")
        self._gen = self.script(observation)
        self._started = False

    def act(self, observation: dict) -> dict:
        if self._gen is None:
            raise RuntimeError("reset() first")
        try:
            if not self._started:
                self._started = True
                return next(self._gen)
            return self._gen.send(observation)
        except StopIteration:
            return submit_action(self.episode_id, [])

    def script(self, obs0: dict) -> Script:  # pragma: no cover — переопределяется
        yield submit_action(self.episode_id, [])

    # ------------------------------------------------------------------ помощники
    @staticmethod
    def read_table(table: str, limit: int = 50) -> Generator[dict, dict, list[dict]]:
        """Прочитать таблицу целиком: `rows = yield from self.read_table("ledger")`."""
        rows: list[dict] = []
        cursor: Any = None
        while True:
            obs = yield {"action": "list_records", "table": table, "cursor": cursor, "limit": limit}
            if obs.get("error"):
                raise TableReadError(obs["error"].get("code", "error"))
            result = obs["result"]
            rows.extend(result["records"])
            cursor = result["next_cursor"]
            if cursor is None:
                return rows
