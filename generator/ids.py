"""Непрозрачные ID: префикс типа + случайный хвост из локального RNG.

Хвост не зависит от порядка генерации и не несёт информации об ошибках
(никаких dup_/err_/последовательных номеров и пропусков).
"""

from __future__ import annotations

import random

ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"


class IdFactory:
    def __init__(self, rng: random.Random, used: set[str] | None = None) -> None:
        self._rng = rng
        self._used: set[str] = set(used or ())

    def new(self, prefix: str, length: int) -> str:
        while True:
            candidate = prefix + "".join(self._rng.choice(ALPHABET) for _ in range(length))
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate
