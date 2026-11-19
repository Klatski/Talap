"""Оценщик (зона B): валидация submit, matcher, метрики, награда, режимы train/eval.

Публичный API:
    score_submission(truth, submission, inspection_steps, *, provenance=None, truncated=False) -> dict
    validate_submission(submission, episode_id=None) -> list[str]
    match(findings, issues) -> MatchResult
Процесс оценщика: python -m grader.service (его запускает среда, см. env/server.py).
"""

from .matcher import match
from .metrics import experimental_reward, prf, shared_reward
from .scoring import score_submission
from .validate import validate_submission
from .version import GRADER_VERSION

__all__ = [
    "GRADER_VERSION",
    "experimental_reward",
    "match",
    "prf",
    "score_submission",
    "shared_reward",
    "validate_submission",
]
