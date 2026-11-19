from __future__ import annotations

from typing import Any


MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_EXPLANATION_LEN = 1000

REQUIRED_FINDING_KEYS = frozenset(
    {
        "type",
        "affected_ids",
        "claimed_amount_minor",
        "currency",
        "evidence_ids",
        "explanation",
    }
)


class SubmissionSchemaError(Exception):
    """Raised when submitted findings violate schema or limits."""


def validate_findings_payload(payload: Any, expected_episode_id: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise SubmissionSchemaError("Payload must be a JSON object")

    if payload.get("episode_id") != expected_episode_id:
        raise SubmissionSchemaError("episode_id does not match current episode")

    findings = payload.get("findings")
    if not isinstance(findings, list):
        raise SubmissionSchemaError("findings must be a list")

    if len(findings) > MAX_FINDINGS:
        raise SubmissionSchemaError(f"Too many findings: {len(findings)} > {MAX_FINDINGS}")

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(findings):
        if not isinstance(item, dict):
            raise SubmissionSchemaError(f"Finding at index {idx} must be an object")

        if not REQUIRED_FINDING_KEYS.issubset(item.keys()):
            raise SubmissionSchemaError(f"Finding at index {idx} is missing required keys")

        f_type = item["type"]
        affected_ids = item["affected_ids"]
        amount = item["claimed_amount_minor"]
        currency = item["currency"]
        evidence_ids = item["evidence_ids"]
        explanation = item["explanation"]

        if not isinstance(f_type, str) or not f_type.strip():
            raise SubmissionSchemaError(f"Finding at index {idx} has invalid 'type'")

        if (
            not isinstance(affected_ids, list)
            or len(affected_ids) == 0
            or not all(isinstance(x, str) and x.strip() for x in affected_ids)
        ):
            raise SubmissionSchemaError(f"Finding at index {idx} has invalid 'affected_ids'")

        if isinstance(amount, bool) or not isinstance(amount, int):
            raise SubmissionSchemaError(f"Finding at index {idx} has non-integer 'claimed_amount_minor'")

        if not isinstance(currency, str) or not currency.strip():
            raise SubmissionSchemaError(f"Finding at index {idx} has invalid 'currency'")

        if (
            not isinstance(evidence_ids, list)
            or len(evidence_ids) > MAX_EVIDENCE_IDS
            or not all(isinstance(x, str) and x.strip() for x in evidence_ids)
        ):
            raise SubmissionSchemaError(f"Finding at index {idx} has invalid 'evidence_ids'")

        if not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION_LEN:
            raise SubmissionSchemaError(f"Finding at index {idx} has invalid 'explanation'")

        normalized.append(
            {
                "type": f_type.strip(),
                "affected_ids": tuple(sorted(x.strip() for x in affected_ids)),
                "claimed_amount_minor": amount,
                "currency": currency.strip(),
                "evidence_ids": frozenset(x.strip() for x in evidence_ids),
                "explanation": explanation,
            }
        )

    return normalized