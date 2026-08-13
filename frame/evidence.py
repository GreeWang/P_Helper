"""Shared evidence validation helpers."""

import re

from .errors import ValidationError


NUMBER_PATTERN = re.compile(r"(?<!\d)[-+]?\d+(?:[.,]\d+)?[%％]?")


def numbers(value: str) -> list[str]:
    return NUMBER_PATTERN.findall(value)


def validate_numbers(text: str, evidence: str, label: str):
    evidence_numbers = {normalise_number(item) for item in numbers(evidence)}
    missing = [item for item in numbers(text)
               if normalise_number(item) not in evidence_numbers]
    if missing:
        raise ValidationError(f"Unsupported numeric value in {label}: {', '.join(missing)}")


def normalise_number(value: str) -> str:
    return value.replace(",", "").replace("％", "%")
