"""Deliberate failures surfaced by P-Helper."""

import re


def safe_error(exc: Exception, *secrets: str, limit: int = 400) -> str:
    value = f"{type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value)
    return value[:limit]


class PHelperError(RuntimeError):
    pass


class ParserError(PHelperError):
    pass


class LLMError(PHelperError):
    pass


class ValidationError(PHelperError):
    pass
