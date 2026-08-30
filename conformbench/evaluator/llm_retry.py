"""Small retry wrapper for evaluator LLM calls."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from loguru import logger


def invoke_with_retries(
    model: Any,
    messages: Sequence[dict[str, Any]],
    *,
    description: str,
    max_attempts: int = 6,
) -> Any:
    """Invoke an LLM call with bounded retries for transient transport failures."""

    delays = [3, 10, 30, 60, 120]
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return model.invoke(messages)
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            delay = delays[min(attempt - 1, len(delays) - 1)]
            logger.warning(
                "{} failed on attempt {}/{} with {}: {}; retrying in {}s",
                description,
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)

    assert last_error is not None
    logger.error(
        "{} failed after {} attempts with {}: {}",
        description,
        max_attempts,
        type(last_error).__name__,
        last_error,
    )
    raise last_error
