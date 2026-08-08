from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import httpx

T = TypeVar("T")

# Retry transient gateway / rate-limit / network failures
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def request_with_retry(
    fn: Callable[[], T],
    *,
    retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    label: str = "http",
) -> T:
    """Call ``fn`` with exponential backoff on transient HTTP / network errors."""
    last: BaseException | None = None
    attempts = max(1, retries + 1)
    for attempt in range(attempts):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            last = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in _RETRY_STATUS or attempt + 1 >= attempts:
                raise
            delay = min(max_delay, base_delay * (2**attempt))
            print(f"[{label}] HTTP {status}; retry {attempt + 1}/{retries} in {delay:.1f}s")
            time.sleep(delay)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            if attempt + 1 >= attempts:
                raise
            delay = min(max_delay, base_delay * (2**attempt))
            print(f"[{label}] {type(exc).__name__}; retry {attempt + 1}/{retries} in {delay:.1f}s")
            time.sleep(delay)
    assert last is not None
    raise last
