"""
In-memory sliding-window rate limiter.
Provides thread-safe request throttling based on authenticated tenant and user identity.
Supports exact Retry-After header calculations, expired entry pruning, and multi-tenant isolation.
Zero external dependencies (pure Python standard library).
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from typing import Callable
from fastapi import Depends, HTTPException, status

from app.core.security import CurrentUser, get_current_user


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter.
    Stores timestamps per rate limit key and automatically prunes expired records.
    """

    def __init__(self, cleanup_interval_ops: int = 100):
        self._lock = threading.Lock()
        self._records: dict[str, list[float]] = defaultdict(list)
        self._cleanup_interval_ops = cleanup_interval_ops
        self._op_counter = 0

    def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
        now: float | None = None,
    ) -> None:
        """
        Records an attempt and raises HTTP 429 if the request count in the window exceeds max_requests.
        Includes a standard Retry-After header indicating when the oldest request expires.
        """
        current_time = time.time() if now is None else now
        cutoff = current_time - window_seconds

        with self._lock:
            self._op_counter += 1
            if self._op_counter >= self._cleanup_interval_ops:
                self._periodic_cleanup(current_time)
                self._op_counter = 0

            # Prune expired timestamps for this key
            timestamps = [ts for ts in self._records[key] if ts > cutoff]

            if len(timestamps) >= max_requests:
                # Calculate time until oldest timestamp leaves the sliding window
                oldest_ts = timestamps[0]
                retry_after = max(1, int(math.ceil(oldest_ts + window_seconds - current_time)))
                self._records[key] = timestamps
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Record current timestamp
            timestamps.append(current_time)
            self._records[key] = timestamps

    def _periodic_cleanup(self, current_time: float) -> None:
        """Internal helper to remove keys with empty or completely expired timestamp lists."""
        expired_keys = []
        for k, timestamps in self._records.items():
            valid = [ts for ts in timestamps if ts > current_time - 3600]
            if not valid:
                expired_keys.append(k)
            else:
                self._records[k] = valid
        for k in expired_keys:
            del self._records[k]

    def clear(self) -> None:
        """Reset all rate limiter state (useful for test isolation)."""
        with self._lock:
            self._records.clear()
            self._op_counter = 0

    def get_tracked_keys_count(self) -> int:
        """Return number of active tracked keys."""
        with self._lock:
            return len(self._records)


# Global rate limiter instance
limiter = SlidingWindowRateLimiter()


def rate_limit(
    max_requests: int,
    window_seconds: int,
    endpoint_tag: str,
) -> Callable:
    """
    FastAPI dependency factory enforcing rate limits keyed by tenant_id, user_id, and endpoint.
    """
    async def dependency(user: CurrentUser = Depends(get_current_user)) -> None:
        key = f"{user.tenant_id}:{user.id}:{endpoint_tag}"
        limiter.check_rate_limit(
            key=key,
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    return dependency
