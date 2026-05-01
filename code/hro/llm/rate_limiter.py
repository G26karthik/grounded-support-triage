"""Async sliding-window rate limiter.

Pre-empts 429s by ensuring we never exceed N requests in any rolling 60s
window. Cheap, deterministic, no third-party dep.

Configured via the `HRO_RPM` env var (defaults to 7 — under the free-tier 10
RPM ceiling on gemini-2.5-flash, with headroom for burstiness in retries).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque


class RateLimiter:
    """Allows up to `rpm` calls per rolling 60-second window."""

    def __init__(self, rpm: int) -> None:
        self.rpm = max(rpm, 1)
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a slot is available, then consume it."""
        while True:
            async with self._lock:
                now = time.monotonic()
                # Drop timestamps older than 60s.
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.rpm:
                    self._timestamps.append(now)
                    return
                # Compute the soonest a slot will free up.
                wait = 60.0 - (now - self._timestamps[0]) + 0.05
            await asyncio.sleep(max(wait, 0.1))


# Process-singleton.
_default_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    global _default_limiter
    if _default_limiter is None:
        rpm = int(os.environ.get("HRO_RPM", "7"))
        _default_limiter = RateLimiter(rpm=rpm)
    return _default_limiter
