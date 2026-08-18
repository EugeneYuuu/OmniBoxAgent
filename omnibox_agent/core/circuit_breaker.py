"""Per-base_url circuit breaker for embedding and LLM API calls.

Fuses open after failure_threshold consecutive failures for open_seconds.
Half-open state allows one probe request; success closes, failure re-opens.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omnibox_agent.core.config import get_config

log = logging.getLogger(__name__)


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Breaker:
    state: BreakerState = BreakerState.CLOSED
    failure_count: int = 0
    opened_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    _probing: bool = False  # Issue #3: atomic single-flight probe flag

    def record_failure(self, open_seconds: int) -> None:
        with self.lock:
            self.failure_count += 1
            cfg = get_config().circuit_breaker
            if self.state == BreakerState.CLOSED and self.failure_count >= cfg.failure_threshold:
                self.state = BreakerState.OPEN
                self.opened_at = time.monotonic()
                log.warning("Circuit breaker opened (failures=%d)", self.failure_count)
            elif self.state == BreakerState.HALF_OPEN:
                self.state = BreakerState.OPEN
                self.opened_at = time.monotonic()
                self.failure_count = 1
                self._probing = False

    def record_success(self) -> None:
        with self.lock:
            self.state = BreakerState.CLOSED
            self.failure_count = 0
            self._probing = False

    def allow(self, open_seconds: int) -> bool:
        with self.lock:
            if self.state == BreakerState.CLOSED:
                return True
            if self.state == BreakerState.OPEN:
                if time.monotonic() - self.opened_at >= open_seconds:
                    self.state = BreakerState.HALF_OPEN
                    self._probing = True
                    return True
                return False
            # HALF_OPEN: issue #3 - true single-flight probe via atomic _probing flag
            if self.state == BreakerState.HALF_OPEN:
                if not self._probing:
                    self._probing = True
                    return True
                return False
            return False


class CircuitBreakerRegistry:
    """Per-base_url circuit breaker registry with LRU eviction."""

    def __init__(self) -> None:
        self._breakers: dict[str, _Breaker] = {}
        self._lock = threading.Lock()

    def _get_or_create(self, base_url: str) -> _Breaker:
        with self._lock:
            if base_url not in self._breakers:
                cfg = get_config().circuit_breaker
                if len(self._breakers) >= cfg.max_breakers:
                    # Simple LRU: evict oldest
                    oldest = next(iter(self._breakers))
                    del self._breakers[oldest]
                self._breakers[base_url] = _Breaker()
            return self._breakers[base_url]

    def allow(self, base_url: str) -> bool:
        cfg = get_config().circuit_breaker
        return self._get_or_create(base_url).allow(cfg.open_seconds)

    def record_success(self, base_url: str) -> None:
        self._get_or_create(base_url).record_success()

    def record_failure(self, base_url: str) -> None:
        cfg = get_config().circuit_breaker
        self._get_or_create(base_url).record_failure(cfg.open_seconds)

    async def call(
        self,
        base_url: str,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call fn(*args, **kwargs) with circuit breaker protection."""
        if not self.allow(base_url):
            raise CircuitBreakerOpenError(f"Circuit breaker open for {base_url}")

        try:
            result = await fn(*args, **kwargs)
            self.record_success(base_url)
            return result
        except Exception:
            self.record_failure(base_url)
            raise


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Module-level singleton
_circuit_breaker_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker() -> CircuitBreakerRegistry:
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry()
    return _circuit_breaker_registry
