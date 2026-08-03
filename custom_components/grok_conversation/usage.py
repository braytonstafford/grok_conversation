"""Token usage tracking for Grok Conversation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.storage import Store

from .const import (
    DEFAULT_INPUT_PRICE_PER_M,
    DEFAULT_OUTPUT_PRICE_PER_M,
    DOMAIN,
    EVENT_USAGE_UPDATED,
    LOGGER,
)

STORAGE_KEY = f"{DOMAIN}.usage"
STORAGE_VERSION = 1


@dataclass
class UsageSnapshot:
    """Aggregated usage counters."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    estimated_cost_usd: float = 0.0
    last_model: str = ""
    last_request_at: str | None = None
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_service: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "last_model": self.last_model,
            "last_request_at": self.last_request_at,
            "by_model": self.by_model,
            "by_service": self.by_service,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> UsageSnapshot:
        """Deserialize."""
        if not data:
            return cls()
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            request_count=int(data.get("request_count", 0)),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            last_model=str(data.get("last_model", "")),
            last_request_at=data.get("last_request_at"),
            by_model=dict(data.get("by_model") or {}),
            by_service=dict(data.get("by_service") or {}),
        )


class UsageTracker:
    """Persist and expose token usage statistics."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        self.hass = hass
        self.entry_id = entry_id
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}_{entry_id}")
        self.snapshot = UsageSnapshot()
        self._listeners: list[callback] = []

    async def async_load(self) -> None:
        """Load from disk."""
        data = await self._store.async_load()
        self.snapshot = UsageSnapshot.from_dict(data)
        LOGGER.debug("Loaded usage stats for %s: %s", self.entry_id, self.snapshot)

    async def async_save(self) -> None:
        """Persist to disk."""
        await self._store.async_save(self.snapshot.to_dict())

    def async_add_listener(self, listener: callback) -> callback:
        """Register a listener fired after each record."""
        self._listeners.append(listener)

        def _remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _remove

    async def async_record(
        self,
        *,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        service: str = "conversation",
        input_price_per_m: float = DEFAULT_INPUT_PRICE_PER_M,
        output_price_per_m: float = DEFAULT_OUTPUT_PRICE_PER_M,
    ) -> None:
        """Record a completed API call."""
        prompt_tokens = max(int(prompt_tokens or 0), 0)
        completion_tokens = max(int(completion_tokens or 0), 0)
        total = prompt_tokens + completion_tokens
        cost = (prompt_tokens / 1_000_000) * input_price_per_m + (
            completion_tokens / 1_000_000
        ) * output_price_per_m

        snap = self.snapshot
        snap.prompt_tokens += prompt_tokens
        snap.completion_tokens += completion_tokens
        snap.total_tokens += total
        snap.request_count += 1
        snap.estimated_cost_usd += cost
        snap.last_model = model or snap.last_model
        snap.last_request_at = datetime.now(timezone.utc).isoformat()

        model_key = model or "unknown"
        model_bucket = snap.by_model.setdefault(
            model_key,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        model_bucket["prompt_tokens"] += prompt_tokens
        model_bucket["completion_tokens"] += completion_tokens
        model_bucket["total_tokens"] += total
        model_bucket["request_count"] += 1
        model_bucket["estimated_cost_usd"] = round(
            float(model_bucket["estimated_cost_usd"]) + cost, 6
        )

        svc_bucket = snap.by_service.setdefault(
            service,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        svc_bucket["prompt_tokens"] += prompt_tokens
        svc_bucket["completion_tokens"] += completion_tokens
        svc_bucket["total_tokens"] += total
        svc_bucket["request_count"] += 1
        svc_bucket["estimated_cost_usd"] = round(
            float(svc_bucket["estimated_cost_usd"]) + cost, 6
        )

        await self.async_save()
        self.hass.bus.async_fire(
            EVENT_USAGE_UPDATED,
            {"entry_id": self.entry_id, **snap.to_dict()},
        )
        for listener in list(self._listeners):
            listener()

    async def async_reset(self) -> None:
        """Zero all counters."""
        self.snapshot = UsageSnapshot()
        await self.async_save()
        self.hass.bus.async_fire(
            EVENT_USAGE_UPDATED,
            {"entry_id": self.entry_id, **self.snapshot.to_dict()},
        )
        for listener in list(self._listeners):
            listener()
