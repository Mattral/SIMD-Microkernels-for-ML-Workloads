"""Redis-backed prompt classification cache.

Architecture note: This cache sits between the FastAPI route handler and
GuardrailService. A cache hit returns in <1ms, enabling us to serve repeated
prompts (common in adversarial scanners and test harnesses) without hitting Triton.

At 20k+ RPS with 5% cache hit rate (conservative), this eliminates 1000 Triton
round-trips per second, directly reducing p99 latency and GPU load.
"""

import hashlib
from typing import Optional
import aioredis
from src.schemas.firewall import GuardrailResult
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """Singleton Redis cache for guardrail classifications."""

    _instance: Optional["CacheService"] = None
    _client: Optional[aioredis.Redis] = None

    def __new__(cls) -> "CacheService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self) -> None:
        """Create the Redis connection pool. Call once at app startup."""
        if self._client is not None:
            return  # Already initialized

        try:
            self._client = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
                socket_connect_timeout=1.0,
                socket_timeout=0.5,  # 500ms timeout — never block hot path
            )
            logger.info("Redis cache initialized", extra={"url": settings.REDIS_URL})
        except Exception as exc:
            logger.warning(
                "Redis cache initialization failed — cache disabled",
                exc_info=exc
            )
            self._client = None

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key from normalized text.
        
        Normalization: strip whitespace and lowercase to catch repeated
        prompts that differ only in formatting.
        """
        normalized = text.strip().lower()
        return f"guardrail:v1:{hashlib.sha256(normalized.encode()).hexdigest()}"

    async def get(self, text: str) -> Optional[GuardrailResult]:
        """Return cached result or None. Fails open — never raises.
        
        Args:
            text: Input prompt
            
        Returns:
            Cached GuardrailResult or None if not in cache or on error
        """
        if not self._client:
            return None

        try:
            raw = await self._client.get(self._cache_key(text))
            if raw:
                return GuardrailResult.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cache GET failed — proceeding without cache",
                exc_info=exc
            )
        return None

    async def set(self, text: str, result: GuardrailResult) -> None:
        """Cache a result. Fails silently — never raises, never blocks caller.
        
        Args:
            text: Input prompt
            result: GuardrailResult to cache
        """
        if not self._client:
            return

        try:
            await self._client.setex(
                name=self._cache_key(text),
                time=settings.CACHE_TTL_SECONDS,
                value=result.model_dump_json(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache SET failed", exc_info=exc)


cache_service = CacheService()
