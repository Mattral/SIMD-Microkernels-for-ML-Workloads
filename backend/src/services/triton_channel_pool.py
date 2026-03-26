"""Multi-channel gRPC connection pool for Triton Inference Server.

HTTP/2 multiplexed connections have a server-side MAX_CONCURRENT_STREAMS
limit (default 100 on Triton). At 20k+ RPS with ~5ms average inference time:
  - Concurrent streams needed: 20000 * 0.005 = 100 streams minimum
  - A single channel with 100 stream slots provides exactly zero headroom

We use a pool of TRITON_POOL_SIZE channels (default 4), providing 400 stream
slots: sufficient for 20k sustained RPS with 4× safety margin.
"""

import asyncio
import itertools
from typing import Iterator, Optional

try:
    import tritonclient.grpc.aio as grpcclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class _TritonChannelPool:
    """Round-robin pool of gRPC channels to a single Triton endpoint."""

    def __init__(self, url: str, pool_size: int = 4) -> None:
        """Initialize the channel pool.
        
        Args:
            url: Triton gRPC endpoint URL
            pool_size: Number of channels to maintain (default 4)
        """
        self._channels: list[grpcclient.InferenceServerClient] = []
        self._cycle: Optional[Iterator[grpcclient.InferenceServerClient]] = None
        self._url = url
        self._pool_size = pool_size
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create the connection pool. Call once at app startup."""
        if not TRITON_AVAILABLE:
            logger.warning("Triton client not available - pool initialization skipped")
            return

        async with self._lock:
            if self._channels:  # Already initialized
                return

            logger.info(
                "Initializing Triton gRPC channel pool",
                extra={"url": self._url, "pool_size": self._pool_size}
            )

            try:
                self._channels = [
                    grpcclient.InferenceServerClient(
                        url=self._url,
                        verbose=False,
                        channel_args=[
                            ("grpc.max_send_message_length", 16 * 1024 * 1024),
                            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                            ("grpc.keepalive_time_ms", 10_000),
                            ("grpc.keepalive_timeout_ms", 5_000),
                            ("grpc.http2.max_pings_without_data", 0),
                        ],
                    )
                    for _ in range(self._pool_size)
                ]
                self._cycle = itertools.cycle(self._channels)
                logger.info("Triton gRPC channel pool initialized successfully")
            except Exception as exc:
                logger.error(
                    "Failed to initialize Triton channel pool",
                    exc_info=exc
                )
                self._channels = []
                self._cycle = None

    def acquire(self) -> Optional[grpcclient.InferenceServerClient]:
        """Return the next channel in round-robin order. Thread-safe via GIL.
        
        Returns:
            Next channel from the pool, or None if pool not initialized
        """
        if self._cycle is None or not self._channels:
            return None
        return next(self._cycle)

    async def close(self) -> None:
        """Close all channels and cleanup resources."""
        for ch in self._channels:
            try:
                await ch.close()
            except Exception as exc:
                logger.warning("Error closing channel", exc_info=exc)
        self._channels.clear()
        self._cycle = None
        logger.info("Triton gRPC channel pool closed")

    def is_initialized(self) -> bool:
        """Check if the pool is initialized and ready."""
        return len(self._channels) > 0
