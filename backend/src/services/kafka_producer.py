"""Kafka telemetry producer for streaming events asynchronously."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from datetime import datetime

from aiokafka import AIOKafkaProducer

from src.core.logging import get_logger
from src.core.config import settings

logger = get_logger(__name__)


class KafkaTelemetryProducer:
    """Async Kafka producer for telemetry events.
    
    Sends firewall events, performance metrics, and drift signals to Kafka topics
    without blocking the request path.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_prefix: str = "guardrail-studio"
    ):
        """Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Comma-separated list of Kafka broker addresses
            topic_prefix: Prefix for all topics (e.g., "guardrail-studio")
        """
        self.bootstrap_servers = bootstrap_servers.split(",") if isinstance(bootstrap_servers, str) else bootstrap_servers
        self.topic_prefix = topic_prefix
        self.producer: Optional[AIOKafkaProducer] = None
        self._running = False

    async def start(self) -> None:
        """Start the Kafka producer."""
        if self._running:
            return

        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks="all",
                compression_type="snappy",
                max_in_flight_requests_per_connection=5,
            )
            await self.producer.start()
            self._running = True
            logger.info("Kafka telemetry producer started", extra={"brokers": self.bootstrap_servers})
        except Exception as exc:
            logger.error("Failed to start Kafka producer: %s", exc, extra={"brokers": self.bootstrap_servers})
            raise

    async def stop(self) -> None:
        """Stop the Kafka producer gracefully."""
        if not self._running or not self.producer:
            return

        try:
            await self.producer.stop()
            self._running = False
            logger.info("Kafka telemetry producer stopped")
        except Exception as exc:
            logger.error("Error stopping Kafka producer: %s", exc)

    async def send_firewall_event(self, event: Dict[str, Any]) -> None:
        """Send a firewall event (request checked, blocked, etc.).
        
        Args:
            event: Dictionary with keys like request_id, blocked, policy, timestamp
        """
        if not self._running or not self.producer:
            logger.warning("Kafka producer not running; discarding firewall event")
            return

        try:
            topic = f"{self.topic_prefix}.firewall-events"
            await self.producer.send_and_wait(topic, value=event)
            logger.debug(
                "Sent firewall event to Kafka",
                extra={"topic": topic, "request_id": event.get("request_id")}
            )
        except Exception as exc:
            logger.error(
                "Failed to send firewall event: %s",
                exc,
                extra={"request_id": event.get("request_id")}
            )

    async def send_performance_metric(self, metric: Dict[str, Any]) -> None:
        """Send a performance metric (latency, throughput, cache hit rate).
        
        Args:
            metric: Dictionary with metric data (latency_ms, endpoint, timestamp, etc.)
        """
        if not self._running or not self.producer:
            logger.warning("Kafka producer not running; discarding performance metric")
            return

        try:
            topic = f"{self.topic_prefix}.performance-metrics"
            await self.producer.send_and_wait(topic, value=metric)
            logger.debug("Sent performance metric to Kafka", extra={"topic": topic})
        except Exception as exc:
            logger.error("Failed to send performance metric: %s", exc)

    async def send_drift_signal(self, signal: Dict[str, Any]) -> None:
        """Send a data drift detection signal.
        
        Args:
            signal: Dictionary with drift data (metric_name, threshold, detected, timestamp)
        """
        if not self._running or not self.producer:
            logger.warning("Kafka producer not running; discarding drift signal")
            return

        try:
            topic = f"{self.topic_prefix}.drift-signals"
            await self.producer.send_and_wait(topic, value=signal)
            logger.info(
                "Sent drift signal to Kafka",
                extra={"topic": topic, "metric": signal.get("metric_name")}
            )
        except Exception as exc:
            logger.error("Failed to send drift signal: %s", exc)


# Global telemetry producer instance
_kafka_producer: Optional[KafkaTelemetryProducer] = None


def get_kafka_producer() -> KafkaTelemetryProducer:
    """Get or create the global Kafka telemetry producer."""
    global _kafka_producer
    if _kafka_producer is None:
        kafka_brokers = getattr(settings, "kafka_brokers", "localhost:9092")
        _kafka_producer = KafkaTelemetryProducer(
            bootstrap_servers=kafka_brokers,
            topic_prefix="guardrail-studio"
        )
    return _kafka_producer
