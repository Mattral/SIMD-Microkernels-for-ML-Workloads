"""Mock inference client for Triton Inference Server (Phase 1)."""
import random
import time
from typing import Dict, Any, List
import numpy as np

from src.core.logging import get_logger
from src.core.config import settings
from src.schemas.firewall import ThreatType

logger = get_logger(__name__)


class MockInferenceClient:
    """Mock client simulating Triton gRPC inference (Phase 1 only).
    
    In Phase 2, this will be replaced with actual tritonclient.grpc.aio implementation.
    """
    
    def __init__(self, model_name: str = "distilroberta-base"):
        """Initialize mock inference client.
        
        Args:
            model_name: Name of the model to simulate
        """
        self.model_name = model_name
        self.initialized = False
        logger.info(f"Mock inference client initialized", extra={"model": model_name})
    
    async def initialize(self) -> None:
        """Initialize the inference client (mock)."""
        self.initialized = True
        logger.info("Mock inference client ready")
    
    async def infer(
        self,
        text: str,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """Perform mock inference on input text.
        
        Args:
            text: Input text to classify
            return_embeddings: Whether to return embeddings (for Qdrant)
            
        Returns:
            Dictionary containing classification results and optional embeddings
        """
        start_time = time.perf_counter()
        
        # Simulate ultra-low latency inference (target < 10ms)
        await self._simulate_inference_delay()
        
        # Mock classification logic
        threat_detected, threat_type, confidence = self._mock_classify(text)
        
        # Mock embeddings (768-dim for DistilRoBERTa)
        embeddings = None
        if return_embeddings:
            embeddings = np.random.rand(768).tolist()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        result = {
            "threat_detected": threat_detected,
            "threat_type": threat_type,
            "confidence": confidence,
            "model_name": self.model_name,
            "latency_ms": latency_ms,
            "embeddings": embeddings
        }
        
        logger.debug(
            f"Mock inference completed",
            extra={
                "threat_detected": threat_detected,
                "latency_ms": latency_ms
            }
        )
        
        return result
    
    async def _simulate_inference_delay(self) -> None:
        """Simulate realistic inference latency."""
        # Simulate 2-8ms latency (well under 10ms target)
        delay_ms = random.uniform(2.0, 8.0)
        await self._sleep_ms(delay_ms)
    
    async def _sleep_ms(self, milliseconds: float) -> None:
        """Async sleep for specified milliseconds."""
        import asyncio
        await asyncio.sleep(milliseconds / 1000.0)
    
    def _mock_classify(self, text: str) -> tuple[bool, str, float]:
        """Mock classification logic with pattern matching.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (threat_detected, threat_type, confidence)
        """
        text_lower = text.lower()
        
        # Prompt injection patterns
        injection_patterns = [
            "ignore all previous instructions",
            "ignore previous",
            "disregard",
            "forget everything",
            "system prompt",
            "reveal your instructions"
        ]
        
        for pattern in injection_patterns:
            if pattern in text_lower:
                confidence = random.uniform(0.85, 0.98)
                return True, ThreatType.PROMPT_INJECTION.value, confidence
        
        # PII patterns
        pii_patterns = [
            "ssn", "social security",
            "credit card", "passport",
            "driver's license", "drivers license"
        ]
        
        for pattern in pii_patterns:
            if pattern in text_lower:
                confidence = random.uniform(0.80, 0.95)
                return True, ThreatType.PII_DETECTION.value, confidence
        
        # Toxicity patterns
        # (In production, use actual toxicity model)
        if len([c for c in text if c.isupper()]) / max(len(text), 1) > 0.5:
            # Excessive caps might indicate toxicity
            confidence = random.uniform(0.70, 0.85)
            return True, ThreatType.TOXICITY.value, confidence
        
        # No threat detected
        confidence = random.uniform(0.05, 0.25)
        return False, ThreatType.NONE.value, confidence
    
    async def close(self) -> None:
        """Close the inference client."""
        self.initialized = False
        logger.info("Mock inference client closed")


# Global inference client instance (Singleton pattern)
inference_client = MockInferenceClient(model_name=settings.triton_model_name)
