"""
Production Triton Inference Client with Circuit Breaker
=======================================================

This module implements a production-grade asynchronous gRPC client for Triton Inference Server
with enterprise patterns: Singleton connection pooling, circuit breaker for fault tolerance,
structured error handling, and comprehensive telemetry.

Architecture:
- Async gRPC via tritonclient.grpc.aio for non-blocking inference
- Singleton pattern ensures single connection pool per process
- Circuit breaker pattern with automatic fallback to regex heuristics
- Tokenization using HuggingFace fast tokenizers (Rust-backed)
- Binary Protocol Buffers for efficient serialization

Author: Principal MLOps Engineer
"""

import asyncio
import itertools
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

try:
    import tritonclient.grpc.aio as grpcclient
    from tritonclient.utils import InferenceServerException
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from transformers import AutoTokenizer

from src.core.config import settings
from src.core.logging import get_logger
from src.core.exceptions import InferenceException
from src.schemas.firewall import ThreatType
from src.services.triton_channel_pool import _TritonChannelPool

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, using fallback
    HALF_OPEN = "half_open"  # Testing if service recovered


class ProductionInferenceClient:
    """
    Production-grade Triton gRPC client with Singleton pattern and circuit breaker.
    
    This client implements:
    1. Singleton pattern for connection pool management
    2. Async gRPC communication with Triton
    3. Circuit breaker for automatic fallback
    4. Structured error handling and telemetry
    5. Token-level input serialization
    
    Thread Safety: This class is thread-safe via asyncio event loop isolation.
    """
    
    _instance: Optional['ProductionInferenceClient'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        """Implement Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ProductionInferenceClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Triton client with multi-channel connection pool."""
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.triton_url = settings.triton_url
        self.model_name = settings.triton_model_name
        self.max_seq_length = 512
        
        # Multi-channel gRPC connection pool (for HTTP/2 multiplexing)
        self._channel_pool: Optional[_TritonChannelPool] = None
        
        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._failure_threshold = 5
        self._recovery_timeout = 30.0  # seconds
        self._last_failure_time = 0.0
        
        # Tokenizer (HuggingFace fast tokenizer with Rust backend)
        self._tokenizer = None
        
        # Performance metrics
        self._total_requests = 0
        self._total_failures = 0
        self._fallback_requests = 0
        
        logger.info(
            "ProductionInferenceClient initialized",
            extra={
                "triton_url": self.triton_url,
                "model_name": self.model_name,
                "triton_pool_size": settings.triton_pool_size,
                "triton_available": TRITON_AVAILABLE
            }
        )
    
    async def initialize(self) -> Nhannel pool and tokenizer.
        
        This method establishes the multi-channel gRPC pool and loads the tokenizer.
        It's safe to call multiple times (idempotent).
        """
        if not TRITON_AVAILABLE:
            logger.warning(
                "Tritonclient not available. Running in fallback mode only."
            )
            return
        
        try:
            # Initialize multi-channel pool
            if self._channel_pool is None:
                self._channel_pool = _TritonChannelPool(
                    url=self.triton_url,
                    pool_size=settings.triton_pool_size
                )
                await self._channel_pool.initialize()
                
                # Verify server is live using the first channel
                ch = self._channel_pool.acquire()
                if ch:
                    is_live = await ch.is_server_live()
                    if not is_live:
                        raise InferenceException(
                            "Triton server is not live",
                            details={"url": self.triton_url}
                        )
                    
                    # Verify model is ready
                    is_ready = await ch.is_model_ready(self.model_name)
                    if not is_ready:
                        raise InferenceException(
                            f"Model {self.model_name} is not ready",
                            details={"model": self.model_name}
                        )
                
                logger.info(
                    "Triton gRPC channel pool connected successfully",
                    extra={
                        "url": self.triton_url,
                        "model": self.model_name,
                        "pool_size": settings.triton_pool_size
                    }
                )
            
            # Initialize tokenizer
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "distilroberta-base",
                    use_fast=True  # Use Rust-backed fast tokenizer
                )
                logger.info("Tokenizer initialized successfully")
            
        except Exception as e:
            logger.error(
                f"Failed to initialize Triton client: {str(e)}",
                exc_info=True
            )
            # Don't raise - fall back to heuristic mode
            self._circuit_state = CircuitState.OPEN
                    extra={"url": self.triton_url, "model": self.model_name}
                )
            
            # Initialize tokenizer
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "distilroberta-base",
                    use_fast=True  # Use Rust-backed fast tokenizer
                )
                logger.info("Tokenizer initialized successfully")
            
        except Exception as e:
            logger.error(
                f"Failed to initialize Triton client: {str(e)}",
                exc_info=True
            )
            # Don't raise - fall back to heuristic mode
            self._circuit_state = CircuitState.OPEN
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows request.
        
        Returns:
            True if request should proceed, False if circuit is open
        """
        if self._circuit_state == CircuitState.CLOSED:
            return True
        
        if self._circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self._last_failure_time > self._recovery_timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self._circuit_state = CircuitState.HALF_OPEN
                return True
            return False
        
        # HALF_OPEN state - allow one request to test recovery
        return True
    
    def _record_success(self) -> None:
        """Record successful inference."""
        if self._circuit_state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker transitioning to CLOSED (service recovered)")
            self._circuit_state = CircuitState.CLOSED
            self._failure_count = 0 using channel pool.
        
        Args:
            text: Input text to classify
            return_embeddings: Whether to return embeddings (not supported in current config)
            
        Returns:
            Dictionary with inference results
            
        Raises:
            InferenceException: If inference fails
        """
        start_time = time.perf_counter()
        
        if self._channel_pool is None or not self._channel_pool.is_initialized():
            raise InferenceException("Triton channel pool not initialized")
        
        # Acquire a channel from the pool (round-robin)
        grpc_client = self._channel_pool.acquire()
        if grpc_client is None:
            raise InferenceException("Failed to acquire channel from pool")
        
        # Tokenize input using fast tokenizer
        tokens = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np"  # Return NumPy arrays directly
        )
        
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        
        # Create Triton InferInput objects (Protocol Buffers under the hood)
        inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        
        # Set binary data from NumPy arrays
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        # Define output
        outputs = [
            grpcclient.InferRequestedOutput("logits")
        ]
        
        # Execute inference via gRPC
        try:
            response = await _length,
            return_tensors="np"  # Return NumPy arrays directly
        )
        
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)
        
        # Create Triton InferInput objects (Protocol Buffers under the hood)
        inputs = [
            grpcclient.InferInput("input_ids", input_ids.shape, "INT64"),
            grpcclient.InferInput("attention_mask", attention_mask.shape, "INT64")
        ]
        
        # Set binary data from NumPy arrays
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        
        # Define output
        outputs = [
            grpcclient.InferRequestedOutput("logits")
        ]
        
        # Execute inference via gRPC
        try:
            response = await self._grpc_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=10.0  # 10ms timeout
            )
            
            # Extract logits
            logits = response.as_numpy("logits").flatten()
            
            # Compute softmax probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / exp_logits.sum()
            
            # Get predicted class and confidence
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            # Map to threat types
            threat_map = {
                0: ThreatType.NONE.value,
                1: ThreatType.PROMPT_INJECTION.value,
                2: ThreatType.PII_DETECTION.value
            }
            
            threat_type = threat_map.get(predicted_class, ThreatType.NONE.value)
            threat_detected = predicted_class != 0
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Check latency threshold
            if latency_ms > 10.0:
                logger.warning(
                    "Inference latency exceeded 10ms threshold",
                    extra={
                        "latency_ms": latency_ms,
                        "threshold_ms": 10.0
                    }
                )
            
            self._record_success()
            
            return {
                "threat_detected": threat_detected,
                "threat_type": threat_type,
                "confidence": confidence,
                "model_name": self.model_name,
                "latency_ms": latency_ms,
                "embeddings": None  # Not supported in current config
            }
            
        except InferenceServerException as e:
            logger.error(
                f"Triton inference failed: {str(e)}",
                extra={"error_code": e.status()}
            )
            self._record_failure()
            raise InferenceException(
                f"Triton inference failed: {str(e)}",
                details={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"Unexpected inference error: {str(e)}", exc_info=True)
            self._record_failure()
            raise InferenceException(f"Inference error: {str(e)}")
    
    def _infer_fallback_heuristic(self, text: str) -> Dict[str, Any]:
        """Lightweight CPU fallback using regex heuristics.
        
        This is a circuit breaker fallback when Triton is unavailable.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with heuristic classification results
        """
        start_time = time.perf_counter()
        self._fallback_requests += 1
        
        text_lower = text.lower()
        
        # Prompt injection patterns
        injection_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"disregard.*instructions",
            r"forget\s+everything",
            r"system\s+prompt",
            r"reveal\s+your\s+instructions"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, text_lower):
                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "threat_detected": True,
                    "threat_type": ThreatType.PROMPT_INJECTION.value,
                    "confidence": 0.85,  # Lower confidence for heuristics
                    "model_name": "fallback_heuristic",
                    "latency_ms": latency_ms,
                    "embeddings": None
                }
        
        # PII patterns
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
            r"\b[A-Z]\d{8}\b"  # Passport
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "threat_detected": True,
                    "threat_type": ThreatType.PII_DETECTION.value,
                    "confidence": 0.75,
                    "model_name": "fallback_heuristic",
                    "latency_ms": latency_ms,
                    "embeddings": None
                }
        
        # No threat detected
        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "threat_detected": False,
            "threat_type": ThreatType.NONE.value,
            "confidence": 0.20,
            "model_name": "fallback_heuristic",
            "latency_ms": latency_ms,
            "embeddings": None
        }
    
    async def infer(
        self,
        text: str,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """Perform inference with automatic fallback.
        
        This is the main entry point for inference. It handles:
        1. Circuit breaker checks
        2. Triton inference (primary)
        3. Fallback to heuristics (if Triton unavailable)
        4. Telemetry anhannel pool and cleanup resources."""
        if self._channel_pool is not None:
            try:
                await self._channel_pool.close()
                logger.info("Triton gRPC channel pool closed")
            except Exception as e:
                logger.error(f"Error closing channel pool
            Dictionary with classification results
        """
        self._total_requests += 1
        
        # Check circuit breaker
        if not self._check_circuit_breaker() or not TRITON_AVAILABLE:
            logger.debug(
                "Using fallback heuristics",
                extra={"circuit_state": self._circuit_state.value}
            )
            return self._infer_fallback_heuristic(text)
        
        # Try Triton inference
        try:
            result = await self._infer_triton(text, return_embeddings)
            return result
        except InferenceException:
            # Fall back to heuristics
            logger.warning("Falling back to heuristic classification")
            return self._infer_fallback_heuristic(text)
    
    async def close(self) -> None:
        """Close gRPC connection and cleanup resources."""
        if self._grpc_client is not None:
            try:
                await self._grpc_client.close()
                logger.info("Triton gRPC client closed")
            except Exception as e:
                logger.error(f"Error closing client: {str(e)}")
        
        # Log final statistics
        logger.info(
            "Inference client statistics",
            extra={
                "total_requests": self._total_requests,
                "total_failures": self._total_failures,
                "fallback_requests": self._fallback_requests,
                "failure_rate": (
                    self._total_failures / self._total_requests
                    if self._total_requests > 0 else 0
                )
            }
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "fallback_requests": self._fallback_requests,
            "circuit_state": self._circuit_state.value,
            "failure_rate": (
                self._total_failures / self._total_requests
                if self._total_requests > 0 else 0.0
            )
        }


# Global singleton instance
inference_client = ProductionInferenceClient()
