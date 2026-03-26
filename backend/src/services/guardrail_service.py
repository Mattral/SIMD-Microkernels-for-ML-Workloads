"""Guardrail service with pluggable strategies (Strategy pattern)."""
from typing import Dict, Any, List
import uuid
import time

from src.services.inference_client import inference_client
from src.db.qdrant import qdrant_manager
from src.schemas.firewall import (
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    ClassificationResult,
    ThreatType
)
from src.core.config import settings
from src.core.logging import get_logger
from src.core.exceptions import InferenceException, ThresholdExceededException

logger = get_logger(__name__)


class GuardrailService:
    """Service for performing guardrail checks with pluggable policies."""
    
    def __init__(self):
        """Initialize guardrail service."""
        self.thresholds = {
            ThreatType.PROMPT_INJECTION: settings.prompt_injection_threshold,
            ThreatType.PII_DETECTION: settings.pii_detection_threshold,
            ThreatType.TOXICITY: settings.toxicity_threshold
        }
    
    async def check_guardrails(
        self,
        request: GuardrailCheckRequest
    ) -> GuardrailCheckResponse:
        """Perform comprehensive guardrail check on input text.
        
        Args:
            request: Guardrail check request
            
        Returns:
            Guardrail check response with classification results
            
        Raises:
            InferenceException: If inference fails
        """
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        start_time = time.perf_counter()
        
        logger.info(
            f"Starting guardrail check",
            extra={
                "request_id": request_id,
                "endpoint": request.endpoint,
                "text_length": len(request.text)
            }
        )
        
        try:
            # Step 1: Run ML inference
            inference_result = await inference_client.infer(
                text=request.text,
                return_embeddings=True
            )
            
            threat_detected = inference_result["threat_detected"]
            threat_type_str = inference_result["threat_type"]
            confidence = inference_result["confidence"]
            embeddings = inference_result["embeddings"]
            
            # Step 2: Check against vector database for similar historical threats
            similar_threats = []
            if threat_detected and embeddings:
                similar_threats = qdrant_manager.search_similar_threats(
                    query_vector=embeddings,
                    limit=3,
                    score_threshold=0.7
                )
            
            # Step 3: Apply threshold-based blocking policy
            threat_type = ThreatType(threat_type_str)
            threshold = self.thresholds.get(threat_type, 0.9)
            blocked = threat_detected and confidence >= threshold
            passed = not blocked
            
            # Create classification result
            classification = ClassificationResult(
                threat_type=threat_type,
                confidence=confidence,
                model_name=inference_result["model_name"],
                latency_ms=inference_result["latency_ms"]
            )
            
            # Generate human-readable message
            if blocked:
                message = f"Request blocked: {threat_type.value} detected (confidence: {confidence:.2f})"
            elif threat_detected:
                message = f"Threat detected but below threshold: {threat_type.value} (confidence: {confidence:.2f})"
            else:
                message = "Request passed all guardrail checks"
            
            total_latency = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                f"Guardrail check completed",
                extra={
                    "request_id": request_id,
                    "blocked": blocked,
                    "threat_detected": threat_detected,
                    "confidence": confidence,
                    "total_latency_ms": total_latency
                }
            )
            
            return GuardrailCheckResponse(
                request_id=request_id,
                passed=passed,
                blocked=blocked,
                classification=classification,
                message=message,
                similar_threats=similar_threats if similar_threats else None
            )
            
        except Exception as e:
            logger.error(
                f"Guardrail check failed",
                extra={"request_id": request_id, "error": str(e)},
                exc_info=True
            )
            raise InferenceException(
                f"Guardrail check failed: {str(e)}",
                details={"request_id": request_id}
            )


# Global guardrail service instance
guardrail_service = GuardrailService()
