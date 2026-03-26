"""Guardrail service with pluggable strategies (Strategy pattern)."""
import asyncio
import hashlib
from typing import Dict, Any, List, Optional
import uuid
import time

from src.services.inference_client import inference_client
from src.db.qdrant import qdrant_manager
from src.repositories.telemetry_repo import telemetry_repo
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
        
        CRITICAL PATH: Only inference blocks the response. Qdrant enrichment
        and telemetry logging are fire-and-forget background tasks (asyncio.ensure_future).
        This ensures p99 latency stays ≤ 10ms at the FastAPI boundary.
        
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
            # CRITICAL PATH: Only ML inference blocks the response
            inference_result = await inference_client.infer(
                text=request.text,
                return_embeddings=True
            )
            
            threat_detected = inference_result["threat_detected"]
            threat_type_str = inference_result["threat_type"]
            confidence = inference_result["confidence"]
            embeddings = inference_result["embeddings"]
            
            # Apply threshold-based blocking policy
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
            
            # Build response immediately — do NOT wait for Qdrant or telemetry
            response = GuardrailCheckResponse(
                request_id=request_id,
                passed=passed,
                blocked=blocked,
                classification=classification,
                message=message,
                similar_threats=None  # Will be populated asynchronously
            )
            
            # BACKGROUND TASK (fire-and-forget): Qdrant enrichment + telemetry logging
            # Never awaited by caller — latency isolated from hot path
            asyncio.ensure_future(
                self._enrich_and_log(
                    text=request.text,
                    request_id=request_id,
                    inference_result=inference_result,
                    response=response,
                    threat_detected=threat_detected
                )
            )
            
            logger.info(
                f"Guardrail check completed (hot path only)",
                extra={
                    "request_id": request_id,
                    "blocked": blocked,
                    "threat_detected": threat_detected,
                    "confidence": confidence,
                    "hot_path_latency_ms": total_latency
                }
            )
            
            return response
            
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
    
    async def _enrich_and_log(
        self,
        text: str,
        request_id: str,
        inference_result: Dict[str, Any],
        response: GuardrailCheckResponse,
        threat_detected: bool,
    ) -> None:
        """Background enrichment: ANN lookup + telemetry. Never awaited by caller.
        
        This task runs asynchronously and NEVER blocks the hot path. If it takes
        200ms or times out, the client has already received their response.
        """
        try:
            # ANN search for similar historical threats (only if threat detected)
            similar_threat_ids: List[str] = []
            if threat_detected and inference_result.get("embeddings"):
                similar_threats = qdrant_manager.search_similar_threats(
                    query_vector=inference_result["embeddings"],
                    limit=3,
                    score_threshold=0.7
                )
                similar_threat_ids = [t.get("id") for t in similar_threats if t.get("id")]
            
            # Log telemetry (hash the text, never store plaintext in telemetry)
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            await telemetry_repo.log(
                request_id=request_id,
                text_hash=text_hash,
                threat_detected=threat_detected,
                confidence=inference_result.get("confidence", 0),
                threat_type=inference_result.get("threat_type"),
                blocked=response.blocked,
                similar_threat_ids=similar_threat_ids,
            )
            
            logger.debug(
                "Background enrichment completed",
                extra={
                    "request_id": request_id,
                    "similar_threats_count": len(similar_threat_ids)
                }
            )
        except Exception as exc:  # noqa: BLE001
            # Background task failure must never bubble up to client
            logger.warning(
                "Background enrichment failed",
                extra={"request_id": request_id, "error": str(exc)},
                exc_info=exc
            )


# Global guardrail service instance
guardrail_service = GuardrailService()




# Global guardrail service instance
guardrail_service = GuardrailService()
