"""Firewall proxy endpoints for guardrail checks."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import time

from src.schemas.firewall import (
    GuardrailCheckRequest,
    GuardrailCheckResponse,
    ProxyRequest,
    ProxyResponse
)
from src.services.guardrail_service import guardrail_service
from src.repositories.telemetry_repo import TelemetryRepository
from src.api.dependencies import get_db_session, get_telemetry_repo
from src.core.logging import get_logger
from src.core.exceptions import InferenceException, GuardRailException

logger = get_logger(__name__)
router = APIRouter(prefix="/firewall", tags=["Firewall"])


@router.post("/check", response_model=GuardrailCheckResponse)
async def check_guardrails(
    request: GuardrailCheckRequest,
    session: AsyncSession = Depends(get_db_session)
) -> GuardrailCheckResponse:
    """Perform guardrail check on input text.
    
    Args:
        request: Guardrail check request
        session: Database session (injected)
        
    Returns:
        Guardrail check response with classification results
        
    Raises:
        HTTPException: If guardrail check fails
    """
    start_time = time.perf_counter()
    
    try:
        # Perform guardrail check
        result = await guardrail_service.check_guardrails(request)
        
        # Save telemetry
        telemetry_repo = TelemetryRepository(session)
        await telemetry_repo.save_request({
            "request_id": result.request_id,
            "endpoint": request.endpoint,
            "method": "POST",
            "input_text": request.text,
            "input_tokens": len(request.text.split()),
            "threat_detected": result.classification.threat_type.value != "none",
            "threat_type": result.classification.threat_type.value,
            "confidence_score": result.classification.confidence,
            "model_name": result.classification.model_name,
            "latency_ms": result.classification.latency_ms,
            "blocked": result.blocked
        })
        
        total_latency = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Guardrail check endpoint completed",
            extra={
                "request_id": result.request_id,
                "total_latency_ms": total_latency
            }
        )
        
        return result
        
    except GuardRailException as e:
        logger.error(f"Guardrail check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=e.status_code,
            detail={"message": e.message, "details": e.details}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Internal server error", "error": str(e)}
        )


@router.post("/proxy", response_model=ProxyResponse)
async def proxy_request(
    request: ProxyRequest,
    session: AsyncSession = Depends(get_db_session)
) -> ProxyResponse:
    """Proxy request to LLM with guardrail protection.
    
    This endpoint demonstrates the full firewall flow:
    1. Check guardrails on user prompt
    2. If passed, forward to LLM (mocked in Phase 1)
    3. Return response or block message
    
    Args:
        request: Proxy request with prompt and LLM parameters
        session: Database session (injected)
        
    Returns:
        Proxy response with guardrail result and optional LLM response
    """
    try:
        # Step 1: Check guardrails
        guardrail_check = GuardrailCheckRequest(
            text=request.prompt,
            endpoint="/api/firewall/proxy"
        )
        
        guardrail_result = await guardrail_service.check_guardrails(guardrail_check)
        
        # Step 2: If passed, forward to LLM (mocked in Phase 1)
        llm_response = None
        if guardrail_result.passed:
            # Mock LLM response (in production, call actual LLM API)
            llm_response = f"[Mock LLM Response] This is a safe response to: {request.prompt[:50]}..."
            logger.info(
                f"Request forwarded to LLM",
                extra={"request_id": guardrail_result.request_id}
            )
        else:
            logger.warning(
                f"Request blocked by guardrail",
                extra={
                    "request_id": guardrail_result.request_id,
                    "reason": guardrail_result.message
                }
            )
        
        # Save telemetry
        telemetry_repo = TelemetryRepository(session)
        await telemetry_repo.save_request({
            "request_id": guardrail_result.request_id,
            "endpoint": "/api/firewall/proxy",
            "method": "POST",
            "input_text": request.prompt,
            "input_tokens": len(request.prompt.split()),
            "threat_detected": guardrail_result.classification.threat_type.value != "none",
            "threat_type": guardrail_result.classification.threat_type.value,
            "confidence_score": guardrail_result.classification.confidence,
            "model_name": guardrail_result.classification.model_name,
            "latency_ms": guardrail_result.classification.latency_ms,
            "blocked": guardrail_result.blocked
        })
        
        return ProxyResponse(
            request_id=guardrail_result.request_id,
            blocked=guardrail_result.blocked,
            guardrail_result=guardrail_result,
            llm_response=llm_response
        )
        
    except Exception as e:
        logger.error(f"Proxy request failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Proxy request failed", "error": str(e)}
        )
