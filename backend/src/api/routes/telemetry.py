"""Telemetry and observability endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime, timedelta, timezone

from src.schemas.telemetry import TelemetryMetrics, ThreatBreakdown, RequestLogEntry
from src.repositories.telemetry_repo import TelemetryRepository
from src.api.dependencies import get_db_session
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.get("/metrics", response_model=TelemetryMetrics)
async def get_metrics(
    hours: int = Query(default=24, ge=1, le=168, description="Time range in hours"),
    session: AsyncSession = Depends(get_db_session)
) -> TelemetryMetrics:
    """Get aggregated telemetry metrics.
    
    Args:
        hours: Time range in hours (default 24, max 168 = 1 week)
        session: Database session (injected)
        
    Returns:
        Aggregated telemetry metrics
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)
    
    telemetry_repo = TelemetryRepository(session)
    metrics = await telemetry_repo.get_metrics(start_time, end_time)
    
    return metrics


@router.get("/threats", response_model=ThreatBreakdown)
async def get_threat_breakdown(
    hours: int = Query(default=24, ge=1, le=168, description="Time range in hours"),
    session: AsyncSession = Depends(get_db_session)
) -> ThreatBreakdown:
    """Get breakdown of threats by type.
    
    Args:
        hours: Time range in hours (default 24, max 168 = 1 week)
        session: Database session (injected)
        
    Returns:
        Threat breakdown by type
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)
    
    telemetry_repo = TelemetryRepository(session)
    breakdown = await telemetry_repo.get_threat_breakdown(start_time, end_time)
    
    return breakdown


@router.get("/requests", response_model=List[RequestLogEntry])
async def get_recent_requests(
    limit: int = Query(default=100, ge=1, le=1000, description="Max number of requests"),
    offset: int = Query(default=0, ge=0, description="Number of requests to skip"),
    session: AsyncSession = Depends(get_db_session)
) -> List[RequestLogEntry]:
    """Get recent firewall requests with pagination.
    
    Args:
        limit: Maximum number of requests to return
        offset: Number of requests to skip (for pagination)
        session: Database session (injected)
        
    Returns:
        List of recent request log entries
    """
    telemetry_repo = TelemetryRepository(session)
    requests = await telemetry_repo.get_recent_requests(limit, offset)
    
    return requests
