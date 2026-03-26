"""Health check and system status endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime, timezone
import time

from src.schemas.telemetry import SystemHealth
from src.api.dependencies import get_db_session
from src.db.qdrant import qdrant_manager
from src.services.inference_client import inference_client
from src.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])

# Track startup time for uptime calculation
startup_time = time.time()


@router.get("/", response_model=SystemHealth)
async def health_check(
    session: AsyncSession = Depends(get_db_session)
) -> SystemHealth:
    """Check system health status.
    
    Returns:
        System health status including all component connections
    """
    # Check database connection
    database_connected = False
    try:
        result = await session.execute(text("SELECT 1"))
        database_connected = True
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
    
    # Check Qdrant connection
    qdrant_connected = qdrant_manager._client is not None
    
    # Check Triton/inference client (mock in Phase 1)
    triton_connected = inference_client.initialized
    
    # Calculate uptime
    uptime = time.time() - startup_time
    
    # Determine overall status
    all_healthy = database_connected and qdrant_connected and triton_connected
    status = "healthy" if all_healthy else "degraded"
    
    return SystemHealth(
        status=status,
        database_connected=database_connected,
        qdrant_connected=qdrant_connected,
        triton_connected=triton_connected,
        uptime_seconds=uptime
    )
