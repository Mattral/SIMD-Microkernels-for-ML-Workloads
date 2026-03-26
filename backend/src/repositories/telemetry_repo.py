"""Repository for telemetry data access (Repository pattern)."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import numpy as np

from src.db.postgres import FirewallRequest
from src.schemas.telemetry import TelemetryMetrics, ThreatBreakdown, RequestLogEntry
from src.core.logging import get_logger
from src.core.exceptions import DatabaseException

logger = get_logger(__name__)


class TelemetryRepository:
    """Repository for accessing telemetry data."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session.
        
        Args:
            session: SQLAlchemy async session
        """
        self.session = session
    
    async def save_request(self, request_data: Dict[str, Any]) -> FirewallRequest:
        """Save firewall request to database.
        
        Args:
            request_data: Dictionary containing request data
            
        Returns:
            Created FirewallRequest instance
            
        Raises:
            DatabaseException: If save operation fails
        """
        try:
            firewall_request = FirewallRequest(**request_data)
            self.session.add(firewall_request)
            await self.session.flush()
            logger.info(
                f"Saved request",
                extra={
                    "request_id": firewall_request.request_id,
                    "blocked": firewall_request.blocked
                }
            )
            return firewall_request
        except Exception as e:
            logger.error(f"Failed to save request: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to save request: {str(e)}")
    
    async def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> TelemetryMetrics:
        """Get aggregated telemetry metrics.
        
        Args:
            start_time: Start of time range (defaults to 24 hours ago)
            end_time: End of time range (defaults to now)
            
        Returns:
            Aggregated telemetry metrics
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        try:
            # Get aggregated counts
            stmt = select(
                func.count(FirewallRequest.id).label('total'),
                func.sum(func.cast(FirewallRequest.blocked, type_=func.Integer())).label('blocked'),
                func.sum(func.cast(FirewallRequest.threat_detected, type_=func.Integer())).label('threats')
            ).where(
                and_(
                    FirewallRequest.timestamp >= start_time,
                    FirewallRequest.timestamp <= end_time
                )
            )
            
            result = await self.session.execute(stmt)
            row = result.first()
            
            total = row.total or 0
            blocked = row.blocked or 0
            threats = row.threats or 0
            
            # Get latency percentiles
            latency_stmt = select(FirewallRequest.latency_ms).where(
                and_(
                    FirewallRequest.timestamp >= start_time,
                    FirewallRequest.timestamp <= end_time
                )
            )
            latency_result = await self.session.execute(latency_stmt)
            latencies = [row[0] for row in latency_result.fetchall()]
            
            if latencies:
                avg_latency = float(np.mean(latencies))
                p50_latency = float(np.percentile(latencies, 50))
                p95_latency = float(np.percentile(latencies, 95))
                p99_latency = float(np.percentile(latencies, 99))
            else:
                avg_latency = p50_latency = p95_latency = p99_latency = 0.0
            
            return TelemetryMetrics(
                total_requests=total,
                blocked_requests=blocked,
                threats_detected=threats,
                avg_latency_ms=avg_latency,
                p50_latency_ms=p50_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency
            )
        except Exception as e:
            logger.error(f"Failed to get metrics: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to get metrics: {str(e)}")
    
    async def get_threat_breakdown(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> ThreatBreakdown:
        """Get breakdown of threats by type.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            Threat breakdown by type
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        try:
            stmt = select(
                FirewallRequest.threat_type,
                func.count(FirewallRequest.id).label('count')
            ).where(
                and_(
                    FirewallRequest.timestamp >= start_time,
                    FirewallRequest.timestamp <= end_time,
                    FirewallRequest.threat_detected == True
                )
            ).group_by(FirewallRequest.threat_type)
            
            result = await self.session.execute(stmt)
            
            breakdown = ThreatBreakdown()
            for row in result:
                threat_type = row.threat_type
                count = row.count
                if threat_type == "prompt_injection":
                    breakdown.prompt_injection = count
                elif threat_type == "pii_detection":
                    breakdown.pii_detection = count
                elif threat_type == "toxicity":
                    breakdown.toxicity = count
                elif threat_type == "malicious_code":
                    breakdown.malicious_code = count
            
            return breakdown
        except Exception as e:
            logger.error(f"Failed to get threat breakdown: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to get threat breakdown: {str(e)}")
    
    async def get_recent_requests(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[RequestLogEntry]:
        """Get recent requests with pagination.
        
        Args:
            limit: Maximum number of requests to return
            offset: Number of requests to skip
            
        Returns:
            List of request log entries
        """
        try:
            stmt = select(FirewallRequest).order_by(
                FirewallRequest.timestamp.desc()
            ).limit(limit).offset(offset)
            
            result = await self.session.execute(stmt)
            requests = result.scalars().all()
            
            return [
                RequestLogEntry(
                    request_id=req.request_id,
                    timestamp=req.timestamp,
                    endpoint=req.endpoint,
                    threat_detected=req.threat_detected,
                    threat_type=req.threat_type,
                    confidence=req.confidence_score,
                    latency_ms=req.latency_ms,
                    blocked=req.blocked,
                    input_tokens=req.input_tokens
                )
                for req in requests
            ]
        except Exception as e:
            logger.error(f"Failed to get recent requests: {str(e)}", exc_info=True)
            raise DatabaseException(f"Failed to get recent requests: {str(e)}")
