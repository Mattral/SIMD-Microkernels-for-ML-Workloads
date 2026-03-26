"""FastAPI dependency injection utilities."""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.postgres import db_manager
from src.repositories.telemetry_repo import TelemetryRepository
from src.core.logging import get_logger

logger = get_logger(__name__)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session.
    
    Yields:
        AsyncSession instance
    """
    async for session in db_manager.get_session():
        yield session


async def get_telemetry_repo(
    session: AsyncSession = None
) -> TelemetryRepository:
    """Dependency for getting telemetry repository.
    
    Args:
        session: Database session (injected)
        
    Returns:
        TelemetryRepository instance
    """
    return TelemetryRepository(session)
