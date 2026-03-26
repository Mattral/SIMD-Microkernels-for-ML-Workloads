"""PostgreSQL database connection and models using SQLAlchemy 2.0."""
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, Integer, Boolean, DateTime, Text, Index
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
import uuid

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class FirewallRequest(Base):
    """Model for storing firewall request telemetry."""
    
    __tablename__ = "firewall_requests"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
    
    # Request metadata
    request_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    endpoint: Mapped[str] = mapped_column(String(255))
    method: Mapped[str] = mapped_column(String(10))
    
    # Content analysis
    input_text: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer)
    
    # Classification results
    threat_detected: Mapped[bool] = mapped_column(Boolean, index=True)
    threat_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    
    # Performance metrics
    latency_ms: Mapped[float] = mapped_column(Float, index=True)
    
    # Action taken
    blocked: Mapped[bool] = mapped_column(Boolean, index=True)
    
    __table_args__ = (
        Index('idx_timestamp_blocked', 'timestamp', 'blocked'),
        Index('idx_threat_confidence', 'threat_detected', 'confidence_score'),
    )


class DatabaseManager:
    """Singleton manager for database connections (Singleton pattern)."""
    
    _instance: Optional['DatabaseManager'] = None
    _engine: Optional[AsyncEngine] = None
    _session_maker: Optional[async_sessionmaker[AsyncSession]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    async def initialize(self, database_url: Optional[str] = None) -> None:
        """Initialize database engine and session maker.
        
        Args:
            database_url: Database connection URL (uses settings if not provided)
        """
        if self._engine is not None:
            logger.info("Database already initialized")
            return
        
        url = database_url or settings.postgres_url
        logger.info(f"Initializing database connection", extra={"url": url.split('@')[-1] if '@' in url else url})
        
        # Check if using SQLite
        is_sqlite = url.startswith("sqlite")
        
        # Create engine with appropriate parameters
        engine_kwargs = {
            "echo": False,
            "pool_pre_ping": True
        }
        
        if not is_sqlite:
            engine_kwargs.update({
                "pool_size": 10,
                "max_overflow": 20
            })
        
        self._engine = create_async_engine(url, **engine_kwargs)
        
        self._session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_maker = None
            logger.info("Database connections closed")
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session (Dependency Injection pattern).
        
        Yields:
            AsyncSession instance
        """
        if self._session_maker is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        async with self._session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Session error: {str(e)}", exc_info=True)
                raise
            finally:
                await session.close()


# Global database manager instance
db_manager = DatabaseManager()
