"""Production PostgreSQL Connection Manager with Optimized Pooling.

This module provides an enhanced database manager for high-throughput
production environments with connection pooling tuned for low-latency writes.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from typing import Optional, AsyncGenerator
import os

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ProductionDatabaseManager:
    """Enhanced database manager for production PostgreSQL.
    
    Features:
    - Optimized connection pooling for high throughput
    - Automatic connection recycling
    - Health check integration
    - Statement timeout enforcement
    """
    
    _instance: Optional['ProductionDatabaseManager'] = None
    _engine: Optional[AsyncEngine] = None
    _session_maker: Optional[async_sessionmaker[AsyncSession]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProductionDatabaseManager, cls).__new__(cls)
        return cls._instance
    
    async def initialize(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_recycle: int = 1800,
        pool_pre_ping: bool = True,
        echo: bool = False
    ) -> None:
        """Initialize database engine with production settings.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Number of persistent connections
            max_overflow: Max temporary connections beyond pool_size
            pool_recycle: Recycle connections after N seconds
            pool_pre_ping: Test connections before using
            echo: Enable SQL query logging
        """
        if self._engine is not None:
            logger.info("Database already initialized")
            return
        
        url = database_url or settings.postgres_url
        is_sqlite = url.startswith("sqlite")
        
        # Production PostgreSQL configuration
        if not is_sqlite:
            self._engine = create_async_engine(
                url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_recycle=pool_recycle,
                pool_pre_ping=pool_pre_ping,
                pool_timeout=30,
                echo=echo,
                connect_args={
                    "statement_timeout": "30000",  # 30s statement timeout
                    "server_settings": {
                        "application_name": "guardrail-studio",
                        "jit": "off"  # Disable JIT for predictable latency
                    }
                }
            )
            
            logger.info(
                "Production PostgreSQL engine initialized",
                extra={
                    "pool_size": pool_size,
                    "max_overflow": max_overflow,
                    "pool_recycle": pool_recycle
                }
            )
        else:
            # Development SQLite configuration
            self._engine = create_async_engine(
                url,
                echo=echo,
                poolclass=NullPool  # SQLite doesn't support pooling
            )
            
            logger.info("SQLite engine initialized (development mode)")
        
        # Create session maker
        self._session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False
        )
        
        # Create tables if needed (development only)
        if is_sqlite:
            from src.db.postgres import Base
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
    
    async def close(self) -> None:
        """Close database engine and cleanup connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_maker = None
            logger.info("Database engine disposed")
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session.
        
        Yields:
            AsyncSession instance
        """
        if self._session_maker is None:
            raise RuntimeError("Database not initialized")
        
        async with self._session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Perform database health check.
        
        Returns:
            True if database is healthy
        """
        if self._engine is None:
            return False
        
        try:
            async with self._engine.connect() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False


# Global production database manager
production_db = ProductionDatabaseManager()
