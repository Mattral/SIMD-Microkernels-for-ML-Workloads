"""GuardRail Studio - Main FastAPI Application.

Enterprise-grade LLM Firewall & Observability Platform.
Phase 1: Local monolithic core with mock inference.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.core.config import settings
from src.core.logging import setup_logging, get_logger
from src.core.exceptions import GuardRailException
from src.db.postgres import db_manager
from src.db.qdrant import qdrant_manager
from src.services.inference_client import inference_client
from src.services.kafka_producer import get_kafka_producer
from src.api.routes import health, firewall, telemetry

# Setup structured logging
setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("="*80)
    logger.info("GuardRail Studio - Phase 1 Starting")
    logger.info("="*80)
    
    try:
        # Initialize database
        logger.info("Initializing PostgreSQL database...")
        await db_manager.initialize()
        
        # Initialize Qdrant
        logger.info("Initializing Qdrant vector database...")
        qdrant_manager.initialize()
        
        # Initialize inference client
        logger.info("Initializing inference client (mock)...")
        await inference_client.initialize()
        
        # Initialize Kafka producer if enabled
        if settings.kafka_enabled:
            try:
                logger.info(f"Initializing Kafka producer with brokers: {settings.kafka_brokers}")
                kafka_producer = get_kafka_producer()
                await kafka_producer.start()
            except Exception as e:
                logger.warning(f"Kafka producer initialization failed: {str(e)}. Continuing without Kafka telemetry.")
        else:
            logger.info("Kafka telemetry disabled (set KAFKA_ENABLED=true to enable)")
        
        # Initialize W&B if API key is provided
        if settings.wandb_api_key:
            try:
                import wandb
                wandb.login(key=settings.wandb_api_key)
                logger.info(f"Weights & Biases initialized: {settings.wandb_project}")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {str(e)}. Continuing without W&B integration.")
        else:
            logger.warning("W&B API key not provided. Experiment tracking disabled.")
        
        logger.info("All systems initialized successfully")
        logger.info(f"Target latency: <{settings.request_timeout_ms}ms")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down GuardRail Studio...")
    
    try:
        await db_manager.close()
        qdrant_manager.close()
        await inference_client.close()
        
        # Shutdown Kafka producer if initialized
        if settings.kafka_enabled:
            try:
                kafka_producer = get_kafka_producer()
                await kafka_producer.stop()
            except Exception as e:
                logger.warning(f"Kafka producer shutdown error: {str(e)}")
        
        logger.info("Shutdown completed successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}", exc_info=True)


# Create FastAPI app
app = FastAPI(
    title="GuardRail Studio",
    description="Ultra-low-latency, high-throughput real-time LLM Firewall & Observability Platform",
    version="1.0.0-phase1",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(GuardRailException)
async def guardrail_exception_handler(request: Request, exc: GuardRailException):
    """Handle custom GuardRail exceptions."""
    logger.error(
        f"GuardRail exception",
        extra={
            "path": request.url.path,
            "message": exc.message,
            "status_code": exc.status_code
        }
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "details": exc.details}
    )


# Include routers with /api prefix
app.include_router(health.router, prefix="/api")
app.include_router(firewall.router, prefix="/api")
app.include_router(telemetry.router, prefix="/api")


@app.get("/api")
async def root():
    """Root endpoint."""
    return {
        "service": "GuardRail Studio",
        "version": "1.0.0-phase1",
        "status": "operational",
        "phase": "Phase 1: Local Monolithic Core",
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
