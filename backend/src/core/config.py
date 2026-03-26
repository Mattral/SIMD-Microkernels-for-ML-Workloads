"""Application configuration using Pydantic Settings."""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """Global application settings with environment variable support."""
    
    # Database Configuration
    postgres_url: str = Field(
        default="sqlite+aiosqlite:///./guardrail_studio.db",
        description="Database connection URL (SQLite for Phase 1)"
    )
    db_name: str = Field(default="guardrail_studio", description="Database name")
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")
    qdrant_collection: str = Field(
        default="adversarial_patterns",
        description="Qdrant collection name for threat vectors"
    )
    
    # Triton Inference Server Configuration (Mock in Phase 1)
    triton_url: str = Field(
        default="localhost:8001",
        description="Triton Inference Server gRPC endpoint"
    )
    triton_model_name: str = Field(
        default="distilroberta_guardrail",
        description="Model name in Triton model repository"
    )
    
    # Guardrail Thresholds
    prompt_injection_threshold: float = Field(
        default=0.85,
        description="Classification confidence threshold for prompt injection"
    )
    pii_detection_threshold: float = Field(
        default=0.80,
        description="Classification confidence threshold for PII detection"
    )
    toxicity_threshold: float = Field(
        default=0.75,
        description="Classification confidence threshold for toxicity"
    )
    
    # Performance Configuration
    max_batch_size: int = Field(default=32, description="Maximum batch size for inference")
    request_timeout_ms: int = Field(
        default=10,
        description="Target latency for guardrail check in milliseconds"
    )
    
    # CORS Configuration
    cors_origins: str = Field(
        default="*",
        description="Allowed CORS origins (comma-separated)"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list."""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins
    
    # Weights & Biases Configuration
    wandb_api_key: str = Field(default="", description="W&B API key for experiment tracking")
    wandb_project: str = Field(
        default="guardrail-studio",
        description="W&B project name"
    )
    wandb_entity: str = Field(default="", description="W&B entity/team name")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Singleton settings instance
settings = Settings()
