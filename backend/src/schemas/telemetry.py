"""Pydantic schemas for telemetry and metrics."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class TelemetryMetrics(BaseModel):
    """Aggregated telemetry metrics."""
    
    total_requests: int = Field(..., description="Total number of requests processed")
    blocked_requests: int = Field(..., description="Number of blocked requests")
    threats_detected: int = Field(..., description="Number of threats detected")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    p50_latency_ms: float = Field(..., description="p50 latency in milliseconds")
    p95_latency_ms: float = Field(..., description="p95 latency in milliseconds")
    p99_latency_ms: float = Field(..., description="p99 latency in milliseconds")


class ThreatBreakdown(BaseModel):
    """Breakdown of threats by type."""
    
    prompt_injection: int = Field(default=0, description="Prompt injection attempts")
    pii_detection: int = Field(default=0, description="PII leakage detections")
    toxicity: int = Field(default=0, description="Toxic content detections")
    malicious_code: int = Field(default=0, description="Malicious code detections")


class SystemHealth(BaseModel):
    """System health status."""
    
    status: str = Field(..., description="Overall system status")
    database_connected: bool = Field(..., description="Database connection status")
    qdrant_connected: bool = Field(..., description="Qdrant connection status")
    triton_connected: bool = Field(..., description="Triton server connection status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")


class RequestLogEntry(BaseModel):
    """Individual request log entry."""
    
    request_id: str = Field(..., description="Request identifier")
    timestamp: datetime = Field(..., description="Request timestamp")
    endpoint: str = Field(..., description="Request endpoint")
    threat_detected: bool = Field(..., description="Whether threat was detected")
    threat_type: Optional[str] = Field(None, description="Type of threat detected")
    confidence: float = Field(..., description="Classification confidence")
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    blocked: bool = Field(..., description="Whether request was blocked")
    input_tokens: int = Field(..., description="Number of input tokens")
