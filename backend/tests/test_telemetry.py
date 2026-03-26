"""Tests for telemetry endpoints."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_metrics(client: AsyncClient):
    """Test metrics endpoint returns valid structure."""
    response = await client.get("/api/telemetry/metrics?hours=24")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "total_requests" in data
    assert "blocked_requests" in data
    assert "threats_detected" in data
    assert "avg_latency_ms" in data
    assert "p50_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "p99_latency_ms" in data
    
    # Check types
    assert isinstance(data["total_requests"], int)
    assert isinstance(data["avg_latency_ms"], float)


@pytest.mark.asyncio
async def test_get_threat_breakdown(client: AsyncClient):
    """Test threat breakdown endpoint."""
    response = await client.get("/api/telemetry/threats?hours=24")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "prompt_injection" in data
    assert "pii_detection" in data
    assert "toxicity" in data
    assert "malicious_code" in data
    
    # All should be integers
    assert all(isinstance(v, int) for v in data.values())


@pytest.mark.asyncio
async def test_get_recent_requests(client: AsyncClient):
    """Test recent requests endpoint."""
    response = await client.get("/api/telemetry/requests?limit=10")
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
