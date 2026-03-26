"""Tests for health check endpoints."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint returns correct status."""
    response = await client.get("/api/health/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "database_connected" in data
    assert "qdrant_connected" in data
    assert "triton_connected" in data
    assert "uptime_seconds" in data
    
    assert isinstance(data["uptime_seconds"], float)
    assert data["uptime_seconds"] >= 0
