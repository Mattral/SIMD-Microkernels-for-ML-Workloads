"""AWS Secrets Manager helper with local caching and IRSA compatibility."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from src.core.logging import get_logger

logger = get_logger(__name__)


class SecretsManagerClient:
    """Wrapper for AWS Secrets Manager with simple in-memory caching."""

    def __init__(self, cache_ttl_seconds: int = 300):
        self._client = boto3.client("secretsmanager")
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)

    def get_secret_value(self, secret_name: str) -> Optional[str]:
        """Fetch a secret string from Secrets Manager.

        Caches values for a short time to avoid repeated AWS API calls.
        """
        now = datetime.utcnow()
        cached = self._cache.get(secret_name)
        if cached and cached["expires_at"] > now:
            return cached["value"]

        try:
            response = self._client.get_secret_value(SecretId=secret_name)
            secret = response.get("SecretString")
            if secret is None and "SecretBinary" in response:
                secret = response["SecretBinary"].decode("utf-8")

            self._cache[secret_name] = {
                "value": secret,
                "expires_at": now + self._cache_ttl,
            }
            return secret
        except ClientError as exc:
            logger.error("Failed to load secret %s from AWS Secrets Manager: %s", secret_name, exc)
            raise

    def get_secret_dict(self, secret_name: str) -> Dict[str, Any]:
        """Load a secret and parse it as JSON if possible."""
        value = self.get_secret_value(secret_name)
        if not value:
            return {}

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Secret %s is not valid JSON; returning raw string under 'value' key", secret_name)
            return {"value": value}


secrets_manager = SecretsManagerClient()
