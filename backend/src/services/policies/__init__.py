"""Strategy Pattern base class for pluggable guardrail evaluation policies."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from src.schemas.firewall import GuardrailResult


class PolicyName(StrEnum):
    """Available guardrail evaluation policies."""
    
    TRITON_ONNX = "triton_onnx"          # Full ML inference via Triton
    REGEX_HEURISTIC = "regex_heuristic"  # CPU-only regex heuristics


@dataclass
class PolicyContext:
    """Request context passed to policy evaluation."""
    
    text: str
    request_id: str
    tenant_id: Optional[str] = None


class GuardrailPolicy(abc.ABC):
    """Abstract base for guardrail evaluation strategies.
    
    Each concrete policy encapsulates a complete evaluation strategy.
    The GuardrailService selects a policy per-request based on:
      - The request's 'x-guardrail-policy' header (for A/B testing)
      - The tenant configuration (for multi-tenant deployments)
      - The circuit breaker state (automatic degradation)

    This is separate from the circuit breaker. The circuit breaker handles
    fault tolerance within a policy. The strategy selects which policy runs.
    """

    @property
    @abc.abstractmethod
    def name(self) -> PolicyName:
        """Return the policy's name."""
        ...

    @abc.abstractmethod
    async def evaluate(self, ctx: PolicyContext) -> GuardrailResult:
        """Evaluate the text and return a classification result.

        Args:
            ctx: Request context including text, tenant, and latency budget.

        Returns:
            GuardrailResult with threat classification and confidence.

        Raises:
            PolicyEvaluationError: If the policy cannot produce a result.
        """
        ...
