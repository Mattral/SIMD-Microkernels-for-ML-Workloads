"""Concrete strategy: CPU-only regex heuristics fallback."""

import re
import time
from typing import List

from src.services.policies import GuardrailPolicy, PolicyContext, PolicyName
from src.schemas.firewall import GuardrailResult, ThreatType
from src.core.logging import get_logger

logger = get_logger(__name__)

# Regex patterns for common prompt injection attack vectors
_INJECTION_PATTERNS: List[re.Pattern] = [
    # Ignore instructions / context escape
    re.compile(r"ignore\s+(previous|prior|all)\s+(instruction|context|prompt)", re.IGNORECASE),
    re.compile(r"(forget|disregard)\s+(everything|your|the)\s+(instruction|context|prompt|rule)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:a\s+)?(?:new|different)\s+(?:ai|assistant|bot)", re.IGNORECASE),
    
    # SQL/Command injection markers
    re.compile(r"(union\s+select|select\s+.*\s+from|drop\s+table|exec\s*\(|system\s*\()", re.IGNORECASE),
    
    # Base64 encoded payloads (often used to bypass filters)
    re.compile(r"base64\s*:\s*[A-Za-z0-9+/]{20,}", re.IGNORECASE),
    re.compile(r"base64\s*decode\s*\(", re.IGNORECASE),
    
    # Prompt template markers
    re.compile(r"\{\{.*?\}\}", re.DOTALL),
    re.compile(r"{%.*?%}", re.DOTALL),
]


class RegexHeuristicPolicy(GuardrailPolicy):
    """Fast regex-only evaluation. Used as circuit-breaker fallback.
    
    This policy prioritizes latency (< 2ms) over accuracy. It detects
    common injection patterns but has higher false-negative rate compared
    to ML inference. Used when:
    - Circuit breaker is OPEN (Triton unhealthy)
    - Tenant configuration specifies low-latency mode
    - For A/B testing of heuristic vs ML accuracy
    
    Inference time: ~1-2 ms for typical prompt (<1000 chars)
    """

    @property
    def name(self) -> PolicyName:
        return PolicyName.REGEX_HEURISTIC

    async def evaluate(self, ctx: PolicyContext) -> GuardrailResult:
        """Run regex heuristic evaluation.
        
        Args:
            ctx: Policy context with text, request_id, tenant_id
            
        Returns:
            GuardrailResult with heuristic classification
        """
        t0 = time.perf_counter()
        
        # Run pattern matching
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(ctx.text):
                latency_ms = (time.perf_counter() - t0) * 1000
                
                return GuardrailResult(
                    request_id=ctx.request_id,
                    threat_detected=True,
                    threat_type=ThreatType.PROMPT_INJECTION,
                    confidence=0.85,  # Fixed heuristic confidence
                    policy_used=self.name,
                    latency_ms=latency_ms,
                    message="Prompt injection pattern detected (heuristic fallback)",
                )
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return GuardrailResult(
            request_id=ctx.request_id,
            threat_detected=False,
            threat_type=None,
            confidence=0.15,  # Default low confidence for no-match
            policy_used=self.name,
            latency_ms=latency_ms,
            message="No injection patterns detected (heuristic fallback)",
        )
