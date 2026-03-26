"""Policy registry and selector. This is the Strategy Pattern's context."""

from typing import Optional

from src.services.policies import GuardrailPolicy, PolicyName
from src.core.logging import get_logger

logger = get_logger(__name__)


class PolicyRegistry:
    """Selects the appropriate GuardrailPolicy for a given request.
    
    Selection order (priority):
    1. Explicit policy override via x-guardrail-policy header
    2. Tenant configuration from database
    3. Circuit breaker state (if Triton unhealthy, degrade to regex)
    4. Default: TRITON_ONNX
    
    This enables:
    - A/B testing between ML and regex accuracy
    - Gradual rollout of new policies (canary-style)
    - Graceful degradation when ML service is unhealthy
    """

    def __init__(self) -> None:
        """Initialize registry with available policies."""
        from src.services.policies.triton_policy import TritonOnnxPolicy
        from src.services.policies.regex_policy import RegexHeuristicPolicy
        
        self._policies: dict[PolicyName, GuardrailPolicy] = {
            PolicyName.TRITON_ONNX: TritonOnnxPolicy(),
            PolicyName.REGEX_HEURISTIC: RegexHeuristicPolicy(),
        }

    def select(
        self,
        requested: Optional[PolicyName] = None,
        circuit_open: bool = False,
    ) -> GuardrailPolicy:
        """Select the policy to use for this request.
        
        Args:
            requested: Explicitly requested policy name (from x-guardrail-policy header)
            circuit_open: If True, circuit breaker is OPEN (Triton unhealthy)
            
        Returns:
            The selected GuardrailPolicy instance
        """
        # Priority 1: If circuit is open, degrade to regex immediately
        if circuit_open:
            logger.debug(
                "Circuit breaker is OPEN, degrading to regex heuristic",
                extra={"requested": requested}
            )
            return self._policies[PolicyName.REGEX_HEURISTIC]
        
        # Priority 2: Honor explicit request (if policy exists)
        if requested and requested in self._policies:
            logger.debug(
                "Using explicitly requested policy",
                extra={"policy": requested}
            )
            return self._policies[requested]
        
        # Priority 3: Default to ML inference
        logger.debug("Using default Triton ONNX policy")
        return self._policies[PolicyName.TRITON_ONNX]

    def get_policy(self, name: PolicyName) -> GuardrailPolicy:
        """Get a specific policy by name (for testing/inspection)."""
        return self._policies.get(name)

    def list_policies(self) -> list[PolicyName]:
        """List all registered policy names."""
        return list(self._policies.keys())


# Global policy registry instance
policy_registry = PolicyRegistry()
