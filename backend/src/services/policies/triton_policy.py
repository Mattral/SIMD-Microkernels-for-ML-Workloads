"""Concrete strategy: ML inference via Triton gRPC."""

from src.services.policies import GuardrailPolicy, PolicyContext, PolicyName
from src.services.inference_client_triton import inference_client
from src.schemas.firewall import GuardrailResult, ThreatType
from src.core.logging import get_logger

logger = get_logger(__name__)


class TritonOnnxPolicy(GuardrailPolicy):
    """Evaluates prompts using DistilRoBERTa served by Triton (ONNX FP16).
    
    This is the primary policy for full ML inference accuracy. It blocks
    only when a threat is detected with confidence ≥ threshold.
    """

    def __init__(self, client=None) -> None:
        """Initialize with inference client (defaults to global singleton)."""
        self._client = client or inference_client

    @property
    def name(self) -> PolicyName:
        return PolicyName.TRITON_ONNX

    async def evaluate(self, ctx: PolicyContext) -> GuardrailResult:
        """Run full DistilRoBERTa inference via Triton.
        
        Args:
            ctx: Policy context with text, request_id, tenant_id
            
        Returns:
            GuardrailResult with ML classification
            
        Raises:
            InferenceException: If Triton is unreachable or returns an error
        """
        result = await self._client.infer(
            text=ctx.text,
            request_id=ctx.request_id,
            return_embeddings=True
        )
        
        return GuardrailResult(
            request_id=ctx.request_id,
            threat_detected=result["threat_detected"],
            threat_type=ThreatType(result["threat_type"]),
            confidence=result["confidence"],
            policy_used=self.name,
            latency_ms=result["latency_ms"],
            message=self._build_message(result),
        )
    
    @staticmethod
    def _build_message(result: dict) -> str:
        """Generate human-readable classification message."""
        threat_type = result["threat_type"]
        confidence = result["confidence"]
        
        if result["threat_detected"]:
            return f"Threat detected: {threat_type} (confidence: {confidence:.2f})"
        else:
            return "No threats detected"
