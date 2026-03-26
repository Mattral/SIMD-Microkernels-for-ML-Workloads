"""Custom exception classes for GuardRail Studio."""
from typing import Optional, Dict, Any


class GuardRailException(Exception):
    """Base exception for all GuardRail Studio errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


class InferenceException(GuardRailException):
    """Exception raised when model inference fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=503
        )


class DatabaseException(GuardRailException):
    """Exception raised when database operations fail."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=500
        )


class ValidationException(GuardRailException):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            details=details,
            status_code=400
        )


class ThresholdExceededException(GuardRailException):
    """Exception raised when guardrail threshold is exceeded."""
    
    def __init__(
        self,
        message: str,
        threat_type: str,
        confidence: float,
        threshold: float
    ):
        super().__init__(
            message=message,
            details={
                "threat_type": threat_type,
                "confidence": confidence,
                "threshold": threshold
            },
            status_code=403
        )
