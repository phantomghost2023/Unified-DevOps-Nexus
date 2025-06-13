from enum import Enum

class DeploymentStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"

class DeploymentError(Exception):
    """Raised when deployment fails"""
    pass

class OptimizationError(Exception):
    """Raised when optimization fails"""
    pass

class ValidationError(Exception):
    """Raised when validation fails"""
    def __init__(self, message="Validation Error", *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        self.message = message