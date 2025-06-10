from typing import Optional

class DevOpsNexusError(Exception):
    """Base exception for all DevOps Nexus errors"""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class ConfigurationError(DevOpsNexusError):
    """Configuration related errors"""
    pass

class ValidationError(DevOpsNexusError):
    """Validation related errors"""
    pass

class OptimizationError(DevOpsNexusError):
    """AI optimization related errors"""
    pass

class ProviderError(DevOpsNexusError):
    """Provider interaction related errors"""
    pass