from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

class ResourceType(Enum):
    """Enum representing different types of cloud resources."""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    DATABASE = "database"
    SECURITY = "security"
    MONITORING = "monitoring"
    OTHER = "other"

class ResourceState(Enum):
    """Enum representing different states a resource can be in."""
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class Resource:
    """Class representing a cloud resource."""
    name: str
    type: ResourceType
    state: ResourceState
    metadata: Dict[str, Any]
    id: Optional[str] = None
    provider: Optional[str] = None
    region: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the resource to a dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "state": self.state.value,
            "provider": self.provider,
            "region": self.region,
            "metadata": self.metadata,
            "tags": self.tags or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resource':
        """Create a Resource instance from a dictionary."""
        return cls(
            id=data.get("id"),
            name=data["name"],
            type=ResourceType(data["type"]),
            state=ResourceState(data["state"]),
            provider=data.get("provider"),
            region=data.get("region"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", {})
        ) 