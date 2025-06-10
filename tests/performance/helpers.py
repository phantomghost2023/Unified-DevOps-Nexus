import random
from typing import Dict, Any

def generate_test_config(size: int = 10, service_type: str = "compute") -> Dict[str, Any]:
    """Generate test configuration of specified size"""
    if service_type not in ["compute", "storage", "network"]:
        raise ValueError("Invalid service type")
        
    node_types = ["t3.small", "t3.medium", "t3.large", "t3.xlarge"]
    return {
        "version": "1.0",
        "metadata": {
            "project": "perf-test",
            "environment": "test"
        },
        "providers": {
            "aws": {
                "enabled": True,
                "services": [{
                    "type": service_type,
                    "resources": [
                        {
                            "specs": {
                                "nodeType": random.choice(node_types),
                                "count": random.randint(1, 5)
                            }
                        } 
                        for _ in range(size)
                    ]
                }]
            }
        }
    }