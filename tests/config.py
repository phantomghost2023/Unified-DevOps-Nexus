import os
from typing import Dict, Any

def get_azure_config() -> Dict[str, Any]:
    """Get Azure configuration from environment variables.
    
    Returns:
        Dict[str, Any]: Azure configuration dictionary
    """
    return {
        "azure_credentials": {
            "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
            "tenant_id": os.getenv("AZURE_TENANT_ID", ""),
            "client_id": os.getenv("AZURE_CLIENT_ID", ""),
            "client_secret": os.getenv("AZURE_CLIENT_SECRET", "")
        },
        "resource_group_name": os.getenv("AZURE_RESOURCE_GROUP", "test-resource-group"),
        "location": os.getenv("AZURE_LOCATION", "eastus")
    }