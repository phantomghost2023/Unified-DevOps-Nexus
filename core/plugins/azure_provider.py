# filepath: core/plugins/azure_provider.py
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.core.exceptions import HttpResponseError
from typing import Dict, Any, Optional, List, Union
from core.exceptions import DeploymentStatus

# Fixed header and removed AWS references
class AzureProvider:
    def __init__(self, config: dict):
        """Initialize Azure provider with configuration.
        
        Args:
            config (dict): Configuration containing Azure credentials
            
        Raises:
            ValueError: If required credentials are missing
        """
        self._validate_config(config)
        self.config = config
        self._client = None

    def _validate_config(self, config: dict) -> None:
        """Validate the configuration dictionary.
        
        Args:
            config (dict): Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
            
        if not config.get("azure_credentials"):
            raise ValueError("Missing Azure credentials")
            
        required_fields = ["subscription_id", "tenant_id", "client_id", "client_secret"]
        missing_fields = [field for field in required_fields 
                         if field not in config["azure_credentials"]]
        
        if missing_fields:
            raise ValueError(f"Missing required Azure credentials: {', '.join(missing_fields)}")

    def _get_client(self) -> ResourceManagementClient:
        """Get or create Azure Resource Management client.
        
        Returns:
            ResourceManagementClient: Azure resource management client
            
        Raises:
            ValueError: If client initialization fails
        """
        if not self._client:
            try:
                self._client = ResourceManagementClient(
                    credential=DefaultAzureCredential(),
                    subscription_id=self.config["azure_credentials"]["subscription_id"]
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Azure client: {str(e)}")
        return self._client

    def _get_resource_group_name(self) -> Optional[str]:
        """Get the resource group name from config.
        
        Returns:
            Optional[str]: Resource group name if present, None otherwise
        """
        return self.config.get("resource_group_name")

    def _create_error_response(self, error: str, message: str) -> Dict[str, str]:
        """Create a standardized error response.
        
        Args:
            error (str): Error description
            message (str): Error message
            
        Returns:
            Dict[str, str]: Standardized error response
        """
        return {
            "status": "error",
            "error": error,
            "message": message
        }

    def _create_success_response(self, data: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Create a standardized success response.
        
        Args:
            data (Dict[str, Any]): Response data
            message (str): Success message
            
        Returns:
            Dict[str, Any]: Standardized success response
        """
        return {
            "status": "success",
            **data,
            "message": message
        }

    def deploy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy or manage Azure resources.
        
        Args:
            config (Dict[str, Any]): The configuration for the Azure provider.

        Returns:
            Dict[str, Any]: Status of the deployment operation.
        """
        try:
            client = self._get_client()
            resource_group_name = config.get("resource_group_name")
            
            if not resource_group_name:
                return self._create_error_response("Configuration Error", "Missing resource_group_name in Azure provider config")
                
            try:
                client.resource_groups.get(resource_group_name)
            except HttpResponseError:
                client.resource_groups.create_or_update(
                    resource_group_name,
                    {"location": config.get("location", "eastus")}
                )
            
            return self._create_success_response({"resources": []}, "Azure deployment successful")
            
        except Exception as e:
            return self._create_error_response("Deployment Error", str(e))

    def get_actual_state(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get the actual state of deployed Azure resources.
        
        Args:
            config (Dict[str, Any]): The configuration for the Azure provider.

        Returns:
            Dict[str, Any]: Current state of Azure resources or error status.
        """
        try:
            client = self._get_client()
            resource_group_name = config.get("resource_group_name")
            
            if not resource_group_name:
                return self._create_error_response("Configuration Error", "Missing resource_group_name in Azure provider config")
            
            resources = list(client.resources.list_by_resource_group(resource_group_name))
            
            return self._create_success_response(
                {
                    "resources": [{
                        "name": r.name,
                        "type": r.type,
                        "location": r.location
                    } for r in resources]
                },
                "Successfully retrieved Azure resource state"
            )
        except Exception as e:
            return self._create_error_response("State Retrieval Error", str(e))

    def remediate_drift(self, desired_state: Dict[str, Any]) -> Dict[str, Any]:
        """Remediate drift between desired and actual state of Azure resources.
        
        Args:
            desired_state (dict): The desired state of resources
            
        Returns:
            Dict[str, Any]: Remediation status or error status.
        """
        try:
            if not isinstance(desired_state, dict):
                return self._create_error_response("Validation Error", "Desired state must be a dictionary")
                
            client = self._get_client()
            resource_group_name = desired_state.get("resource_group_name")
            
            if not resource_group_name:
                return self._create_error_response("Configuration Error", "Missing resource_group_name in desired_state for Azure remediation")
            
            current_resources = list(client.resources.list_by_resource_group(resource_group_name))
            current_state = {r.name: r for r in current_resources}
            remediated = self._remediate_resources(client, resource_group_name, current_state, desired_state)
            
            return self._create_success_response(
                {"remediated": remediated},
                "Successfully remediated Azure resource drift"
            )
        except Exception as e:
            return self._create_error_response("Remediation Error", str(e))

    def _remediate_resources(
        self,
        client: ResourceManagementClient,
        resource_group_name: str,
        current_state: Dict[str, Any],
        desired_state: Dict[str, Any]
    ) -> List[str]:
        """Remediate individual resources.
        
        Args:
            client (ResourceManagementClient): Azure client
            resource_group_name (str): Name of the resource group
            current_state (Dict[str, Any]): Current state of resources
            desired_state (Dict[str, Any]): Desired state of resources
            
        Returns:
            List[str]: List of remediated resource names
        """
        remediated = []
        for resource_name, desired_config in desired_state.items():
            if (resource_name not in current_state or 
                current_state[resource_name] != desired_config):
                client.resources.begin_create_or_update(
                    resource_group_name,
                    resource_name,
                    desired_config
                )
                remediated.append(resource_name)
        return remediated