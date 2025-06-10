from typing import Dict, Any
import openai
import logging
import yaml
from openai import AsyncOpenAI  # Add this import
import copy
from datetime import datetime  # Import datetime
from ..exceptions import OptimizationError, ValidationError

class AIOptimizer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self.logger = logging.getLogger(__name__)

    async def generate_infrastructure(self, description: str) -> Dict[str, Any]:
        """Generate infrastructure code from natural language"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are an infrastructure expert. Generate YAML configuration."
                }, {
                    "role": "user",
                    "content": description
                }]
            )
            return self._parse_ai_response(response)
        except Exception as e:
            self.logger.error(f"AI generation failed: {str(e)}")
            raise
            
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration with detailed errors"""
        if config is None:
            raise ValidationError("Configuration cannot be None")
        
        if not isinstance(config, dict):
            raise ValidationError("Configuration must be a dictionary")
            
        missing_fields = [field for field in ["version", "metadata", "providers"] 
                         if field not in config]
        if missing_fields:
            raise ValidationError(
                "Missing required fields in configuration",
                {"missing_fields": missing_fields}
            )

        # Validate providers
        if not isinstance(config["providers"], dict):
            raise ValidationError("Providers must be a dictionary")

        # Validate service types
        for provider_name, provider in config["providers"].items():
            if provider is None:
                raise ValidationError(f"Provider {provider_name} configuration cannot be None")
            if not isinstance(provider, dict):
                raise ValidationError(f"Provider {provider_name} configuration must be a dictionary")
            
            services = provider.get("services", [])
            if not isinstance(services, list):
                raise ValidationError(f"Services for provider {provider_name} must be a list")
            
            for service in services:
                if not isinstance(service, dict):
                    raise ValidationError(f"Service configuration must be a dictionary")
                if "type" not in service:
                    raise ValidationError(f"Service type is required")
                if "resources" not in service:
                    raise ValidationError(f"Service resources are required")
                
                resources = service["resources"]
                if not isinstance(resources, list):
                    raise ValidationError(f"Resources must be a list")
                
                for resource in resources:
                    if not isinstance(resource, dict):
                        raise ValidationError(f"Resource configuration must be a dictionary")
                    if "specs" not in resource:
                        raise ValidationError(f"Resource specs are required")
                    if not isinstance(resource["specs"], dict):
                        raise ValidationError(f"Resource specs must be a dictionary")

    def optimize_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize infrastructure configuration"""
        try:
            self.validate_config(config)
            optimized = copy.deepcopy(config)
            return self._process_configuration(optimized)
        except ValidationError as e:
            self.logger.error(f"Configuration validation failed: {e.message}")
            raise
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_ai_response(self, response) -> Dict[str, Any]:
        """Parse and validate AI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse AI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing AI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _process_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process and optimize configuration"""
        try:
            for provider in config["providers"].values():
                for service in provider.get("services", []):
                    if service["type"] == "compute":
                        for resource in service["resources"]:
                            if "specs" in resource and "nodeType" in resource["specs"]:
                                # Optimize node type
                                current_type = resource["specs"]["nodeType"]
                                if current_type == "t3.large":
                                    resource["specs"]["nodeType"] = "t3.medium"
                                elif current_type == "t3.xlarge":
                                    resource["specs"]["nodeType"] = "t3.large"
            return config
        except Exception as e:
            self.logger.error(f"Error processing configuration: {str(e)}")
            raise OptimizationError(f"Failed to process configuration: {str(e)}")