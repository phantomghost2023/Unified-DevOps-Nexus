from typing import Dict, Any
import logging
import yaml
import copy
from datetime import datetime
from typing import Dict, Any

import openai
from openai import AsyncOpenAI
import google.generativeai as genai
import httpx

from ..exceptions import OptimizationError, ValidationError

class AIOptimizer:
    def __init__(self, openai_api_key: str = None, gemini_api_key: str = None, openrouter_api_key: str = None):
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.openrouter_api_key = openrouter_api_key
        self.openai_client = None
        self.gemini_client = None
        self.openrouter_client = None
        self.logger = logging.getLogger(__name__)

    def set_openai_client(self, client=None):
        """Set the OpenAI client, or initialize if not provided."""
        if client is not None:
            self.openai_client = client
        else:
            if self.openai_api_key:
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            else:
                raise ValueError("OpenAI API key not provided.")

    def set_gemini_client(self):
        """Initialize the Gemini client."""
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-pro')
        else:
            raise ValueError("Gemini API key not provided.")

    def set_openrouter_client(self):
        """Initialize the OpenRouter client."""
        if self.openrouter_api_key:
            self.openrouter_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key,
                http_client=httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)),
            )
        else:
            raise ValueError("OpenRouter API key not provided.")



    async def generate_infrastructure(self, description: str, model_name: str = "gpt-4") -> Dict[str, Any]:
        """Generate infrastructure code from natural language using the specified model."""
        try:
            if model_name == "gpt-4":
                if self.openai_client is None:
                    self.set_openai_client()
                if self.openai_client is None:
                    raise RuntimeError("OpenAI client is not initialized.")
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{
                        "role": "system",
                        "content": "You are an infrastructure expert. Generate YAML configuration."
                    }, {
                        "role": "user",
                        "content": description
                    }]
                )
                return self._parse_openai_response(response)
            elif model_name.startswith("openrouter/"):
                if self.openrouter_client is None:
                    self.set_openrouter_client()
                if self.openrouter_client is None:
                    raise RuntimeError("OpenRouter client is not initialized.")
                response = await self.openrouter_client.chat.completions.create(
                    model=model_name.split("openrouter/")[1], # Extract model name after "openrouter/"
                    messages=[
                        {"role": "system", "content": "You are an infrastructure expert. Generate YAML configuration."},
                        {"role": "user", "content": description}
                    ]
                )
                return self._parse_openrouter_response(response)
            elif model_name == "gemini-pro":
                if self.gemini_client is None:
                    self.set_gemini_client()
                if self.gemini_client is None:
                    raise RuntimeError("Gemini client is not initialized.")
                response = await self.gemini_client.generate_content(
                    f"You are an infrastructure expert. Generate YAML configuration.\n\nUser: {description}\n\nYAML:",
                    generation_config=genai.types.GenerationConfig(temperature=0.2)
                )
                return self._parse_gemini_response(response)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            self.logger.error(f"AI generation failed: {str(e)}")
            raise

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")

            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
            # Apply predictive analytics after initial processing
            config = self._predict_resource_needs(config)
            return config
        except Exception as e:
            self.logger.error(f"Error processing configuration: {str(e)}")
            raise OptimizationError(f"Failed to process configuration: {str(e)}")

    def _predict_resource_needs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate predictive analytics for resource needs and adjust configuration."""
        self.logger.info("Performing predictive analytics for resource optimization...")
        # This is a placeholder for actual predictive analytics logic.
        # In a real-world scenario, this would involve:
        # 1. Analyzing historical usage data (e.g., from monitoring systems).
        # 2. Applying machine learning models to forecast future resource demands.
        # 3. Adjusting resource allocations in the configuration based on predictions.

        # In a real-world scenario, this would involve:
        # 1. Analyzing historical usage data (e.g., from monitoring systems).
        # 2. Applying machine learning models to forecast future resource demands.
        # 3. Adjusting resource allocations in the configuration based on predictions.

        # For demonstration, we'll simulate a simple predictive model.
        # Let's assume we have some 'historical_data' (e.g., average CPU utilization over time).
        # In a real system, this would come from a database or monitoring service.
        historical_data = {
            "web_server": {"cpu": [0.6, 0.7, 0.75, 0.8, 0.85], "memory": [4.0, 4.2, 4.1, 4.3, 4.5]},
            "database": {"memory": [0.5, 0.55, 0.6, 0.62, 0.65]}
        }

        modified_config = copy.deepcopy(config)
        for provider_name, provider_data in modified_config.get("providers", {}).items():
            for service in provider_data.get("services", []):
                service_name = service.get("name")
                if service.get("type") == "compute" and service_name in historical_data:
                    for resource in service.get("resources", []):
                        specs = resource.get("specs", {})
                        if "cpu" in specs and "cpu" in historical_data[service_name]:
                            # Simple moving average prediction for CPU
                            cpu_unit = specs['cpu'].split(' ')[1] if ' ' in specs['cpu'] else ''
                            avg_cpu = sum(historical_data[service_name]["cpu"]) / len(historical_data[service_name]["cpu"])
                            predicted_cpu = avg_cpu * 1.1 # 10% buffer
                            self.logger.info(f"Predicting new CPU for {service_name}: Current {specs['cpu']} -> Predicted {predicted_cpu:.2f}")
                            resource["specs"]["cpu"] = f"{round(predicted_cpu, 2)} {cpu_unit}".strip()
                        if "memory" in specs and "memory" in historical_data[service_name]:
                            memory_unit = specs['memory'].split(' ')[1] if ' ' in specs['memory'] else ''
                            avg_memory = sum(historical_data[service_name]["memory"]) / len(historical_data[service_name]["memory"])
                            predicted_memory = avg_memory * 1.05 # 5% buffer
                            self.logger.info(f"Predicting new Memory for {service_name}: Current {specs['memory']} -> Predicted {predicted_memory:.2f}")
                            resource["specs"]["memory"] = f"{round(predicted_memory, 2)} {memory_unit}".strip()
        self.logger.info("Predictive analytics completed.")
        return modified_config

    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing Gemini response: {str(e)}"
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
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationError(f"Failed to optimize configuration: {str(e)}")
    def _parse_openai_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenAI-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenAI response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_openrouter_response(self, response) -> Dict[str, Any]:
        """Parse and validate OpenRouter-generated configuration"""
        try:
            if not response.choices:
                raise ValueError("Invalid response format: no choices available")
            content = response.choices[0].message.content
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error parsing OpenRouter response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _parse_gemini_response(self, response) -> Dict[str, Any]:
        """Parse and validate Gemini-generated configuration"""
        try:
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid YAML: root element must be a mapping")
            return config
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse Gemini response: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            # Access the text from the parts attribute
            content = response.text
            config = yaml.safe_load(content)
            if not isinstance(config, dict):
                raise ValueError("Invalid response format: no choices available")
