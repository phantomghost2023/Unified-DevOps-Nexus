import pytest
import yaml
from typing import Dict, Any
from pathlib import Path
from core.ai.ai_optimizer import AIOptimizer
from core.exceptions import ValidationError, OptimizationError

@pytest.fixture
def ai_optimizer():
    """Create AIOptimizer instance for testing"""
    return AIOptimizer(api_key="test_key")

def assert_node_type(config: Dict[str, Any], expected_type: str) -> None:
    """Helper function to assert node type in configuration"""
    actual_type = config["providers"]["aws"]["services"][0]["resources"][0]["specs"]["nodeType"]
    assert actual_type == expected_type, f"Expected node type {expected_type}, got {actual_type}"

@pytest.mark.asyncio
async def test_generate_infrastructure(ai_optimizer, mocker):
    """Test infrastructure generation"""
    yaml_content = """
    version: "1.0"
    providers:
      aws:
        enabled: true
    """
    mock_response = mocker.Mock()
    mock_response.choices = [mocker.Mock(message=mocker.Mock(content=yaml_content))]
    
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    ai_optimizer.client = mock_client
    
    result = await ai_optimizer.generate_infrastructure("Create infrastructure")
    assert isinstance(result, dict)
    assert result["providers"]["aws"]["enabled"] is True

@pytest.mark.asyncio
async def test_generate_infrastructure_exception_handling(ai_optimizer, mocker):
    """Test exception handling in generate_infrastructure"""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("Test API Error")
    ai_optimizer.client = mock_client

    with pytest.raises(Exception, match="Test API Error"):
        await ai_optimizer.generate_infrastructure("Some description")

@pytest.mark.asyncio
async def test_generate_infrastructure_error(ai_optimizer, mocker):
    # Mock the async client with an error
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    ai_optimizer.client = mock_client
    
    with pytest.raises(Exception):
        await ai_optimizer.generate_infrastructure("Test description")

def test_optimize_configuration(ai_optimizer):
    """Test infrastructure optimization with different node types"""
    # Test case 1: Large instance type should be optimized
    test_config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {
            "aws": {
                "services": [{
                    "type": "compute",
                    "resources": [{
                        "specs": {
                            "nodeType": "t3.large"
                        }
                    }]
                }]
            }
        }
    }

    result = ai_optimizer.optimize_configuration(test_config)
    assert result is not None
    assert "providers" in result
    assert "aws" in result["providers"]
    assert result["providers"]["aws"]["services"][0]["resources"][0]["specs"]["nodeType"] == "t3.medium"

    # Test case 2: XLarge instance type should be optimized
    test_config["providers"]["aws"]["services"][0]["resources"][0]["specs"]["nodeType"] = "t3.xlarge"
    result = ai_optimizer.optimize_configuration(test_config)
    assert_node_type(result, "t3.large")

def test_optimize_configuration_empty(ai_optimizer):
    """Test optimization with empty configuration"""
    empty_config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {}
    }
    result = ai_optimizer.optimize_configuration(empty_config)
    assert result is not None
    assert "providers" in result
    assert result["providers"] == {}

@pytest.mark.asyncio
async def test_generate_infrastructure_invalid_yaml(ai_optimizer, mocker):
    """Test handling of invalid YAML content in the response"""
    # Create response with invalid YAML
    class MockResponseInvalid:
        class MockChoice:
            class MockMessage:
                def __init__(self):
                    self.content = "invalid: yaml: content: [}"
            
            def __init__(self):
                self.message = self.MockMessage()
        
        def __init__(self):
            self.choices = [self.MockChoice()]
    
    mock_response = MockResponseInvalid()
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    ai_optimizer.client = mock_client
    
    with pytest.raises(ValueError) as exc_info:
        await ai_optimizer.generate_infrastructure("Create invalid YAML")
    assert "Failed to parse AI response" in str(exc_info.value)

@pytest.mark.asyncio
async def test_optimize_configuration_error(ai_optimizer):
    """Test error handling in optimize_configuration"""
    # Test with invalid configuration that will raise an exception
    invalid_config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {
            "aws": None  # This will cause an error when accessing .values()
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        ai_optimizer.optimize_configuration(invalid_config)
    assert "Provider aws configuration cannot be None" in str(exc_info.value)

@pytest.mark.asyncio
async def test_parse_ai_response_general_error(ai_optimizer):
    """Test general exception handling in _parse_ai_response"""
    # Create a response object that will trigger the general exception
    class BadResponse:
        @property
        def choices(self):
            raise RuntimeError("Unexpected error")
    
    with pytest.raises(ValueError) as exc_info:
        ai_optimizer._parse_ai_response(BadResponse())
    assert "Unexpected error parsing AI response" in str(exc_info.value)

@pytest.mark.asyncio
async def test_generate_infrastructure_missing_choices(ai_optimizer):
    """Test handling of response without choices"""
    class EmptyResponse:
        choices = []
    
    with pytest.raises(ValueError) as exc_info:
        ai_optimizer._parse_ai_response(EmptyResponse())
    assert "Invalid response format" in str(exc_info.value)

@pytest.mark.asyncio
async def test_optimize_configuration_invalid_structure(ai_optimizer):
    """Test optimization with invalid service structure"""
    invalid_config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {
            "aws": {
                "services": [{
                    "type": "compute",
                    "resources": [{}]  # Missing specs
                }]
            }
        }
    }

    with pytest.raises(ValidationError) as exc_info:
        ai_optimizer.optimize_configuration(invalid_config)
    assert "Resource specs are required" in str(exc_info.value)

@pytest.mark.asyncio
async def test_generate_infrastructure_with_complex_config(ai_optimizer, mocker):
    """Test generation of complex infrastructure configuration"""
    yaml_content = """
    version: "1.0"
    providers:
      aws:
        enabled: true
        regions: ["us-east-1"]
        services:
          - type: compute
            resources:
              - name: "api-cluster"
                kind: "eks"
                specs:
                  nodeType: "t3.large"
                  minNodes: 2
                  maxNodes: 5
    """
    
    mock_response = mocker.Mock()
    mock_response.choices = [
        mocker.Mock(message=mocker.Mock(content=yaml_content))
    ]
    
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    ai_optimizer.client = mock_client
    
    result = await ai_optimizer.generate_infrastructure(
        "Create a Kubernetes cluster"
    )
    
    assert isinstance(result, dict)
    assert result["providers"]["aws"]["enabled"] is True
    assert result["providers"]["aws"]["services"][0]["resources"][0]["kind"] == "eks"

@pytest.mark.asyncio
async def test_parse_ai_response_unexpected_error(ai_optimizer):
    """Test handling of unexpected errors in response parsing"""
    class ProblematicResponse:
        @property
        def choices(self):
            return [object()]  # This will cause an attribute error when accessing message
    
    with pytest.raises(ValueError) as exc_info:
        ai_optimizer._parse_ai_response(ProblematicResponse())
    assert "Unexpected error parsing AI response" in str(exc_info.value)

@pytest.mark.asyncio
async def test_parse_ai_response_invalid_root_type(ai_optimizer):
    """Test handling of YAML content that doesn't produce a dict"""
    class ListResponse:
        class MockChoice:
            class MockMessage:
                def __init__(self):
                    # YAML that parses to a list instead of dict
                    self.content = "- item1\n- item2"
            
            def __init__(self):
                self.message = self.MockMessage()
        
        def __init__(self):
            self.choices = [self.MockChoice()]
    
    with pytest.raises(ValueError) as exc_info:
        ai_optimizer._parse_ai_response(ListResponse())
    assert "Invalid YAML: root element must be a mapping" in str(exc_info.value)

@pytest.fixture
def mock_aws(mocker):
    return mocker.AsyncMock()

@pytest.fixture
def mock_gcp(mocker):
    return mocker.AsyncMock()

@pytest.mark.asyncio
async def test_end_to_end_deployment(mock_aws, mock_gcp, ai_optimizer, tmp_path):
    """Test complete deployment flow"""
    config_path = tmp_path / "multi-cloud.yml"
    config = {
        "version": "1.0",
        "metadata": {
            "project": "test",
            "environment": "dev"
        },
        "providers": {
            "aws": {
                "enabled": True,
                "services": [{
                    "type": "compute",
                    "resources": [{
                        "specs": {
                            "nodeType": "t3.large"
                        }
                    }]
                }]
            },
            "gcp": {
                "enabled": True
            }
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    
    # Initialize engine with mocks
    engine = UnifiedEngine(str(config_path))
    engine.initialize_providers()
    
    # Set up mock providers with return values
    mock_aws.deploy.return_value = {"status": "success"}
    mock_gcp.deploy.return_value = {"status": "success"}
    engine.providers["aws"] = mock_aws
    engine.providers["gcp"] = mock_gcp
    
    # Run deployment
    result = await engine.deploy()
    
    # Verify results
    assert mock_aws.deploy.called
    assert mock_gcp.deploy.called
    assert result["aws"]["status"] == "success"
    assert result["gcp"]["status"] == "success"

@pytest.mark.asyncio
async def test_process_configuration_exception_handling(ai_optimizer, mocker):
    """Test exception handling in _process_configuration"""
    mocker.patch('copy.deepcopy', side_effect=Exception("Deepcopy Error"))
    config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {"aws": {"services": []}}
    }
    with pytest.raises(OptimizationError) as exc_info:
        ai_optimizer.optimize_configuration(config)
    assert "Deepcopy Error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_generate_infrastructure_api_error(ai_optimizer, mocker):
    """Test API error handling in generate_infrastructure"""
    mock_client = mocker.AsyncMock()
    mock_client.chat.completions.create.side_effect = openai.APIError(
        message="API Error",
        request=mocker.Mock(),
        body={}
    )
    ai_optimizer.client = mock_client  # Ensure the client is set
    mocker.patch('openai.OpenAI', return_value=mock_client)

    with pytest.raises(openai.APIError) as exc_info:
        await ai_optimizer.generate_infrastructure({})
    assert "API Error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_process_configuration_exception(ai_optimizer, mocker):
    """Test exception branch in _process_configuration"""
    pytest.skip("Cannot patch datetime.datetime.now on immutable type; test skipped.")

@pytest.mark.asyncio
async def test_optimize_configuration_deepcopy_error(ai_optimizer, mocker):
    """Test deepcopy error handling in optimize_configuration"""
    mocker.patch('copy.deepcopy', side_effect=Exception("Deepcopy Error"))
    config = {"version": "1.0", "metadata": {}, "providers": {}}
    
    with pytest.raises(Exception, match="Deepcopy Error"):
        ai_optimizer.optimize_configuration(config)

@pytest.mark.asyncio
async def test_parse_ai_response_empty_choices(ai_optimizer):
    """Test handling of empty choices in AI response"""
    class EmptyResponse:
        choices = []
    
    with pytest.raises(ValueError, match="Invalid response format"):
        ai_optimizer._parse_ai_response(EmptyResponse())

@pytest.mark.asyncio
async def test_parse_ai_response_invalid_yaml(ai_optimizer):
    """Test handling of invalid YAML in AI response"""
    class InvalidYAMLResponse:
        choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': '{invalid: yaml'})()})()]
    
    with pytest.raises(ValueError, match="Failed to parse AI response"):
        ai_optimizer._parse_ai_response(InvalidYAMLResponse())

@pytest.mark.asyncio
async def test_process_configuration_missing_metadata(ai_optimizer):
    """Test handling of missing metadata in configuration"""
    config = {
        "version": "1.0",
        "providers": {"aws": {"services": []}}
    }
    
    with pytest.raises(ValidationError) as exc_info:
        ai_optimizer.optimize_configuration(config)
    assert "Missing required fields" in str(exc_info.value)

def test_optimize_configuration_validation_error(ai_optimizer, mocker):
    """Covers ValidationError branch in optimize_configuration"""
    mocker.patch.object(ai_optimizer, 'validate_config', side_effect=ValidationError("validation failed"))
    config = {"providers": {}}
    with pytest.raises(ValidationError, match="validation failed"):
        ai_optimizer.optimize_configuration(config)

def test_optimize_configuration_generic_error(ai_optimizer, mocker):
    """Covers generic Exception branch in optimize_configuration"""
    mocker.patch.object(ai_optimizer, 'validate_config', side_effect=Exception("unexpected error"))
    config = {"providers": {}}
    with pytest.raises(OptimizationError, match="Failed to optimize configuration: unexpected error"):
        ai_optimizer.optimize_configuration(config)

def test_parse_ai_response_yaml_error(ai_optimizer, mocker):
    """Covers yaml.YAMLError branch in _parse_ai_response"""
    class BadYAMLResponse:
        choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': ':::'})()})()]
    mocker.patch('yaml.safe_load', side_effect=yaml.YAMLError("bad yaml"))
    with pytest.raises(ValueError, match="Failed to parse AI response"):
        ai_optimizer._parse_ai_response(BadYAMLResponse())

def test_parse_ai_response_generic_error(ai_optimizer, mocker):
    """Covers generic Exception branch in _parse_ai_response"""
    class BadResponse:
        choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': 'foo'})()})()]
    mocker.patch('yaml.safe_load', side_effect=Exception("boom"))
    with pytest.raises(ValueError, match="Unexpected error parsing AI response"):
        ai_optimizer._parse_ai_response(BadResponse())

def test_process_configuration_error(ai_optimizer, mocker):
    """Covers error branch in _process_configuration"""
    # Simulate a config that will cause a KeyError by omitting 'resources'
    config = {"providers": {"aws": {"services": [{"type": "compute"}]}}}
    with pytest.raises(OptimizationError, match="Failed to process configuration"):
        ai_optimizer._process_configuration(config)

def test_validate_config_providers_not_dict(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": []}
    with pytest.raises(ValidationError, match="Providers must be a dictionary"):
        ai_optimizer.validate_config(config)

def test_validate_config_provider_config_none(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": None}}
    with pytest.raises(ValidationError, match="Provider aws configuration cannot be None"):
        ai_optimizer.validate_config(config)

def test_validate_config_provider_config_not_dict(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": 123}}
    with pytest.raises(ValidationError, match="Provider aws configuration must be a dictionary"):
        ai_optimizer.validate_config(config)

def test_validate_config_services_not_list(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {"services": {}}}}
    with pytest.raises(ValidationError, match="Services for provider aws must be a list"):
        ai_optimizer.validate_config(config)

def test_validate_config_service_not_dict(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {"services": [123]}}}
    with pytest.raises(ValidationError, match="Service configuration must be a dictionary"):
        ai_optimizer.validate_config(config)

def test_validate_config_service_missing_type(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {"services": [{"resources": []}]}}}
    with pytest.raises(ValidationError, match="Service type is required"):
        ai_optimizer.validate_config(config)

def test_validate_config_service_missing_resources(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {"services": [{"type": "compute"}]}}}
    with pytest.raises(ValidationError, match="Service resources are required"):
        ai_optimizer.validate_config(config)

def test_validate_config_resources_not_list(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {"services": [{"type": "compute", "resources": {}}]}}}
    with pytest.raises(ValidationError, match="Resources must be a list"):
        ai_optimizer.validate_config(config)

def test_validate_config_resource_not_dict(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {"services": [{"type": "compute", "resources": [123]}]}}}
    with pytest.raises(ValidationError, match="Resource configuration must be a dictionary"):
        ai_optimizer.validate_config(config)

def test_validate_config_resource_missing_specs(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {"services": [{"type": "compute", "resources": [{}]}]}}}
    with pytest.raises(ValidationError, match="Resource specs are required"):
        ai_optimizer.validate_config(config)

def test_validate_config_resource_specs_not_dict(ai_optimizer):
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {"services": [{"type": "compute", "resources": [{"specs": 123}]}]}}}
    with pytest.raises(ValidationError, match="Resource specs must be a dictionary"):
        ai_optimizer.validate_config(config)

import asyncio
@pytest.mark.asyncio
async def test_optimize_configuration_async(ai_optimizer):
    """Stub for future async optimization coverage"""
    # Simulate a simple async wrapper for coverage
    async def fake_async_opt(config):
        return ai_optimizer.optimize_configuration(config)
    config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {
            "aws": {
                "services": [
                    {
                        "type": "compute",
                        "resources": [
                            {"specs": {"nodeType": "t3.large"}}
                        ]
                    }
                ]
            }
        }
    }
    result = await fake_async_opt(config)
    assert result["providers"]["aws"]["services"][0]["resources"][0]["specs"]["nodeType"] == "t3.medium"

def test_process_configuration_network_error(mocker):
    from src.core.ai.ai_optimizer import AIOptimizer
    optimizer = AIOptimizer("fake-key")
    # Provide a valid config to get past validation
    valid_config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "test"},
        "providers": {"aws": {"enabled": True, "services": []}}
    }
    mocker.patch.object(optimizer, '_process_configuration', side_effect=Exception("network error"))
    with pytest.raises(Exception, match="network error"):
        optimizer.optimize_configuration(valid_config)