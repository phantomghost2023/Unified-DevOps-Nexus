import pytest
import pytest_asyncio
from core.ai.ai_optimizer import AIOptimizer
from core.engine.unified_engine import UnifiedEngine
from core.exceptions import ValidationError, OptimizationError
import yaml
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import logging

@pytest.fixture
def ai_optimizer():
    return AIOptimizer(api_key="test_key")

@pytest.fixture
def mock_cloud_clients():
    """Fixture that provides mock cloud clients for testing."""
    aws_mock = MagicMock()
    return {"aws": aws_mock}

@pytest.mark.asyncio
@pytest.mark.integration
async def test_end_to_end_deployment(mock_cloud_clients, ai_optimizer, tmp_path):
    """Test full deployment flow with AI optimization"""
    # Create config file
    test_config = {
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
            }
        }
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(test_config, f)

    # 1. AI Optimization
    optimized_config = ai_optimizer.optimize_configuration(test_config)

    # 2. Engine Deployment
    engine = UnifiedEngine(str(config_path))
    
    # Initialize providers with mocks
    engine.initialize_providers()
    engine.providers["aws"] = mock_cloud_clients["aws"]
    mock_cloud_clients["aws"].deploy = AsyncMock(return_value={"status": "success"})

    result = await engine.deploy()

    # Verify results
    assert mock_cloud_clients["aws"].deploy.called
    assert result["aws"]["status"] == "success"