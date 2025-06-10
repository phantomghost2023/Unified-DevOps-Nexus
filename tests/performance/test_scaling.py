import pytest
import yaml
import logging
from pathlib import Path
from core.ai.ai_optimizer import AIOptimizer
from src.core.engine.unified_engine import UnifiedEngine
from src.core.exceptions import OptimizationError, ValidationError

@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration file"""
    config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {
            "aws": {
                "enabled": True,
                "services": [{
                    "type": "compute",
                    "resources": [{
                        "specs": {"nodeType": "t3.medium"}
                    }]
                }]
            }
        }
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    return str(config_path)

@pytest.mark.benchmark(
    min_rounds=100,
    max_time=2.0,
    warmup=True,
    warmup_iterations=5
)
@pytest.mark.performance
def test_ai_optimizer_performance(benchmark):
    """Test AIOptimizer performance"""
    optimizer = AIOptimizer(api_key="test_key")
    
    test_config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {
            "aws": {
                "services": [{
                    "type": "compute",
                    "resources": [{
                        "specs": {"nodeType": "t3.large"}
                    }]
                }]
            }
        }
    }
    
    result = benchmark(optimizer.optimize_configuration, test_config)
    assert result is not None
    assert "providers" in result
    assert "aws" in result["providers"]
    assert result["providers"]["aws"]["services"][0]["resources"][0]["specs"]["nodeType"] == "t3.medium"

@pytest.mark.asyncio
@pytest.mark.performance
async def test_engine_deployment_performance(test_config, tmp_path, caplog):
    """Test engine deployment performance"""
    caplog.set_level(logging.INFO)

    # Save config
    config_path = tmp_path / "perf_deploy.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump({
            "version": "1.0",
            "metadata": {"project": "test", "environment": "dev"},
            "providers": {
                "aws": {
                    "enabled": True,
                    "services": [{
                        "type": "compute",
                        "resources": [{
                            "specs": {"nodeType": "t3.medium"}
                        }]
                    }]
                }
            }
        }, f)

    # Initialize engine
    engine = UnifiedEngine(str(config_path))
    engine.initialize_providers()

    # Test deployment
    result = await engine.deploy()

    # Verify results
    assert result["aws"]["status"] == "success"
    assert "Deployed to aws:" in caplog.text

    # Test error path
    engine.providers = {}  # Clear providers to force error
    with pytest.raises(ValueError) as exc_info:
        await engine._deploy_provider("aws", {"enabled": True})
    assert "Provider aws not initialized" in str(exc_info.value)

@pytest.mark.performance
def test_ai_optimizer_edge_cases(benchmark):
    """Test AIOptimizer edge cases"""
    optimizer = AIOptimizer(api_key="test_key")
    
    # Test with empty configuration
    empty_config = {
        "version": "1.0",
        "metadata": {"project": "test", "environment": "dev"},
        "providers": {}
    }
    
    result = benchmark(optimizer.optimize_configuration, empty_config)
    assert result is not None
    assert "providers" in result
    assert result["providers"] == {}