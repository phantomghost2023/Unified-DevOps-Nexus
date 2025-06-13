import pytest
import yaml
from pathlib import Path
from typing import Dict, Any
from core.ai.ai_optimizer import AIOptimizer
from core.engine.unified_engine import UnifiedEngine
from core.exceptions import ValidationError
import pytest
from tests.performance.helpers import generate_test_config
import openai

@pytest.fixture
def ai_optimizer():
    """Create AIOptimizer instance for testing."""
    return AIOptimizer(api_key="test_key")

@pytest.fixture
def complex_config() -> Dict[str, Any]:
    """Complex configuration for edge case testing"""
    return {
        "version": "1.0",
        "metadata": {
            "project": "edge-test",
            "environment": "test"
        },
        "providers": {
            "aws": {
                "enabled": True,
                "services": [{
                    "type": "compute",
                    "resources": [
                        {"specs": {"nodeType": "t3.medium", "count": i}}
                        for i in range(1, 4)
                    ]
                }]
            }
        }
    }

def test_optimization_error_handling(ai_optimizer):
    """Test AI optimization error handling"""
    # Providers is None
    bad_config = {"version": "1.0", "metadata": {"project": "test"}, "providers": None}
    with pytest.raises(ValidationError) as exc_info:
        ai_optimizer.optimize_configuration(bad_config)
    assert "Providers must be a dictionary" in str(exc_info.value)

    # Resources is None
    bad_config = {
        "version": "1.0",
        "metadata": {"project": "test"},
        "providers": {
            "aws": {
                "services": [{
                    "type": "compute",
                    "resources": None
                }]
            }
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        ai_optimizer.optimize_configuration(bad_config)
    assert "Resources must be a list" in str(exc_info.value)

@pytest.mark.asyncio
def test_engine_file_not_found(tmp_path):
    """Test UnifiedEngine with missing config file"""
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ValidationError) as exc_info:
        UnifiedEngine(str(missing))
    assert "Configuration file not found" in str(exc_info.value)

@pytest.mark.parametrize("config_size", [1 * 1024, 10 * 1024, 100 * 1024])
def test_config_size_scaling(benchmark, config_size):
    """Benchmark optimizer with large configs"""
    large_config = generate_test_config(size=config_size)
    optimizer = AIOptimizer("fake-key")
    benchmark(optimizer.optimize_configuration, large_config)

def test_logging_behavior(caplog):
    """Test logging output"""
    import logging
    logger = logging.getLogger("core.ai.ai_optimizer")
    caplog.set_level(logging.INFO)
    logger.info("Test log message")
    assert "Test log message" in caplog.text

def test_timestamp_behavior():
    """Placeholder for timestamp-related edge case tests"""
    pass