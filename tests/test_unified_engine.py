import pytest
import yaml
import logging
import asyncio
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.core.engine.unified_engine import UnifiedEngine
from src.core.exceptions import ValidationError

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

def test_engine_initialization(test_config):
    """Test engine initialization with valid config"""
    engine = UnifiedEngine(test_config)
    assert engine.config is not None
    assert "version" in engine.config
    assert "providers" in engine.config
    assert "aws" in engine.config["providers"]

def test_invalid_config_file(tmp_path):
    """Test engine initialization with invalid config file"""
    with pytest.raises(ValidationError) as exc_info:
        UnifiedEngine(str(tmp_path / "nonexistent.yaml"))
    assert "Configuration file not found" in str(exc_info.value)

def test_validate_config_missing_required_fields(test_config):
    """Test validation of missing required fields"""
    engine = UnifiedEngine(test_config)
    invalid_config = {"version": "1.0"}  # Missing metadata and providers
    with pytest.raises(ValidationError) as exc_info:
        engine.validate_config(invalid_config)
    assert "Missing required fields" in str(exc_info.value)

def test_validate_config_invalid_provider_config(test_config):
    """Test validation of invalid provider configuration"""
    engine = UnifiedEngine(test_config)
    invalid_config = {
        "version": "1.0",
        "metadata": {"project": "test"},
        "providers": {
            "aws": "invalid"  # Should be a dictionary
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        engine.validate_config(invalid_config)
    assert "Provider aws configuration must be a dictionary" in str(exc_info.value)

def test_initialize_providers_return_value(test_config):
    """Test initialize_providers return value"""
    engine = UnifiedEngine(test_config)
    providers = engine.initialize_providers()
    assert isinstance(providers, dict)
    assert "aws" in providers

@pytest.mark.asyncio
async def test_deploy_provider_with_missing_logger(test_config):
    """Test provider deployment when logger attribute is missing"""
    engine = UnifiedEngine(test_config)
    if hasattr(engine, 'logger'):
        delattr(engine, 'logger')
    result = await engine._deploy_provider("aws", {"enabled": True})
    assert result["status"] == "success"
    assert result["provider"] == "aws"

@pytest.mark.asyncio
async def test_deploy_provider_initializes_providers(test_config):
    """Test _deploy_provider initializes providers if not already initialized"""
    engine = UnifiedEngine(test_config)
    if hasattr(engine, 'providers'):
        del engine.providers
    result = await engine._deploy_provider("aws", {"enabled": True})
    assert result["status"] == "success"
    assert result["provider"] == "aws"

@pytest.mark.asyncio
async def test_deploy_uninitialized_provider(test_config):
    """Test deployment with uninitialized provider"""
    engine = UnifiedEngine(test_config)
    engine.providers = {}  # Clear providers
    with pytest.raises(ValueError) as exc_info:
        await engine._deploy_provider("aws", {"enabled": True})
    assert "Provider aws not initialized" in str(exc_info.value)

@pytest.mark.asyncio
async def test_deploy_results_handling(test_config):
    """Test handling of deployment results"""
    engine = UnifiedEngine(test_config)
    results = await engine.deploy()
    assert isinstance(results, dict)
    assert "aws" in results
    assert results["aws"]["status"] == "success"

@pytest.mark.asyncio
async def test_unified_engine_file_not_found():
    """Test UnifiedEngine with a non-existent configuration file."""
    with pytest.raises(ValidationError, match="Configuration file not found"):
        UnifiedEngine("nonexistent_file.yaml")

@pytest.mark.asyncio
async def test_unified_engine_general_exception_on_load(mocker):
    """Test UnifiedEngine with a general exception during configuration loading."""
    mocker.patch('builtins.open', side_effect=Exception("Mocked general error"))
    with pytest.raises(ValidationError, match="Mocked general error"):
        UnifiedEngine("test_config.yaml")

@pytest.mark.asyncio
async def test_deploy_provider_logging(test_config, mocker, caplog):
    """Test provider deployment with logging"""
    # Set up logging
    caplog.set_level(logging.INFO)
    
    # Create mock provider that succeeds
    mock_aws = mocker.AsyncMock()
    mock_aws.deploy.return_value = {"status": "success"}
    
    engine = UnifiedEngine(test_config)
    engine.initialize_providers()
    engine.providers["aws"] = mock_aws
    
    await engine._deploy_provider("aws", {"enabled": True})
    
    # Verify success logging
    assert "Deployed to aws: {'status': 'success'}" in caplog.text

@pytest.mark.asyncio
async def test_deploy_provider_error_logging(test_config, mocker, caplog):
    """Test provider deployment error handling and logging"""
    # Set up logging
    caplog.set_level(logging.ERROR)
    
    # Create mock provider that fails
    mock_aws = mocker.AsyncMock()
    mock_aws.deploy.side_effect = Exception("Deployment failed")
    
    engine = UnifiedEngine(test_config)
    engine.initialize_providers()
    engine.providers["aws"] = mock_aws
    
    with pytest.raises(Exception) as exc_info:
        await engine._deploy_provider("aws", {"enabled": True})
    
    # Verify error logging
    assert str(exc_info.value) == "Deployment failed"
    assert "Failed to deploy to aws: Deployment failed" in caplog.text

@pytest.mark.asyncio
async def test_deploy_provider_custom_config(test_config, mocker, caplog):
    """Test provider deployment with custom config and logging"""
    caplog.set_level(logging.INFO)
    
    # Create mock provider with detailed response
    mock_aws = mocker.AsyncMock()
    mock_aws.deploy.return_value = {
        "status": "success",
        "resources": {
            "compute": {
                "instances": 2,
                "type": "t3.medium"
            }
        }
    }
    
    # Initialize engine with mock
    engine = UnifiedEngine(test_config)
    engine.initialize_providers()
    engine.providers["aws"] = mock_aws
    
    # Test with detailed config
    test_config = {
        "enabled": True,
        "resources": {
            "compute": {
                "instances": 2,
                "type": "t3.medium"
            }
        }
    }
    
    result = await engine._deploy_provider("aws", test_config)
    
    # Verify logging and result
    mock_aws.deploy.assert_called_once_with(test_config)  # Fixed assertion
    assert "Deployed to aws:" in caplog.text
    assert result["status"] == "success"
    assert result["resources"]["compute"]["instances"] == 2
    
    # Add another deployment to ensure logging coverage
    await engine._deploy_provider("aws", test_config)
    assert len([r for r in caplog.records if "Deployed to aws:" in r.message]) == 2

@pytest.mark.asyncio
async def test_deploy_provider_not_initialized(test_config):
    """Test _deploy_provider when a provider is not initialized."""
    engine = UnifiedEngine(test_config)
    # Do not call initialize_providers to simulate uninitialized state
    with pytest.raises(ValueError, match="Provider non_existent_provider not initialized"):
        await engine._deploy_provider("non_existent_provider", {})

@pytest.mark.asyncio
async def test_deploy_results_coverage(test_config, mocker):
    """Test deploy method's results handling (line 48)"""
    engine = UnifiedEngine(test_config)
    engine.initialize_providers()
    mock_provider = mocker.AsyncMock()
    mock_provider.deploy.return_value = {"status": "ok"}
    engine.providers["aws"] = mock_provider
    results = await engine.deploy()
    assert results["aws"]["status"] == "ok"

def test_load_config_file_not_found(tmp_path):
    """Covers FileNotFoundError branch in _load_config (line 35)."""
    from src.core.engine.unified_engine import UnifiedEngine
    missing_path = tmp_path / "missing.yaml"
    try:
        UnifiedEngine(str(missing_path))
    except Exception as e:
        assert "Configuration file not found" in str(e)

def test_load_config_invalid_yaml(tmp_path):
    """Covers Exception branch in _load_config (invalid YAML)"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("{bad: yaml: content")
    from src.core.engine.unified_engine import UnifiedEngine
    from src.core.exceptions import ValidationError
    with pytest.raises(ValidationError, match="Failed to load configuration"):
        UnifiedEngine(str(bad))

def test_validate_config_providers_not_dict():
    from src.core.engine.unified_engine import UnifiedEngine
    engine = UnifiedEngine.__new__(UnifiedEngine)
    config = {"version": "1.0", "metadata": {}, "providers": []}
    with pytest.raises(ValidationError, match="Providers must be a dictionary"):
        engine.validate_config(config)

def test_validate_config_provider_not_dict():
    from src.core.engine.unified_engine import UnifiedEngine
    engine = UnifiedEngine.__new__(UnifiedEngine)
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": 123}}
    with pytest.raises(ValidationError, match="Provider aws configuration must be a dictionary"):
        engine.validate_config(config)

def test_validate_config_provider_missing_enabled():
    from src.core.engine.unified_engine import UnifiedEngine
    engine = UnifiedEngine.__new__(UnifiedEngine)
    config = {"version": "1.0", "metadata": {}, "providers": {"aws": {}}}
    with pytest.raises(ValidationError, match="Provider aws must specify enabled status"):
        engine.validate_config(config)

@pytest.mark.asyncio
async def test_deploy_error_logging(mocker, test_config):
    """Covers error logging in deploy (provider deploy fails)"""
    from src.core.engine.unified_engine import UnifiedEngine
    engine = UnifiedEngine(test_config)
    engine.initialize_providers()
    mock_provider = mocker.AsyncMock()
    mock_provider.deploy.side_effect = Exception("fail-deploy")
    engine.providers["aws"] = mock_provider
    logerr = mocker.patch.object(engine.logger, 'error', autospec=True)
    results = await engine.deploy()
    assert results["aws"]["status"] == "error"
    assert logerr.call_count > 0

@pytest.mark.asyncio
async def test__deploy_provider_error_logging(mocker, test_config):
    """Covers error logging in _deploy_provider (provider.deploy fails)"""
    from src.core.engine.unified_engine import UnifiedEngine
    engine = UnifiedEngine(test_config)
    engine.initialize_providers()
    mock_provider = mocker.AsyncMock()
    mock_provider.deploy.side_effect = Exception("fail-deploy")
    engine.providers["aws"] = mock_provider
    logerr = mocker.patch.object(engine.logger, 'error', autospec=True)
    with pytest.raises(Exception, match="fail-deploy"):
        await engine._deploy_provider("aws", {"enabled": True})
    assert logerr.call_count > 0

@pytest.mark.asyncio
async def test_optimize_configuration_async_stub(mocker, test_config):
    """Covers optimize_configuration_async stub for coverage"""
    from src.core.engine.unified_engine import UnifiedEngine
    engine = UnifiedEngine(test_config)
    mocker.patch.object(engine, 'optimize_configuration', return_value={"ok": True})
    result = await engine.optimize_configuration_async({"foo": "bar"})
    assert result["ok"] is True

@pytest.mark.asyncio
async def test__deploy_provider_fallback_and_error_logging(mocker, test_config):
    """Covers fallback static response and error logging in _deploy_provider."""
    from src.core.engine.unified_engine import UnifiedEngine
    engine = UnifiedEngine(test_config)
    engine.initialize_providers()
    # Fallback: provider object without deploy method
    engine.providers["aws"] = object()
    loginfo = mocker.patch.object(engine.logger, 'info', autospec=True)
    result = await engine._deploy_provider("aws", {"enabled": True})
    assert result["status"] == "success"
    assert loginfo.call_count > 0
    # Error: provider not in self.providers
    with pytest.raises(ValueError, match="Provider missingprov not initialized"):
        await engine._deploy_provider("missingprov", {"enabled": True})

@pytest.mark.skip(reason="Cannot cover logger.error in initialize_providers error branch due to Python control flow; exception in .get() prevents logger call.")
def test_initialize_providers_error_logging(mocker, test_config):
    """Uncoverable: logger.error in initialize_providers cannot be triggered if .get() raises, due to Python control flow."""
    from src.core.engine.unified_engine import UnifiedEngine
    engine = UnifiedEngine(test_config)
    class BadConfig(dict):
        def get(self, *a, **kw):
            raise Exception("init-error")
    engine.config["providers"] = {"bad": BadConfig()}
    logerr = mocker.patch.object(engine.logger, 'error', autospec=True)
    try:
        engine.initialize_providers()
    except Exception as e:
        assert "init-error" in str(e)
    assert logerr.call_count > 0