import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from core.plugins.gcp_provider import GcpProvider

@pytest.fixture
def gcp_config():
    return {
        "project_id": "test-project-id",
        "region": "us-central1"
    }

@pytest.fixture
def gcp_provider(gcp_config):
    return GcpProvider(gcp_config)

def test_gcp_provider_initialization(gcp_config):
    provider = GcpProvider(gcp_config)
    assert provider.config == gcp_config
    assert provider._client is None

def test_gcp_provider_initialization_invalid_config():
    with pytest.raises(ValueError, match="Config must be a dictionary"):
        GcpProvider("invalid_config")

def test_gcp_provider_initialization_missing_required_fields():
    with pytest.raises(ValueError, match="Missing required configuration fields"):
        GcpProvider({"subscription_id": "test"})

@pytest.mark.asyncio
async def test_gcp_provider_deploy_success(gcp_provider, mocker):
    # Mock the GCP client
    mock_client = mocker.Mock(spec=storage.Client)
    
    # Mock buckets
    mock_bucket1 = MagicMock()
    mock_bucket1.name = "test-bucket-1"
    mock_bucket2 = MagicMock()
    mock_bucket2.name = "test-bucket-2"
    
    # Setup the mock client to return our test buckets
    mock_client.list_buckets.return_value = [mock_bucket1, mock_bucket2]
    
    # Mock the storage client
    with patch('core.plugins.gcp_provider.storage.Client', return_value=mock_client):
        result = await gcp_provider.deploy()
        
        assert result["status"] == "success"
        assert "test-bucket-1" in result["buckets"]
        assert "test-bucket-2" in result["buckets"]
        assert len(result["buckets"]) == 2
        assert "message" in result

@pytest.mark.asyncio
async def test_gcp_provider_deploy_with_custom_config(gcp_provider, mocker):
    custom_config = {
        "project_id": "custom-project-id",
        "region": "us-east1"
    }
    
    mock_client = mocker.Mock(spec=storage.Client)
    mock_client.list_buckets.return_value = []
    
    with patch('core.plugins.gcp_provider.storage.Client', return_value=mock_client):
        result = await gcp_provider.deploy(custom_config)
        
        assert result["status"] == "success"
        assert isinstance(result["buckets"], list)
        assert "message" in result

@pytest.mark.asyncio
async def test_gcp_provider_deploy_with_gcp_error(gcp_provider, mocker):
    mock_client = mocker.Mock(spec=storage.Client)
    mock_client.list_buckets.side_effect = GoogleAPIError("Test error")
    
    with patch('core.plugins.gcp_provider.storage.Client', return_value=mock_client):
        result = await gcp_provider.deploy()
        assert result["status"] == "error"
        assert "error" in result
        assert "message" in result

@pytest.mark.asyncio
async def test_gcp_provider_deploy_with_invalid_config(gcp_provider):
    with pytest.raises(ValueError, match="Config must be a dictionary"):
        await gcp_provider.deploy("invalid_config")

@pytest.mark.asyncio
async def test_gcp_provider_get_actual_state_success(gcp_provider, mocker):
    mock_client = mocker.Mock(spec=storage.Client)
    mock_client.list_buckets.return_value = []
    
    with patch('core.plugins.gcp_provider.storage.Client', return_value=mock_client):
        result = await gcp_provider.get_actual_state()
        assert result["status"] == "success"
        assert "buckets" in result
        assert "message" in result

@pytest.mark.asyncio
async def test_gcp_provider_get_actual_state_error(gcp_provider, mocker):
    mock_client = mocker.Mock(spec=storage.Client)
    mock_client.list_buckets.side_effect = Exception("Test error")
    
    with patch('core.plugins.gcp_provider.storage.Client', return_value=mock_client):
        result = await gcp_provider.get_actual_state()
        assert result["status"] == "error"
        assert "error" in result
        assert "message" in result

@pytest.mark.asyncio
async def test_gcp_provider_remediate_drift_success(gcp_provider, mocker):
    mock_client = mocker.Mock(spec=storage.Client)
    
    with patch('core.plugins.gcp_provider.storage.Client', return_value=mock_client):
        result = await gcp_provider.remediate_drift({"desired": "state"})
        assert result["status"] == "success"
        assert "remediated" in result
        assert "message" in result

@pytest.mark.asyncio
async def test_gcp_provider_remediate_drift_error(gcp_provider, mocker):
    mock_client = mocker.Mock(spec=storage.Client)
    mock_client.list_buckets.side_effect = Exception("Test error")
    
    with patch('core.plugins.gcp_provider.storage.Client', return_value=mock_client):
        result = await gcp_provider.remediate_drift({"desired": "state"})
        assert result["status"] == "error"
        assert "error" in result
        assert "message" in result

@pytest.mark.asyncio
async def test_gcp_provider_remediate_drift_invalid_state(gcp_provider):
    with pytest.raises(ValueError, match="Desired state must be a dictionary"):
        await gcp_provider.remediate_drift("invalid_state")