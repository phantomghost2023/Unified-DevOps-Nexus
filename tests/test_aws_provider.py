import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import boto3
import botocore
from botocore.exceptions import ClientError
from core.plugins.aws_provider import AwsProvider

@pytest.fixture
def aws_config():
    return {
        "region": "us-east-1",
        "credentials": {
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret"
        }
    }

@pytest.fixture
def aws_provider(aws_config):
    return AwsProvider(aws_config)

@pytest.mark.asyncio
async def test_aws_provider_initialization(aws_config):
    provider = AwsProvider(aws_config)
    assert provider.config == aws_config

@pytest.mark.asyncio
async def test_aws_provider_deploy_success(aws_provider, mocker):
    # Mock the AWS session and client
    mock_session = mocker.Mock(spec=boto3.Session)
    mock_s3 = mocker.Mock()
    
    # Mock bucket data
    mock_buckets = {
        'Buckets': [
            {'Name': 'test-bucket-1', 'CreationDate': '2024-01-01'},
            {'Name': 'test-bucket-2', 'CreationDate': '2024-01-02'}
        ]
    }
    mock_s3.list_buckets.return_value = mock_buckets
    
    # Setup the mock session to return our mock client
    mock_session.client.return_value = mock_s3
    
    # Mock boto3.Session
    with patch('boto3.Session', return_value=mock_session):
        result = await aws_provider.deploy()
        
        assert result["status"] == "success"
        assert len(result["buckets"]) == 2
        assert result["buckets"][0]["Name"] == "test-bucket-1"
        assert result["buckets"][1]["Name"] == "test-bucket-2"

@pytest.mark.asyncio
async def test_aws_provider_deploy_with_custom_config(aws_provider, mocker):
    custom_config = {
        "region": "us-west-2",
        "credentials": {
            "aws_access_key_id": "custom-key",
            "aws_secret_access_key": "custom-secret"
        }
    }
    
    mock_session = mocker.Mock(spec=boto3.Session)
    mock_s3 = mocker.Mock()
    mock_s3.list_buckets.return_value = {'Buckets': []}
    mock_session.client.return_value = mock_s3
    
    with patch('boto3.Session', return_value=mock_session):
        result = await aws_provider.deploy(custom_config)
        
        assert result["status"] == "success"
        assert isinstance(result["buckets"], list)

@pytest.mark.asyncio
async def test_aws_provider_deploy_with_client_error(aws_provider, mocker):
    mock_session = mocker.Mock(spec=boto3.Session)
    mock_s3 = mocker.Mock()
    mock_s3.list_buckets.side_effect = ClientError(
        {'Error': {'Code': 'AccessDenied', 'Message': 'Access Denied'}},
        'ListBuckets'
    )
    mock_session.client.return_value = mock_s3
    
    with patch('boto3.Session', return_value=mock_session):
        with pytest.raises(ClientError):
            await aws_provider.deploy()

@pytest.mark.asyncio
async def test_aws_provider_get_actual_state(aws_provider, mocker):
    mock_session = mocker.Mock(spec=boto3.Session)
    mock_s3 = mocker.Mock()
    mock_s3.list_buckets.return_value = {
        'Buckets': [
            {'Name': 'test-bucket-1'},
            {'Name': 'test-bucket-2'}
        ]
    }
    mock_session.client.return_value = mock_s3
    
    with patch('boto3.Session', return_value=mock_session):
        result = await aws_provider.get_actual_state()
        
        assert "test-bucket-1" in result["buckets"]
        assert "test-bucket-2" in result["buckets"]
        assert len(result["buckets"]) == 2

@pytest.mark.asyncio
async def test_aws_provider_remediate_drift(aws_provider, mocker):
    # Mock the AWS session and client
    mock_session = mocker.Mock(spec=boto3.Session)
    mock_s3 = mocker.Mock()
    mock_session.client.return_value = mock_s3
    
    # Mock get_actual_state to return existing buckets
    mock_s3.list_buckets.return_value = {
        'Buckets': [{'Name': 'existing-bucket'}]
    }
    
    # Setup for create_bucket
    mock_s3.create_bucket.return_value = {}
    
    with patch('boto3.Session', return_value=mock_session):
        desired_state = {
            "buckets": ["existing-bucket", "new-bucket-1", "new-bucket-2"]
        }
        
        result = await aws_provider.remediate_drift(desired_state)
        
        assert "new-bucket-1" in result["created_buckets"]
        assert "new-bucket-2" in result["created_buckets"]
        assert len(result["created_buckets"]) == 2

@pytest.mark.asyncio
async def test_aws_provider_remediate_drift_bucket_exists(aws_provider, mocker):
    mock_session = mocker.Mock(spec=boto3.Session)
    mock_s3 = mocker.Mock()
    mock_session.client.return_value = mock_s3
    
    # Mock get_actual_state
    mock_s3.list_buckets.return_value = {
        'Buckets': [{'Name': 'existing-bucket'}]
    }
    
    # Mock create_bucket to raise BucketAlreadyExists error
    mock_s3.create_bucket.side_effect = ClientError(
        {'Error': {'Code': 'BucketAlreadyExists', 'Message': 'Bucket already exists'}},
        'CreateBucket'
    )
    
    with patch('boto3.Session', return_value=mock_session):
        desired_state = {
            "buckets": ["existing-bucket", "existing-global-bucket"]
        }
        
        result = await aws_provider.remediate_drift(desired_state)
        
        assert len(result["created_buckets"]) == 0