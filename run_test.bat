@echo off
set PYTHONPATH=%CD%
python -m src.tests.test_azure_provider
pause 