import subprocess
import sys
import os
import subprocess

import pytest
from pathlib import Path

def test_cli_drift_detect(tmp_path):
    # Write a minimal config file
    config_file = tmp_path / "nexus-infra.yml"
    config_file.write_text("""
version: '1.0'
providers:
  aws:
    enabled: true
    regions: [us-east-1]
    buckets: []
""")
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    cli_script_path = project_root / "cli" / "nexus_cli.py"

    result = subprocess.run(
        [sys.executable, str(cli_script_path), "drift-detect", str(config_file)],
        capture_output=True, text=True, cwd=project_root
    )
    assert "Drift detection started..." in result.stdout
    assert "Drift detected:" in result.stdout or "No drift detected." in result.stdout