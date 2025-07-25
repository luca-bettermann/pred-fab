import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path

from src.lbp_package.utils.log_manager import LBPLogger
from examples.file_data_interface import FileDataInterface
from tests.test_data import create_test_data_files, get_mock_study_params, get_mock_exp_params, get_designed_paths, get_measured_paths

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def mock_study_params():
    """Load mock study parameters."""
    return get_mock_study_params()

@pytest.fixture
def mock_exp_params():
    """Load mock experiment data."""
    return get_mock_exp_params()

@pytest.fixture
def mock_config():
    """Load example configuration from examples/config.yaml."""
    config_path = Path(__file__).parent.parent / "examples" / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def mock_data_interface(temp_dir):
    """Create mock data interface with test data files."""
    # Generate test data files using the shared utility
    create_test_data_files(temp_dir, study_code="test", exp_nr=1)
    
    # Return interface pointing to temp directory with generated files
    return FileDataInterface(temp_dir)

@pytest.fixture
def test_logger(temp_dir):
    """Create test logger."""
    log_dir = os.path.join(temp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return LBPLogger("TestLogger", log_dir)

@pytest.fixture
def setup_test_data(temp_dir):
    """Setup test data files using the shared utility."""
    # Generate complete test data structure
    create_test_data_files(temp_dir, study_code="test", exp_nr=1)
    
    return {
        "study_dir": os.path.join(temp_dir, "test"),
        "exp_dir": os.path.join(temp_dir, "test", "test_001"),
        "designed_paths": get_designed_paths(),
        "measured_paths": get_measured_paths()
    }
