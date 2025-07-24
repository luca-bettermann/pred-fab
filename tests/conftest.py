import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path

from src.lbp_package.utils.log_manager import LBPLogger
from examples.mock_data_interface import ExampleDataInterface

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def mock_study_params():
    """Load mock study parameters."""
    return {
        "target_deviation": 0.0,
        "max_deviation": 0.5,
        "target_energy": 0.0,
        "max_energy": 1000.0,
        "power_rating": 50.0
    }

@pytest.fixture
def mock_exp_params():
    """Load mock experiment data."""
    return {
        "n_layers": 2,
        "n_segments": 2,
        "layerTime": 30.0,
    }

@pytest.fixture
def mock_config():
    """Load example configuration from examples/config.yaml."""
    config_path = Path(__file__).parent.parent / "examples" / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def mock_data_interface(temp_dir, mock_study_params, mock_exp_params):
    """Create mock data interface with test data."""
    return ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)

@pytest.fixture
def test_logger(temp_dir):
    """Create test logger."""
    log_dir = os.path.join(temp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return LBPLogger("TestLogger", log_dir)

@pytest.fixture
def setup_test_data(temp_dir):
    """Setup test data files."""
    # Create study directory
    study_dir = os.path.join(temp_dir, "TEST_STUDY")
    os.makedirs(study_dir, exist_ok=True)
    
    # Create experiment directory
    exp_dir = os.path.join(study_dir, "TEST_STUDY_001")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create designed paths data
    designed_paths = {
        "layers": [
            {
                "layer_id": 0,
                "segments": [
                    {
                        "segment_id": 0,
                        "path_points": [
                            {"x": 10.0, "y": 20.0, "z": 0.2},
                            {"x": 15.0, "y": 25.0, "z": 0.2},
                            {"x": 20.0, "y": 30.0, "z": 0.2}
                        ]
                    },
                    {
                        "segment_id": 1,
                        "path_points": [
                            {"x": 25.0, "y": 35.0, "z": 0.2},
                            {"x": 30.0, "y": 40.0, "z": 0.2},
                            {"x": 35.0, "y": 45.0, "z": 0.2}
                        ]
                    }
                ]
            },
            {
                "layer_id": 1,
                "segments": [
                    {
                        "segment_id": 0,
                        "path_points": [
                            {"x": 10.0, "y": 20.0, "z": 0.4},
                            {"x": 15.0, "y": 25.0, "z": 0.4},
                            {"x": 20.0, "y": 30.0, "z": 0.4}
                        ]
                    },
                    {
                        "segment_id": 1,
                        "path_points": [
                            {"x": 25.0, "y": 35.0, "z": 0.4},
                            {"x": 30.0, "y": 40.0, "z": 0.4},
                            {"x": 35.0, "y": 45.0, "z": 0.4}
                        ]
                    }
                ]
            }
        ]
    }
    
    # Create measured paths data with small deviations
    measured_paths = {
        "layers": [
            {
                "layer_id": 0,
                "segments": [
                    {
                        "segment_id": 0,
                        "path_points": [
                            {"x": 10.05, "y": 20.02, "z": 0.19},
                            {"x": 15.03, "y": 25.01, "z": 0.21},
                            {"x": 20.02, "y": 30.03, "z": 0.20}
                        ]
                    },
                    {
                        "segment_id": 1,
                        "path_points": [
                            {"x": 25.01, "y": 35.04, "z": 0.19},
                            {"x": 30.02, "y": 40.01, "z": 0.22},
                            {"x": 35.03, "y": 45.02, "z": 0.20}
                        ]
                    }
                ]
            },
            {
                "layer_id": 1,
                "segments": [
                    {
                        "segment_id": 0,
                        "path_points": [
                            {"x": 10.02, "y": 20.01, "z": 0.41},
                            {"x": 15.01, "y": 25.02, "z": 0.39},
                            {"x": 20.01, "y": 30.01, "z": 0.40}
                        ]
                    },
                    {
                        "segment_id": 1,
                        "path_points": [
                            {"x": 25.02, "y": 35.01, "z": 0.41},
                            {"x": 30.01, "y": 40.02, "z": 0.39},
                            {"x": 35.01, "y": 45.01, "z": 0.40}
                        ]
                    }
                ]
            }
        ]
    }
    
    # Save path data files
    with open(os.path.join(exp_dir, "TEST_STUDY_001_designed_paths.json"), 'w') as f:
        json.dump(designed_paths, f, indent=2)
    
    with open(os.path.join(exp_dir, "TEST_STUDY_001_measured_paths.json"), 'w') as f:
        json.dump(measured_paths, f, indent=2)
    
    return {
        "study_dir": study_dir,
        "exp_dir": exp_dir,
        "designed_paths": designed_paths,
        "measured_paths": measured_paths
    }
