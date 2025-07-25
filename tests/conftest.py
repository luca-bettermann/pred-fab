import pytest
import tempfile
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any

from src.lbp_package.utils.log_manager import LBPLogger
from examples.file_data_interface import FileDataInterface


def get_mock_study_params() -> Dict[str, Any]:
    """Return standard study parameters."""
    return {
        "target_deviation": 0.0,
        "max_deviation": 0.5,
        "target_energy": 0.0,
        "max_energy": 10000.0,
        "power_rating": 50.0
    }

def get_mock_exp_params() -> Dict[str, Any]:
    """Return standard experiment parameters."""
    return {
        "n_layers": 2,
        "n_segments": 2,
        "layerTime": 30.0,
    }

def get_designed_paths() -> Dict[str, Any]:
    """Return standard designed path data."""
    return {
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

def get_measured_paths() -> Dict[str, Any]:
    """Return standard measured path data with small deviations."""
    return {
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

def create_test_data_files(base_folder: str, study_code: str = "test", exp_nr: int = 1):
    """Generate all test data files in the specified folder structure."""
    
    # Create directory structure
    study_dir = os.path.join(base_folder, study_code)
    exp_code = f"{study_code}_{exp_nr:03d}"
    exp_dir = os.path.join(study_dir, exp_code)
    
    os.makedirs(study_dir, exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create study_params.json
    study_data = {
        "study_name": f"Test Study {study_code}",
        "description": "Generated test study",
        "parameters": get_mock_study_params()
    }
    with open(os.path.join(study_dir, "study_params.json"), 'w') as f:
        json.dump(study_data, f, indent=2)
    
    # Create performance_records.json
    perf_data = {
        "records": [
            {"Code": "path_deviation", "Name": "Path Deviation", "Active": True, "Unit": "mm"},
            {"Code": "energy_consumption", "Name": "Energy Consumption", "Active": True, "Unit": "Wh"}
        ]
    }
    with open(os.path.join(study_dir, "performance_records.json"), 'w') as f:
        json.dump(perf_data, f, indent=2)
    
    # Create exp_params.json
    exp_data = {
        "exp_name": f"Test Experiment {exp_code}",
        "description": "Generated test experiment",
        "parameters": get_mock_exp_params(),
        "status": "completed"
    }
    with open(os.path.join(exp_dir, "exp_params.json"), 'w') as f:
        json.dump(exp_data, f, indent=2)
    
    # Create path data files
    with open(os.path.join(exp_dir, f"{exp_code}_designed_paths.json"), 'w') as f:
        json.dump(get_designed_paths(), f, indent=2)
    
    with open(os.path.join(exp_dir, f"{exp_code}_measured_paths.json"), 'w') as f:
        json.dump(get_measured_paths(), f, indent=2)



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
