import pytest
import tempfile
import os
import json
import yaml
from random import randint

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

def get_mock_exp_params(n_layers: int = 2, n_segments: int = 2, layer_time: float = 30.0, layer_height: float = 0.2) -> Dict[str, Any]:
    """Return standard experiment parameters."""
    return {
        "n_layers": n_layers,
        "n_segments": n_segments,
        "layerTime": layer_time,
        "layerHeight": layer_height
    }

def generate_path_data(n_layers: int = 2, n_segments: int = 2, noise: bool = False) -> Dict[str, Any]:
    """Return designed path data, add random noise to mock a measured path.
    
    Args:
        path_deviation: Amount of noise to add to coordinates
        n_layers: Number of layers to generate
        n_segments: Number of segments per layer
    """
    layers = []
    xy_deviation = 0.5 if noise else 0.0
    z_deviation = 0.02 if noise else 0.0
    
    for layer_id in range(n_layers):
        segments = []
        z_position = 0.2 * (layer_id + 1)
        
        for segment_id in range(n_segments):
            # Generate path points for each segment
            # Base coordinates shift by segment
            x_offset = segment_id * 15
            y_offset = segment_id * 15
            
            # Create 3 path points per segment
            path_points = [
                {"x": 10.0 + x_offset + _add_noise(xy_deviation), "y": 20.0 + y_offset + _add_noise(xy_deviation), "z": z_position + _add_noise(z_deviation)},
                {"x": 15.0 + x_offset + _add_noise(xy_deviation), "y": 25.0 + y_offset + _add_noise(xy_deviation), "z": z_position + _add_noise(z_deviation)},
                {"x": 20.0 + x_offset + _add_noise(xy_deviation), "y": 30.0 + y_offset + _add_noise(xy_deviation), "z": z_position + _add_noise(z_deviation)}
            ]
            
            segments.append({
                "segment_id": segment_id,
                "path_points": path_points
            })
        
        layers.append({
            "layer_id": layer_id,
            "segments": segments
        })
    
    return {"layers": layers}

def generate_energy_consumption_data(n_layers: int = 2, n_segments: int = 2, noise: bool = False) -> Dict[str, Any]:
    """Return designed energy consumption data, add random noise to mock measured values.

    Args:
        n_layers: Number of layers to generate
        n_segments: Number of segments per layer
        noise: Amount of noise to add to coordinates
    """
    deviation = 1.0 if noise else 0.0
    layers = []
    for layer_id in range(n_layers):
        layers.append({
            "layer_id": layer_id,
            "energy_consumption": 10.0 * n_segments + _add_noise(deviation)
        })
    return {"layers": layers}

def _add_noise(magnitude: float) -> float:
    """Add random noise to path points."""
    return randint(-10, 10) * 0.1 * magnitude if magnitude != 0 else 0

def generate_temperature_data(base_temp: int = 20, fluctuation: int = 2) -> Dict[str, Any]:
    """Generate a time series of temperature data with continuous changes."""
    temperatures = []
    current_temp = base_temp
    for i in range(10):
        # Simulate a temperature reading with some fluctuation
        temp = current_temp + randint(-fluctuation, fluctuation)
        temperatures.append(temp)
        current_temp = temp
    return {"temperature": temperatures}

def create_study_json_files(base_folder: str, study_code: str = "test"):
    """Generate the file structure for an experiment."""
    
    # Create directory structure
    study_dir = os.path.join(base_folder, study_code)
    os.makedirs(study_dir, exist_ok=True)
    
    # Create study_params.json (overwrite if it exists)
    study_data = {
        "study_name": f"Test Study {study_code}",
        "description": "Generated test study",
        "Parameters": get_mock_study_params()
    }
    with open(os.path.join(study_dir, "study_params.json"), 'w') as f:
        json.dump(study_data, f, indent=2)

    # Create performance_records.json (overwrite if it exists)
    perf_data = {
        "records": [
            {"Code": "path_deviation", "Performance": "Path Deviation", "Optimal Value": 0.0, "Description": "Deviation from designed path"},
            {"Code": "energy_consumption", "Performance": "Energy Consumption", "Optimal Value": 0.0, "Description": "Energy consumed during operation"}
        ]
    }
    with open(os.path.join(study_dir, "performance_records.json"), 'w') as f:
        json.dump(perf_data, f, indent=2)

def create_exp_json_files(
        base_folder: str, 
        study_code: str = "test", 
        exp_nr: int = 1, 
        layer_time: float = 30.0,
        layer_height: float = 0.2,
        n_layers: int = 2,
        n_segments: int = 2
        ):
    """Generate all experimental data files in the specified folder structure. 
    Use the input parameters to customize the data generation."""

    # Create directory structure
    study_dir = os.path.join(base_folder, study_code)
    exp_code = f"{study_code}_{exp_nr:03d}"
    exp_dir = os.path.join(study_dir, exp_code)

    os.makedirs(exp_dir, exist_ok=True)

    # Create exp_params.json (overwrite if it exists)
    exp_data = {
        "Code": exp_code,
        "System Performance": None,
        "Parameters": get_mock_exp_params(n_layers=n_layers, n_segments=n_segments, layer_time=layer_time, layer_height=layer_height),
        "Status": "Evaluation"
    }
    with open(os.path.join(exp_dir, "exp_params.json"), 'w') as f:
        json.dump(exp_data, f, indent=2)

    # Create designed path data files (overwrite if they exist)
    with open(os.path.join(exp_dir, f"{exp_code}_designed_paths.json"), 'w') as f:
        json.dump(generate_path_data(
            n_layers=n_layers,
            n_segments=n_segments,
            noise=False
        ), f, indent=2)

    # Generate measured path data
    with open(os.path.join(exp_dir, f"{exp_code}_measured_paths.json"), 'w') as f:
        json.dump(generate_path_data(
            n_layers=n_layers,
            n_segments=n_segments,
            noise=True
        ), f, indent=2)

    # Generate energy consumption data
    with open(os.path.join(exp_dir, f"{exp_code}_energy_consumption.json"), 'w') as f:
        json.dump(generate_energy_consumption_data(
            n_layers=n_layers,
            n_segments=n_segments,
            noise=True
        ), f, indent=2)

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
    create_study_json_files(temp_dir, study_code="test")
    create_exp_json_files(temp_dir, study_code="test", exp_nr=1)
    
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
    create_study_json_files(temp_dir, study_code="test")
    create_exp_json_files(temp_dir, study_code="test", exp_nr=1)
    
    return {
        "study_dir": os.path.join(temp_dir, "test"),
        "exp_dir": os.path.join(temp_dir, "test", "test_001"),
        "designed_paths": generate_path_data(),
        "measured_paths": generate_path_data(noise=True),
    }
