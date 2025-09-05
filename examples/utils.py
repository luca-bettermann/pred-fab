from typing import Dict, Any, List
from random import randint

def generate_path_data(n_layers: int = 2, n_segments: int = 2, noise: bool = False) -> Dict[str, Any]:
    """Return designed path data, add random noise to mock a measured path.
    
    Args:
        path_deviation: Amount of noise to add to coordinates
        n_layers: Number of layers to generate
        n_segments: Number of segments per layer
    """
    layers = []
    xy_deviation = 3 if noise else 0.0
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


def generate_temperature_data(base_temp: int = 20, fluctuation: int = 2) -> List[int]:
    """Generate a time series of temperature data with continuous changes."""
    temperatures = []
    current_temp = base_temp
    for i in range(10):
        # Simulate a temperature reading with some fluctuation
        temp = current_temp + randint(-fluctuation, fluctuation)
        temperatures.append(temp)
        current_temp = temp
    return temperatures


def _add_noise(magnitude: float) -> float:
    """Add random noise to path points."""
    return randint(-10, 10) * 0.1 * magnitude if magnitude != 0 else 0