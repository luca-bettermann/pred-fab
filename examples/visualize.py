
from typing import Dict, Any, List, Optional, Tuple, Type
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_geometry(
    exp_code: str, 
    layer_id: int, 
    designed_layer: List[Dict[str, float]], 
    measured_layer: List[Dict[str, float]], 
    avg_deviation: List[float],
    z_axis_length: int = 4) -> None:
    """
    Visualize the results of the path deviation analysis.

    Args:
    exp_code: Experiment code identifier
    layer_id: Layer identifier number
    designed_layer: List of coordinate points as dictionaries ('x', 'y', 'z')
    measured_layer: List of coordinate points as dictionaries ('x', 'y', 'z')
    avg_deviation: List of deviation values from which we take the average
    z_axis_length: Length of the z-axis for visualization

    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    designed_x = [point['x'] for point in designed_layer]
    designed_y = [point['y'] for point in designed_layer]
    designed_z = [point['z'] for point in designed_layer]

    measured_x = [point['x'] for point in measured_layer]
    measured_y = [point['y'] for point in measured_layer]
    measured_z = [point['z'] for point in measured_layer]

    # Plot 3D paths
    ax.plot(designed_x, designed_y, designed_z, color='gray', label='Designed Path', linewidth=2)
    ax.plot(measured_x, measured_y, measured_z, color='green', label='Measured Path', linewidth=2)

    # Draw lines between corresponding points to show deviation
    for i in range(len(designed_layer)):
        ax.plot([designed_x[i], measured_x[i]], 
                [designed_y[i], measured_y[i]], 
                [designed_z[i], measured_z[i]], 'r--', alpha=0.5)
    
    # Set z-axis limits to exact input length
    all_z = designed_z + measured_z
    z_center = (min(all_z) + max(all_z)) / 2
    ax.set_zlim(z_center - z_axis_length / 2, z_center + z_axis_length / 2) # type: ignore

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore

    avg_dev = sum(avg_deviation) / len(avg_deviation)
    ax.set_title(f"EXP: '{exp_code}' - Layer {layer_id}\n\nAvg Path Deviation = {avg_dev:.3f}")
    ax.legend()
    plt.show()

def visualize_temperature(temperature_data: Dict[str, Any]) -> None:
    """Visualize the temperature data."""
    plt.figure(figsize=(10, 6))
    plt.plot(temperature_data["time"], temperature_data["temperature"], label="Temperature")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Over Time")
    plt.legend()
    plt.show()