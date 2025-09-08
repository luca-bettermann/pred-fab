
from typing import Dict, Any, List, Optional, Tuple, Type
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_geometry(
    exp_code: str, 
    designed_layer: Dict[int, List[Dict[str, float]]], 
    measured_layer: Dict[int, List[Dict[str, float]]], 
    avg_deviation: float,
    z_axis_length: int = 4) -> None:
    """
    Visualize the results of the path deviation analysis for all layers.

    Args:
    exp_code: Experiment code identifier
    designed_layer: Dict of layer_id and List of coordinate points as dictionaries ('x', 'y', 'z')
    measured_layer: Dict of layer_id and List of coordinate points as dictionaries ('x', 'y', 'z')
    avg_deviation: Dict of layer_id and List of deviation values from which we take the average
    z_axis_length: Length of the z-axis for visualization

    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    all_z = []
    
    # Iterate through all layers
    for layer_id in designed_layer.keys():
        if layer_id not in measured_layer:
            continue
            
        # Extract coordinates for current layer
        designed_x = [point['x'] for point in designed_layer[layer_id]]
        designed_y = [point['y'] for point in designed_layer[layer_id]]
        designed_z = [point['z'] for point in designed_layer[layer_id]]

        measured_x = [point['x'] for point in measured_layer[layer_id]]
        measured_y = [point['y'] for point in measured_layer[layer_id]]
        measured_z = [point['z'] for point in measured_layer[layer_id]]

        # Add to all_z for axis calculation
        all_z.extend(designed_z + measured_z)

        # Plot 3D paths for current layer
        ax.plot(designed_x, designed_y, designed_z, color='gray', 
                label='Designed Path' if layer_id == list(designed_layer.keys())[0] else "", 
                linewidth=2)
        ax.plot(measured_x, measured_y, measured_z, color='green', 
                label='Measured Path' if layer_id == list(designed_layer.keys())[0] else "", 
                linewidth=2)

        # Draw lines between corresponding points to show deviation
        for i in range(min(len(designed_layer[layer_id]), len(measured_layer[layer_id]))):
            ax.plot([designed_x[i], measured_x[i]], 
                    [designed_y[i], measured_y[i]], 
                    [designed_z[i], measured_z[i]], 'r--', alpha=0.5)
    
    # Set z-axis limits to exact input length
    # z_center = (min(all_z) + max(all_z)) / 2
    # ax.set_zlim(z_center - z_axis_length / 2, z_center + z_axis_length / 2) # type: ignore

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore

    ax.set_title(f"EXP: '{exp_code}' - {len(designed_layer)} Layers\n\nAvg Path Deviation = {avg_deviation:.3f}")
    ax.legend()
    plt.show(block=True)
