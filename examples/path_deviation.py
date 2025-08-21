import json
import os
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Type
from dataclasses import dataclass

from lbp_package import EvaluationModel, FeatureModel
from lbp_package.utils import runtime_parameter, model_parameter, exp_parameter

@dataclass
class PathEvaluation(EvaluationModel):
    """Example evaluation model for path deviation assessment."""

    # Model parameters
    target_deviation: Optional[float] = model_parameter(default=0.1)
    max_deviation: Optional[float] = model_parameter()
    
    # Experiment parameters
    n_layers: Optional[int] = exp_parameter()
    n_segments: Optional[int] = exp_parameter()

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _declare_dimensions(self) -> List[Tuple[str, str, str]]:
        """Declare dimensions for path evaluation with the corresponding structure."""
        dimension_names = [
            ('layers', 'layer_id', 'n_layers'),
            ('segments', 'segment_id', 'n_segments')
        ]
        return dimension_names
    
    def _declare_feature_model_type(self) -> Type[FeatureModel]:
        """Declare the feature model type to use for feature extraction."""
        return PathDeviationFeature

    def _compute_target_value(self) -> Optional[float]:
        """Return target deviation (ideally 0)."""
        return self.target_deviation

    def _declare_scaling_factor(self) -> Optional[float]:
        """Return maximum acceptable deviation."""
        return self.max_deviation


@dataclass
class PathDeviationFeature(FeatureModel):
    """Example feature model for path deviation calculation."""
    
    # Model parameters
    tolerance_xyz: Optional[float] = model_parameter(0.1)

    # Runtime parameters
    layer_id: Optional[int] = runtime_parameter()
    segment_id: Optional[int] = runtime_parameter()

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize feature storage for path deviation
        self.features["path_deviation"] = np.empty([])

    def _load_data(self, exp_code: str, exp_folder: str) -> Dict[str, Any]:
        """Load designed and measured path data."""        
        # Load designed paths
        designed_path = os.path.join(exp_folder, f"{exp_code}_designed_paths.json")
        with open(designed_path, 'r') as f:
            designed_data = json.load(f)

        # Load measured paths
        measured_path = os.path.join(exp_folder, f"{exp_code}_measured_paths.json")
        with open(measured_path, 'r') as f:
            measured_data = json.load(f)
            
        return {"designed": designed_data, "measured": measured_data}
    
    def _compute_features(self, data: Dict, visualize_flag: bool) -> Dict[str, float]:
        """Calculate path deviation for current layer/segment."""
        designed_layer = data["designed"]["layers"][self.layer_id]
        measured_layer = data["measured"]["layers"][self.layer_id]
        
        designed_segment = designed_layer["segments"][self.segment_id]
        measured_segment = measured_layer["segments"][self.segment_id]
        
        # Calculate average deviation across all points in segment
        total_deviation = 0.0
        point_count = len(designed_segment["path_points"])
        
        for i in range(point_count):
            d_point = designed_segment["path_points"][i]
            m_point = measured_segment["path_points"][i]
            
            # Calculate 3D Euclidean distance
            dx = d_point["x"] - m_point["x"]
            dy = d_point["y"] - m_point["y"] 
            dz = d_point["z"] - m_point["z"]
            
            deviation = math.sqrt(dx**2 + dy**2 + dz**2)
            total_deviation += deviation
            
        avg_deviation = total_deviation / point_count
        return {"path_deviation": avg_deviation}

