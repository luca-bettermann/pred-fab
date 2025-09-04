import math
from typing import Dict, Any, List, Optional, Tuple, Type
from dataclasses import dataclass

from lbp_package import EvaluationModel, FeatureModel
from lbp_package.utils import dim_parameter, study_parameter, exp_parameter
from utils import generate_path_data
from visualize import visualize_geometry

@dataclass
class PathEvaluation(EvaluationModel):
    """Example evaluation model for path deviation assessment."""

    # Model parameters
    target_deviation: Optional[float] = study_parameter(default=0.1)
    max_deviation: Optional[float] = study_parameter()
    
    # Experiment parameters
    n_layers: Optional[int] = exp_parameter()
    n_segments: Optional[int] = exp_parameter()

    # Dimensionality parameters
    layer_id: Optional[int] = dim_parameter()
    segment_id: Optional[int] = dim_parameter()

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
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
    
    # === OPTIONAL METHODS ===

    # We only use this for visualization during the _cleanup step, i.e.
    def _cleanup_step(self, exp_code: str, exp_folder: str, visualize_flag: bool, debug_flag: bool) -> None:
        # skip if we don't visualize
        if not visualize_flag:
            return

        # Action: layer is finished
        assert isinstance(self.n_segments, int), "n_segments must be an integer"
        if self.segment_id == self.n_segments - 1:

            # validate that the feature model has path coordinates
            feature_model = self.feature_model
            if not isinstance(feature_model, PathDeviationFeature):
                raise ValueError("Feature model is missing path coordinates")
            
            # Type cast to access attributes after runtime check
            assert isinstance(self.layer_id, int), "layer_id must be an integer"
            avg_deviation_list = list(feature_model.features['path_deviation'][self.layer_id, :])
            visualize_geometry(
                exp_code,
                self.layer_id,
                feature_model.designed_path_coords, 
                feature_model.measured_path_coords, 
                avg_deviation_list
                )
            
            # reset storage
            feature_model.designed_path_coords = []
            feature_model.measured_path_coords = []


@dataclass
class PathDeviationFeature(FeatureModel):
    """Example feature model for path deviation calculation."""
    
    # Model parameters
    tolerance_xyz: Optional[float] = study_parameter(0.1)

    # Experiment parameters
    n_layers: Optional[int] = exp_parameter()
    n_segments: Optional[int] = exp_parameter()

    # Dimensionality parameters
    layer_id: Optional[int] = dim_parameter()
    segment_id: Optional[int] = dim_parameter()

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Store designed and measured filament coordinates for visualizations
        self.designed_path_coords: List[Dict[str, float]] = []
        self.measured_path_coords: List[Dict[str, float]] = []

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    def _load_data(self, exp_code: str, exp_folder: str, debug_flag: bool) -> Dict[str, Any]:
        """Mock loading of raw data by generating designed and measured filament paths."""

        if self.n_layers is None or self.n_segments is None:
            raise ValueError("Layer and segment information must be provided.")

        # Generate designed paths, without noise
        designed_data = generate_path_data(
            n_layers=self.n_layers,
            n_segments=self.n_segments,
            noise=False
        )

        # Generate measured paths, with noise
        measured_data = generate_path_data(
            n_layers=self.n_layers,
            n_segments=self.n_segments,
            noise=True
        )
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

            # Store coordinates for visualizations
            if visualize_flag:
                self.designed_path_coords.append(d_point)
                self.measured_path_coords.append(m_point)

            # Calculate 3D Euclidean distance
            dx = d_point["x"] - m_point["x"]
            dy = d_point["y"] - m_point["y"] 
            dz = d_point["z"] - m_point["z"]
            
            deviation = math.sqrt(dx**2 + dy**2 + dz**2)
            total_deviation += deviation
            
        avg_deviation = total_deviation / point_count
        return {"path_deviation": avg_deviation}
