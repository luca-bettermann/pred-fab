import json
import os
import math
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from src.lbp_package.evaluation import EvaluationModel, FeatureModel
from src.lbp_package.utils.parameter_handler import runtime_parameter, model_parameter, exp_parameter

@dataclass
class PathDeviationEvaluation(EvaluationModel):
    """Example evaluation model for path deviation assessment."""
    
    # Model parameters
    target_deviation: float = model_parameter(0.0) # type: ignore
    max_deviation: float = model_parameter(0.5) # type: ignore
    n_layers: int = model_parameter(2) # type: ignore
    n_segments: int = model_parameter(2) # type: ignore

    def __init__(
            self, 
            performance_code: str, 
            folder_navigator, 
            logger, 
            **study_params
            ):
        """Initialize path deviation evaluation."""
        dimension_names = [
            ('layers', 'layer_id', 'n_layers'),
            ('segments', 'segment_id', 'n_segments')
        ]
        
        super().__init__(
            performance_code=performance_code,
            folder_navigator=folder_navigator,
            dimension_names=dimension_names,
            feature_model_type=PathDeviationFeature,
            logger=logger,
            **study_params
        )
    
    def _compute_target_value(self) -> float:
        """Return target deviation (ideally 0)."""
        return self.target_deviation
        
    def _compute_scaling_factor(self) -> float:
        """Return maximum acceptable deviation."""
        return self.max_deviation


@dataclass
class PathDeviationFeature(FeatureModel):
    """Example feature model for path deviation calculation."""
    
    # Model parameters
    tolerance_xyz: float = model_parameter(0.1) # type: ignore

    # Experiment parameters
    n_layers: int = exp_parameter() # type: ignore
    n_segments: int = exp_parameter() # type: ignore

    # Runtime parameters
    layer_id: int = runtime_parameter() # type: ignore
    segment_id: int = runtime_parameter() # type: ignore


    def __init__(
            self, 
            performance_code: str, 
            folder_navigator, 
            logger, 
            round_digits: int,
            **study_params
            ):
        
        """Initialize path deviation feature model."""
        super().__init__(
            performance_code=performance_code,
            folder_navigator=folder_navigator,
            logger=logger,
            round_digits=round_digits,
            **study_params
        )
        
        # Initialize feature storage for path deviation
        self.features["path_deviation"] = np.empty([])
    
    def _load_data(self, exp_nr: int) -> Dict[str, Any]:
        """Load designed and measured path data."""
        exp_code = self.nav.get_experiment_code(exp_nr)
        
        # Load designed paths
        designed_path = os.path.join(self.nav.get_experiment_folder(exp_nr), 
                                   f"{exp_code}_designed_paths.json")
        with open(designed_path, 'r') as f:
            designed_data = json.load(f)
            
        # Load measured paths  
        measured_path = os.path.join(self.nav.get_experiment_folder(exp_nr),
                                   f"{exp_code}_measured_paths.json")
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


@dataclass
class EnergyConsumption(EvaluationModel):
    """Example energy consumption evaluation model."""

    # Model parameters
    target_energy: float = model_parameter(100.0) # type: ignore
    max_energy: float = model_parameter(1000.0) # type: ignore

    def __init__(
            self, 
            performance_code: str, 
            folder_navigator,
            logger, 
            **study_params
            ):
        
        """Initialize energy consumption evaluation model."""
        dimension_names = []  # No dimensions for energy consumption

        super().__init__(
            performance_code=performance_code,
            folder_navigator=folder_navigator,
            dimension_names=dimension_names,
            feature_model_type=EnergyFeature,
            logger=logger,
            **study_params
        )
    
    def _compute_target_value(self) -> float:
        """Return target energy value."""
        return self.target_energy
    
    def _compute_scaling_factor(self) -> float:
        """Return maximum energy for normalization."""
        return self.max_energy


@dataclass
class EnergyFeature(FeatureModel):
    """Example feature model for energy consumption extraction."""

    # Model parameters
    power_rating: float = model_parameter(50.0)  # type: ignore

    # Experiment parameters
    layerTime: float = exp_parameter() # type: ignore

    def __init__(
            self, 
            performance_code: str, 
            folder_navigator, 
            logger, 
            round_digits: int,
            **study_params
            ):
        
        """Initialize energy feature model."""
        super().__init__(
            performance_code=performance_code,
            folder_navigator=folder_navigator,
            logger=logger,
            round_digits=round_digits,
            **study_params
        )
        
        # Initialize feature storage for energy consumption
        self.features["energy_consumption"] = np.empty([])

    def _load_data(self, exp_nr: int) -> Any:
        """No data loading required for energy calculation."""
        return None

    def _compute_features(self, data: Any, visualize_flag: bool) -> Dict[str, float]:
        """Compute energy feature from power rating and layer time."""
        energy_consumption = self.power_rating * self.layerTime  # Watts * seconds
        return {"energy_consumption": energy_consumption}
