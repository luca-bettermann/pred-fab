import json
import os
import math
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from src.lbp_package.evaluation import EvaluationModel, FeatureModel
from src.lbp_package.utils.parameter_handler import runtime_parameter, model_parameter, exp_parameter

@dataclass
class EnergyConsumption(EvaluationModel):
    """Example energy consumption evaluation model."""

    # Model parameters
    target_energy: float = model_parameter(0.0) # type: ignore

    # Experiment parameters
    max_energy: float = exp_parameter() # type: ignore

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
