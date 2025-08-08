import numpy as np
from typing import Dict, Any, List, Tuple, Type, Optional
from dataclasses import dataclass

from lbp_package import EvaluationModel, FeatureModel
from lbp_package.utils import runtime_parameter, model_parameter, exp_parameter

@dataclass
class EnergyConsumption(EvaluationModel):
    """Example energy consumption evaluation model."""

    # Model parameters
    target_energy: Optional[float] = model_parameter(0.0)

    # Experiment parameters
    max_energy: Optional[float] = exp_parameter()

    # Passing initialization parameters to the parent class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _declare_dimensions(self) -> List[Tuple[str, str, str]]:
        """Declare dimensions for energy evaluation (no dimensions)."""
        return []  # No dimensions for energy consumption
    
    def _declare_feature_model_type(self) -> Type[FeatureModel]:
        """Declare the feature model type to use for feature extraction."""
        return EnergyFeature
    
    def _compute_target_value(self) -> Optional[float]:
        """Return target energy value."""
        return self.target_energy
    
    def _declare_scaling_factor(self) -> Optional[float]:
        """Return maximum energy for normalization."""
        return self.max_energy


@dataclass
class EnergyFeature(FeatureModel):
    """Example feature model for energy consumption extraction."""

    # Model parameters
    power_rating: Optional[float] = model_parameter(50.0)

    # Experiment parameters
    layerTime: Optional[float] = exp_parameter()

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
            associated_code=performance_code,
            folder_navigator=folder_navigator,
            logger=logger,
            round_digits=round_digits,
            **study_params
        )
        
        # Initialize feature storage for energy consumption
        self.features["energy_consumption"] = np.empty([])

    def _load_data(self, exp_nr: int, debug_flag: bool = False) -> Any:
        """No data loading required for energy calculation."""
        return None

    def _compute_features(self, data: Any, visualize_flag: bool) -> Dict[str, float]:
        """Compute energy feature from power rating and layer time."""
        if self.power_rating is None or self.layerTime is None:
            raise ValueError("Power rating and layer time must be set to compute energy consumption.")
        energy_consumption = self.power_rating * self.layerTime  # Watts * seconds
        return {"energy_consumption": energy_consumption}
