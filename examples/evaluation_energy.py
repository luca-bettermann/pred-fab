from typing import Dict, Any, List, Tuple, Type, Optional
from dataclasses import dataclass

from lbp_package import IEvaluationModel, IFeatureModel
from lbp_package.utils import dim_parameter, study_parameter, exp_parameter

@dataclass
class EnergyConsumption(IEvaluationModel):
    """Example energy consumption evaluation model."""

    # Model parameters
    target_energy: Optional[float] = study_parameter(0.0)
    max_energy: Optional[float] = study_parameter()

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def feature_model_type(self) -> Type[IFeatureModel]:
        """Declare the feature model type to use for feature extraction."""
        return EnergyFeature
        
    @property
    def dim_names(self) -> List[str]:
        """Return the dimension names."""
        return []

    @property
    def dim_param_names(self) -> List[str]:
        """Return the dimension parameter names."""
        return []

    @property
    def dim_iterator_names(self) -> List[str]:
        """Return the dimension iterator names."""
        return []
    
    @property
    def target_value(self) -> Optional[float]:
        """Return target energy value."""
        return self.target_energy

    @property
    def scaling_factor(self) -> Optional[float]:
        """Return maximum energy for normalization."""
        return self.max_energy


@dataclass
class EnergyFeature(IFeatureModel):
    """Example feature model for energy consumption extraction."""

    # Model parameters
    power_rating: Optional[float] = study_parameter(50.0)

    # Experiment parameters
    layerTime: Optional[float] = exp_parameter()

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_data(self, exp_code: str, exp_folder: str, debug_flag: bool) -> Any:
        """No data loading required for energy calculation."""
        return None

    def _compute_features(self, data: Any, visualize_flag: bool) -> Dict[str, float]:
        """Compute energy feature from power rating and layer time."""
        if self.power_rating is None or self.layerTime is None:
            raise ValueError("Power rating and layer time must be set to compute energy consumption.")
        energy_consumption = self.power_rating * self.layerTime  # Watts * seconds
        return {"energy_consumption": energy_consumption}
