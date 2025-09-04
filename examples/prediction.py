from typing import Dict, Any, List, Type
from numpy import ndarray

from utils import generate_temperature_data
from visualize import visualize_temperature
from lbp_package import PredictionModel, FeatureModel

class PredictExample(PredictionModel):
    """
    Example prediction model for demonstration purposes.
    """

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _declare_inputs(self) -> List[str]:
        """
        Declare the input keys required for this prediction model.
        """
        return ["layerTime", "layerHeight", "temperature"]
    
    def _declare_outputs(self) -> List[str]:
        """
        Declare the output keys produced by this prediction model.
        """
        return ["path_deviation", "energy_consumption"]

    def _declare_feature_model_types(self) -> Dict[str, Type[FeatureModel]]:
        """
        Declare the feature model types this prediction model uses.
        """
        return {"temperature": TemperatureExtraction}

    def train(self, X: Dict[str, ndarray], y: ndarray) -> None:
        ...

    def predict(self, X: Dict[str, ndarray]) -> Dict[str, ndarray]:
        """
        Perform prediction based on the input features.
        """
        ...

class TemperatureExtraction(FeatureModel):
    """
    Example feature model that loads and extracts temperature data.
    """

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_data(self, exp_code: str, exp_folder: str) -> Any:
        """
        Mock loading of raw temperature data by generating it.
        """

        # Generate mock temperature data
        temperature_time_series = generate_temperature_data(
            base_temp=20,
            fluctuation=3
        )
        return {"temperature": temperature_time_series}

    def _compute_features(self, data: Dict[str, Any], visualize_flag: bool) -> Dict[str, float]:
        """
        Extract temperature features from the provided data.
        """

        if visualize_flag:
            visualize_temperature(data["temperature"])

        # No feature extraction needed, as we use the raw temperature data directly
        return data

    

