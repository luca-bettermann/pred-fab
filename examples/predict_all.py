from typing import Dict, Any, List, Type, Optional
from numpy import ndarray

from lbp_package import PredictionModel, FeatureModel
from tests.conftest import generate_temperature_data

class PredictExample(PredictionModel):
    """
    Example prediction model for demonstration purposes.
    """

    # Passing initialization parameters to the parent class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _declare_inputs(self) -> List[str]:
        """
        Declare the input keys required for this prediction model.
        """
        return ["layerTime", "layerHeight", "temperature"]
    
    def _declare_feature_model_types(self) -> Dict[str, Type[FeatureModel]]:
        """
        Declare the feature model types this prediction model uses.
        """
        return {}
    
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_data(self, exp_nr: int) -> Any:
        """
        Load the data required for temperature feature extraction.
        """
        # For demonstration, we generate mock temperature data
        return generate_temperature_data()

    def _compute_features(self, data: Dict[str, Any]) -> Dict[str, ndarray]:
        """
        Extract temperature features from the provided data.
        """
        # Example extraction logic
        return {
            "temperature": data.get("temperature", 0.0)
        }
    
    def get_required_inputs(self) -> List[str]:
        return ["temperature"]