from typing import Dict, Any, List, Type
from numpy import ndarray

from utils import generate_temperature_data
from lbp_package import PredictionModel, FeatureModel

class PredictExample(PredictionModel):
    """
    Example prediction model for demonstration purposes.
    """

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def input(self) -> List[str]:
        """
        Declare the input keys required for this prediction model.
        """
        return ["layerTime", "layerHeight", "temperature"]
        
    @property
    def dataset_type(self) -> PredictionModel.DatasetType:
        """
        Specify the type of dataset this prediction model works with.
        """
        return PredictionModel.DatasetType.AGGR_METRICS

    @property
    def feature_model_types(self) -> Dict[str, Type[FeatureModel]]:
        """
        Declare the feature model types this prediction model uses.
        """
        return {"temperature": TemperatureExtraction}

    def train(self, X: Dict[str, ndarray], y: ndarray) -> None:
        pass

    def predict(self, X: Dict[str, ndarray]) -> Dict[str, ndarray]:
        """
        Perform prediction based on the input features.
        """
        return {}

class TemperatureExtraction(FeatureModel):
    """
    Example feature model that loads and extracts temperature data.
    """

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_data(self, exp_code: str, exp_folder: str, debug_flag: bool) -> Any:
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
        temp_data = data["temperature"]
        if len(temp_data) == 0:
            raise ValueError("Temperature data is empty.")
        # No feature extraction needed, as we use the raw temperature data directly
        return {"temperature": sum(temp_data)/len(temp_data)}

    

