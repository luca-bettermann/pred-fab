from typing import Dict, Any, List, Type
from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from utils import generate_temperature_data
from lbp_package import IPredictionModel, IFeatureModel

class PredictExample(IPredictionModel):
    """
    Example prediction model for demonstration purposes.
    """

    # Passing initialization parameters to the parent class
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.is_trained = False
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )

    @property
    def input(self) -> List[str]:
        """
        Declare the input keys required for this prediction model.
        """
        return ["layerTime", "layerHeight", "temperature"]
        
    @property
    def dataset_type(self) -> IPredictionModel.DatasetType:
        """
        Specify the type of dataset this prediction model works with.
        """
        return IPredictionModel.DatasetType.AGGR_METRICS

    @property
    def feature_model_types(self) -> Dict[str, Type[IFeatureModel]]:
        """
        Declare the feature model types this prediction model uses.
        """
        return {"temperature": TemperatureExtraction}

    def train(self, X: ndarray, y: ndarray) -> Dict[str, Any]:
        """
        Train the Random Forest model and return training metrics.
        
        Args:
            X: Input features with shape (n_samples, n_features)
            y: Target variables with shape (n_samples, n_targets)
            
        Returns:
            Dictionary with required training metrics
        """
        self.logger.info(f"Training Random Forest with X shape: {X.shape}, y shape: {y.shape}")
        
        # Check if we have data to train on
        if X.shape[0] == 0:
            raise ValueError("No training data provided")
        
        # For multi-output regression, Random Forest can handle it directly
        self.model.fit(X, y)
        self.is_trained = True
        
        # Compute training metrics
        training_score = self.model.score(X, y)
        
        # Log additional info (user can log whatever they want)
        feature_importance = self.model.feature_importances_
        feature_importance_dict = {self.input[i]: round(feature_importance[i], 3) for i in range(len(self.input))}
        self.logger.info(f"Model trained successfully. Feature importances: {feature_importance_dict}")
        
        # Return only required metrics
        return {
            "training_score": round(training_score, 4),
            "training_samples": X.shape[0]
        }

    def predict(self, X: ndarray) -> ndarray:
        """
        Perform prediction based on the input features.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples, n_targets)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.logger.debug(f"Making predictions for X shape: {X.shape}")
        predictions = self.model.predict(X)
        
        # Ensure we return the right shape - if single sample, reshape to (1, n_targets)
        if predictions.ndim == 1 and len(self.output) > 1:
            predictions = predictions.reshape(1, -1)
        elif predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            
        self.logger.debug(f"Predictions shape: {predictions.shape}")
        return predictions

class TemperatureExtraction(IFeatureModel):
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

    

