"""
Integration test for DataModule with Agent and PredictionSystem.

Tests the complete workflow:
1. Agent initialization with prediction models
2. DataModule creation and configuration
3. Training with normalization
4. Prediction with validation and denormalization
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass

from lbp_package.core import Dataset, DatasetSchema, DataModule
from lbp_package.core.data_objects import DataReal, DataInt
from lbp_package.core.data_blocks import DataBlock
from lbp_package.interfaces.prediction import IPredictionModel
from lbp_package.orchestration.prediction import PredictionSystem
from lbp_package.utils import LBPLogger


@dataclass
class SimplePredictionModel(IPredictionModel):
    """Simple prediction model for testing."""
    
    
    @property
    def feature_names(self):
        return ['feature_1']
    
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        self.is_trained = False
        self.X_train = None
        self.y_train = None
        self.training_kwargs = None  # Store kwargs for verification
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """Store training data and kwargs (mock training)."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.training_kwargs = kwargs
        self.is_trained = True
    
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return simple predictions."""
        if not self.is_trained or self.y_train is None:
            raise RuntimeError("Model not trained")
        
        # Simple prediction: average of training y
        predictions = pd.DataFrame({
            'feature_1': [self.y_train['feature_1'].mean()] * len(X)
        })
        return predictions


@pytest.fixture
def sample_dataset_with_features():
    """Create dataset with experiments containing features."""
    schema = DatasetSchema()
    schema.parameters.add("param_1", DataReal("param_1"))
    schema.parameters.add("param_2", DataInt("param_2"))
    
    dataset = Dataset(
        name="test",
        schema=schema,
        schema_id="test_001",
        local_data=None,
        external_data=None
    )
    
    # Add experiments with features
    np.random.seed(42)
    for i in range(10):
        exp_code = f"exp_{i:03d}"
        
        exp_data = dataset.add_experiment(
            exp_code=exp_code,
            exp_params={"param_1": float(i) * 0.1, "param_2": i + 10}
        )
        
        # Add features
        exp_data.features = DataBlock()
        exp_data.features.add("feature_1", DataReal("feature_1"))
        exp_data.features.set_value("feature_1", float(i) * 2.0)
    
    return dataset


class TestDataModuleIntegration:
    """Test DataModule integration with PredictionSystem."""
    
    def test_prediction_system_train_with_datamodule(self, sample_dataset_with_features):
        """Test training prediction models with DataModule."""
        dataset = sample_dataset_with_features
        logger = LBPLogger("test", "/tmp")
        
        # Create prediction system
        system = PredictionSystem(dataset, logger)
        
        # Add prediction model
        model = SimplePredictionModel(logger=logger)
        system.add_prediction_model(model)
        
        # Create DataModule with normalization
        datamodule = DataModule(dataset, normalize='standard')
        
        # Train
        system.train(datamodule)
        
        # Verify model was trained
        assert model.is_trained
        assert model.X_train is not None
        assert model.y_train is not None
        
        # Verify normalization was applied (mean~0, std~1)
        assert model.y_train['feature_1'].mean() == pytest.approx(0.0, abs=1e-6)
        assert model.y_train['feature_1'].std() == pytest.approx(1.0, abs=1e-6)
    
    def test_prediction_system_predict_with_validation(self, sample_dataset_with_features):
        """Test prediction with input validation."""
        dataset = sample_dataset_with_features
        logger = LBPLogger("test", "/tmp")
        
        # Create and train system
        system = PredictionSystem(dataset, logger)
        model = SimplePredictionModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(dataset, normalize='standard')
        system.train(datamodule)
        
        # Valid prediction
        X_new = pd.DataFrame({'param_1': [0.5], 'param_2': [15]})
        predictions = system.predict(X_new)
        
        assert 'feature_1' in predictions.columns
        assert len(predictions) == 1
    
    def test_prediction_system_validates_input(self, sample_dataset_with_features):
        """Test that predict() validates input columns."""
        dataset = sample_dataset_with_features
        logger = LBPLogger("test", "/tmp")
        
        system = PredictionSystem(dataset, logger)
        model = SimplePredictionModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(dataset, normalize='none')
        system.train(datamodule)
        
        # Invalid input - unexpected column
        X_bad = pd.DataFrame({'param_1': [0.5], 'param_2': [15], 'invalid_param': [99]})
        
        with pytest.raises(ValueError, match="Unexpected parameter columns"):
            system.predict(X_bad)
    
    def test_prediction_without_training_fails(self, sample_dataset_with_features):
        """Test that predict() before train() raises error."""
        dataset = sample_dataset_with_features
        logger = LBPLogger("test", "/tmp")
        
        system = PredictionSystem(dataset, logger)
        model = SimplePredictionModel(logger=logger)
        system.add_prediction_model(model)
        
        X_new = pd.DataFrame({'param_1': [0.5], 'param_2': [15]})
        
        with pytest.raises(RuntimeError, match="not trained yet"):
            system.predict(X_new)
    
    def test_datamodule_immutability_after_training(self, sample_dataset_with_features):
        """Test that modifying DataModule after training doesn't affect predictions."""
        dataset = sample_dataset_with_features
        logger = LBPLogger("test", "/tmp")
        
        system = PredictionSystem(dataset, logger)
        model = SimplePredictionModel(logger=logger)
        system.add_prediction_model(model)
        
        # Train with standard normalization
        datamodule = DataModule(dataset, normalize='standard')
        system.train(datamodule)
        
        # Modify original datamodule
        datamodule.set_feature_normalize('feature_1', 'minmax')
        
        # System should still use original (standard) normalization
        # This is ensured by storing a copy
        assert system.datamodule is not None
        assert system.datamodule is not datamodule
        assert system.datamodule._feature_overrides != datamodule._feature_overrides
    
    def test_normalization_round_trip(self, sample_dataset_with_features):
        """Test that normalization->denormalization preserves scale."""
        dataset = sample_dataset_with_features
        logger = LBPLogger("test", "/tmp")
        
        system = PredictionSystem(dataset, logger)
        model = SimplePredictionModel(logger=logger)
        system.add_prediction_model(model)
        
        # Disable splits to use all data (test_size=0.0, val_size=0.0)
        datamodule = DataModule(dataset, normalize='standard', test_size=0.0, val_size=0.0)
        system.train(datamodule)
        
        # Get original feature values from all data
        _, y_original = datamodule.extract_all()
        original_mean = y_original['feature_1'].mean()
        
        # Predict (should denormalize to original scale)
        X_new = pd.DataFrame({'param_1': [0.5], 'param_2': [15]})
        predictions = system.predict(X_new)
        
        # Prediction should be in original scale (not normalized)
        # Model predicts mean of training data, which should match original mean
        assert predictions['feature_1'].iloc[0] == pytest.approx(original_mean, abs=1e-6)
