"""
Tests for **kwargs pass-through in training pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from lbp_package.core import Dataset, DatasetSchema, DataModule
from lbp_package.core.data_objects import DataReal, DataInt, DataArray
from lbp_package.core.data_blocks import DataBlock, MetricArrays
from lbp_package.interfaces.prediction import IPredictionModel
from lbp_package.orchestration.prediction import PredictionSystem
from lbp_package.utils.logger import LBPLogger


class KwargsCapturingModel(IPredictionModel):
    """Model that captures kwargs passed during training."""
    
    
    @property
    def feature_output_codes(self) -> List[str]:
        return ['test_feature']
    
    def __init__(self):
        self.training_kwargs = None
        self.X_train = None
        self.y_train = None
        self.is_trained = False
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """Capture training data and kwargs."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.training_kwargs = kwargs
        self.is_trained = True
    
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """Simple mean prediction."""
        if not self.is_trained or self.y_train is None:
            raise RuntimeError("Model not trained")
        mean_val = self.y_train['test_feature'].mean()
        return pd.DataFrame({'test_feature': [mean_val] * len(X)})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {
            "is_trained": self.is_trained,
            "mean_value": self.y_train['test_feature'].mean() if self.y_train is not None else None
        }
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.is_trained = artifacts.get("is_trained", False)
        if artifacts.get("mean_value") is not None:
            self.y_train = pd.DataFrame({'test_feature': [artifacts["mean_value"]]})


@pytest.fixture
def simple_dataset():
    """Create minimal dataset for testing."""
    schema = DatasetSchema()
    schema.parameters.add("x", DataReal("x"))
    schema.parameters.add("y", DataReal("y"))
    
    dataset = Dataset(
        name="test",
        schema=schema,
        schema_id="test_001",
        local_data=None,
        external_data=None
    )
    
    # Add experiments with features in metric_arrays
    np.random.seed(42)
    
    for i in range(3):
        exp_code = f"exp_{i}"
        exp_data = dataset.load_experiment(
            exp_code=exp_code,
            exp_params={"x": float(i), "y": float(i * 2)}
        )
        
        # Initialize metric_arrays and add features
        exp_data.features = MetricArrays()
        test_feature_arr = DataArray(code="test_feature", shape=())
        exp_data.features.add("test_feature", test_feature_arr)
        exp_data.features.set_value("test_feature", np.array(float(i + 1)))
    
    return dataset


def test_kwargs_passed_to_model_train(simple_dataset):
    """Test that **kwargs are passed from PredictionSystem.train to model.train."""
    # Setup
    logger = LBPLogger("test", "/tmp")
    model = KwargsCapturingModel()
    pred_system = PredictionSystem(simple_dataset, logger)
    pred_system.add_prediction_model(model)
    
    datamodule = DataModule(simple_dataset, normalize='none')
    
    # Train with custom kwargs
    custom_kwargs = {
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'verbose': True,
        'early_stopping': False
    }
    
    pred_system.train(datamodule, **custom_kwargs)
    
    # Verify all kwargs were passed through
    assert model.training_kwargs is not None
    assert model.training_kwargs == custom_kwargs
    assert model.training_kwargs['learning_rate'] == 0.001
    assert model.training_kwargs['epochs'] == 100
    assert model.training_kwargs['batch_size'] == 32
    assert model.training_kwargs['verbose'] is True
    assert model.training_kwargs['early_stopping'] is False


def test_empty_kwargs(simple_dataset):
    """Test that training works without any additional kwargs."""
    # Setup
    logger = LBPLogger("test", "/tmp")
    model = KwargsCapturingModel()
    pred_system = PredictionSystem(simple_dataset, logger)
    pred_system.add_prediction_model(model)
    
    datamodule = DataModule(simple_dataset, normalize='none')
    
    # Train without kwargs
    pred_system.train(datamodule)
    
    # Verify empty kwargs dict
    assert model.training_kwargs is not None
    assert model.training_kwargs == {}


def test_kwargs_with_normalization(simple_dataset):
    """Test that kwargs work correctly with normalization enabled."""
    # Setup
    logger = LBPLogger("test", "/tmp")
    model = KwargsCapturingModel()
    pred_system = PredictionSystem(simple_dataset, logger)
    pred_system.add_prediction_model(model)
    
    datamodule = DataModule(simple_dataset, normalize='standard')
    
    # Train with kwargs
    pred_system.train(datamodule, learning_rate=0.01, regularization=0.001)
    
    # Verify kwargs and that training still worked
    assert model.training_kwargs is not None
    assert model.training_kwargs['learning_rate'] == 0.01
    assert model.training_kwargs['regularization'] == 0.001
    assert model.X_train is not None
    assert model.y_train is not None


def test_various_kwarg_types(simple_dataset):
    """Test that different types of kwargs are handled correctly."""
    # Setup
    logger = LBPLogger("test", "/tmp")
    model = KwargsCapturingModel()
    pred_system = PredictionSystem(simple_dataset, logger)
    pred_system.add_prediction_model(model)
    
    datamodule = DataModule(simple_dataset, normalize='none')
    
    # Train with various types
    pred_system.train(
        datamodule,
        int_param=42,
        float_param=3.14,
        str_param="test",
        bool_param=True,
        list_param=[1, 2, 3],
        dict_param={'key': 'value'}
    )
    
    # Verify all types preserved
    assert model.training_kwargs is not None
    assert model.training_kwargs['int_param'] == 42
    assert model.training_kwargs['float_param'] == 3.14
    assert model.training_kwargs['str_param'] == "test"
    assert model.training_kwargs['bool_param'] is True
    assert model.training_kwargs['list_param'] == [1, 2, 3]
    assert model.training_kwargs['dict_param'] == {'key': 'value'}
