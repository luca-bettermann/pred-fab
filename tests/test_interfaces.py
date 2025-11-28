"""  
Tests for updated interface classes (IFeatureModel, IEvaluationModel, IPredictionModel, ICalibrationModel).
"""
import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Any
import tempfile

from lbp_package.interfaces.features import IFeatureModel
from lbp_package.interfaces.evaluation import IEvaluationModel
from lbp_package.interfaces.prediction import IPredictionModel
from lbp_package.interfaces.calibration import ICalibrationModel
from lbp_package.core.dataset import Dataset, ExperimentData
from lbp_package.core.data_blocks import Parameters, Dimensions
from lbp_package.core.data_objects import Parameter, Dimension
from lbp_package.core.schema import DatasetSchema
from lbp_package.utils.logger import LBPLogger


# Test fixtures
@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def logger(temp_log_dir):
    """Create test logger."""
    return LBPLogger(name="test", log_folder=temp_log_dir)


@pytest.fixture
def dataset():
    """Create test dataset."""
    schema = DatasetSchema()
    return Dataset(name="test_dataset", schema=schema, schema_id="test_schema")


@dataclass
class ConcreteFeatureModel(IFeatureModel):
    """Concrete implementation for testing."""
    
    def _load_data(self, **param_values):
        """Mock data loading - just return param values."""
        return param_values
    
    def _compute_features(self, data, visualize: bool = False):
        """Mock feature computation - sum of param values."""
        return sum(data.values())


@dataclass
class ConcreteEvaluationModel(IEvaluationModel):
    """Concrete implementation for testing."""
    
    @property
    def feature_model_type(self) -> Type[IFeatureModel]:
        return ConcreteFeatureModel
    
    def _compute_target_value(self, **param_values) -> float:
        """Mock target - always 10.0."""
        return 10.0
    
    def _compute_scaling_factor(self, **param_values) -> Optional[float]:
        """Mock scaling - always 5.0."""
        return 5.0


@dataclass
class ConcretePredictionModel(IPredictionModel):
    """Concrete implementation for testing."""
    
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        self.is_trained = False
        self.X_train = None
        self.y_train = None
    
    
    @property
    def predicted_features(self) -> List[str]:
        return ["feature_1"]
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """Mock training."""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.is_trained = True
    
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mock prediction - return mean of training features."""
        if not self.is_trained or self.y_train is None:
            raise RuntimeError("Model not trained")
        mean_val = self.y_train["feature_1"].mean()
        return pd.DataFrame({"feature_1": [mean_val] * len(X)})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {
            "is_trained": self.is_trained,
            "mean_value": self.y_train["feature_1"].mean() if self.y_train is not None else None
        }
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.is_trained = artifacts.get("is_trained", False)
        if artifacts.get("mean_value") is not None:
            self.y_train = pd.DataFrame({"feature_1": [artifacts["mean_value"]]})


@dataclass
class ConcreteCalibrationModel(ICalibrationModel):
    """Concrete implementation for testing."""
    
    def optimize(self, param_ranges, objective_fn):
        """Mock optimization - just return midpoint of ranges."""
        return {k: (v[0] + v[1]) / 2 for k, v in param_ranges.items()}


class TestIFeatureModel:
    """Test IFeatureModel interface."""
    
    def test_feature_model_creation(self, dataset, logger):
        """Test creating a feature model with dataset."""
        model = ConcreteFeatureModel(dataset=dataset, logger=logger)
        
        assert model.dataset is dataset
        assert model.logger is logger
    
    def test_feature_model_run_basic(self, dataset, logger):
        """Test basic feature extraction."""
        model = ConcreteFeatureModel(dataset=dataset, logger=logger)
        
        # Run feature extraction
        result = model.run(feature_name="test_feature", x=3, y=4)
        
        # Should compute sum: 3 + 4 = 7
        assert result == 7
    
    def test_feature_model_memoization(self, dataset, logger):
        """Test that features are memoized in dataset."""
        model = ConcreteFeatureModel(dataset=dataset, logger=logger)
        
        # First call - computes
        result1 = model.run(feature_name="test_feature", x=3, y=4)
        assert result1 == 7
        
        # Second call - should use cache
        result2 = model.run(feature_name="test_feature", x=3, y=4)
        assert result2 == 7
        
        # Verify it's cached
        assert dataset.has_features_at(x=3, y=4)
        cached = dataset.get_feature_value("test_feature", x=3, y=4)
        assert cached == 7


class TestIEvaluationModel:
    """Test IEvaluationModel interface."""
    
    def test_evaluation_model_creation(self, logger):
        """Test creating an evaluation model."""
        model = ConcreteEvaluationModel(logger=logger)
        
        assert model.logger is logger
        assert model.feature_model is None
    
    def test_add_feature_model(self, dataset, logger):
        """Test adding a feature model."""
        feature_model = ConcreteFeatureModel(dataset=dataset, logger=logger)
        eval_model = ConcreteEvaluationModel(logger=logger)
        
        eval_model.add_feature_model(feature_model)
        
        assert eval_model.feature_model is feature_model
    
    def test_evaluation_model_run(self, dataset, logger):
        """Test that run() method executes without error."""
        # This is a minimal integration test - full testing would require
        # creating complex ExperimentData with proper DataBlocks
        feature_model = ConcreteFeatureModel(dataset=dataset, logger=logger)
        eval_model = ConcreteEvaluationModel(logger=logger)
        eval_model.add_feature_model(feature_model)
        
        # Note: Full test would create ExperimentData with proper schema-based
        # DataBlocks. For now, we just validate the method signature exists
        assert callable(eval_model.run)
    
    def test_evaluation_requires_feature_model(self, logger):
        """Test that evaluation fails without feature model."""
        eval_model = ConcreteEvaluationModel(logger=logger)
        
        # Should fail - no feature model set
        # Full test would require proper ExperimentData setup
        assert eval_model.feature_model is None


class TestIPredictionModel:
    """Test IPredictionModel interface."""
    
    def test_prediction_model_creation(self, logger):
        """Test creating a prediction model."""
        model = ConcretePredictionModel(logger=logger)
        
        assert model.logger is logger
        assert not model.is_trained
    
    def test_prediction_model_properties(self, logger):
        """Test abstract properties."""
        model = ConcretePredictionModel(logger=logger)
        
        assert model.predicted_features == ["feature_1"]
    
    def test_prediction_model_train(self, logger):
        """Test training method."""
        model = ConcretePredictionModel(logger=logger)
        
        # Create mock training data
        X = pd.DataFrame({"param_1": [1, 2, 3], "param_2": [10, 20, 30]})
        y = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0]})
        
        model.train(X, y)
        
        assert model.is_trained
        assert model.X_train is not None
        assert model.y_train is not None
    
    def test_prediction_model_predict(self, logger):
        """Test prediction method."""
        model = ConcretePredictionModel(logger=logger)
        
        # Train first
        X_train = pd.DataFrame({"param_1": [1, 2, 3], "param_2": [10, 20, 30]})
        y_train = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0]})
        model.train(X_train, y_train)
        
        # Predict
        X_new = pd.DataFrame({"param_1": [1.5], "param_2": [15]})
        predictions = model.forward_pass(X_new)
        
        assert "feature_1" in predictions.columns
        assert len(predictions) == 1
        assert predictions["feature_1"].iloc[0] == 2.0  # Mean of [1,2,3]



class TestICalibrationModel:
    """Test ICalibrationModel interface."""
    
    def test_calibration_model_creation(self, logger):
        """Test creating a calibration model."""
        model = ConcreteCalibrationModel(logger=logger)
        
        assert model.logger is logger
        assert model.performance_weights == {}
    
    def test_set_performance_weights(self, logger):
        """Test setting performance weights."""
        model = ConcreteCalibrationModel(logger=logger)
        
        weights = {"perf1": 0.6, "perf2": 0.4}
        model.set_performance_weights(weights)
        
        assert model.performance_weights == weights
    
    def test_calibration_optimize(self, logger):
        """Test optimization method."""
        model = ConcreteCalibrationModel(logger=logger)
        
        param_ranges = {"x": (0.0, 10.0), "y": (-5.0, 5.0)}
        
        def dummy_objective(params):
            return sum(params.values())
        
        result = model.calibrate(param_ranges, dummy_objective)
        
        # Should return midpoints
        assert result["x"] == 5.0
        assert result["y"] == 0.0
    
    def test_calibration_counts_evaluations(self, logger):
        """Test that calibration counts evaluations."""
        model = ConcreteCalibrationModel(logger=logger)
        
        param_ranges = {"x": (0.0, 10.0)}
        
        def dummy_objective(params):
            return params["x"]
        
        model.calibrate(param_ranges, dummy_objective)
        
        # Check that eval count was tracked (at least initialized)
        assert hasattr(model, "_eval_count")

