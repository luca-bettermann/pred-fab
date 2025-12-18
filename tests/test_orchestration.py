"""
Tests for orchestration layer (EvaluationSystem, PredictionSystem).
"""
import pytest
import tempfile
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Any, Tuple

from lbp_package.orchestration.evaluation import EvaluationSystem
from lbp_package.orchestration.prediction import PredictionSystem
from lbp_package.interfaces.features import IFeatureModel
from lbp_package.interfaces.evaluation import IEvaluationModel
from lbp_package.interfaces.prediction import IPredictionModel
from lbp_package.core.dataset import Dataset, ExperimentData
from lbp_package.core.schema import DatasetSchema
from lbp_package.core.data_blocks import Parameters, DataBlock, Features
from lbp_package.core.data_objects import Parameter, DataReal, DataArray
from lbp_package.core.datamodule import DataModule
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
    schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
    return Dataset(name="test_dataset", schema=schema, schema_id="test_schema")


# Mock implementations
@dataclass
class MockFeatureModel(IFeatureModel):
    """Mock feature model for testing."""
    
    def _load_data(self, **param_values):
        return param_values
    
    def _compute_feature_logic(self, data):
        return sum(data.values())


@dataclass
class MockEvaluationModel(IEvaluationModel):
    """Mock evaluation model for testing."""
    
    @property
    def feature_model_class(self) -> Type[IFeatureModel]:
        return MockFeatureModel
    
    def _compute_target_value(self, **param_values) -> float:
        return 10.0
    
    def _compute_scaling_factor(self, **param_values) -> Optional[float]:
        return 5.0


@dataclass
class MockPredictionModel(IPredictionModel):
    """Mock prediction model for testing."""
    
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        self.is_trained = False
        self.X_train = None
        self.y_train = None
        self.mean_val = 0.0
    
    @property
    def outputs(self) -> List[str]:
        return ["test_feature"]
    
    def train(self, train_batches: List[Tuple[np.ndarray, np.ndarray]], val_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs):
        # Concatenate batches for storage/verification
        if train_batches:
            self.X_train = np.vstack([b[0] for b in train_batches])
            self.y_train = np.vstack([b[1] for b in train_batches])
            self.mean_val = np.mean(self.y_train)
        else:
            self.X_train = np.array([])
            self.y_train = np.array([])
            self.mean_val = 0.0
            
        self.is_trained = True
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        # Return mean value for all inputs
        return np.full((len(X), 1), self.mean_val)
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {
            "is_trained": self.is_trained,
            "mean_value": self.mean_val
        }
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.is_trained = artifacts.get("is_trained", False)
        self.mean_val = artifacts.get("mean_value", 0.0)


class TestEvaluationSystem:
    """Test EvaluationSystem orchestration."""
    
    def test_evaluation_system_creation(self, dataset, logger):
        """Test creating an evaluation system."""
        system = EvaluationSystem(dataset=dataset, logger=logger)
        
        assert system.dataset is dataset
        assert system.logger is logger
        assert len(system.models) == 0
    
    def test_add_evaluation_model(self, dataset, logger):
        """Test adding an evaluation model."""
        system = EvaluationSystem(dataset=dataset, logger=logger)
        
        eval_model = MockEvaluationModel(logger=logger)
        
        system.add_evaluation_model(
            performance_code="test_perf",
            evaluation_model=eval_model,
            feature_model_class=MockFeatureModel
        )
        
        assert "test_perf" in system.models
        assert system.models["test_perf"] is eval_model
        assert eval_model.feature_model is not None
    
    def test_get_evaluation_models(self, dataset, logger):
        """Test retrieving evaluation models."""
        system = EvaluationSystem(dataset=dataset, logger=logger)
        
        eval_model = MockEvaluationModel(logger=logger)
        system.add_evaluation_model(
            performance_code="test_perf",
            evaluation_model=eval_model,
            feature_model_class=MockFeatureModel
        )
        
        models = system.get_evaluation_model_dict()
        
        assert len(models) == 1
        assert "test_perf" in models


class TestPredictionSystem:
    """Test PredictionSystem orchestration."""
    
    def test_prediction_system_creation(self, dataset, logger):
        """Test creating a prediction system."""
        system = PredictionSystem(dataset=dataset, logger=logger)
        
        assert system.dataset is dataset
        assert system.logger is logger
        assert len(system.models) == 0
        assert system.datamodule is None
    
    def test_add_prediction_model(self, dataset, logger):
        """Test adding a prediction model."""
        system = PredictionSystem(dataset=dataset, logger=logger)
        
        pred_model = MockPredictionModel(logger=logger)
        
        system.add_prediction_model(pred_model)
        
        assert "test_feature" in system.feature_to_model
        assert system.feature_to_model["test_feature"] is pred_model
        assert pred_model in system.models
    
    def test_train_prediction_model(self, dataset, logger):
        """Test training a prediction model (requires DataModule)."""
        # Note: This test would require a proper dataset with features
        # and a DataModule. Skipping actual training test here.
        system = PredictionSystem(dataset=dataset, logger=logger)
        
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        # Just verify model is registered
        assert pred_model in system.models
        assert "test_feature" in system.feature_to_model
    
    def test_predict_with_model(self, dataset, logger):
        """Test making predictions (requires training first)."""
        # Note: This test would require proper training with DataModule
        # Skipping actual prediction test here.
        system = PredictionSystem(dataset=dataset, logger=logger)
        
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        # Just verify model is registered
        assert pred_model in system.models
        assert "test_feature" in system.feature_to_model
    
    def test_get_prediction_models(self, dataset, logger):
        """Test retrieving prediction models."""
        system = PredictionSystem(dataset=dataset, logger=logger)
        
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        # Check prediction_models list and feature_to_model mapping
        assert len(system.models) == 1
        assert pred_model in system.models
        assert "test_feature" in system.feature_to_model
        assert system.feature_to_model["test_feature"] is pred_model


class TestPredictionSystemWithSplits:
    """Test PredictionSystem with train/val/test splits."""
    
    @pytest.fixture
    def dataset_with_features(self):
        """Create dataset with experiments containing features."""
        schema = DatasetSchema()
        schema.parameters.add("param_1", DataReal("param_1"))
        schema.parameters.add("param_2", DataReal("param_2"))
        
        dataset = Dataset(
            name="test",
            schema=schema,
            schema_id="test_001",
            local_data=None,
            external_data=None
        )
        
        # Add 20 experiments with features in metric_arrays
        np.random.seed(42)
        
        for i in range(20):
            exp_code = f"exp_{i:03d}"
            
            # Create experiment
            exp_data = dataset.load_experiment(
                exp_code=exp_code,
                exp_params={
                    "param_1": float(i) * 0.1,
                    "param_2": float(i) * 0.5
                }
            )
            
            # Initialize metric_arrays and add features
            exp_data.features = Features()
            test_feature_arr = DataArray(code="test_feature", shape=())
            exp_data.features.add("test_feature", test_feature_arr)
            feature_val = exp_data.parameters.get_value("param_1") * 2.0 + \
                         exp_data.parameters.get_value("param_2") * 3.0 + \
                         np.random.randn() * 0.1  # Add noise
            exp_data.features.set_value("test_feature", np.array(feature_val))
        
        return dataset
    
    def test_train_with_splits(self, dataset_with_features, logger):
        """Test training uses only training split."""
        system = PredictionSystem(dataset=dataset_with_features, logger=logger)
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        # Create datamodule with splits
        datamodule = DataModule(
            dataset_with_features,
            test_size=0.2,
            val_size=0.1,
            random_seed=42
        )
        
        # Train
        system.train(datamodule)
        
        # Check model was trained on training set only (14 samples)
        assert pred_model.is_trained
        assert len(pred_model.X_train) == 14
    
    def test_validate_on_val_set(self, dataset_with_features, logger):
        """Test validation uses validation split."""
        system = PredictionSystem(dataset=dataset_with_features, logger=logger)
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        datamodule = DataModule(
            dataset_with_features,
            test_size=0.2,
            val_size=0.1,
            random_seed=42
        )
        
        system.train(datamodule)
        
        # Validate on validation set
        results = system.validate(use_test=False)
        
        assert "test_feature" in results
        assert "mae" in results["test_feature"]
        assert "rmse" in results["test_feature"]
        assert "r2" in results["test_feature"]
        assert "n_samples" in results["test_feature"]
        assert results["test_feature"]["n_samples"] == 2  # Val set size
    
    def test_validate_on_test_set(self, dataset_with_features, logger):
        """Test validation can use test split."""
        system = PredictionSystem(dataset=dataset_with_features, logger=logger)
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        datamodule = DataModule(
            dataset_with_features,
            test_size=0.2,
            val_size=0.1,
            random_seed=42
        )
        
        system.train(datamodule)
        
        # Validate on test set
        results = system.validate(use_test=True)
        
        assert "test_feature" in results
        assert results["test_feature"]["n_samples"] == 4  # Test set size
    
    def test_validate_before_train_raises(self, dataset_with_features, logger):
        """Test validation before training raises error."""
        system = PredictionSystem(dataset=dataset_with_features, logger=logger)
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        with pytest.raises(RuntimeError, match="not trained yet"):
            system.validate()
    
    def test_validate_metrics_reasonable(self, dataset_with_features, logger):
        """Test validation metrics are in reasonable ranges."""
        system = PredictionSystem(dataset=dataset_with_features, logger=logger)
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        datamodule = DataModule(
            dataset_with_features,
            test_size=0.2,
            val_size=0.1,
            random_seed=42,
            normalize='standard'
        )
        
        system.train(datamodule)
        results = system.validate(use_test=True)
        
        metrics = results["test_feature"]
        
        # Metrics should be non-negative
        assert metrics["mae"] >= 0.0
        assert metrics["rmse"] >= 0.0
        
        # R² should be between -inf and 1.0 (can be negative for bad models)
        assert metrics["r2"] <= 1.0
        
        # RMSE should be >= MAE (by definition)
        assert metrics["rmse"] >= metrics["mae"]
    
    def test_validate_empty_val_split_raises(self, dataset_with_features, logger):
        """Test validation with val_size=0 raises clear error."""
        system = PredictionSystem(dataset=dataset_with_features, logger=logger)
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        # Create datamodule with no validation split
        datamodule = DataModule(
            dataset_with_features,
            test_size=0.2,
            val_size=0.0,  # No validation split
            random_seed=42
        )
        
        system.train(datamodule)
        
        # Should raise clear error about empty val split
        with pytest.raises(ValueError, match="Cannot validate on val set: split is empty"):
            system.validate(use_test=False)
        
        # But test split should work
        results = system.validate(use_test=True)
        assert "test_feature" in results
    
    def test_validate_empty_test_split_raises(self, dataset_with_features, logger):
        """Test validation with test_size=0 raises clear error."""
        system = PredictionSystem(dataset=dataset_with_features, logger=logger)
        pred_model = MockPredictionModel(logger=logger)
        system.add_prediction_model(pred_model)
        
        # Create datamodule with no test split
        datamodule = DataModule(
            dataset_with_features,
            test_size=0.0,  # No test split
            val_size=0.1,
            random_seed=42
        )
        
        system.train(datamodule)
        
        # Should raise clear error about empty test split
        with pytest.raises(ValueError, match="Cannot validate on test set: split is empty"):
            system.validate(use_test=True)
        
        # But val split should work
        results = system.validate(use_test=False)
        assert "test_feature" in results
