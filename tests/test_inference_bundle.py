"""
Tests for InferenceBundle export/import and production inference.

Tests the complete export/import workflow:
1. Train models with PredictionSystem
2. Export to inference bundle
3. Load bundle in new context
4. Predict with validation and denormalization
5. Verify round-trip consistency
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from dataclasses import dataclass
from typing import Dict, Any, List

from lbp_package.core import Dataset, DatasetSchema, DataModule
from lbp_package.core.data_objects import DataReal
from lbp_package.interfaces.prediction import IPredictionModel
from lbp_package.orchestration.prediction import PredictionSystem
from lbp_package.orchestration.inference_bundle import InferenceBundle
from lbp_package.utils.logger import LBPLogger


@dataclass
class SimpleLinearModel(IPredictionModel):
    """Simple linear model for testing: y = 2*x + 1."""
    
    param_x: DataReal = DataReal("param_x", min_val=0.0, max_val=10.0)
    
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        self.is_trained = False
        self.coef = None
        self.intercept = None
    
    @property
    def feature_names(self) -> List[str]:
        return ["output_y"]
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """Fit simple linear model."""
        # Simple linear regression: y = coef * x + intercept
        x_vals = X["param_x"].values
        y_vals = y["output_y"].values
        
        # Least squares solution
        x_mean = np.mean(x_vals)
        y_mean = np.mean(y_vals)
        self.coef = np.sum((x_vals - x_mean) * (y_vals - y_mean)) / np.sum((x_vals - x_mean) ** 2)
        self.intercept = y_mean - self.coef * x_mean
        self.is_trained = True
    
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict using learned coefficients."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        x_vals = X["param_x"].values
        y_pred = self.coef * x_vals + self.intercept
        return pd.DataFrame({"output_y": y_pred})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        """Export coefficients."""
        if not self.is_trained or self.coef is None or self.intercept is None:
            raise RuntimeError("Cannot export untrained model")
        return {
            "coef": float(self.coef),
            "intercept": float(self.intercept),
            "is_trained": self.is_trained
        }
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore coefficients."""
        self.coef = artifacts["coef"]
        self.intercept = artifacts["intercept"]
        self.is_trained = artifacts["is_trained"]


@dataclass
class MultiOutputModel(IPredictionModel):
    """Model that predicts multiple features."""
    
    param_a: DataReal = DataReal("param_a", min_val=0.0, max_val=5.0)
    param_b: DataReal = DataReal("param_b", min_val=0.0, max_val=5.0)
    
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        self.is_trained = False
        self.means = {}
    
    @property
    def feature_names(self) -> List[str]:
        return ["feat_1", "feat_2"]
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """Store mean values."""
        self.means = {
            "feat_1": float(y["feat_1"].mean()),
            "feat_2": float(y["feat_2"].mean())
        }
        self.is_trained = True
    
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return constant predictions."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        n = len(X)
        return pd.DataFrame({
            "feat_1": [self.means["feat_1"]] * n,
            "feat_2": [self.means["feat_2"]] * n
        })
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        """Export mean values."""
        return {"means": self.means, "is_trained": self.is_trained}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore mean values."""
        self.means = artifacts["means"]
        self.is_trained = artifacts["is_trained"]


@pytest.fixture
def logger(tmp_path):
    """Create test logger."""
    return LBPLogger("test", str(tmp_path / "logs"))


@pytest.fixture
def simple_dataset():
    """Create simple dataset with linear relationship y = 2*x + 1."""
    schema = DatasetSchema()
    schema.parameters.add("param_x", DataReal("param_x", min_val=0.0, max_val=10.0))
    
    dataset = Dataset(
        name="linear_test",
        schema_id="linear_001",
        schema=schema
    )
    
    # Add training data
    for i in range(10):
        x = float(i)
        y = 2 * x + 1  # True relationship
        
        exp_data = dataset.add_experiment(
            exp_code=f"exp_{i:03d}",
            exp_params={"param_x": x}
        )
        
        # Add features manually
        from lbp_package.core.data_blocks import DataBlock
        exp_data.features = DataBlock()
        exp_data.features.add("output_y", DataReal("output_y"))
        exp_data.features.set_value("output_y", y)
    
    return dataset


@pytest.fixture
def multi_output_dataset():
    """Create dataset with multiple output features."""
    schema = DatasetSchema()
    schema.parameters.add("param_a", DataReal("param_a", min_val=0.0, max_val=5.0))
    schema.parameters.add("param_b", DataReal("param_b", min_val=0.0, max_val=5.0))
    
    dataset = Dataset(
        name="multi_test",
        schema_id="multi_001",
        schema=schema
    )
    
    # Add training data
    from lbp_package.core.data_blocks import DataBlock
    for i in range(5):
        exp_data = dataset.add_experiment(
            exp_code=f"exp_{i:03d}",
            exp_params={"param_a": float(i), "param_b": float(i) * 0.5}
        )
        
        # Add features
        exp_data.features = DataBlock()
        exp_data.features.add("feat_1", DataReal("feat_1"))
        exp_data.features.add("feat_2", DataReal("feat_2"))
        exp_data.features.set_value("feat_1", 10.0 + i)
        exp_data.features.set_value("feat_2", 20.0 + i * 2)
    
    return dataset


class TestInferenceBundleBasics:
    """Test basic export/import functionality."""
    
    def test_export_and_load(self, simple_dataset, logger, tmp_path):
        """Test round-trip export and load."""
        # Train model
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="none")
        system.train(datamodule)
        
        # Export bundle
        bundle_path = str(tmp_path / "test_bundle.pkl")
        system.export_inference_bundle(bundle_path)
        
        # Verify file exists
        assert os.path.exists(bundle_path)
        
        # Load bundle
        bundle = InferenceBundle.load(bundle_path)
        
        # Verify bundle structure
        assert len(bundle.prediction_models) == 1
        assert bundle.feature_names == ["output_y"]
        assert bundle.normalization_state is not None
        assert bundle.schema is not None
    
    def test_predict_with_bundle(self, simple_dataset, logger, tmp_path):
        """Test predictions from loaded bundle match original model."""
        # Train and export
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="none")
        system.train(datamodule)
        
        bundle_path = str(tmp_path / "test_bundle.pkl")
        system.export_inference_bundle(bundle_path)
        
        # Predict with original system
        X_test = pd.DataFrame({"param_x": [5.0, 7.5, 10.0]})
        y_original = system.predict(X_test)
        
        # Load bundle and predict
        bundle = InferenceBundle.load(bundle_path)
        y_bundle = bundle.predict(X_test)
        
        # Predictions should match
        pd.testing.assert_frame_equal(y_original, y_bundle)
    
    def test_bundle_validates_inputs(self, simple_dataset, logger, tmp_path):
        """Test that bundle validates unknown parameters."""
        # Train and export
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="none")
        system.train(datamodule)
        
        bundle_path = str(tmp_path / "test_bundle.pkl")
        system.export_inference_bundle(bundle_path)
        
        # Load and try invalid input
        bundle = InferenceBundle.load(bundle_path)
        
        with pytest.raises(ValueError, match="Unknown parameter"):
            bundle.predict(pd.DataFrame({"invalid_param": [1.0, 2.0]}))


class TestInferenceBundleNormalization:
    """Test normalization/denormalization in bundle."""
    
    def test_denormalization_standard(self, simple_dataset, logger, tmp_path):
        """Test standard normalization is correctly reversed."""
        # Train with normalization
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="standard")
        system.train(datamodule)
        
        # Export and load
        bundle_path = str(tmp_path / "norm_bundle.pkl")
        system.export_inference_bundle(bundle_path)
        bundle = InferenceBundle.load(bundle_path)
        
        # Predict
        X_test = pd.DataFrame({"param_x": [5.0]})
        y_pred = bundle.predict(X_test)
        
        # Should be denormalized (not in [-1, 1] range)
        assert y_pred["output_y"].iloc[0] > 5.0  # True value ~11.0
    
    def test_denormalization_minmax(self, simple_dataset, logger, tmp_path):
        """Test minmax normalization is correctly reversed."""
        # Train with minmax
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="minmax")
        system.train(datamodule)
        
        # Export and load
        bundle_path = str(tmp_path / "minmax_bundle.pkl")
        system.export_inference_bundle(bundle_path)
        bundle = InferenceBundle.load(bundle_path)
        
        # Predict
        X_test = pd.DataFrame({"param_x": [0.0, 9.0]})
        y_pred = bundle.predict(X_test)
        
        # Check values are denormalized
        assert y_pred["output_y"].iloc[0] < 5.0  # Should be ~1.0
        assert y_pred["output_y"].iloc[1] > 15.0  # Should be ~19.0


class TestMultipleModels:
    """Test bundles with multiple prediction models."""
    
    def test_multiple_models_export(self, multi_output_dataset, logger, tmp_path):
        """Test exporting multiple models in one bundle."""
        system = PredictionSystem(dataset=multi_output_dataset, logger=logger)
        
        model = MultiOutputModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(multi_output_dataset, normalize="none")
        system.train(datamodule)
        
        # Export
        bundle_path = str(tmp_path / "multi_bundle.pkl")
        system.export_inference_bundle(bundle_path)
        
        # Load
        bundle = InferenceBundle.load(bundle_path)
        
        # Verify all features available
        assert set(bundle.feature_names) == {"feat_1", "feat_2"}
    
    def test_multiple_models_predict(self, multi_output_dataset, logger, tmp_path):
        """Test predictions from multiple models."""
        system = PredictionSystem(dataset=multi_output_dataset, logger=logger)
        
        model = MultiOutputModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(multi_output_dataset, normalize="none")
        system.train(datamodule)
        
        # Export and load
        bundle_path = str(tmp_path / "multi_bundle.pkl")
        system.export_inference_bundle(bundle_path)
        bundle = InferenceBundle.load(bundle_path)
        
        # Predict
        X_test = pd.DataFrame({
            "param_a": [1.0, 2.0],
            "param_b": [0.5, 1.0]
        })
        y_pred = bundle.predict(X_test)
        
        # Check both features present
        assert "feat_1" in y_pred.columns
        assert "feat_2" in y_pred.columns
        assert len(y_pred) == 2


class TestExportValidation:
    """Test export validation and error handling."""
    
    def test_export_before_training_fails(self, simple_dataset, logger, tmp_path):
        """Test that export fails if models not trained."""
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        # Try to export without training
        with pytest.raises(RuntimeError, match="Cannot export before training"):
            system.export_inference_bundle(str(tmp_path / "fail.pkl"))
    
    def test_round_trip_validation(self, simple_dataset, logger, tmp_path):
        """Test that export validates artifacts via round-trip."""
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="none")
        system.train(datamodule)
        
        # Export should succeed (includes round-trip validation)
        bundle_path = str(tmp_path / "validated.pkl")
        result_path = system.export_inference_bundle(bundle_path)
        assert result_path == bundle_path


class TestBundleRepresentation:
    """Test bundle string representation."""
    
    def test_repr(self, simple_dataset, logger, tmp_path):
        """Test bundle __repr__ shows useful info."""
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="none")
        system.train(datamodule)
        
        bundle_path = str(tmp_path / "repr_bundle.pkl")
        system.export_inference_bundle(bundle_path)
        
        bundle = InferenceBundle.load(bundle_path)
        repr_str = repr(bundle)
        
        assert "InferenceBundle" in repr_str
        assert "models=1" in repr_str
        assert "features=1" in repr_str


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe_prediction(self, simple_dataset, logger, tmp_path):
        """Test prediction with empty DataFrame."""
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="none")
        system.train(datamodule)
        
        bundle_path = str(tmp_path / "empty_test.pkl")
        system.export_inference_bundle(bundle_path)
        
        bundle = InferenceBundle.load(bundle_path)
        
        # Predict with empty DataFrame
        X_empty = pd.DataFrame({"param_x": []})
        y_pred = bundle.predict(X_empty)
        
        assert len(y_pred) == 0
        assert "output_y" in y_pred.columns
    
    def test_feature_names_property(self, simple_dataset, logger, tmp_path):
        """Test feature_names property."""
        system = PredictionSystem(dataset=simple_dataset, logger=logger)
        model = SimpleLinearModel(logger=logger)
        system.add_prediction_model(model)
        
        datamodule = DataModule(simple_dataset, normalize="none")
        system.train(datamodule)
        
        bundle_path = str(tmp_path / "feature_test.pkl")
        system.export_inference_bundle(bundle_path)
        
        bundle = InferenceBundle.load(bundle_path)
        
        assert isinstance(bundle.feature_names, list)
        assert bundle.feature_names == ["output_y"]
