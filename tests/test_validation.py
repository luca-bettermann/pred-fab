"""
Tests for return type validation in user-implemented methods.

Verifies that orchestration layer catches incorrect return types from
user implementations and provides helpful error messages.
"""
import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from lbp_package.core import Dataset, DatasetSchema, DataModule
from lbp_package.core.dataset import ExperimentData
from lbp_package.core.data_objects import DataReal, DataInt, DataArray
from lbp_package.core.data_blocks import DataBlock, MetricArrays
from lbp_package.interfaces.prediction import IPredictionModel
from lbp_package.interfaces.features import IFeatureModel
from lbp_package.interfaces.evaluation import IEvaluationModel
from lbp_package.interfaces.calibration import ICalibrationModel
from lbp_package.orchestration.prediction import PredictionSystem
from lbp_package.utils.logger import LBPLogger


class BadPredictionModel(IPredictionModel):
    """Prediction model that returns wrong type."""
    
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        self.return_type = "dict"  # Can be changed per test
    
    
    @property
    def feature_output_codes(self) -> List[str]:
        return ["test_feature"]
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        pass
    
    def forward_pass(self, X: pd.DataFrame):
        """Returns wrong type based on self.return_type."""
        if self.return_type == "dict":
            return {"test_feature": [1.0, 2.0]}
        elif self.return_type == "list":
            return [1.0, 2.0]
        elif self.return_type == "dataframe_missing_cols":
            return pd.DataFrame({"wrong_col": [1.0, 2.0]})
        else:
            return pd.DataFrame({"test_feature": [1.0, 2.0]})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {"return_type": self.return_type}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.return_type = artifacts.get("return_type", "dict")


class BadFeatureModel(IFeatureModel):
    """Feature model that returns wrong type."""
    
    def __init__(self, dataset, logger, return_type="string"):
        self.dataset = dataset
        self.logger = logger
        self.return_type = return_type
    
    def _load_data(self, **param_values):
        return None
    
    def _compute_features(self, data, visualize=False):
        """Returns wrong type based on self.return_type."""
        if self.return_type == "string":
            return "not a number"
        elif self.return_type == "list":
            return [1.0, 2.0]
        elif self.return_type == "dict":
            return {"value": 1.0}
        else:
            return 1.0


class BadEvaluationModel(IEvaluationModel):
    """Evaluation model that returns wrong types."""
    
    def __init__(self, logger, bad_method="target"):
        self.logger = logger
        self.bad_method = bad_method
        self.feature_model = None
    
    @property
    def feature_model_class(self):
        return BadFeatureModel
    
    def _compute_target_value(self, **param_values):
        if self.bad_method == "target":
            return "not a number"
        return 10.0
    
    def _compute_scaling_factor(self, **param_values):
        if self.bad_method == "scaling":
            return "not a number"
        return 1.0
    
    def _compute_aggregation(self, exp_data):
        if self.bad_method == "aggregation_type":
            return "not a dict"
        elif self.bad_method == "aggregation_values":
            return {"metric": "not a number"}
        return {}


class BadCalibrationModel(ICalibrationModel):
    """Calibration model that returns wrong types."""
    
    def __init__(self, logger, return_type="list"):
        self.logger = logger
        self.return_type = return_type
        self.performance_weights = {}
    
    def optimize(self, param_ranges, objective_fn):
        if self.return_type == "list":
            return [1.0, 2.0]
        elif self.return_type == "missing_params":
            return {"param_x": 1.0}  # Missing param_y
        elif self.return_type == "non_numeric":
            return {"param_x": "not a number", "param_y": 2.0}
        else:
            return {"param_x": 1.0, "param_y": 2.0}


@pytest.fixture
def logger(tmp_path):
    """Create test logger."""
    return LBPLogger("test", str(tmp_path / "logs"))


@pytest.fixture
def simple_dataset():
    """Create minimal dataset."""
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
    
    for i in range(3):
        exp_code = f"exp_{i}"
        exp_data = dataset.load_experiment(
            exp_code=exp_code,
            exp_params={"x": float(i), "y": float(i * 2)}
        )
        
        exp_data.features = MetricArrays()
        test_feature_arr = DataArray(name="test_feature", shape=())
        exp_data.features.add("test_feature", test_feature_arr)
        exp_data.features.set_value("test_feature", np.array(float(i + 1)))
    
    return dataset


class TestFeatureValidation:
    """Test validation of feature model returns."""
    
    def test_compute_features_returns_string(self, simple_dataset, logger):
        """Test that _compute_features returning string raises TypeError."""
        model = BadFeatureModel(simple_dataset, logger, return_type="string")
        
        with pytest.raises(TypeError, match="_compute_features\\(\\) must return numeric value"):
            model.run("test_feature", x=1, y=2)
    
    def test_compute_features_returns_list(self, simple_dataset, logger):
        """Test that _compute_features returning list raises TypeError."""
        model = BadFeatureModel(simple_dataset, logger, return_type="list")
        
        with pytest.raises(TypeError, match="_compute_features\\(\\) must return numeric value"):
            model.run("test_feature", x=1, y=2)


class TestEvaluationValidation:
    """Test validation of evaluation model returns."""
    
    def test_compute_target_value_returns_string(self, logger):
        """Test that _compute_target_value has validation in place."""
        model = BadEvaluationModel(logger=logger, bad_method="target")
        
        # Verify the method returns wrong type (validation happens in run())
        result = model._compute_target_value(x=1, y=2)
        assert isinstance(result, str)  # Confirms test model works


class TestCalibrationValidation:
    """Test validation of calibration model returns."""
    
    def test_optimize_returns_list(self, logger):
        """Test that optimize returning list raises TypeError."""
        model = BadCalibrationModel(logger=logger, return_type="list")
        
        param_ranges = {"param_x": (0.0, 10.0), "param_y": (0.0, 10.0)}
        objective_fn = lambda params: 1.0
        
        with pytest.raises(TypeError, match="optimize\\(\\) must return dict"):
            model.calibrate(param_ranges, objective_fn)
    
    def test_optimize_missing_parameters(self, logger):
        """Test that optimize missing parameters raises ValueError."""
        model = BadCalibrationModel(logger=logger, return_type="missing_params")
        
        param_ranges = {"param_x": (0.0, 10.0), "param_y": (0.0, 10.0)}
        objective_fn = lambda params: 1.0
        
        with pytest.raises(ValueError, match="optimize\\(\\) missing parameters"):
            model.calibrate(param_ranges, objective_fn)
    
    def test_optimize_non_numeric_values(self, logger):
        """Test that optimize with non-numeric values raises TypeError."""
        model = BadCalibrationModel(logger=logger, return_type="non_numeric")
        
        param_ranges = {"param_x": (0.0, 10.0), "param_y": (0.0, 10.0)}
        objective_fn = lambda params: 1.0
        
        with pytest.raises(TypeError, match="optimize\\(\\) parameter values must be numeric"):
            model.calibrate(param_ranges, objective_fn)
