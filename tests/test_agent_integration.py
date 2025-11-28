"""Test LBPAgent integration with new AIXD architecture."""

import pytest
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Type, Optional, Any

from lbp_package.core.dataset import Dataset
from lbp_package.core.schema import DatasetSchema
from lbp_package.core.data_objects import DataReal, DataInt
from lbp_package.interfaces.evaluation import IEvaluationModel
from lbp_package.interfaces.features import IFeatureModel
from lbp_package.interfaces.prediction import IPredictionModel
from lbp_package.orchestration.agent import LBPAgent
from lbp_package.utils.logger import LBPLogger


# === TEST FIXTURES ===

@dataclass
class SimpleFeatureModel(IFeatureModel):
    """Simple feature model for testing."""
    
    param_x: DataReal = DataReal("param_x", min_val=0.0, max_val=10.0)
    param_y: DataInt = DataInt("param_y", min_val=1, max_val=100)
    
    def _load_data(self, **param_values):
        """Mock data loading."""
        return param_values
    
    def _compute_features(self, data):
        """Mock feature computation."""
        return sum(data.values())


@dataclass
class SimpleEvaluationModel(IEvaluationModel):
    """Simple evaluation model for testing."""
    
    @property
    def feature_model_type(self) -> Type[IFeatureModel]:
        return SimpleFeatureModel
    
    def _compute_target_value(self, **param_values) -> float:
        """Mock target computation."""
        return 10.0
    
    def _compute_scaling_factor(self, **param_values) -> Optional[float]:
        """Mock scaling factor."""
        return 1.0


@dataclass
class SimplePredictionModel(IPredictionModel):
    """Simple prediction model for testing."""
    
    param_x: DataReal = DataReal("param_x", min_val=0.0, max_val=10.0)
    param_y: DataInt = DataInt("param_y", min_val=1, max_val=100)
    
    def __init__(self, logger=None, **kwargs):
        self.logger = logger
        self.is_trained = False
    
    @property
    def predicted_features(self):
        return ["predicted_feature"]
    
    def train(self, X, y, **kwargs):
        """Mock training."""
        self.is_trained = True
    
    def forward_pass(self, X):
        """Mock prediction."""
        # Return mock predictions
        return pd.DataFrame({"predicted_feature": [1.0] * len(X)})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {"is_trained": self.is_trained}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.is_trained = artifacts.get("is_trained", False)


@pytest.fixture
def logger(tmp_path):
    """Create a test logger."""
    log_folder = str(tmp_path / "logs")
    return LBPLogger("TestAgent", log_folder)


@pytest.fixture
def agent(tmp_path, logger):
    """Create a test agent."""
    root_folder = str(tmp_path)
    local_folder = str(tmp_path / "local")
    log_folder = str(tmp_path / "logs")
    
    agent = LBPAgent(
        root_folder=root_folder,
        local_folder=local_folder,
        log_folder=log_folder,
        debug_flag=True,
        recompute_flag=True
    )
    
    return agent


# === TESTS ===

def test_agent_creation(agent):
    """Test agent can be created."""
    assert agent is not None
    assert agent.eval_system is None  # Not initialized until dataset is created
    assert agent.pred_system is None


def test_register_evaluation_model(agent):
    """Test registering an evaluation model."""
    agent.register_evaluation_model(
        performance_code="perf_metric",
        evaluation_class=SimpleEvaluationModel
    )
    
    assert "perf_metric" in agent._evaluation_model_specs
    eval_class, eval_kwargs = agent._evaluation_model_specs["perf_metric"]
    assert eval_class == SimpleEvaluationModel


def test_register_prediction_model(agent):
    """Test registering a prediction model."""
    agent.register_prediction_model(
        performance_codes=["perf_metric"],
        prediction_class=SimplePredictionModel
    )
    
    assert "perf_metric" in agent._prediction_model_specs
    pred_class, pred_kwargs = agent._prediction_model_specs["perf_metric"]
    assert pred_class == SimplePredictionModel


def test_schema_generation(agent):
    """Test schema generation from registered models."""
    # Register models
    agent.register_evaluation_model(
        performance_code="perf_metric",
        evaluation_class=SimpleEvaluationModel
    )
    agent.register_prediction_model(
        performance_codes=["perf_metric"],
        prediction_class=SimplePredictionModel
    )
    
    # Generate schema
    schema = agent.generate_schema_from_registered_models()
    
    assert isinstance(schema, DatasetSchema)
    assert len(schema.performance_attrs.data_objects) == 1
    assert "perf_metric" in schema.performance_attrs.data_objects
    
    # Should have extracted param_x and param_y from models
    assert "param_x" in schema.parameters.data_objects
    assert "param_y" in schema.parameters.data_objects


def test_dataset_initialization(agent):
    """Test dataset initialization with registered models."""
    # Register models
    agent.register_evaluation_model(
        performance_code="perf_metric",
        evaluation_class=SimpleEvaluationModel
    )
    agent.register_prediction_model(
        performance_codes=["perf_metric"],
        prediction_class=SimplePredictionModel
    )
    
    # Initialize dataset
    static_params = {"param_x": 5.0, "param_y": 50}
    dataset = agent.initialize(static_params=static_params)
    
    # Agent is now stateless - returns dataset without storing it
    assert isinstance(dataset, Dataset)
    assert dataset.schema is not None
    assert dataset.schema_id == "schema_001"
    
    # Systems should be created and have models registered
    # Note: Systems are created during initialize() but agent doesn't expose them
    # (they're used internally during evaluate_experiment() calls)

def test_post_initialization_registration_prohibited(agent):
    """Test that registration after initialize() raises error."""
    # Register and initialize
    agent.register_evaluation_model(
        performance_code="perf_metric",
        evaluation_class=SimpleEvaluationModel
    )
    
    # Initialize with empty static params (SimpleEvaluationModel has no parameters)
    agent.initialize(static_params={})
    
    # Attempt to register after initialization should fail
    with pytest.raises(RuntimeError, match="Cannot register models after initialize"):
        agent.register_evaluation_model(
            performance_code="another_metric",
            evaluation_class=SimpleEvaluationModel
        )
    
    with pytest.raises(RuntimeError, match="Cannot register models after initialize"):
        agent.register_prediction_model(
            performance_codes=["another_metric"],
            prediction_class=SimplePredictionModel
        )
