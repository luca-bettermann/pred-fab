"""
Tests for PredictionSystem API behavior.

Ensures that the prediction system correctly handles:
- Model-driven feature declarations (models declare what they predict)
- Input declarations (models declare what they need)
- Feature model type declarations and instance sharing
- Correct internal data structures (list vs dict)
"""

import pytest
import pandas as pd
from typing import List, Dict, Type, Any
from dataclasses import dataclass

from lbp_package.core.schema import DatasetSchema
from lbp_package.core.dataset import Dataset
from lbp_package.core.data_objects import DataReal
from lbp_package.interfaces.prediction import IPredictionModel
from lbp_package.interfaces.features import IFeatureModel
from lbp_package.orchestration.prediction import PredictionSystem
from lbp_package.utils.logger import LBPLogger


# ============================================================================
# TEST FIXTURES AND MOCK MODELS
# ============================================================================

class MockFeatureModel1(IFeatureModel):
    """Mock feature model type 1."""
    
    def __init__(self, dataset=None, logger=None, **kwargs):
        super().__init__(dataset=dataset, logger=logger, **kwargs)
        self.call_count = 0
    
    def _load_data(self, exp_data):
        """Mock data loading."""
        return {"data": [1.0, 2.0, 3.0]}
    
    def _compute_features(self, data_dict):
        """Mock feature computation."""
        self.call_count += 1
        return pd.DataFrame({"computed_feature_1": [1.0]})


class MockFeatureModel2(IFeatureModel):
    """Mock feature model type 2."""
    
    def __init__(self, dataset=None, logger=None, **kwargs):
        super().__init__(dataset=dataset, logger=logger, **kwargs)
        self.call_count = 0
    
    def _load_data(self, exp_data):
        """Mock data loading."""
        return {"data": [4.0, 5.0, 6.0]}
    
    def _compute_features(self, data_dict):
        """Mock feature computation."""
        self.call_count += 1
        return pd.DataFrame({"computed_feature_2": [2.0]})


@dataclass
class SingleFeaturePredictionModel(IPredictionModel):
    """Model that predicts a single feature."""
    
    param_1: DataReal = DataReal("param_1")
    param_2: DataReal = DataReal("param_2")
    
    def __post_init__(self):
        self.logger = None
        self._feature_models = {}
        self.is_trained = False
    
    @property
    def feature_names(self) -> List[str]:
        return ["energy"]
    
    def train(self, X, y, **kwargs):
        self.is_trained = True
    
    def forward_pass(self, X):
        return pd.DataFrame({"energy": [1.0] * len(X)})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {"is_trained": self.is_trained}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.is_trained = artifacts.get("is_trained", False)


@dataclass
class MultiFeaturePredictionModel(IPredictionModel):
    """Model that predicts multiple features."""
    
    param_1: DataReal = DataReal("param_1")
    param_2: DataReal = DataReal("param_2")
    param_3: DataReal = DataReal("param_3")
    
    def __post_init__(self):
        self.logger = None
        self._feature_models = {}
        self.is_trained = False
    
    @property
    def feature_names(self) -> List[str]:
        return ["feature_a", "feature_b", "feature_c"]  # Multiple features
    
    def train(self, X, y, **kwargs):
        self.is_trained = True
    
    def forward_pass(self, X):
        return pd.DataFrame({
            "feature_a": [1.0] * len(X),
            "feature_b": [2.0] * len(X),
            "feature_c": [3.0] * len(X),
        })
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {"is_trained": self.is_trained}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.is_trained = artifacts.get("is_trained", False)


@dataclass
class ModelWithFeatureModelDependency(IPredictionModel):
    """Model that depends on feature models."""
    
    param_1: DataReal = DataReal("param_1")
    
    def __post_init__(self):
        self.logger = None
        self._feature_models = {}
        self.is_trained = False
    
    @property
    def feature_names(self) -> List[str]:
        return ["predicted_output"]
    
    @property
    def feature_model_types(self) -> Dict[str, Type[IFeatureModel]]:
        return {
            "feat1": MockFeatureModel1,
            "feat2": MockFeatureModel2,
        }
    
    def train(self, X, y, **kwargs):
        self.is_trained = True
    
    def forward_pass(self, X):
        # Could use self._feature_models["feat1"] here
        return pd.DataFrame({"predicted_output": [4.0] * len(X)})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {"is_trained": self.is_trained}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.is_trained = artifacts.get("is_trained", False)


@dataclass
class AnotherModelWithSameFeatureModel(IPredictionModel):
    """Another model that depends on MockFeatureModel1 (should share instance)."""
    
    param_2: DataReal = DataReal("param_2")
    
    def __post_init__(self):
        self.logger = None
        self._feature_models = {}
        self.is_trained = False
    
    @property
    def feature_names(self) -> List[str]:
        return ["other_output"]
    
    @property
    def feature_model_types(self) -> Dict[str, Type[IFeatureModel]]:
        return {
            "shared_feat": MockFeatureModel1,  # Same type as first model
        }
    
    def train(self, X, y, **kwargs):
        self.is_trained = True
    
    def forward_pass(self, X):
        return pd.DataFrame({"other_output": [5.0] * len(X)})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {"is_trained": self.is_trained}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.is_trained = artifacts.get("is_trained", False)


@pytest.fixture
def logger(tmp_path):
    """Create test logger."""
    return LBPLogger("test", str(tmp_path))


@pytest.fixture
def dataset():
    """Create test dataset."""
    schema = DatasetSchema()
    schema.parameters.add("param_1", DataReal("param_1"))
    schema.parameters.add("param_2", DataReal("param_2"))
    schema.parameters.add("param_3", DataReal("param_3"))
    
    return Dataset(
        name="test_dataset",
        schema_id="test_schema",
        schema=schema
    )


# ============================================================================
# TEST PREDICTION SYSTEM API
# ============================================================================

class TestPredictionSystemAPI:
    """Test that PredictionSystem API works as designed."""
    
    def test_add_prediction_model_signature(self, dataset, logger):
        """Test that add_prediction_model() takes only model parameter (no feature_name)."""
        system = PredictionSystem(dataset, logger)
        model = SingleFeaturePredictionModel()
        
        # New API: only pass model
        system.add_prediction_model(model)
        
        # Verify model is in list
        assert model in system.prediction_models
        assert len(system.prediction_models) == 1
    
    def test_prediction_models_is_list_not_dict(self, dataset, logger):
        """Test that prediction_models is a List, not a Dict."""
        system = PredictionSystem(dataset, logger)
        
        # Should be an empty list
        assert isinstance(system.prediction_models, list)
        assert len(system.prediction_models) == 0
        
        # Add models
        model1 = SingleFeaturePredictionModel()
        model2 = MultiFeaturePredictionModel()
        system.add_prediction_model(model1)
        system.add_prediction_model(model2)
        
        # Should be a list of models
        assert isinstance(system.prediction_models, list)
        assert len(system.prediction_models) == 2
        assert model1 in system.prediction_models
        assert model2 in system.prediction_models
    
    def test_feature_to_model_mapping_auto_generated(self, dataset, logger):
        """Test that feature_to_model mapping is automatically created from model.feature_names."""
        system = PredictionSystem(dataset, logger)
        model = SingleFeaturePredictionModel()
        
        system.add_prediction_model(model)
        
        # feature_to_model should be auto-created
        assert "energy" in system.feature_to_model
        assert system.feature_to_model["energy"] is model
    
    def test_multi_feature_model_creates_multiple_mappings(self, dataset, logger):
        """Test that models predicting multiple features create mappings for each."""
        system = PredictionSystem(dataset, logger)
        model = MultiFeaturePredictionModel()
        
        system.add_prediction_model(model)
        
        # All features should map to same model
        assert "feature_a" in system.feature_to_model
        assert "feature_b" in system.feature_to_model
        assert "feature_c" in system.feature_to_model
        assert system.feature_to_model["feature_a"] is model
        assert system.feature_to_model["feature_b"] is model
        assert system.feature_to_model["feature_c"] is model
    
    def test_model_declares_outputs_via_feature_names(self):
        """Test that models declare outputs via feature_names property."""
        """Test that models declare outputs via feature_names property."""
        model1 = SingleFeaturePredictionModel()
        model2 = MultiFeaturePredictionModel()
        
        # Single feature model
        assert model1.feature_names == ["energy"]
        
        # Multi-feature model
        assert model2.feature_names == ["feature_a", "feature_b", "feature_c"]


class TestFeatureModelDependencies:
    """Test feature model type declarations and instance sharing."""
    
    def test_model_declares_feature_model_types(self):
        """Test that models can declare feature model dependencies."""
        model = ModelWithFeatureModelDependency()
        
        # Should declare feature model types
        feature_types = model.feature_model_types
        assert "feat1" in feature_types
        assert "feat2" in feature_types
        assert feature_types["feat1"] == MockFeatureModel1
        assert feature_types["feat2"] == MockFeatureModel2
    
    def test_model_without_feature_models_returns_empty_dict(self):
        """Test that models without feature dependencies return empty dict."""
        model = SingleFeaturePredictionModel()
        
        # Default should be empty dict
        assert model.feature_model_types == {}
    
    def test_add_feature_model_method_attaches_instance(self, dataset, logger):
        """Test that add_feature_model() attaches IFeatureModel instance."""
        model = ModelWithFeatureModelDependency()
        feature_instance = MockFeatureModel1(dataset=dataset, logger=logger)
        
        # Attach feature model
        model.add_feature_model("feat1", feature_instance)
        
        # Should be stored in _feature_models
        assert "feat1" in model._feature_models
        assert model._feature_models["feat1"] is feature_instance
    
    def test_multiple_models_can_share_feature_model_instance(self, dataset, logger):
        """Test that multiple models can receive the same feature model instance."""
        model1 = ModelWithFeatureModelDependency()
        model2 = AnotherModelWithSameFeatureModel()
        
        # Create a single feature model instance
        shared_instance = MockFeatureModel1(dataset=dataset, logger=logger)
        
        # Attach to both models
        model1.add_feature_model("feat1", shared_instance)
        model2.add_feature_model("shared_feat", shared_instance)
        
        # Both should reference the same instance
        assert model1._feature_models["feat1"] is shared_instance
        assert model2._feature_models["shared_feat"] is shared_instance
        assert model1._feature_models["feat1"] is model2._feature_models["shared_feat"]
    
    def test_feature_model_instance_sharing_prevents_duplication(self, dataset, logger):
        """Test that sharing feature model instances prevents creating duplicates."""
        # Create two models that both need MockFeatureModel1
        model1 = ModelWithFeatureModelDependency()
        model2 = AnotherModelWithSameFeatureModel()
        
        # Create a single shared instance
        shared_instance = MockFeatureModel1(dataset=dataset, logger=logger)
        
        # Attach to both
        model1.add_feature_model("feat1", shared_instance)
        model2.add_feature_model("shared_feat", shared_instance)
        
        # Modify the shared instance
        shared_instance.call_count = 42
        
        # Both models should see the modification (proving it's the same instance)
        assert model1._feature_models["feat1"].call_count == 42
        assert model2._feature_models["shared_feat"].call_count == 42


class TestBackwardCompatibility:
    """Test that the new API doesn't break existing patterns."""
    
    def test_system_still_works_with_single_feature_models(self, dataset, logger):
        """Test that traditional single-feature models still work."""
        system = PredictionSystem(dataset, logger)
        model = SingleFeaturePredictionModel()
        
        system.add_prediction_model(model)
        
        # Should work exactly as before
        assert len(system.prediction_models) == 1
        assert "energy" in system.feature_to_model
    
    def test_models_without_feature_dependencies_still_work(self, dataset, logger):
        """Test that models without feature_model_types work fine."""
        system = PredictionSystem(dataset, logger)
        model = SingleFeaturePredictionModel()
        
        # Should not fail even though feature_model_types returns {}
        system.add_prediction_model(model)
        
        assert model.feature_model_types == {}
        assert len(model._feature_models) == 0
