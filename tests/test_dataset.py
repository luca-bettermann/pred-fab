"""
Tests for Dataset with ExperimentData migration.
"""
import pytest
import numpy as np

from lbp_package.core.dataset import Dataset, ExperimentData
from lbp_package.core.schema import DatasetSchema
from lbp_package.core.data_objects import Parameter, Performance, DataArray, DataDimension
from lbp_package.core.data_blocks import DataBlock


class TestExperimentDataDataclass:
    """Test ExperimentData dataclass."""
    
    def test_experiment_data_creation(self):
        """Test creating ExperimentData with minimal fields."""
        params = DataBlock()
        params.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        params.set_value("lr", 0.01)
        
        exp_data = ExperimentData(exp_code="exp_001", parameters=params)
        
        assert exp_data.exp_code == "exp_001"
        assert exp_data.parameters is not None
        # All data blocks are auto-initialized (not None)
        assert exp_data.performance is not None
        assert exp_data.metric_arrays is not None
        assert exp_data.predicted_metric_arrays is not None
    
    def test_experiment_data_dimensions_property(self):
        """Test dimensions property extracts dimensional params."""
        
        params = DataBlock()
        params.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        params.set_value("lr", 0.01)
        
        # Add dimensional parameter
        dim_obj = DataDimension(dim_name="timestep", dim_param_name="trajectory", dim_iterator_name="t")
        params.add("trajectory", dim_obj)
        params.set_value("trajectory", 100)
        
        exp_data = ExperimentData(exp_code="exp_001", parameters=params)
        
        dims = exp_data.dimensions
        assert "trajectory" in dims
        assert dims["trajectory"] == 100
        assert "lr" not in dims


class TestDatasetCreation:
    """Test Dataset initialization."""
    
    def test_dataset_creation(self):
        """Test creating empty dataset."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        dataset = Dataset(name="test_dataset", schema=schema, schema_id="schema_123")
        
        assert dataset.name == "test_dataset"
        assert dataset.schema_id == "schema_123"
        assert len(dataset.get_experiment_codes()) == 0
    
    def test_dataset_set_static_values(self):
        """Test setting static parameter values."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema.parameters.add("model_type", Parameter.categorical(categories=["A", "B"]))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        dataset.set_static_values({"lr": 0.001, "model_type": "A"})
        
        assert dataset.get_static_value("lr") == 0.001
        assert dataset.get_static_value("model_type") == "A"
    
    def test_dataset_set_invalid_static_value(self):
        """Test that invalid static values raise error."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        with pytest.raises(ValueError, match="Unknown parameter"):
            dataset.set_static_values({"unknown_param": 123})


class TestDatasetAddExperiment:
    """Test adding experiments to dataset."""
    
    def test_add_simple_experiment(self):
        """Test adding experiment with parameters only."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema.parameters.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        dataset.set_static_values({"lr": 0.001})
        
        dataset.add_experiment(
            exp_code="exp_001",
            exp_params={"batch_size": 32}
        )
        
        assert dataset.has_experiment("exp_001")
        assert "exp_001" in dataset.get_experiment_codes()
    
    def test_add_experiment_with_performance(self):
        """Test adding experiment with performance metrics."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema.performance_attrs.add("accuracy", Performance.real(min_val=0.0, max_val=1.0))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        dataset.set_static_values({"lr": 0.001})
        
        dataset.add_experiment(
            exp_code="exp_001",
            exp_params={},
            performance={"accuracy": 0.95}
        )
        
        exp_data = dataset.get_experiment("exp_001")
        assert exp_data.performance is not None
        assert exp_data.performance.get_value("accuracy") == 0.95
    
    def test_add_experiment_with_metric_arrays(self):
        """Test adding experiment with metric arrays."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema.metric_arrays.add("energy", DataArray(name="energy", shape=(100,)))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        dataset.set_static_values({"lr": 0.001})
        
        energy_data = np.random.rand(100)
        dataset.add_experiment(
            exp_code="exp_001",
            exp_params={},
            metric_arrays={"energy": energy_data}
        )
        
        exp_data = dataset.get_experiment("exp_001")
        assert exp_data.metric_arrays is not None
        assert np.array_equal(exp_data.metric_arrays.get_value("energy"), energy_data)
    
    def test_add_experiment_combines_static_and_dynamic(self):
        """Test that static values are copied into experiment parameters."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema.parameters.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        dataset.set_static_values({"lr": 0.001})
        
        dataset.add_experiment(exp_code="exp_001", exp_params={"batch_size": 32})
        
        exp_data = dataset.get_experiment("exp_001")
        assert exp_data.parameters.get_value("lr") == 0.001  # Static
        assert exp_data.parameters.get_value("batch_size") == 32  # Dynamic
    
    def test_add_experiment_unknown_param_fails(self):
        """Test that unknown parameters raise error."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        with pytest.raises(ValueError, match="Unknown parameter"):
            dataset.add_experiment(
                exp_code="exp_001",
                exp_params={"unknown_param": 123}
            )


class TestDatasetAccessors:
    """Test dataset accessor methods."""
    
    def test_get_experiment(self):
        """Test retrieving ExperimentData."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        dataset.set_static_values({"lr": 0.001})
        dataset.add_experiment(exp_code="exp_001", exp_params={})
        
        exp_data = dataset.get_experiment("exp_001")
        
        assert isinstance(exp_data, ExperimentData)
        assert exp_data.exp_code == "exp_001"
    
    def test_get_experiment_not_found(self):
        """Test that getting nonexistent experiment raises KeyError."""
        schema = DatasetSchema()
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        with pytest.raises(KeyError, match="not found"):
            dataset.get_experiment("nonexistent")
    
    def test_get_experiment_params(self):
        """Test getting experiment parameters as dict."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema.parameters.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        dataset.set_static_values({"lr": 0.001})
        dataset.add_experiment(exp_code="exp_001", exp_params={"batch_size": 32})
        
        params = dataset.get_experiment_params("exp_001")
        
        assert params["lr"] == 0.001
        assert params["batch_size"] == 32
    
    def test_has_experiment(self):
        """Test checking experiment existence."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        dataset.set_static_values({"lr": 0.001})
        dataset.add_experiment(exp_code="exp_001", exp_params={})
        
        assert dataset.has_experiment("exp_001")
        assert not dataset.has_experiment("exp_999")


class TestFeatureMemoization:
    """Test feature caching for IFeatureModel."""
    
    def test_has_features_at(self):
        """Test checking if features are cached."""
        schema = DatasetSchema()
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        assert not dataset.has_features_at(lr=0.001, batch_size=32)
        
        dataset.set_feature_value("feature1", 123, lr=0.001, batch_size=32)
        
        assert dataset.has_features_at(lr=0.001, batch_size=32)
    
    def test_set_and_get_feature_value(self):
        """Test caching and retrieving feature values."""
        schema = DatasetSchema()
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        dataset.set_feature_value("temperature", 25.5, location="A", time=100)
        dataset.set_feature_value("pressure", 101.3, location="A", time=100)
        
        assert dataset.get_feature_value("temperature", location="A", time=100) == 25.5
        assert dataset.get_feature_value("pressure", location="A", time=100) == 101.3
    
    def test_get_feature_value_not_cached(self):
        """Test that retrieving uncached feature raises KeyError."""
        schema = DatasetSchema()
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        with pytest.raises(KeyError, match="No features cached"):
            dataset.get_feature_value("feature1", lr=0.001)
    
    def test_feature_cache_different_params(self):
        """Test that different params create separate cache entries."""
        schema = DatasetSchema()
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        dataset.set_feature_value("result", 10, x=1, y=2)
        dataset.set_feature_value("result", 20, x=1, y=3)
        
        assert dataset.get_feature_value("result", x=1, y=2) == 10
        assert dataset.get_feature_value("result", x=1, y=3) == 20
    
    def test_feature_cache_param_order_irrelevant(self):
        """Test that parameter order doesn't affect cache key."""
        schema = DatasetSchema()
        dataset = Dataset(name="test", schema=schema, schema_id="schema_1")
        
        dataset.set_feature_value("result", 42, a=1, b=2, c=3)
        
        # Different order should retrieve same cached value
        assert dataset.get_feature_value("result", c=3, a=1, b=2) == 42
        assert dataset.get_feature_value("result", b=2, c=3, a=1) == 42


class TestDatasetIntegration:
    """Integration tests for complete dataset workflow."""
    
    def test_complete_workflow(self):
        """Test complete dataset usage: create, populate, access."""
        # Create schema
        schema = DatasetSchema()
        schema.parameters.add("learning_rate", Parameter.real(min_val=0.0, max_val=1.0))
        schema.parameters.add("num_layers", Parameter.integer(min_val=1, max_val=10))
        schema.parameters.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        schema.performance_attrs.add("accuracy", Performance.real(min_val=0.0, max_val=1.0))
        schema.performance_attrs.add("loss", Performance.real(min_val=0.0, max_val=100.0))
        schema.metric_arrays.add("energy", DataArray(name="energy", shape=(100,)))
        
        # Create dataset
        dataset = Dataset(name="robot_study", schema=schema, schema_id="schema_abc")
        
        # Set static values
        dataset.set_static_values({
            "learning_rate": 0.001,
            "num_layers": 5
        })
        
        # Add experiments
        for i in range(3):
            exp_code = f"exp_{i:03d}"
            energy_data = np.random.rand(100) * 10
            
            dataset.add_experiment(
                exp_code=exp_code,
                exp_params={"batch_size": 32 * (i + 1)},
                performance={
                    "accuracy": 0.9 + i * 0.01,
                    "loss": 10.0 - i * 0.5
                },
                metric_arrays={"energy": energy_data}
            )
        
        # Verify
        assert len(dataset.get_experiment_codes()) == 3
        
        exp_data = dataset.get_experiment("exp_001")
        assert exp_data.parameters.get_value("learning_rate") == 0.001  # Static
        assert exp_data.parameters.get_value("batch_size") == 64  # Dynamic
        assert exp_data.performance.get_value("accuracy") == 0.91
        assert exp_data.metric_arrays.get_value("energy").shape == (100,)
        
        # Feature caching
        dataset.set_feature_value("processed_energy", 123.45, batch_size=64)
        assert dataset.get_feature_value("processed_energy", batch_size=64) == 123.45
