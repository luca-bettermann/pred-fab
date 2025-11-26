"""
Tests for data blocks including Parameters, Dimensions, PerformanceAttributes, and MetricArrays.
"""
import pytest
import numpy as np
from lbp_package.core.data_blocks import (
    DataBlock,
    Parameters,
    Dimensions,
    PerformanceAttributes,
    MetricArrays,
)
from lbp_package.core.data_objects import (
    Parameter,
    Performance,
    Dimension,
    DataArray,
)


class TestDataBlockBase:
    """Test base DataBlock functionality with value storage."""
    
    def test_datablock_add_and_get(self):
        """Test adding and retrieving DataObjects from block."""
        block = DataBlock()
        param = Parameter.real(min_val=0.0, max_val=1.0)
        
        block.add("learning_rate", param)
        
        assert block.has("learning_rate")
        retrieved = block.get("learning_rate")
        assert retrieved.dtype == float
    
    def test_datablock_value_storage(self):
        """Test storing and retrieving values."""
        block = DataBlock()
        block.add("learning_rate", Parameter.real(min_val=0.0, max_val=1.0))
        block.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        
        # Initially no values
        assert not block.has_value("learning_rate")
        assert not block.has_value("batch_size")
        
        # Set values
        block.set_value("learning_rate", 0.001)
        block.set_value("batch_size", 32)
        
        # Check values
        assert block.has_value("learning_rate")
        assert block.has_value("batch_size")
        assert block.get_value("learning_rate") == 0.001
        assert block.get_value("batch_size") == 32
    
    def test_get_value_missing_key(self):
        """Test that getting missing value raises KeyError."""
        block = DataBlock()
        
        with pytest.raises(KeyError):
            block.get_value("nonexistent")
    
    def test_has_value_missing_key(self):
        """Test has_value returns False for missing key."""
        block = DataBlock()
        assert not block.has_value("nonexistent")
    
    def test_validate_value(self):
        """Test value validation."""
        block = DataBlock()
        block.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        # Valid value
        assert block.validate_value("lr", 0.5) is True
        
        # Invalid value
        with pytest.raises(ValueError):
            block.validate_value("lr", 2.0)


class TestParameters:
    """Test unified Parameters block."""
    
    def test_parameters_creation(self):
        """Test creating Parameters block."""
        params = Parameters()
        assert len(list(params.keys())) == 0
    
    def test_parameters_add_objects(self):
        """Test adding various parameter types."""
        params = Parameters()
        
        params.add("learning_rate", Parameter.real(min_val=0.0, max_val=1.0))
        params.add("num_layers", Parameter.integer(min_val=1, max_val=10))
        params.add("optimizer", Parameter.categorical(categories=["adam", "sgd"]))
        params.add("use_dropout", Parameter.boolean())
        
        assert len(list(params.keys())) == 4
        assert params.has("learning_rate")
        assert params.has("num_layers")
        assert params.has("optimizer")
        assert params.has("use_dropout")
    
    def test_parameters_value_storage(self):
        """Test storing parameter values."""
        params = Parameters()
        params.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        params.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        
        params.set_value("lr", 0.001)
        params.set_value("batch_size", 32)
        
        assert params.get_value("lr") == 0.001
        assert params.get_value("batch_size") == 32


class TestDimensions:
    """Test Dimensions block for dimensional metadata."""
    
    def test_dimensions_creation(self):
        """Test creating Dimensions block."""
        dims = Dimensions()
        assert len(list(dims.keys())) == 0
    
    def test_dimensions_add_objects(self):
        """Test adding dimension objects."""
        dims = Dimensions()
        
        dims.add("trajectory.timestep", Dimension.integer("trajectory", "timestep", "t", min_val=0, max_val=100))
        dims.add("sensor_data.channel", Dimension.integer("sensor_data", "channel", "c", min_val=0, max_val=10))
        
        assert len(list(dims.keys())) == 2
        assert dims.has("trajectory.timestep")
        assert dims.has("sensor_data.channel")


class TestPerformanceAttributes:
    """Test PerformanceAttributes block."""
    
    def test_performance_creation(self):
        """Test creating PerformanceAttributes block."""
        perf = PerformanceAttributes()
        assert len(list(perf.keys())) == 0
        assert perf.calibration_weights == {}
    
    def test_performance_add_objects(self):
        """Test adding performance metrics."""
        perf = PerformanceAttributes()
        
        perf.add("accuracy", Performance.real(min_val=0.0, max_val=1.0))
        perf.add("loss", Performance.real(min_val=0.0, max_val=100.0))
        perf.add("num_errors", Performance.integer(min_val=0, max_val=1000))
        
        assert len(list(perf.keys())) == 3
        assert perf.has("accuracy")
        assert perf.has("loss")
        assert perf.has("num_errors")
    
    def test_performance_value_storage(self):
        """Test storing performance values."""
        perf = PerformanceAttributes()
        perf.add("accuracy", Performance.real(min_val=0.0, max_val=1.0))
        
        perf.set_value("accuracy", 0.95)
        assert perf.get_value("accuracy") == 0.95
    
    def test_performance_calibration_weights(self):
        """Test calibration weights."""
        perf = PerformanceAttributes()
        perf.add("accuracy", Performance.real(min_val=0.0, max_val=1.0))
        perf.add("speed", Performance.real(min_val=0.0, max_val=100.0))
        
        perf.calibration_weights = {"accuracy": 0.7, "speed": 0.3}
        
        assert perf.calibration_weights["accuracy"] == 0.7
        assert perf.calibration_weights["speed"] == 0.3


class TestMetricArrays:
    """Test MetricArrays block for numpy array storage."""
    
    def test_metric_arrays_creation(self):
        """Test creating MetricArrays block."""
        arrays = MetricArrays()
        assert len(list(arrays.keys())) == 0
    
    def test_metric_arrays_add_objects(self):
        """Test adding DataArray objects."""
        arrays = MetricArrays()
        
        arrays.add("energy", DataArray(name="energy", shape=(100,), dtype=np.float64))
        arrays.add("position", DataArray(name="position", shape=(100, 3), dtype=np.float32))
        
        assert len(list(arrays.keys())) == 2
        assert arrays.has("energy")
        assert arrays.has("position")
    
    def test_metric_arrays_value_storage(self):
        """Test storing and retrieving array values."""
        arrays = MetricArrays()
        arrays.add("data", DataArray(name="data", shape=(5,), dtype=np.float64))
        
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arrays.set_value("data", arr)
        
        assert arrays.has_value("data")
        assert np.array_equal(arrays.get_value("data"), arr)


class TestDataBlockIntegration:
    """Integration tests for using multiple blocks together."""
    
    def test_complete_experiment_structure(self):
        """Test creating a complete experiment structure with all blocks."""
        # Parameters
        params = Parameters()
        params.add("learning_rate", Parameter.real(min_val=0.0, max_val=1.0))
        params.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        params.set_value("learning_rate", 0.001)
        params.set_value("batch_size", 32)
        
        # Dimensions
        dims = Dimensions()
        dims.add("trajectory.timestep", Dimension.integer("trajectory", "timestep", "t", min_val=0, max_val=100))
        
        # Performance
        perf = PerformanceAttributes()
        perf.add("accuracy", Performance.real(min_val=0.0, max_val=1.0))
        perf.set_value("accuracy", 0.95)
        perf.calibration_weights = {"accuracy": 1.0}
        
        # Metric Arrays
        arrays = MetricArrays()
        arrays.add("energy", DataArray(name="energy", shape=(100,), dtype=np.float64))
        energy_data = np.random.rand(100)
        arrays.set_value("energy", energy_data)
        
        # Verify structure
        assert params.get_value("learning_rate") == 0.001
        assert perf.get_value("accuracy") == 0.95
        assert np.array_equal(arrays.get_value("energy"), energy_data)
        assert dims.has("trajectory.timestep")
