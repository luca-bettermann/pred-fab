"""
Tests for core data objects including DataReal, DataInt, DataArray, and factory classes.
"""
import pytest
import numpy as np

from lbp_package.core.data_objects import (
    DataReal,
    DataInt,
    DataBool,
    DataCategorical,
    DataDimension,
    DataArray,
)
from lbp_package.utils.dataclass_fields import (
    Parameter,
    Performance,
    Dimension,
)
from lbp_package.core.data_blocks import DataBlock


class TestDataReal:
    """Test DataReal parameter type."""
    
    def test_datareal_creation(self):
        """Test creating DataReal with constraints."""
        param = DataReal(name="learning_rate", min_val=0.0, max_val=1.0)
        
        assert param.name == "learning_rate"
        assert param.dtype == float
        assert param.constraints["min"] == 0.0
        assert param.constraints["max"] == 1.0
        # DataObject doesn't store values - values are stored in DataBlock
    
    def test_datareal_value_storage(self):
        """Test storing value in DataReal."""
        param = DataReal(name="lr", min_val=0.0, max_val=1.0)
        param.value = 0.01
        
        assert param.value == 0.01
    
    def test_datareal_validation_success(self):
        """Test valid value passes validation."""
        param = DataReal(name="lr", min_val=0.0, max_val=1.0)
        
        assert param.validate(0.5) is True
        assert param.validate(0.0) is True  # Min boundary
        assert param.validate(1.0) is True  # Max boundary
    
    def test_datareal_validation_out_of_range(self):
        """Test value outside range fails validation."""
        param = DataReal(name="lr", min_val=0.0, max_val=1.0)
        
        with pytest.raises(ValueError, match="below minimum"):
            param.validate(-0.1)
        
        with pytest.raises(ValueError, match="above maximum"):
            param.validate(1.5)


class TestDataInt:
    """Test DataInt parameter type."""
    
    def test_dataint_creation(self):
        """Test creating DataInt with constraints."""
        param = DataInt(name="batch_size", min_val=1, max_val=256)
        
        assert param.name == "batch_size"
        assert param.dtype == int
        assert param.constraints["min"] == 1
        assert param.constraints["max"] == 256
    
    def test_dataint_validation_success(self):
        """Test valid integers pass validation."""
        param = DataInt(name="num_layers", min_val=1, max_val=10)
        
        assert param.validate(5) is True
        assert param.validate(1) is True
        assert param.validate(10) is True
    
    def test_dataint_validation_fail(self):
        """Test invalid integers fail validation."""
        param = DataInt(name="num_layers", min_val=1, max_val=10)
        
        with pytest.raises(ValueError):
            param.validate(0)
        
        with pytest.raises(ValueError):
            param.validate(11)


class TestDataCategorical:
    """Test DataCategorical parameter type."""
    
    def test_datacategorical_creation(self):
        """Test creating categorical parameter."""
        param = DataCategorical(name="optimizer", categories=["adam", "sgd", "rmsprop"])
        
        assert param.name == "optimizer"
        assert param.dtype == str
        assert param.constraints["categories"] == ["adam", "sgd", "rmsprop"]
    
    def test_datacategorical_validation(self):
        """Test categorical validation."""
        param = DataCategorical(name="optimizer", categories=["adam", "sgd"])
        
        assert param.validate("adam") is True
        assert param.validate("sgd") is True
        
        with pytest.raises(ValueError, match="not in allowed categories"):
            param.validate("invalid_opt")


class TestDataArray:
    """Test DataArray for numpy array wrapping."""
    
    def test_dataarray_creation(self):
        """Test creating DataArray with shape and dtype."""
        arr = DataArray(name="test_array", shape=(10, 3), dtype=np.float64)
        
        assert arr.name == "test_array"
        assert arr.constraints["shape"] == (10, 3)
        # dtype is stored as class type in constraints
        assert arr.constraints["dtype"] == np.float64.__class__ or np.float64
        # DataObject doesn't store values - values are stored in DataBlock
    
    def test_dataarray_set_valid_value(self):
        """Test setting valid numpy array via DataBlock."""
        
        arr = DataArray(name="test_array", shape=(5, 2), dtype=np.float32)
        data = np.random.rand(5, 2).astype(np.float32)
        
        # Values are stored in DataBlock, not DataObject
        block = DataBlock()
        block.add("test_array", arr)
        block.set_value("test_array", data)
        
        stored_value = block.get_value("test_array")
        assert stored_value is not None
        assert np.array_equal(stored_value, data)
        assert stored_value.dtype == np.float32
    
    def test_dataarray_shape_mismatch(self):
        """Test that mismatched shape raises ValueError."""
        
        arr = DataArray(name="test_array", shape=(5, 2), dtype=np.float64)
        wrong_data = np.random.rand(3, 4)
        
        block = DataBlock()
        block.add("test_array", arr)
        
        with pytest.raises(ValueError, match="shape mismatch"):
            block.set_value("test_array", wrong_data)
    
    def test_dataarray_dtype_mismatch(self):
        """Test that mismatched dtype raises ValueError."""
        
        arr = DataArray(name="test_array", shape=(5, 2), dtype=np.float32)
        wrong_dtype = np.random.rand(5, 2).astype(np.float64)
        
        block = DataBlock()
        block.add("test_array", arr)
        
        with pytest.raises(ValueError, match="dtype mismatch"):
            block.set_value("test_array", wrong_dtype)
    
    def test_dataarray_not_ndarray(self):
        """Test that non-ndarray raises TypeError."""
        
        arr = DataArray(name="test_array", shape=(5, 2), dtype=np.float64)
        
        block = DataBlock()
        block.add("test_array", arr)
        
        with pytest.raises(TypeError, match="must be np.ndarray"):
            block.set_value("test_array", [[1, 2], [3, 4]])


class TestParameterFactory:
    """Test Parameter factory class."""
    
    def test_parameter_real(self):
        """Test creating real parameter via factory."""
        param = Parameter.real(min_val=0.0, max_val=1.0)
        
        assert isinstance(param, DataReal)
        assert param.dtype == float
        assert param.constraints["min"] == 0.0
        assert param.constraints["max"] == 1.0
    
    def test_parameter_integer(self):
        """Test creating integer parameter via factory."""
        param = Parameter.integer(min_val=1, max_val=10)
        
        assert isinstance(param, DataInt)
        assert param.dtype == int
        assert param.constraints["min"] == 1
        assert param.constraints["max"] == 10
    
    def test_parameter_categorical(self):
        """Test creating categorical parameter via factory."""
        param = Parameter.categorical(categories=["adam", "sgd"])
        
        assert isinstance(param, DataCategorical)
        assert param.constraints["categories"] == ["adam", "sgd"]
    
    def test_parameter_boolean(self):
        """Test creating boolean parameter via factory."""
        param = Parameter.boolean()
        
        assert isinstance(param, DataBool)
        assert param.dtype == bool


class TestPerformanceFactory:
    """Test Performance factory class."""
    
    def test_performance_real(self):
        """Test creating real performance metric via factory."""
        perf = Performance.real(min_val=0.0, max_val=1.0)
        
        assert isinstance(perf, DataReal)
        assert perf.constraints["min"] == 0.0
        assert perf.constraints["max"] == 1.0
    
    def test_performance_integer(self):
        """Test creating integer performance metric via factory."""
        perf = Performance.integer(min_val=0, max_val=100)
        
        assert isinstance(perf, DataInt)
        assert perf.constraints["min"] == 0
        assert perf.constraints["max"] == 100


class TestDimensionFactory:
    """Test Dimension factory class."""
    
    def test_dimension_integer(self):
        """Test creating integer dimension via factory."""
        dim = Dimension.integer(
            param_name="trajectory",
            dim_name="timestep",
            iterator_name="t",
            min_val=0,
            max_val=100
        )
        
        assert isinstance(dim, DataDimension)
        # name is the param_name in DataDimension\n        assert dim.name == \"trajectory\"\n        assert dim.dim_name == \"timestep\"\n        assert dim.dim_iterator_name == \"t\"\n        assert dim.constraints[\"min\"] == 0\n        assert dim.constraints[\"max\"] == 100


class TestRounding:
    """Test round_digits functionality."""
    
    def test_datareal_rounding(self):
        """Test DataReal with round_digits via DataBlock."""
        
        param = DataReal(name="energy", round_digits=3)
        
        block = DataBlock()
        block.add("energy", param)
        block.set_value("energy", 1.23456789)
        
        assert block.get_value("energy") == 1.235
    
    def test_datareal_no_rounding(self):
        """Test DataReal without round_digits preserves full precision."""
        
        param = DataReal(name="energy")
        
        block = DataBlock()
        block.add("energy", param)
        block.set_value("energy", 1.23456789)
        
        assert block.get_value("energy") == 1.23456789
    
    def test_datareal_rounding_zero_digits(self):
        """Test DataReal with zero decimal places."""
        
        param = DataReal(name="energy", round_digits=0)
        
        block = DataBlock()
        block.add("energy", param)
        block.set_value("energy", 1.6)
        
        assert block.get_value("energy") == 2.0
    
    def test_datareal_serialization_with_round_digits(self):
        """Test that round_digits is preserved in serialization."""
        param = DataReal(name="energy", round_digits=4)
        param_dict = param.to_dict()
        
        assert param_dict["round_digits"] == 4
        
        # Test deserialization
        restored = DataReal._from_dict_impl("energy", param_dict["constraints"], param_dict.get("round_digits"))
        assert restored.round_digits == 4
    
    def test_factory_with_round_digits(self):
        """Test factory classes support round_digits."""
        param_field = Parameter.real(min_val=0.0, max_val=1.0, round_digits=5)
        assert param_field.metadata['aixd_schema'].round_digits == 5
        
        perf_field = Performance.real(min_val=0.0, max_val=1.0, round_digits=3)
        assert perf_field.metadata['aixd_schema'].round_digits == 3
