"""
DataBlock collections for organizing related parameters.

DataBlocks group DataObjects into logical collections and store their values.
They provide both schema structure and data storage.
"""

import itertools
from typing import Dict, Any, List, Optional, Tuple, Type
import numpy as np
from .data_objects import DataObject, DataDimension, DataArray, DataReal
from .data_objects import Parameter, Feature, PerformanceAttribute


class DataBlock:
    """
    Container for DataObjects with value storage.
    
    Provides validation, access methods, and value management for typed parameter collections.
    """
    
    def __init__(self, data_object_type: Type[Any]):
        """Initialize empty DataBlock."""
        self.data_object_type = data_object_type

        self.data_objects: Dict[str, DataObject] = {}  # Schema structure
        self.values: Dict[str, Any] = {}  # Actual values
    
    def add(self, name: str, data_obj: DataObject) -> None:
        """Add a DataObject to the block."""
        if not isinstance(data_obj, self.data_object_type):
            raise TypeError(f"Expected data object of type {self.data_object_type.__name__}, got {type(data_obj).__name__}")
        self.data_objects[name] = data_obj
    
    def get(self, name: str) -> DataObject:
        """Get a DataObject by name."""
        return self.data_objects[name]
    
    def has(self, name: str) -> bool:
        """Check if parameter exists in block."""
        return name in self.data_objects
    
    def validate_value(self, name: str, value: Any) -> bool:
        """Validate a value against the corresponding DataObject."""
        if name not in self.data_objects:
            raise KeyError(f"Parameter '{name}' not defined in {self.__class__.__name__}")
        return self.data_objects[name].validate(value)
    
    def validate_all(self, values: Dict[str, Any]) -> bool:
        """Validate multiple values at once."""
        for name, value in values.items():
            self.validate_value(name, value)
        return True
    
    def set_value(self, name: str, value: Any) -> None:
        """Set value for a parameter after validation."""
        self.validate_value(name, value)  # Raises if invalid
        
        # Apply rounding for numeric types if configured
        data_obj = self.data_objects[name]
        if data_obj.round_digits is not None and isinstance(value, (int, float)):
            value = round(float(value), data_obj.round_digits)
        
        self.values[name] = value

    def set_values(self, values: Dict[str, Any]) -> None:
        """Set multiple values at once, ignoring unknown parameters."""
        for name, value in values.items():
            if self.has(name):
                self.set_value(name, value)
    
    def get_value(self, name: str) -> Any:
        """Get value for a parameter."""
        if name not in self.values:
            raise KeyError(f"No value set for parameter '{name}'")
        return self.values[name]
    
    def has_value(self, name: str) -> bool:
        """Check if value is set for parameter."""
        return name in self.values
    
    def to_numpy(self, dtype: type = np.float64) -> np.ndarray:
        """Convert all values to numpy array for ML."""
        values = []
        for name in self.data_objects.keys():
            if name in self.values:
                val = self.values[name]
                if not isinstance(val, (int, float, np.integer, np.floating, bool)):
                     raise ValueError(f"Parameter '{name}' has non-numeric value: {val} (type: {type(val)})")
                values.append(val)
        return np.array(values, dtype=dtype)
    
    def keys(self) -> Any:
        """Return iterator over parameter names."""
        return self.data_objects.keys()
    
    def values_iter(self) -> Any:
        """Return iterator over DataObjects (renamed from values to avoid conflict)."""
        return self.data_objects.values()
    
    def items(self) -> Any:
        """Return iterator over (name, DataObject) pairs."""
        return self.data_objects.items()
    
    def get_values_dict(self) -> Dict[str, Any]:
        """Extract all set values as a simple dictionary."""
        return dict(self.values)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for schema storage."""
        return {
            "type": self.__class__.__name__,
            "data_objects": {
                name: obj.to_dict() 
                for name, obj in self.data_objects.items()
            },
            "values": self.values  # Include current values
        }
    
    def is_compatible(self, other_block) -> bool:
        """Helper function to check compatibility between two data blocks."""
        self_objects = set(self.data_objects)
        other_objects = set(other_block.data_objects)
        if self_objects != other_objects:
            return False
        return True
    
    @classmethod
    def from_dict(cls, data_object_type: Type[Any], data: Dict[str, Any]) -> Any:
        """Reconstruct from dictionary."""
        block = cls(data_object_type)
        for name, obj_data in data.get("data_objects", {}).items():
            data_obj = DataObject.from_dict(obj_data)
            block.add(name, data_obj)
        # Restore values if present
        for name, value in data.get("values", {}).items():
            if name in block.data_objects:
                block.set_value(name, value)
        return block
    
    @classmethod
    def from_list(cls, data_object_type: Type[Any], data_objs: List[Any]) -> Any:
        """Reconstruct from list."""
        block = cls(data_object_type)
        for data_obj in data_objs:
            block.add(data_obj.name, data_obj)
        return block


class Parameters(DataBlock):
    """
    Unified parameter block for ALL parameters.
    
    Parameters may be:
    - Static: Same value across all experiments in dataset
    - Dynamic: Vary per experiment
    - Dimensional: Subset of parameters used for iteration
    """
    
    def __init__(self):
        """Initialize Parameters block."""
        super().__init__(Parameter)

    def get_dim_objects(self) -> Dict[str, DataDimension]:
        """Get view of dimension DataObjects from parameters."""
        dim_objs = {
            name: obj for name, obj in self.data_objects.items() 
            if isinstance(obj, DataDimension)
        }
        return dim_objs
        
    def get_dim_names(self) -> List[str]:
        """Get list of dimension parameter names."""
        dim_object_dict = self.get_dim_objects()
        return list(dim_object_dict.keys())
    
    def get_dim_iterator_names(self) -> Dict[str, str]:
        """Get list of dimension iterator names."""
        return {name: obj.iterator_code for name, obj in self.get_dim_objects().items()}

    def get_dim_values(self) -> Dict[str, int]:
        """Get view of dimension DataObjects from parameters."""
        return {name: self.get_value(name) for name in self.get_dim_names()}

    def validate_dimensions(self) -> None:
        """Validate dimension levels."""
        dim_objs = self.get_dim_objects()
        if not dim_objs:
            return

        levels = []
        for name, dim in dim_objs.items():
            levels.append((dim.level, name))
        levels.sort()
        
        # Check for duplicates
        seen_levels = set()
        for level, name in levels:
            if level in seen_levels:
                raise ValueError(f"Duplicate dimension level {level} found for '{name}'")
            seen_levels.add(level)
            
        # Check start and sequence
        if levels[0][0] != 1:
            raise ValueError(f"Dimension levels must start at 1, found {levels[0][0]} for '{levels[0][1]}'")
            
        for i in range(len(levels) - 1):
            if levels[i+1][0] != levels[i][0] + 1:
                raise ValueError(f"Gap in dimension levels between {levels[i][0]} and {levels[i+1][0]}")

    def get_dim_by_level(self, level: int) -> Optional[DataDimension]:
        """Get dimension object by level."""
        for dim in self.get_dim_objects().values():
            if dim.level == level:
                return dim
        return None

    def get_sorted_dimensions(self) -> List[DataDimension]:
        """Get dimensions sorted by level (ascending: 1, 2, ...)."""
        dim_objs = list(self.get_dim_objects().values())
        dim_objs.sort(key=lambda x: x.level)
        return dim_objs

    def get_dimension_strides(self) -> Dict[str, int]:
        """
        Calculate stride (block size) for each dimension level.
        
        Stride for level L is the product of sizes of all levels > L.
        Lowest level (highest index) has stride 1.
        """
        sorted_dims = self.get_sorted_dimensions()
        strides = {}
        current_stride = 1
        
        # Iterate backwards (from lowest level / highest index)
        for dim in reversed(sorted_dims):
            strides[dim.code] = current_stride
            # Get size of this dimension
            size = self.get_value(dim.code)
            current_stride *= size
        return strides
    
    def get_start_and_end_indices(self, dimension: str, step_index: int) -> Tuple[int, int]:
        # Calculate range based on explicit step_index
        strides = self.get_dimension_strides()
        if dimension not in strides:
             raise ValueError(f"Dimension '{dimension}' not found in experiment parameters.")
             
        stride = strides[dimension]
        start = step_index * stride
        end = (step_index + 1) * stride
        return start, end

    def get_dim_combinations(self, dim_codes: List[str], evaluate_from: int = 0, evaluate_to: Optional[int] = None) -> List[Tuple[int, ...]]:
        """Get all combinations of dimension indices for specified dimensions and the respective iterator names."""
        # Extract dimension values
        dim_values = self.get_dim_values()
        dim_values = [dim_values[dim] for dim in dim_codes if dim in dim_values]
        if len(dim_values) < len(dim_codes):
            missing_dims = [dim for dim in dim_codes if dim not in dim_values]
            raise KeyError(f"Missing dimensions: {', '.join(missing_dims)}")

        # Generate dimensional combinations
        dim_ranges = [range(size) for size in dim_values]
        dim_combinations = list(itertools.product(*dim_ranges))

        # Slice combinations if needed
        if evaluate_to is None:
            dim_combinations = dim_combinations[evaluate_from:]
        else:
            dim_combinations = dim_combinations[evaluate_from:evaluate_to]
        return dim_combinations
    
    def filter_for_dims(self, dim_codes: List[str]) -> List[str]:
        """Create new Parameters block containing only specified dimensions."""
        return [name for name, obj in self.data_objects.items() if name in dim_codes and isinstance(obj, DataDimension)]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameters':
        """Reconstruct from dictionary including weights."""
        return super().from_dict(Parameters, data)
    
    @classmethod
    def from_list(cls, data_objs: List[Any]) -> 'Parameters':
        """Reconstruct from list including weights."""
        return super().from_list(Parameters, data_objs)

class PerformanceAttributes(DataBlock):
    """
    Evaluation outputs (performance metrics).
    
    Includes calibration weights for multi-objective optimization.
    Examples: temperature deviation, path accuracy, energy consumption.
    """
    
    def __init__(self):
        """Initialize PerformanceAttributes block."""
        super().__init__(PerformanceAttribute)
        self.calibration_weights: Dict[str, float] = {}
    
    def set_weight(self, perf_code: str, weight: float) -> None:
        """Set calibration weight for a performance attribute."""
        if perf_code not in self.data_objects:
            raise KeyError(f"Performance code '{perf_code}' not defined")
        
        if weight < 0:
            raise ValueError(f"Calibration weight must be non-negative, got {weight}")
        
        self.calibration_weights[perf_code] = weight
    
    def get_weight(self, perf_code: str) -> Optional[float]:
        """Get calibration weight for a performance attribute."""
        return self.calibration_weights.get(perf_code)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize including calibration weights."""
        data = super().to_dict()
        data["calibration_weights"] = self.calibration_weights
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceAttributes':
        """Reconstruct from dictionary including weights."""
        block = cls()
        for name, obj_data in data.get("data_objects", {}).items():
            data_obj = DataReal.from_dict(obj_data)
            block.add(name, data_obj)
        block.calibration_weights = data.get("calibration_weights", {})
        # Restore values if present
        for name, value in data.get("values", {}).items():
            if name in block.data_objects:
                block.set_value(name, value)
        return block
    
    @classmethod
    def from_list(cls, data_objs: List[Any]) -> 'PerformanceAttributes':
        """Reconstruct from list including weights."""
        return super().from_list(PerformanceAttributes, data_objs)

class Features(DataBlock):
    """
    Multi-dimensional metric arrays using DataArray objects.
    
    Stores numpy arrays with validation for feature extraction and evaluation results.
    Each DataArray wraps a numpy array with shape/dtype constraints.
    """
    
    def __init__(self):
        """Initialize MetricArrays block."""
        super().__init__(Feature)

    def set_dim_codes(self, metric_code: str, dim_codes: List[str]) -> None:
        """Set associated dimension codes for a given metric array."""
        if metric_code not in self.data_objects:
            raise KeyError(f"Metric array code '{metric_code}' not defined")
        
        data_array = self.data_objects[metric_code]
        if not isinstance(data_array, DataArray):
            raise TypeError(f"DataObject for code '{metric_code}' is not a DataArray")
        
        # Set dimension codes for the DataArray object
        data_array.set_dim_codes(dim_codes)

    def initialize_array(self, metric_code: str, shape: Tuple[int, ...]):
        """Initialize numpy array for a given metric code."""
        if metric_code not in self.data_objects:
            raise KeyError(f"Metric array code '{metric_code}' not defined")
        
        if metric_code in self.values:
            raise ValueError(f"Metric array '{metric_code}' already initialized")
        
        # Create empty numpy array with specified shape
        self.set_value(metric_code, np.empty(shape))
        self.data_objects[metric_code].set_shape_constraint(shape) # type: ignore

    def initialize_arrays(self, parameters: Parameters) -> None:
        """Initialize all metric arrays with the same shape."""
        for metric_code in self.data_objects.keys():
            data_array = self.data_objects[metric_code]
            if not isinstance(data_array, DataArray):
                raise TypeError(f"DataObject for code '{metric_code}' is not a DataArray")
            if data_array.dim_codes is None:
                raise ValueError(f"Dimension codes not set for metric array '{metric_code}'")
            dim_values = parameters.get_dim_values()
            shape = (int(np.prod(list(dim_values.values()))), len(dim_values) + 1)
            self.initialize_array(metric_code, shape)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Features':
        """Reconstruct from dictionary including weights."""
        return super().from_dict(Features, data)
    
    @classmethod
    def from_list(cls, data_objs: List[Any]) -> 'Features':
        """Reconstruct from list including weights."""
        return super().from_list(Features, data_objs)