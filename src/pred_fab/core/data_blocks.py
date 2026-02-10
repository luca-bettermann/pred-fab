"""
DataBlock collections for organizing related parameters.

DataBlocks group DataObjects into logical collections and store their values.
They provide both schema structure and data storage.
"""

from abc import abstractmethod, ABC
import itertools
from typing import Dict, Any, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
from .data_objects import DataObject, DataDimension, DataArray, DataReal
from .data_objects import Parameter, Feature, PerformanceAttribute
from ..utils.enum import Roles
from ..utils.logger import PfabLogger


class DataBlock(ABC):
    """
    Container for DataObjects with value storage.
    
    Provides validation, access methods, and value management for typed parameter collections.
    """
    
    def __init__(self):
        """Initialize empty DataBlock."""
        self.data_objects: Dict[str, DataObject] = {}  # Schema structure
        self.values: Dict[str, Any] = {}  # Actual values
        self.populated_status: Dict[str, bool] = {} # Track if value is populated or just initialized

    @property
    @abstractmethod
    def role(self) -> Roles:
        ...
    
    def add(self, name: str, data_obj: DataObject) -> None:
        """Add a DataObject to the block."""
        if not issubclass(type(data_obj), DataObject):
            raise TypeError(f"data_obj must be a DataObject, got {type(data_obj)}")
            
        if self.role and data_obj.role != self.role:
            raise ValueError(f"Expected data object with role '{self.role}', got '{data_obj.role}'")
            
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
    
    def set_value(self, name: str, value: Any, as_populated: bool = True) -> None:
        """Set value for a parameter after validation."""
        self.validate_value(name, value)  # Raises if invalid
        
        # Apply rounding for numeric types if configured
        data_obj = self.get(name)
        if data_obj.round_digits is not None and isinstance(value, (int, float)):
            value = round(float(value), data_obj.round_digits)
        
        self.values[name] = value
        self.populated_status[name] = as_populated

    def set_values_from_dict(self, values: Dict[str, Any], logger: PfabLogger, as_populated: bool = True) -> None:
        """Set multiple values at once, ignoring unknown parameters."""
        for name, value in values.items():
            if self.has(name):
                self.set_value(name, value, as_populated=as_populated)
            else:
                logger.warning(f"Object '{name}' not found in {self.__class__}. Skip assigning value: {value}.")

    def set_values_from_df(self, df: pd.DataFrame, logger: PfabLogger, as_populated: bool = True) -> None:
        columns: List[str] = df.columns # type: ignore
        name: str = columns[-1]
        array = df.values

        # set array
        if self.has(name):
            self.set_value(name, array, as_populated=as_populated)
            
            # set dims
            obj = self.data_objects[name]
            if isinstance(obj, DataArray):
                obj.set_columns(list(columns))
            else:
                raise ValueError(f"Object has wrong type. Expected 'DataArray', got {obj.__class__}.")
        else:
            logger.warning(f"Object '{name}' not found in {self.__class__}. Skip assigning array.")            
    
    def get_value(self, name: str) -> Any:
        """Get value for a parameter."""
        if name not in self.values:
            raise KeyError(f"No value set for parameter '{name}'")
        return self.values[name]
        # return self.data_objects[name].dtype(self.values[name])
    
    def has_value(self, name: str) -> bool:
        """Check if value is set for parameter."""
        return name in self.values
    
    def is_populated(self, name: str) -> bool:
        """Check if value is populated (not just initialized)."""
        return self.populated_status.get(name, False)
    
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
    
    def items(self) -> Any:
        """Return iterator over (name, DataObject) pairs."""
        return self.data_objects.items()
    
    def get_values_dict(self) -> Dict[str, Any]:
        """Extract all set values as a simple dictionary."""
        return dict(self.values)
    
    def is_compatible(self, other_block) -> bool:
        """Helper function to check compatibility between two data blocks."""
        self_objects = set(self.data_objects)
        other_objects = set(other_block.data_objects)
        if self_objects != other_objects:
            return False
        return True
    
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """Reconstruct from dictionary."""
        # Create instance (subclass sets its own types)
        block = cls()

        # Restore DataObjects
        for name, obj_data in data.get("data_objects", {}).items():
            data_obj = DataObject.from_dict(obj_data)
            block.add(name, data_obj)

        # Restore values if present
        for name, value in data.get("values", {}).items():
            if name in block.data_objects:
                block.set_value(name, value)
        return block
    
    @classmethod
    def from_list(cls, data_objs: List[Any]) -> Any:
        """Reconstruct from list."""
        # Create instance (subclass sets its own types)
        block = cls()

        # Add DataObjects
        for data_obj in data_objs:
            block.add(data_obj.code, data_obj)
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
        super().__init__()
    
    @property
    def role(self) -> Roles:
        return Roles.PARAMETER

    def get_dim_objects(self, codes: Optional[List[str]] = None) -> List[DataDimension]:
        """Get view of dimension DataObjects from parameters."""
        return [
            obj for name, obj in self.data_objects.items() 
            if isinstance(obj, DataDimension) and (codes is None or name in codes)
        ]
        
    def get_dim_names(self) -> List[str]:
        """Get list of dimension parameter names."""
        return [name for name, obj in self.data_objects.items() if isinstance(obj, DataDimension)]
    
    def get_dim_iterator_codes(self, codes: Optional[List[str]] = None) -> List[str]:
        """Get list of dimension iterator names."""
        return [obj.iterator_code for obj in self.get_dim_objects(codes)]

    def get_dim_values(self, codes: Optional[List[str]] = None) -> List[int]:
        """Get view of dimension DataObjects from parameters."""
        return [self.get_value(dim.code) for dim in self.get_dim_objects(codes)]

    def validate_dimensions(self) -> None:
        """Validate dimension levels."""
        levels = []
        for dim in self.get_dim_objects():
            levels.append((dim.level, dim.code))
        levels.sort()

        for i, (level, name) in enumerate(levels, start=1):
            if level != i:
                raise ValueError(f"Dimension levels must be consecutive starting from 1. "
                                 f"Expected level {i} for '{name}', got {level}")

    def get_dim_by_level(self, level: int) -> Optional[DataDimension]:
        """Get dimension object by level."""
        for dim in self.get_dim_objects():
            if dim.level == level:
                return dim
        return None

    def get_sorted_dimensions(self) -> List[DataDimension]:
        """Get dimensions sorted by level (ascending: 1, 2, ...)."""
        dim_objs = list(self.get_dim_objects())
        dim_objs.sort(key=lambda x: x.level)
        return dim_objs

    def _get_dimension_strides(self) -> Dict[str, int]:
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
        strides = self._get_dimension_strides()
        if dimension not in strides:
             raise ValueError(f"Dimension '{dimension}' not found in experiment parameters.")
             
        stride = strides[dimension]
        start = step_index * stride
        end = (step_index + 1) * stride
        return start, end

    def get_dim_combinations(self, dim_codes: List[str], evaluate_from: int = 0, evaluate_to: Optional[int] = None) -> List[Tuple[int, ...]]:
        """Get all combinations of dimension indices for specified dimensions and the respective iterator names."""
        # Extract dimension values
        dim_values = self.get_dim_values(dim_codes)
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

class Features(DataBlock):
    """
    Multi-dimensional metric arrays using DataArray objects.
    
    Stores numpy arrays with validation for feature extraction and evaluation results.
    Each Feature wraps a numpy array with shape/dtype constraints.
    """
    
    def __init__(self):
        """Initialize MetricArrays block."""
        super().__init__()
    
    @property
    def role(self) -> Roles:
        return Roles.FEATURE
    
    def _initialize_array(self, metric_code: str, shape: Tuple[int, ...], recompute_flag: bool) -> None:
        """Initialize numpy array for a given metric code."""
        if metric_code not in self.data_objects:
            raise KeyError(f"Metric array code '{metric_code}' not defined")
        
        # Skip if already initialized and not recomputing
        if metric_code in self.values and not recompute_flag:
            return
        
        # Get expected dtype from DataArray definition in constraints
        data_obj = self.data_objects[metric_code]
        dtype_str = data_obj.constraints.get("dtype", "float64")
        
        # Create array filled with NaNs using correct dtype
        self.set_value(metric_code, np.full(shape, np.nan, dtype=np.dtype(dtype_str)), as_populated=False)

    def initialize_arrays(self, parameters: Parameters, recompute_flag: bool = False) -> None:
        """Initialize all metric arrays with the respective shape."""
        for metric_code in self.data_objects.keys():
            data_array = self.data_objects[metric_code]
            if not isinstance(data_array, DataArray):
                raise TypeError(f"DataObject for code '{metric_code}' is not a DataArray")
            if len(data_array.columns) == 0:
                raise ValueError(f"Columns not set for metric array '{metric_code}'")
                        
            # compute shape based on parameter dimensions
            dim_codes = [obj.code for obj in parameters.get_dim_objects() if obj.iterator_code in data_array.columns]
            dim_values = parameters.get_dim_values(dim_codes)
            shape = (int(np.prod(dim_values)), len(dim_values) + 1)

            # Initialize array with computed shape
            self._initialize_array(metric_code, shape, recompute_flag)


class PerformanceAttributes(DataBlock):
    """
    Evaluation outputs (performance metrics).
    
    Includes calibration weights for multi-objective optimization.
    Examples: temperature deviation, path accuracy, energy consumption.
    """
    
    def __init__(self):
        """Initialize PerformanceAttributes block."""
        super().__init__()
    
    @property
    def role(self) -> Roles:
        return Roles.PERFORMANCE
