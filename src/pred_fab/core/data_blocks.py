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
from .data_objects import DataObject, DataDimension, DataArray
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

    def sanitize_values(
        self,
        values: Dict[str, Any],
        ignore_unknown: bool = False
    ) -> Dict[str, Any]:
        """Coerce and validate a parameter dictionary according to schema data objects."""
        sanitized: Dict[str, Any] = {}
        for code, value in values.items():
            if not self.has(code):
                if ignore_unknown:
                    sanitized[code] = value
                    continue
                raise KeyError(f"Parameter '{code}' not defined in {self.__class__.__name__}")
            obj = self.get(code)
            coerced = obj.coerce(value)
            obj.validate(coerced)
            sanitized[code] = coerced
        return sanitized

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
        """Initialize all feature tensors with shapes derived from iterator columns."""
        for metric_code in self.data_objects.keys():
            # Resolve feature schema object and required iterator columns.
            data_array = self.data_objects[metric_code]
            if not isinstance(data_array, DataArray):
                raise TypeError(f"DataObject for code '{metric_code}' is not a DataArray")
            if len(data_array.columns) == 0:
                raise ValueError(f"Columns not set for metric array '{metric_code}'")
                        
            iterator_cols = data_array.columns[:-1]
            if not iterator_cols:
                shape: Tuple[int, ...] = ()
            else:
                # Translate iterator columns to canonical tensor shape.
                dim_sizes = self._get_dim_sizes_from_iterators(parameters, iterator_cols)
                shape = tuple(dim_sizes)

            # Allocate tensor storage in canonical shape.
            self._initialize_array(metric_code, shape, recompute_flag)

    def set_values_from_df(
        self,
        df: pd.DataFrame,
        logger: PfabLogger,
        as_populated: bool = True,
        parameters: Optional[Parameters] = None
    ) -> None:
        """Set feature values from tabular format by transforming to canonical tensor format."""
        columns: List[str] = list(df.columns)  # type: ignore
        metric_code = columns[-1]
        if not self.has(metric_code):
            logger.warning(f"Object '{metric_code}' not found in {self.__class__}. Skip assigning array.")
            return
        if parameters is None:
            raise ValueError("Parameters are required to convert tabular feature values to tensor representation.")

        tensor = self.table_to_tensor(metric_code, df.values, parameters)
        self.set_value(metric_code, tensor, as_populated=as_populated)

    def table_to_tensor(self, feature_code: str, table: np.ndarray, parameters: Parameters) -> np.ndarray:
        """Convert tabular feature representation [iterators..., value] to canonical tensor."""
        # Read schema metadata to determine iterator columns and dtype.
        data_obj = self.get(feature_code)
        if not isinstance(data_obj, DataArray):
            raise TypeError(f"Feature '{feature_code}' is not backed by DataArray.")

        iterator_cols = data_obj.columns[:-1]
        dtype = np.dtype(data_obj.constraints.get("dtype", "float64"))
        arr = np.asarray(table, dtype=dtype)

        if not iterator_cols:
            # Handle scalar features represented without iterator columns.
            if arr.ndim == 0:
                return arr
            if arr.ndim == 1:
                if arr.size == 0:
                    return np.array(np.nan, dtype=dtype)
                return np.array(arr.flat[0], dtype=dtype)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                return np.array(arr[0, -1], dtype=dtype)
            raise ValueError(f"Invalid table shape for scalar feature '{feature_code}': {arr.shape}")

        dim_sizes = self._get_dim_sizes_from_iterators(parameters, iterator_cols)
        tensor = np.full(tuple(dim_sizes), np.nan, dtype=dtype)

        if arr.ndim != 2 or arr.shape[1] < len(iterator_cols) + 1:
            raise ValueError(
                f"Invalid table shape for feature '{feature_code}'. "
                f"Expected [n_rows, {len(iterator_cols)+1}], got {arr.shape}."
            )

        # Fill tensor cells from tabular index/value rows.
        for row in arr:
            idx = tuple(int(round(row[i])) for i in range(len(iterator_cols)))
            if any(dim < 0 or dim >= tensor.shape[i] for i, dim in enumerate(idx)):
                raise ValueError(
                    f"Row index {idx} out of bounds for feature '{feature_code}' with shape {tensor.shape}."
                )
            tensor[idx] = row[-1]

        return tensor

    def tensor_to_table(self, feature_code: str, tensor: np.ndarray, parameters: Parameters) -> np.ndarray:
        """Convert canonical tensor feature representation to tabular [iterators..., value]."""
        # Read schema metadata to determine iterator columns and dtype.
        data_obj = self.get(feature_code)
        if not isinstance(data_obj, DataArray):
            raise TypeError(f"Feature '{feature_code}' is not backed by DataArray.")

        iterator_cols = data_obj.columns[:-1]
        dtype = np.dtype(data_obj.constraints.get("dtype", "float64"))
        arr = np.asarray(tensor, dtype=dtype)

        if not iterator_cols:
            # Emit scalar feature as a single-row one-column table.
            scalar = float(arr) if arr.ndim == 0 else float(arr.flat[0])
            return np.array([[scalar]], dtype=dtype)

        dim_codes = self._get_dim_codes_from_iterators(parameters, iterator_cols)
        dim_combinations = parameters.get_dim_combinations(dim_codes)
        table = np.empty((len(dim_combinations), len(iterator_cols) + 1), dtype=dtype)

        # Flatten tensor into row-wise iterator/value pairs.
        for i, idx in enumerate(dim_combinations):
            table[i, :len(iterator_cols)] = idx
            table[i, -1] = arr[idx]

        return table

    def value_at(self, feature_code: str, parameters: Parameters, iterator_values: Dict[str, Any]) -> Optional[float]:
        """Read one feature value from canonical tensor for given iterator coordinates."""
        # Fast-path for missing feature values.
        if not self.has_value(feature_code):
            return None
        data_obj = self.get(feature_code)
        if not isinstance(data_obj, DataArray):
            raise TypeError(f"Feature '{feature_code}' is not backed by DataArray.")

        iterator_cols = data_obj.columns[:-1]
        arr = self.get_value(feature_code)

        if not iterator_cols:
            # Scalar features have no iterator index.
            return float(arr) if np.asarray(arr).ndim == 0 else float(np.asarray(arr).flat[0])

        idx = []
        # Build tensor index from iterator column names.
        for col in iterator_cols:
            if col not in iterator_values:
                return None
            idx.append(int(iterator_values[col]))
        return float(np.asarray(arr)[tuple(idx)])

    def _get_dim_codes_from_iterators(self, parameters: Parameters, iterator_cols: List[str]) -> List[str]:
        iterator_to_dim = {dim.iterator_code: dim.code for dim in parameters.get_dim_objects()}
        dim_codes = []
        for iterator in iterator_cols:
            if iterator not in iterator_to_dim:
                raise KeyError(f"Iterator '{iterator}' not found in Parameters dimensions.")
            dim_codes.append(iterator_to_dim[iterator])
        return dim_codes

    def _get_dim_sizes_from_iterators(self, parameters: Parameters, iterator_cols: List[str]) -> List[int]:
        dim_codes = self._get_dim_codes_from_iterators(parameters, iterator_cols)
        return [int(parameters.get_value(code)) for code in dim_codes]


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
