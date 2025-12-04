"""
DataBlock collections for organizing related parameters.

DataBlocks group DataObjects into logical collections and store their values.
They provide both schema structure and data storage.
"""

import itertools
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from .data_objects import DataObject


class DataBlock:
    """
    Container for DataObjects with value storage.
    
    Provides validation, access methods, and value management for typed parameter collections.
    """
    
    def __init__(self):
        """Initialize empty DataBlock."""
        self.data_objects: Dict[str, DataObject] = {}  # Schema structure
        self.values: Dict[str, Any] = {}  # Actual values
    
    def add(self, name: str, data_obj: DataObject) -> None:
        """Add a DataObject to the block."""
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
        """
        Convert all values to numpy array for ML.
        
        Args:
            dtype: Target numpy dtype (default: float64)
            
        Returns:
            Numpy array of values
            
        Raises:
            ValueError: If non-numeric values are present
        """
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
    def from_dict(cls, data: Dict[str, Any]) -> 'DataBlock':
        """Reconstruct from dictionary."""
        block = cls()
        for name, obj_data in data.get("data_objects", {}).items():
            data_obj = DataObject.from_dict(obj_data)
            block.add(name, data_obj)
        # Restore values if present
        for name, value in data.get("values", {}).items():
            if name in block.data_objects:
                block.set_value(name, value)
        return block


class Parameters(DataBlock):
    """
    Unified parameter block for ALL parameters (replaces Static/Dynamic split).
    
    Parameters may be:
    - Static: Same value across all experiments in dataset
    - Dynamic: Vary per experiment
    - Dimensional: Subset of parameters used for iteration
    """
    
    def __init__(self):
        """Initialize Parameters block."""
        super().__init__()


class Dimensions(DataBlock):
    """
    Dimensional metadata block using DataDimension objects.
    
    References parameters from Parameters block that define iteration structure.
    Each DataDimension contains: param_name, dim_name, iterator_name.
    """
    
    def __init__(self):
        """Initialize Dimensions block."""
        super().__init__()

    def get_dim_combinations(self) -> List[Tuple[int, ...]]:
        """Get mapping of param_name to iterator_name for all dimensions."""
        # Extract dimension values
        dim_values = [value for value in self.values.values()]

        # Generate dimensional combinations
        dim_ranges = [range(size) for size in dim_values]
        return list(itertools.product(*dim_ranges))


class PerformanceAttributes(DataBlock):
    """
    Evaluation outputs (performance metrics).
    
    Includes calibration weights for multi-objective optimization.
    Examples: temperature deviation, path accuracy, energy consumption.
    """
    
    def __init__(self):
        """Initialize PerformanceAttributes block."""
        super().__init__()
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
            data_obj = DataObject.from_dict(obj_data)
            block.add(name, data_obj)
        block.calibration_weights = data.get("calibration_weights", {})
        # Restore values if present
        for name, value in data.get("values", {}).items():
            if name in block.data_objects:
                block.set_value(name, value)
        return block


class MetricArrays(DataBlock):
    """
    Multi-dimensional metric arrays using DataArray objects.
    
    Stores numpy arrays with validation for feature extraction and evaluation results.
    Each DataArray wraps a numpy array with shape/dtype constraints.
    """
    
    def __init__(self):
        """Initialize MetricArrays block."""
        super().__init__()
