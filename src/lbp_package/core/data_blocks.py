"""
DataBlock collections for organizing related parameters.

DataBlocks group DataObjects into logical collections representing
different parameter categories in a dataset schema.
"""

from typing import Dict, Any, Optional
from .data_objects import DataObject


class DataBlock:
    """
    Base container for organizing related DataObjects.
    
    Provides validation and access methods for typed parameter collections.
    """
    
    def __init__(self):
        """Initialize empty DataBlock."""
        self.data_objects: Dict[str, DataObject] = {}
    
    def add(self, name: str, data_obj: DataObject) -> None:
        """
        Add a DataObject to the block.
        
        Args:
            name: Parameter name (key)
            data_obj: DataObject instance defining type and constraints
        """
        self.data_objects[name] = data_obj
    
    def get(self, name: str) -> DataObject:
        """
        Get a DataObject by name.
        
        Args:
            name: Parameter name
            
        Returns:
            DataObject instance
            
        Raises:
            KeyError: If name not found
        """
        return self.data_objects[name]
    
    def has(self, name: str) -> bool:
        """Check if parameter exists in block."""
        return name in self.data_objects
    
    def validate_value(self, name: str, value: Any) -> bool:
        """
        Validate a value against the corresponding DataObject.
        
        Args:
            name: Parameter name
            value: Value to validate
            
        Returns:
            True if valid
            
        Raises:
            KeyError: If name not in block
            TypeError: If value has wrong type
            ValueError: If value violates constraints
        """
        if name not in self.data_objects:
            raise KeyError(f"Parameter '{name}' not defined in {self.__class__.__name__}")
        
        return self.data_objects[name].validate(value)
    
    def validate_all(self, values: Dict[str, Any]) -> bool:
        """
        Validate multiple values at once.
        
        Args:
            values: Dictionary mapping parameter names to values
            
        Returns:
            True if all valid
            
        Raises:
            TypeError/ValueError: If any validation fails
        """
        for name, value in values.items():
            self.validate_value(name, value)
        return True
    
    def keys(self):
        """Return iterator over parameter names."""
        return self.data_objects.keys()
    
    def values(self):
        """Return iterator over DataObjects."""
        return self.data_objects.values()
    
    def items(self):
        """Return iterator over (name, DataObject) pairs."""
        return self.data_objects.items()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for schema storage."""
        return {
            "type": self.__class__.__name__,
            "data_objects": {
                name: obj.to_dict() 
                for name, obj in self.data_objects.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataBlock':
        """Reconstruct from dictionary."""
        block = cls()
        for name, obj_data in data.get("data_objects", {}).items():
            data_obj = DataObject.from_dict(obj_data)
            block.add(name, data_obj)
        return block


class ParametersStatic(DataBlock):
    """
    Fixed parameters shared across all experiments in a dataset.
    
    Sourced from study_params during dataset initialization.
    Examples: target values, physical constants, study-level configuration.
    """
    
    def __init__(self):
        """Initialize ParametersStatic block."""
        super().__init__()


class ParametersDynamic(DataBlock):
    """
    Parameters that vary per experiment.
    
    Sourced from exp_params in experiment records.
    Examples: process parameters, design variables (excluding dimensional params).
    """
    
    def __init__(self):
        """Initialize ParametersDynamic block."""
        super().__init__()


class ParametersDimensional(DataBlock):
    """
    Runtime iteration variables using DataDimension.
    
    Defines dimensional structure for multi-dimensional evaluation.
    Examples: layer_id, segment_id with three-aspect mapping.
    """
    
    def __init__(self):
        """Initialize ParametersDimensional block."""
        super().__init__()


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
        """
        Set calibration weight for a performance attribute.
        
        Args:
            perf_code: Performance code (must exist in data_objects)
            weight: Calibration weight (typically 0.0 to 1.0)
            
        Raises:
            KeyError: If perf_code not defined
            ValueError: If weight is negative
        """
        if perf_code not in self.data_objects:
            raise KeyError(f"Performance code '{perf_code}' not defined")
        
        if weight < 0:
            raise ValueError(f"Calibration weight must be non-negative, got {weight}")
        
        self.calibration_weights[perf_code] = weight
    
    def get_weight(self, perf_code: str) -> Optional[float]:
        """
        Get calibration weight for a performance attribute.
        
        Args:
            perf_code: Performance code
            
        Returns:
            Weight value or None if not set
        """
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
        return block
