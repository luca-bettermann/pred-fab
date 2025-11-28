"""
DataObject type system for AIXD schema definitions.

DataObjects represent variable types in dataset schemas and hold actual values.
They provide validation, type information, and value storage.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Tuple, Literal
import numpy as np


NormalizeStrategy = Literal['default', 'standard', 'minmax', 'robust', 'none', 'categorical']


class DataObject(ABC):
    """
    Base class for schema type definitions with value storage.
    
    Represents a variable's type, constraints, and holds actual value.
    """
    
    def __init__(self, name: str, dtype: type, constraints: Optional[Dict[str, Any]] = None, 
                 round_digits: Optional[int] = None):
        """Initialize DataObject with name, dtype, constraints, and optional rounding."""
        self.name = name
        self.dtype = dtype
        self.constraints = constraints or {}
        self.round_digits = round_digits
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """
        Get normalization strategy for this data type.
        
        Returns:
            - 'default': Use DataModule's default normalization
            - 'standard': Standard scaling (mean=0, std=1)
            - 'minmax': Min-max scaling to [0, 1]
            - 'robust': Robust scaling using median and IQR
            - 'none': No normalization
            - 'categorical': One-hot encoding (for categorical data)
        """
        return 'default'
    
    @abstractmethod
    def validate(self, value: Any) -> bool:
        """
        Validate value against type and constraints.
        
        Args:
            value: Value to validate
            
        Returns:
            True if valid
            
        Raises:
            TypeError: If value has wrong type
            ValueError: If value violates constraints
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for schema storage."""
        data = {
            "name": self.name,
            "type": self.__class__.__name__,
            "dtype": self.dtype.__name__,
            "constraints": self.constraints
        }
        if self.round_digits is not None:
            data["round_digits"] = self.round_digits
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataObject':
        """Reconstruct from dictionary."""
        obj_type = data["type"]
        name = data["name"]
        constraints = data["constraints"]
        round_digits = data.get("round_digits")
        
        # Map type name to class
        type_map = {
            "DataReal": DataReal,
            "DataInt": DataInt,
            "DataBool": DataBool,
            "DataCategorical": DataCategorical,
            "DataDimension": DataDimension,
            "DataArray": DataArray
        }
        
        if obj_type not in type_map:
            raise ValueError(f"Unknown DataObject type: {obj_type}")
        
        return type_map[obj_type]._from_dict_impl(name, constraints, round_digits)
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any], 
                       round_digits: Optional[int] = None) -> 'DataObject':
        """Implementation-specific reconstruction. Override in subclasses."""
        raise NotImplementedError


class DataReal(DataObject):
    """Floating-point numeric parameter."""
    
    def __init__(self, name: str, min_val: Optional[float] = None, max_val: Optional[float] = None,
                 round_digits: Optional[int] = None):
        """Initialize DataReal with optional min/max bounds and rounding."""
        constraints = {}
        if min_val is not None:
            constraints["min"] = min_val
        if max_val is not None:
            constraints["max"] = max_val
        super().__init__(name, float, constraints, round_digits)
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Use DataModule default normalization (typically 'standard')."""
        return 'default'
    
    def validate(self, value: Any) -> bool:
        """Validate float value against constraints."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be numeric, got {type(value).__name__}")
        
        value = float(value)
        
        if "min" in self.constraints and value < self.constraints["min"]:
            raise ValueError(f"{self.name}={value} below minimum {self.constraints['min']}")
        
        if "max" in self.constraints and value > self.constraints["max"]:
            raise ValueError(f"{self.name}={value} above maximum {self.constraints['max']}")
        
        return True
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataReal':
        return cls(name, constraints.get("min"), constraints.get("max"), round_digits)


class DataInt(DataObject):
    """Integer numeric parameter."""
    
    def __init__(self, name: str, min_val: Optional[int] = None, max_val: Optional[int] = None,
                 round_digits: Optional[int] = None):
        """Initialize DataInt with optional min/max bounds."""
        constraints = {}
        if min_val is not None:
            constraints["min"] = min_val
        if max_val is not None:
            constraints["max"] = max_val
        super().__init__(name, int, constraints, round_digits)
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Use DataModule default normalization (typically 'standard')."""
        return 'default'
    
    def validate(self, value: Any) -> bool:
        """Validate integer value against constraints."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{self.name} must be int, got {type(value).__name__}")
        
        if "min" in self.constraints and value < self.constraints["min"]:
            raise ValueError(f"{self.name}={value} below minimum {self.constraints['min']}")
        
        if "max" in self.constraints and value > self.constraints["max"]:
            raise ValueError(f"{self.name}={value} above maximum {self.constraints['max']}")
        
        return True
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataInt':
        return cls(name, constraints.get("min"), constraints.get("max"), round_digits)


class DataBool(DataObject):
    """Boolean parameter."""
    
    def __init__(self, name: str):
        """Initialize DataBool."""
        super().__init__(name, bool, {})
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """No normalization needed - already 0/1."""
        return 'none'
    
    def validate(self, value: Any) -> bool:
        """Validate boolean value."""
        if not isinstance(value, bool):
            raise TypeError(f"{self.name} must be bool, got {type(value).__name__}")
        return True
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataBool':
        return cls(name)


class DataCategorical(DataObject):
    """Categorical parameter with fixed set of allowed values."""
    
    def __init__(self, name: str, categories: List[str]):
        """Initialize DataCategorical with allowed categories."""
        if not categories:
            raise ValueError("Categories list cannot be empty")
        super().__init__(name, str, {"categories": categories})
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Categorical data requires one-hot encoding."""
        return 'categorical'
    
    def validate(self, value: Any) -> bool:
        """Validate value is in allowed categories."""
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be str, got {type(value).__name__}")
        
        if value not in self.constraints["categories"]:
            raise ValueError(
                f"{self.name}={value} not in allowed categories: {self.constraints['categories']}"
            )
        
        return True
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataCategorical':
        return cls(name, constraints["categories"])


class DataDimension(DataObject):
    """
    Three-aspect dimensional parameter for iteration.
    
    Maps the three related dimensional concepts:
    - dim_name: Human-readable name (e.g., "layers")
    - dim_param_name: Parameter name for size (e.g., "n_layers")
    - dim_iterator_name: Iterator variable name (e.g., "layer_id")
    """
    
    def __init__(self, dim_name: str, dim_param_name: str, dim_iterator_name: str):
        """Initialize DataDimension with three naming aspects."""
        self.dim_name = dim_name
        self.dim_param_name = dim_param_name
        self.dim_iterator_name = dim_iterator_name
        
        constraints = {
            "dim_name": dim_name,
            "dim_iterator_name": dim_iterator_name
        }
        
        # dim_param_name is the actual parameter (must be positive integer)
        super().__init__(dim_param_name, int, constraints)
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Dimensional indices use minmax to preserve ordinal structure."""
        return 'minmax'
    
    def validate(self, value: Any) -> bool:
        """Validate dimensional parameter is positive integer."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"Dimensional parameter {self.name} must be int, got {type(value).__name__}"
            )
        
        if value < 1:
            raise ValueError(f"Dimensional parameter {self.name}={value} must be positive")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize including all three aspects."""
        data = super().to_dict()
        data.update({
            "dim_name": self.dim_name,
            "dim_param_name": self.dim_param_name,
            "dim_iterator_name": self.dim_iterator_name
        })
        return data
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataDimension':
        return cls(
            constraints["dim_name"],
            name,  # dim_param_name
            constraints["dim_iterator_name"]
        )


class DataArray(DataObject):
    """
    Wrapper for numpy arrays with shape and dtype validation.
    
    Used for metric_arrays storage in ExperimentData.
    """
    
    def __init__(self, name: str, shape: Optional[Tuple[int, ...]] = None, dtype: Optional[np.dtype] = None):
        """Initialize DataArray with optional shape and dtype constraints."""
        self.shape_constraint = shape
        self.dtype_constraint = dtype or np.float64
        super().__init__(name, np.ndarray, {
            "shape": shape,
            "dtype": str(dtype) if dtype else "float64"
        })
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Use DataModule default normalization (typically 'standard')."""
        return 'default'
    
    def validate(self, value: Any) -> bool:
        """Validate numpy array against shape and dtype constraints."""
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{self.name} must be np.ndarray, got {type(value).__name__}")
        
        # Validate dtype if specified
        if self.dtype_constraint and value.dtype != self.dtype_constraint:
            raise ValueError(
                f"{self.name} dtype mismatch: expected {self.dtype_constraint}, got {value.dtype}"
            )
        
        # Validate shape if specified (allowing -1 for dynamic dimensions)
        if self.shape_constraint:
            if len(value.shape) != len(self.shape_constraint):
                raise ValueError(
                    f"{self.name} shape rank mismatch: expected {len(self.shape_constraint)}D, got {len(value.shape)}D"
                )
            for i, (expected, actual) in enumerate(zip(self.shape_constraint, value.shape)):
                if expected != -1 and expected != actual:
                    raise ValueError(
                        f"{self.name} shape mismatch at dimension {i}: expected {expected}, got {actual}"
                    )
        
        return True
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataArray':
        shape = constraints.get("shape")
        dtype_str = constraints.get("dtype", "float64")
        dtype = np.dtype(dtype_str) if dtype_str else None
        return cls(name, shape, dtype)


# === FACTORY CLASSES ===

class Parameter:
    """Factory for creating parameter DataObjects."""
    
    @staticmethod
    def real(min_val: Optional[float] = None, max_val: Optional[float] = None,
             round_digits: Optional[int] = None) -> DataReal:
        """Create a real-valued parameter."""
        return DataReal(name="", min_val=min_val, max_val=max_val, round_digits=round_digits)
    
    @staticmethod
    def integer(min_val: Optional[int] = None, max_val: Optional[int] = None) -> DataInt:
        """Create an integer parameter."""
        return DataInt(name="", min_val=min_val, max_val=max_val)
    
    @staticmethod
    def categorical(categories: List[str]) -> DataCategorical:
        """Create a categorical parameter."""
        return DataCategorical(name="", categories=categories)
    
    @staticmethod
    def boolean() -> DataBool:
        """Create a boolean parameter."""
        return DataBool(name="")


class Performance:
    """Factory for creating performance attribute DataObjects."""
    
    @staticmethod
    def real(min_val: Optional[float] = None, max_val: Optional[float] = None,
             round_digits: Optional[int] = None) -> DataReal:
        """Create a real-valued performance attribute."""
        return DataReal(name="", min_val=min_val, max_val=max_val, round_digits=round_digits)
    
    @staticmethod
    def integer(min_val: Optional[int] = None, max_val: Optional[int] = None) -> DataInt:
        """Create an integer performance attribute."""
        return DataInt(name="", min_val=min_val, max_val=max_val)


class Dimension:
    """Factory for creating dimensional parameter DataObjects."""
    
    @staticmethod
    def integer(param_name: str, dim_name: str, iterator_name: str, 
                min_val: int = 1, max_val: Optional[int] = None) -> DataDimension:
        """
        Create a dimensional parameter.
        
        Args:
            param_name: Parameter name (e.g., "n_layers")
            dim_name: Human-readable dimension name (e.g., "layers")
            iterator_name: Iterator variable name (e.g., "layer_id")
            min_val: Minimum value (default 1)
            max_val: Maximum value (optional)
        """
        dim_obj = DataDimension(dim_name, param_name, iterator_name)
        # Add min/max constraints
        if min_val is not None:
            dim_obj.constraints["min"] = min_val
        if max_val is not None:
            dim_obj.constraints["max"] = max_val
        return dim_obj
