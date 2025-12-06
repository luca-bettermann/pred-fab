"""
DataObject type system for AIXD schema definitions.

DataObjects represent variable types in dataset schemas and hold actual values.
They provide validation, type information, and value storage.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Tuple, Literal
from dataclasses import field
import numpy as np


NormalizeStrategy = Literal['default', 'standard', 'minmax', 'robust', 'none', 'categorical']


class DataObject(ABC):
    """
    Base class for schema type definitions with value storage.
    
    Represents a variable's type, constraints, and holds actual value.
    """
    
    def __init__(self, code: str, dtype: type, constraints: Optional[Dict[str, Any]] = None, 
                 round_digits: Optional[int] = None):
        """Initialize DataObject with name, dtype, constraints, and optional rounding."""
        self.code = code
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
            "code": self.code,
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
        code = data["code"]
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
        
        return type_map[obj_type]._from_dict_impl(code, constraints, round_digits)
    
    @classmethod
    def _from_dict_impl(cls, code: str, constraints: Dict[str, Any], 
                       round_digits: Optional[int] = None) -> 'DataObject':
        """Implementation-specific reconstruction. Override in subclasses."""
        raise NotImplementedError


class DataReal(DataObject):
    """Floating-point numeric parameter."""
    
    def __init__(self, code: str, min_val: Optional[float] = None, max_val: Optional[float] = None,
                 round_digits: Optional[int] = None):
        """Initialize DataReal with optional min/max bounds and rounding."""
        constraints = {}
        if min_val is not None:
            constraints["min"] = min_val
        if max_val is not None:
            constraints["max"] = max_val
        super().__init__(code, float, constraints, round_digits)
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Use DataModule default normalization (typically 'standard')."""
        return 'default'
    
    def validate(self, value: Any) -> bool:
        """Validate float value against constraints."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.code} must be numeric, got {type(value).__name__}")
        
        value = float(value)
        
        if "min" in self.constraints and value < self.constraints["min"]:
            raise ValueError(f"{self.code}={value} below minimum {self.constraints['min']}")
        
        if "max" in self.constraints and value > self.constraints["max"]:
            raise ValueError(f"{self.code}={value} above maximum {self.constraints['max']}")
        
        return True
    
    @classmethod
    def _from_dict_impl(cls, code: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataReal':
        return cls(code, constraints.get("min"), constraints.get("max"), round_digits)


class DataInt(DataObject):
    """Integer numeric parameter."""
    
    def __init__(self, code: str, min_val: Optional[int] = None, max_val: Optional[int] = None):
        """Initialize DataInt with optional min/max bounds."""
        constraints = {}
        if min_val is not None:
            constraints["min"] = min_val
        if max_val is not None:
            constraints["max"] = max_val
        super().__init__(code, int, constraints, 0)
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Use DataModule default normalization (typically 'standard')."""
        return 'default'
    
    def validate(self, value: Any) -> bool:
        """Validate integer value against constraints."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{self.code} must be int, got {type(value).__name__}")
        
        if "min" in self.constraints and value < self.constraints["min"]:
            raise ValueError(f"{self.code}={value} below minimum {self.constraints['min']}")
        
        if "max" in self.constraints and value > self.constraints["max"]:
            raise ValueError(f"{self.code}={value} above maximum {self.constraints['max']}")
        
        return True
    
    @classmethod
    def _from_dict_impl(cls, code: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataInt':
        return cls(code, constraints.get("min"), constraints.get("max"))


class DataBool(DataObject):
    """Boolean parameter."""
    
    def __init__(self, code: str):
        """Initialize DataBool."""
        super().__init__(code, bool, {})
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """No normalization needed - already 0/1."""
        return 'none'
    
    def validate(self, value: Any) -> bool:
        """Validate boolean value."""
        if not isinstance(value, bool):
            raise TypeError(f"{self.code} must be bool, got {type(value).__name__}")
        return True
    
    @classmethod
    def _from_dict_impl(cls, code: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataBool':
        return cls(code)

class DataCategorical(DataObject):
    """Categorical parameter with fixed set of allowed values."""
    
    def __init__(self, code: str, categories: List[str]):
        """Initialize DataCategorical with allowed categories."""
        if not categories:
            raise ValueError("Categories list cannot be empty")
        super().__init__(code, str, {"categories": categories})
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Categorical data requires one-hot encoding."""
        return 'categorical'
    
    def validate(self, value: Any) -> bool:
        """Validate value is in allowed categories."""
        if not isinstance(value, str):
            raise TypeError(f"{self.code} must be str, got {type(value).__name__}")
        
        if value not in self.constraints["categories"]:
            raise ValueError(
                f"{self.code}={value} not in allowed categories: {self.constraints['categories']}"
            )
        
        return True
    
    @classmethod
    def _from_dict_impl(cls, code: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataCategorical':
        return cls(code, constraints["categories"])


class DataDimension(DataInt):
    """
    Three-aspect dimensional parameter for iteration.
    
    Maps the three related dimensional concepts:
    - code: Parameter name for size (e.g., "n_layers")
    - dim_iterator_name: Iterator variable name (e.g., "layer_id")
    """
    
    def __init__(self, code: str, iterator_code: str, min_val: int = 1, max_val: Optional[int] = None):
        """Initialize DataDimension with three naming aspects."""
        self.code = code
        self.dim_iterator_code = iterator_code        
        super().__init__(code, min_val, max_val)
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Dimensional indices use minmax to preserve ordinal structure."""
        return 'minmax'
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize including all three aspects."""
        data = super().to_dict()
        data.update({
            "code": self.code,
            "dim_iterator_code": self.dim_iterator_code
        })
        return data
    
    @classmethod
    def _from_dict_impl(cls, code: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataDimension':
        return cls(
            code,
            constraints["dim_iterator_code"],
            constraints["min_val"],
            constraints["max_val"]
        )


class DataArray(DataObject):
    """
    Wrapper for numpy arrays with shape and dtype validation.
    
    Used for metric_arrays storage in ExperimentData.
    """
    
    def __init__(self, code: str, shape: Optional[Tuple[int, ...]] = None, dtype: Optional[np.dtype] = None):
        """Initialize DataArray with optional shape and dtype constraints."""
        self.shape_constraint = shape
        self.dtype_constraint = dtype or np.float64
        self.dim_codes: Optional[List[str]] = None
        super().__init__(code, np.ndarray, {
            "dtype": str(dtype) if dtype else "float64"
        })
    
    @property
    def normalize_strategy(self) -> NormalizeStrategy:
        """Use DataModule default normalization (typically 'standard')."""
        return 'default'
    
    def set_shape_constraint(self, shape: Tuple[int, ...]) -> None:
        """Set shape constraint for this DataArray."""
        self.shape_constraint = shape
    
    def set_dim_codes(self, dim_codes: List[str]) -> None:
        """Set associated dimension codes for this DataArray."""
        self.dim_codes = dim_codes
    
    def validate(self, value: Any) -> bool:
        """Validate numpy array against shape and dtype constraints."""
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{self.code} must be np.ndarray, got {type(value).__name__}")
        
        # Validate dtype if specified
        if self.dtype_constraint and value.dtype != self.dtype_constraint:
            raise ValueError(
                f"{self.code} dtype mismatch: expected {self.dtype_constraint}, got {value.dtype}"
            )
        
        # Validate shape if specified (allowing -1 for dynamic dimensions)
        if self.shape_constraint:
            if len(value.shape) != len(self.shape_constraint):
                raise ValueError(
                    f"{self.code} shape rank mismatch: expected {len(self.shape_constraint)}D, got {len(value.shape)}D"
                )
            for i, (expected, actual) in enumerate(zip(self.shape_constraint, value.shape)):
                if expected != -1 and expected != actual:
                    raise ValueError(
                        f"{self.code} shape mismatch at dimension {i}: expected {expected}, got {actual}"
                    )
        
        return True
    
    @classmethod
    def _from_dict_impl(cls, code: str, constraints: Dict[str, Any],
                       round_digits: Optional[int] = None) -> 'DataArray':
        shape = constraints.get("shape")
        dtype_str = constraints.get("dtype", "float64")
        dtype = np.dtype(dtype_str) if dtype_str else None
        return cls(code, shape, dtype)



# -------- Factories for Parameter declaration -----------

class Parameter:
    """Factory for creating parameter DataObjects."""
    
    @staticmethod
    def real(code: str, min_val: Optional[float] = None, max_val: Optional[float] = None,
             round_digits: Optional[int] = None, default: Optional[float] = None) -> Any:
        """Create a real-valued parameter."""
        schema = DataReal(code=code, min_val=min_val, max_val=max_val, round_digits=round_digits)
        return field(default=default, metadata={'role': 'parameter', 'schema': schema})
    
    @staticmethod
    def integer(code: str, min_val: Optional[int] = None, max_val: Optional[int] = None, default: Optional[int] = None) -> Any:
        """Create an integer parameter."""
        schema = DataInt(code=code, min_val=min_val, max_val=max_val)
        return field(default=default, metadata={'role': 'parameter', 'schema': schema})
    
    @staticmethod
    def categorical(code: str, categories: List[str], default: Optional[str] = None) -> Any:
        """Create a categorical parameter."""
        schema = DataCategorical(code=code, categories=categories)
        return field(default=default, metadata={'role': 'parameter', 'schema': schema})
    
    @staticmethod
    def boolean(code: str, default: Optional[bool] = None) -> Any:
        """Create a boolean parameter."""
        schema = DataBool(code=code)
        return field(default=default, metadata={'role': 'parameter', 'schema': schema})


class Dimension:
    """Factory for creating dimensional parameter DataObjects."""
    
    @staticmethod
    def integer(code: str, iterator_code: str, 
                min_val: int = 1, max_val: Optional[int] = None) -> Any:
        """
        Create a dimensional parameter.
        
        Args:
            code: Parameter code (e.g., "n_layers")
            dim_name: Human-readable dimension name (e.g., "layers")
            iterator_name: Iterator variable name (e.g., "layer_id")
            min_val: Minimum value (default 1)
            max_val: Maximum value (optional)
        """
        dim_obj = DataDimension(code, iterator_code)
        # Add min/max constraints
        if min_val is not None:
            dim_obj.constraints["min"] = min_val
        if max_val is not None:
            dim_obj.constraints["max"] = max_val
        return field(default=None, metadata={'role': 'dimension', 'schema': dim_obj})