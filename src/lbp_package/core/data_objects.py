"""
DataObject type system for AIXD schema definitions.

DataObjects represent variable types in dataset schemas, providing validation
and type information. They are NOT used for runtime model configuration 
(that's handled by ParameterHandling decorators).
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict


class DataObject(ABC):
    """
    Base class for schema type definitions.
    
    Represents a variable's type and constraints in a DatasetSchema.
    Separate from runtime model configuration (ParameterHandling).
    """
    
    def __init__(self, name: str, dtype: type, constraints: Optional[Dict[str, Any]] = None):
        """
        Initialize DataObject.
        
        Args:
            name: Variable name
            dtype: Python type (int, float, bool, str)
            constraints: Type-specific constraints (min, max, categories, etc.)
        """
        self.name = name
        self.dtype = dtype
        self.constraints = constraints or {}
    
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
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "dtype": self.dtype.__name__,
            "constraints": self.constraints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataObject':
        """Reconstruct from dictionary."""
        obj_type = data["type"]
        name = data["name"]
        constraints = data["constraints"]
        
        # Map type name to class
        type_map = {
            "DataReal": DataReal,
            "DataInt": DataInt,
            "DataBool": DataBool,
            "DataCategorical": DataCategorical,
            "DataString": DataString,
            "DataDimension": DataDimension
        }
        
        if obj_type not in type_map:
            raise ValueError(f"Unknown DataObject type: {obj_type}")
        
        return type_map[obj_type]._from_dict_impl(name, constraints)
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any]) -> 'DataObject':
        """Implementation-specific reconstruction. Override in subclasses."""
        raise NotImplementedError


class DataReal(DataObject):
    """Floating-point numeric parameter."""
    
    def __init__(self, name: str, min_val: Optional[float] = None, max_val: Optional[float] = None):
        """
        Initialize DataReal.
        
        Args:
            name: Variable name
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
        """
        constraints = {}
        if min_val is not None:
            constraints["min"] = min_val
        if max_val is not None:
            constraints["max"] = max_val
        super().__init__(name, float, constraints)
    
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
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any]) -> 'DataReal':
        return cls(name, constraints.get("min"), constraints.get("max"))


class DataInt(DataObject):
    """Integer numeric parameter."""
    
    def __init__(self, name: str, min_val: Optional[int] = None, max_val: Optional[int] = None):
        """
        Initialize DataInt.
        
        Args:
            name: Variable name
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
        """
        constraints = {}
        if min_val is not None:
            constraints["min"] = min_val
        if max_val is not None:
            constraints["max"] = max_val
        super().__init__(name, int, constraints)
    
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
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any]) -> 'DataInt':
        return cls(name, constraints.get("min"), constraints.get("max"))


class DataBool(DataObject):
    """Boolean parameter."""
    
    def __init__(self, name: str):
        """
        Initialize DataBool.
        
        Args:
            name: Variable name
        """
        super().__init__(name, bool, {})
    
    def validate(self, value: Any) -> bool:
        """Validate boolean value."""
        if not isinstance(value, bool):
            raise TypeError(f"{self.name} must be bool, got {type(value).__name__}")
        return True
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any]) -> 'DataBool':
        return cls(name)


class DataCategorical(DataObject):
    """Categorical parameter with fixed set of allowed values."""
    
    def __init__(self, name: str, categories: List[str]):
        """
        Initialize DataCategorical.
        
        Args:
            name: Variable name
            categories: List of allowed string values
        """
        if not categories:
            raise ValueError("Categories list cannot be empty")
        super().__init__(name, str, {"categories": categories})
    
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
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any]) -> 'DataCategorical':
        return cls(name, constraints["categories"])


class DataString(DataObject):
    """Arbitrary string parameter."""
    
    def __init__(self, name: str):
        """
        Initialize DataString.
        
        Args:
            name: Variable name
        """
        super().__init__(name, str, {})
    
    def validate(self, value: Any) -> bool:
        """Validate string value."""
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be str, got {type(value).__name__}")
        return True
    
    @classmethod
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any]) -> 'DataString':
        return cls(name)


class DataDimension(DataObject):
    """
    Three-aspect dimensional parameter for iteration.
    
    Maps the three related dimensional concepts:
    - dim_name: Human-readable name (e.g., "layers")
    - dim_param_name: Parameter name for size (e.g., "n_layers")
    - dim_iterator_name: Iterator variable name (e.g., "layer_id")
    """
    
    def __init__(self, dim_name: str, dim_param_name: str, dim_iterator_name: str):
        """
        Initialize DataDimension.
        
        Args:
            dim_name: Human-readable dimension name (e.g., "layers", "segments")
            dim_param_name: Parameter name defining size (e.g., "n_layers", "n_segments")
            dim_iterator_name: Iterator variable name (e.g., "layer_id", "segment_id")
        """
        self.dim_name = dim_name
        self.dim_param_name = dim_param_name
        self.dim_iterator_name = dim_iterator_name
        
        constraints = {
            "dim_name": dim_name,
            "dim_iterator_name": dim_iterator_name
        }
        
        # dim_param_name is the actual parameter (must be positive integer)
        super().__init__(dim_param_name, int, constraints)
    
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
    def _from_dict_impl(cls, name: str, constraints: Dict[str, Any]) -> 'DataDimension':
        return cls(
            constraints["dim_name"],
            name,  # dim_param_name
            constraints["dim_iterator_name"]
        )
