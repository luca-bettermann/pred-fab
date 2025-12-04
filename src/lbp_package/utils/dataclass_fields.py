"""
Utilities for defining AIXD schema-aware dataclasses.

This module provides factory classes for creating dataclass fields with attached
DataObject schema metadata, and a mixin for handling these fields.
"""

from typing import Any, List, Optional, Dict
from dataclasses import field, fields, dataclass
from lbp_package.core.data_objects import (
    DataObject, DataReal, DataInt, DataCategorical, DataBool, DataDimension
)


class Parameter:
    """Factory for creating parameter DataObjects."""
    
    @staticmethod
    def real(min_val: Optional[float] = None, max_val: Optional[float] = None,
             round_digits: Optional[int] = None, default: Any = None) -> Any:
        """Create a real-valued parameter."""
        schema = DataReal(name="", min_val=min_val, max_val=max_val, round_digits=round_digits)
        return field(default=default, metadata={'role': 'parameter', 'schema': schema})
    
    @staticmethod
    def integer(min_val: Optional[int] = None, max_val: Optional[int] = None, default: Any = None) -> Any:
        """Create an integer parameter."""
        schema = DataInt(name="", min_val=min_val, max_val=max_val)
        return field(default=default, metadata={'role': 'parameter', 'schema': schema})
    
    @staticmethod
    def categorical(categories: List[str], default: Any = None) -> Any:
        """Create a categorical parameter."""
        schema = DataCategorical(name="", categories=categories)
        return field(default=default, metadata={'role': 'parameter', 'schema': schema})
    
    @staticmethod
    def boolean(default: Any = None) -> Any:
        """Create a boolean parameter."""
        schema = DataBool(name="")
        return field(default=default, metadata={'role': 'parameter', 'schema': schema})


class Performance:
    """Factory for creating performance attribute DataObjects."""
    
    @staticmethod
    def real(min_val: Optional[float] = None, max_val: Optional[float] = None,
             round_digits: Optional[int] = None) -> Any:
        """Create a real-valued performance attribute."""
        schema = DataReal(name="", min_val=min_val, max_val=max_val, round_digits=round_digits)
        return field(default=None, metadata={'role': 'performance', 'schema': schema})
    
    @staticmethod
    def integer(min_val: Optional[int] = None, max_val: Optional[int] = None) -> Any:
        """Create an integer performance attribute."""
        schema = DataInt(name="", min_val=min_val, max_val=max_val)
        return field(default=None, metadata={'role': 'performance', 'schema': schema})


class Dimension:
    """Factory for creating dimensional parameter DataObjects."""
    
    @staticmethod
    def integer(param_name: str, dim_name: str, iterator_name: str, 
                min_val: int = 1, max_val: Optional[int] = None) -> Any:
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
        return field(default=None, metadata={'role': 'dimension', 'schema': dim_obj})


@dataclass
class DataclassMixin:
    """
    Mixin to automatically populate dataclass fields based on AIXD metadata.
    """

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Populate fields marked as 'parameter' from a dictionary.
        Performs validation against the defined DataObject schema.
        """
        for f in fields(self):
            # Check if this field is a Parameter
            if f.metadata.get('role') == 'parameter':
                if f.name in params:
                    value = params[f.name]
                    
                    # Optional: Validate using the schema stored in metadata
                    schema: Optional[DataObject] = f.metadata.get('schema')
                    if schema:
                        # We temporarily set the name to the field name for meaningful error messages
                        schema.name = f.name 
                        schema.validate(value)
                    
                    setattr(self, f.name, value)

    def set_dimensions(self, dims: Dict[str, Any]) -> None:
        """Populate fields marked as 'dimension'."""
        for f in fields(self):
            if f.metadata.get('role') == 'dimension':
                # Dimensions might be passed by their param_name (e.g. 'n_layers') 
                # or the field name.
                schema = f.metadata.get('schema')
                target_key = schema.dim_param_name if schema else f.name
                
                if target_key in dims:
                    value = dims[target_key]
                    if schema:
                        schema.name = f.name
                        schema.validate(value)
                    setattr(self, f.name, value)

    def get_schema_objects(self) -> Dict[str, DataObject]:
        """
        Extract the DataObjects for schema generation.
        Used by the Agent to build the DatasetSchema automatically.
        """
        schema_objs = {}
        for f in fields(self):
            if 'schema' in f.metadata:
                schema_obj = f.metadata['schema']
                # Update the name to match the class attribute name
                schema_obj.name = f.name 
                schema_objs[f.name] = schema_obj
        return schema_objs
