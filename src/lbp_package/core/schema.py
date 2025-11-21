"""
DatasetSchema for defining dataset structure.

Schema represents the structure of a dataset (what CAN exist),
not the actual values (which are stored in Dataset instances).
"""

import json
import hashlib
from typing import Dict, Any, Set
from .data_blocks import (
    ParametersStatic,
    ParametersDynamic,
    ParametersDimensional,
    PerformanceAttributes
)


class DatasetSchema:
    """
    Defines the structure and types of a dataset.
    
    Schema specifies:
    - Static parameters (study-level, shared across experiments)
    - Dynamic parameters (vary per experiment)
    - Dimensional parameters (iteration structure)
    - Performance attributes (evaluation outputs)
    
    Schema hash provides deterministic ID generation via SchemaRegistry.
    """
    
    def __init__(self):
        """Initialize empty schema with four DataBlocks."""
        self.static_params = ParametersStatic()
        self.dynamic_params = ParametersDynamic()
        self.dimensional_params = ParametersDimensional()
        self.performance_attrs = PerformanceAttributes()
    
    def _compute_schema_hash(self) -> str:
        """
        Compute deterministic hash from schema structure.
        
        Hash includes parameter names, DataObject types, and constraints
        (min/max/categories). This ensures same structure → same hash,
        even if parameter values differ.
        
        Returns:
            SHA256 hex digest (64 characters)
        """
        structure = {
            "static": self._block_to_hash_structure(self.static_params),
            "dynamic": self._block_to_hash_structure(self.dynamic_params),
            "dimensional": self._block_to_hash_structure(self.dimensional_params),
            "performance": self._block_to_hash_structure(self.performance_attrs)
        }
        
        # Sort keys for determinism
        hash_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def _block_to_hash_structure(self, block) -> Dict[str, Any]:
        """Convert DataBlock to hashable structure (types + constraints only)."""
        return {
            name: {
                "type": obj.__class__.__name__,
                **obj.constraints  # Include min/max/categories for collision prevention
            }
            for name, obj in block.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize schema to dictionary for storage.
        
        Returns:
            Dictionary with all blocks and metadata
        """
        return {
            "static_params": self.static_params.to_dict(),
            "dynamic_params": self.dynamic_params.to_dict(),
            "dimensional_params": self.dimensional_params.to_dict(),
            "performance_attrs": self.performance_attrs.to_dict(),
            "schema_hash": self._compute_schema_hash()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSchema':
        """
        Reconstruct schema from dictionary.
        
        Args:
            data: Dictionary from to_dict()
            
        Returns:
            DatasetSchema instance
        """
        schema = cls()
        schema.static_params = ParametersStatic.from_dict(data["static_params"])
        schema.dynamic_params = ParametersDynamic.from_dict(data["dynamic_params"])
        schema.dimensional_params = ParametersDimensional.from_dict(data["dimensional_params"])
        schema.performance_attrs = PerformanceAttributes.from_dict(data["performance_attrs"])
        return schema
    
    def is_compatible_with(self, other: 'DatasetSchema') -> bool:
        """
        Check structural compatibility with another schema.
        
        Uses detailed comparison (not hash) to provide informative
        error messages about specific differences.
        
        Args:
            other: Schema to compare against
            
        Returns:
            True if compatible
            
        Raises:
            ValueError: If incompatible, with detailed message
        """
        errors = []
        
        # Check static parameters
        self_static = set(self.static_params.keys())
        other_static = set(other.static_params.keys())
        if self_static != other_static:
            missing = other_static - self_static
            extra = self_static - other_static
            errors.append(
                f"Static params mismatch: "
                f"missing {missing if missing else 'none'}, "
                f"unexpected {extra if extra else 'none'}"
            )
        
        # Check dynamic parameters
        self_dynamic = set(self.dynamic_params.keys())
        other_dynamic = set(other.dynamic_params.keys())
        if self_dynamic != other_dynamic:
            missing = other_dynamic - self_dynamic
            extra = self_dynamic - other_dynamic
            errors.append(
                f"Dynamic params mismatch: "
                f"missing {missing if missing else 'none'}, "
                f"unexpected {extra if extra else 'none'}"
            )
        
        # Check dimensional parameters
        self_dim = set(self.dimensional_params.keys())
        other_dim = set(other.dimensional_params.keys())
        if self_dim != other_dim:
            missing = other_dim - self_dim
            extra = self_dim - other_dim
            errors.append(
                f"Dimensional params mismatch: "
                f"missing {missing if missing else 'none'}, "
                f"unexpected {extra if extra else 'none'}"
            )
        
        # Check performance attributes
        self_perf = set(self.performance_attrs.keys())
        other_perf = set(other.performance_attrs.keys())
        if self_perf != other_perf:
            missing = other_perf - self_perf
            extra = self_perf - other_perf
            errors.append(
                f"Performance attributes mismatch: "
                f"missing {missing if missing else 'none'}, "
                f"unexpected {extra if extra else 'none'}"
            )
        
        # Check types match for common parameters
        for name in self_static & other_static:
            if type(self.static_params.get(name)) != type(other.static_params.get(name)):
                errors.append(
                    f"Type mismatch for static param '{name}': "
                    f"{type(self.static_params.get(name)).__name__} vs "
                    f"{type(other.static_params.get(name)).__name__}"
                )
        
        for name in self_dynamic & other_dynamic:
            if type(self.dynamic_params.get(name)) != type(other.dynamic_params.get(name)):
                errors.append(
                    f"Type mismatch for dynamic param '{name}': "
                    f"{type(self.dynamic_params.get(name)).__name__} vs "
                    f"{type(other.dynamic_params.get(name)).__name__}"
                )
        
        if errors:
            raise ValueError(
                "Schema compatibility check failed:\n" + 
                "\n".join(f"  - {error}" for error in errors)
            )
        
        return True
    
    def get_all_param_names(self) -> Set[str]:
        """Get all parameter names across all blocks."""
        return (
            set(self.static_params.keys()) |
            set(self.dynamic_params.keys()) |
            set(self.dimensional_params.keys())
        )
    
    def get_all_performance_codes(self) -> Set[str]:
        """Get all performance codes."""
        return set(self.performance_attrs.keys())
