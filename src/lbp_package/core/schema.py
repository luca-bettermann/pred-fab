"""
DatasetSchema for defining dataset structure.

Schema represents the structure of a dataset (what CAN exist),
not the actual values (which are stored in Dataset instances).
"""

import json
import hashlib
from typing import Dict, Any, Set
from .data_blocks import (
    Parameters,
    Dimensions,
    PerformanceAttributes,
    MetricArrays
)


class DatasetSchema:
    """
    Defines the structure and types of a dataset.
    
    Schema specifies:
    - Parameters (unified - no static/dynamic split)
    - Dimensions (dimensional metadata)
    - Performance attributes (evaluation outputs)
    - Metric arrays (multi-dimensional data storage)
    
    Schema hash provides deterministic ID generation via SchemaRegistry.
    """
    
    def __init__(self, default_round_digits: int = 3):
        """
        Initialize empty schema with four DataBlocks.
        
        Args:
            default_round_digits: Default rounding precision for numeric DataObjects
        """
        self.parameters = Parameters()
        self.dimensions = Dimensions()
        self.performance_attrs = PerformanceAttributes()
        self.metric_arrays = MetricArrays()
        self.default_round_digits = default_round_digits
    
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
            "parameters": self._block_to_hash_structure(self.parameters),
            "dimensions": self._block_to_hash_structure(self.dimensions),
            "performance": self._block_to_hash_structure(self.performance_attrs),
            "metric_arrays": self._block_to_hash_structure(self.metric_arrays)
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
            "parameters": self.parameters.to_dict(),
            "dimensions": self.dimensions.to_dict(),
            "performance_attrs": self.performance_attrs.to_dict(),
            "metric_arrays": self.metric_arrays.to_dict(),
            "default_round_digits": self.default_round_digits,
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
        default_round_digits = data.get("default_round_digits", 3)
        schema = cls(default_round_digits=default_round_digits)
        schema.parameters = Parameters.from_dict(data["parameters"])
        schema.dimensions = Dimensions.from_dict(data["dimensions"])
        schema.performance_attrs = PerformanceAttributes.from_dict(data["performance_attrs"])
        schema.metric_arrays = MetricArrays.from_dict(data.get("metric_arrays", {"type": "MetricArrays", "data_objects": {}}))
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
        
        # Check parameters
        self_params = set(self.parameters.keys())
        other_params = set(other.parameters.keys())
        if self_params != other_params:
            missing = other_params - self_params
            extra = self_params - other_params
            errors.append(
                f"Parameters mismatch: "
                f"missing {missing if missing else 'none'}, "
                f"unexpected {extra if extra else 'none'}"
            )
        
        # Check dimensions
        self_dim = set(self.dimensions.keys())
        other_dim = set(other.dimensions.keys())
        if self_dim != other_dim:
            missing = other_dim - self_dim
            extra = self_dim - other_dim
            errors.append(
                f"Dimensions mismatch: "
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
        
        # Check metric arrays
        self_arrays = set(self.metric_arrays.keys())
        other_arrays = set(other.metric_arrays.keys())
        if self_arrays != other_arrays:
            missing = other_arrays - self_arrays
            extra = self_arrays - other_arrays
            errors.append(
                f"Metric arrays mismatch: "
                f"missing {missing if missing else 'none'}, "
                f"unexpected {extra if extra else 'none'}"
            )
        
        # Check types match for common parameters
        for name in self_params & other_params:
            if type(self.parameters.get(name)) != type(other.parameters.get(name)):
                errors.append(
                    f"Type mismatch for parameter '{name}': "
                    f"{type(self.parameters.get(name)).__name__} vs "
                    f"{type(other.parameters.get(name)).__name__}"
                )
        
        if errors:
            raise ValueError(
                "Schema compatibility check failed:\n" + 
                "\n".join(f"  - {error}" for error in errors)
            )
        
        return True
    
    def get_all_param_names(self) -> Set[str]:
        """Get all parameter names."""
        return set(self.parameters.keys())
    
    def get_dimension_names(self) -> Set[str]:
        """Get all dimension parameter names."""
        return set(self.dimensions.keys())
    
    def get_all_performance_codes(self) -> Set[str]:
        """Get all performance codes."""
        return set(self.performance_attrs.keys())
