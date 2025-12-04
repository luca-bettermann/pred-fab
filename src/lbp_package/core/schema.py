"""
DatasetSchema for defining dataset structure.

Schema represents the structure of a dataset (what CAN exist),
not the actual values (which are stored in Dataset instances).

SchemaRegistry maps schema hashes to human-readable IDs for deterministic
folder path generation.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Set, Optional
from ..utils import LBPLogger
from .data_objects import DataDimension
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
        """Initialize empty schema with four DataBlocks."""
        self.parameters = Parameters()
        self.dimensions = Dimensions()
        self.performance_attrs = PerformanceAttributes()
        self.features = MetricArrays()
        self.default_round_digits = default_round_digits
    
    def _compute_schema_hash(self) -> str:
        """Compute deterministic hash from schema structure."""
        structure = {
            "parameters": self._block_to_hash_structure(self.parameters),
            "dimensions": self._block_to_hash_structure(self.dimensions),
            "performance": self._block_to_hash_structure(self.performance_attrs),
            "metric_arrays": self._block_to_hash_structure(self.features)
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
        """Serialize schema to dictionary for storage."""
        return {
            "parameters": self.parameters.to_dict(),
            "dimensions": self.dimensions.to_dict(),
            "performance_attrs": self.performance_attrs.to_dict(),
            "metric_arrays": self.features.to_dict(),
            "default_round_digits": self.default_round_digits,
            "schema_hash": self._compute_schema_hash()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSchema':
        """Reconstruct schema from dictionary."""
        default_round_digits = data.get("default_round_digits", 3)
        schema = cls(default_round_digits=default_round_digits)
        schema.parameters = Parameters.from_dict(data["parameters"])
        schema.dimensions = Dimensions.from_dict(data["dimensions"])
        schema.performance_attrs = PerformanceAttributes.from_dict(data["performance_attrs"])
        schema.features = MetricArrays.from_dict(data.get("metric_arrays", {"type": "MetricArrays", "data_objects": {}}))
        return schema
    
    @classmethod
    def from_model_specs(
        cls,
        eval_specs: Dict[str, Any],
        pred_specs: Dict[str, Any],
        logger: LBPLogger
    ) -> 'DatasetSchema':
        """Build schema from evaluation and prediction system specifications."""        
        schema = cls()
        
        # 1. Merge input parameters from both systems
        all_inputs = {**eval_specs["inputs"], **pred_specs["inputs"]}
        
        # Separate dimensions from parameters
        for param_name, data_obj in all_inputs.items():
            if isinstance(data_obj, DataDimension):
                schema.dimensions.add(param_name, data_obj)
                logger.info(f"  Added dimension: {param_name}")
            else:
                schema.parameters.add(param_name, data_obj)
                logger.info(f"  Added parameter: {param_name}")
        
        # 2. Add performance attributes
        for perf_code, perf_obj in eval_specs["outputs"].items():
            schema.performance_attrs.add(perf_code, perf_obj)
            logger.info(f"  Added performance attribute: {perf_code}")
        
        logger.info(
            f"Generated schema with {len(schema.parameters.data_objects)} parameters, "
            f"{len(schema.dimensions.data_objects)} dimensions, "
            f"{len(schema.performance_attrs.data_objects)} performance attributes"
        )
        
        return schema
    
    def is_compatible_with(self, other: 'DatasetSchema') -> bool:
        """Check structural compatibility with another schema."""        
        # Check all blocks using helper function
        block_checks = [
            (self.parameters, other.parameters),
            (self.dimensions, other.dimensions),
            (self.performance_attrs, other.performance_attrs),
            (self.features, other.features)
        ]
        
        for self_block, other_block in block_checks:
            if not self_block._is_identical(other_block):
                raise ValueError(
                    f"Schema block '{self_block.__class__.__name__}' is not identical "
                    f"to {other_block.__class__.__name__}."
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


class SchemaRegistry:
    """
    Registry mapping schema hashes to human-readable schema IDs.
    
    - Deterministic schema_id generation from structural hash
    - Stored as JSON in {local_folder}/.lbp/schema_registry.json
    - Single-user access model
    """
    
    def __init__(self, local_folder: str):
        """
        Initialize registry.
        
        Args:
            local_folder: Base local data folder
        """
        self.local_folder = local_folder
        self.registry_path = os.path.join(self.local_folder, "schema_registry.json")
        self.registry: Dict[str, Dict[str, Any]] = {}
        
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from file, create if doesn't exist."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            # Create .lbp folder if needed
            os.makedirs(self.local_folder, exist_ok=True)
            self.registry = {}
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        os.makedirs(self.local_folder, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_or_create_schema_id(
        self, 
        schema_hash: str, 
        schema_struct: Dict[str, Any],
        preferred_name: Optional[str] = None
    ) -> str:
        """Get existing schema_id or create new one deterministically based on schema hash."""
        # Check if hash already registered
        if schema_hash in self.registry:
            return self.registry[schema_hash]["schema_id"]
        
        # Generate new ID
        if preferred_name and not self._id_exists(preferred_name):
            schema_id = preferred_name
        else:
            schema_id = self._generate_next_id()
        
        # Register
        self.registry[schema_hash] = {
            "schema_id": schema_id,
            "created": datetime.now().isoformat(),
            "structure": schema_struct
        }
        
        self._save_registry()
        return schema_id
    
    def _id_exists(self, schema_id: str) -> bool:
        """Check if schema_id is already used."""
        return any(
            entry["schema_id"] == schema_id 
            for entry in self.registry.values()
        )
    
    def _generate_next_id(self) -> str:
        """Generate next sequential schema ID (schema_001, schema_002, ...)."""
        if not self.registry:
            return "schema_001"
        
        # Find max existing number
        max_num = 0
        for entry in self.registry.values():
            schema_id = entry["schema_id"]
            if schema_id.startswith("schema_"):
                try:
                    num = int(schema_id.split("_")[1])
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    continue
        
        return f"schema_{(max_num + 1):03d}"
    
    def get_schema_by_id(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Get schema structure by schema_id."""
        for entry in self.registry.values():
            if entry["schema_id"] == schema_id:
                return entry["structure"]
        return None
    
    def get_hash_by_id(self, schema_id: str) -> Optional[str]:
        """Get schema hash by schema_id."""
        for schema_hash, entry in self.registry.items():
            if entry["schema_id"] == schema_id:
                return schema_hash
        return None
    
    def list_schemas(self) -> Dict[str, str]:
        """List all registered schemas as dict mapping schema_id to creation timestamp."""
        return {
            entry["schema_id"]: entry["created"]
            for entry in self.registry.values()
        }
    
    def export(self) -> Dict[str, Any]:
        """Export complete registry dictionary for portability."""
        return self.registry.copy()
    
    def import_registry(self, data: Dict[str, Any]) -> None:
        """Import and merge registry data with existing entries."""
        self.registry.update(data)
        self._save_registry()
