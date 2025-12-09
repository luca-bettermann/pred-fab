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
from typing import Dict, Any, Set, Optional, List

from .data_objects import DataArray, DataDimension, Feature, Parameter, PerformanceAttribute
import copy
from .data_blocks import (
    DataBlock,
    Parameters,
    PerformanceAttributes,
    Features
)

class DatasetSchema:
    """
    Defines the structure and types of a dataset.
    
    Schema specifies:
    - Parameters (including dimensions)
    - Performance attributes (evaluation outputs)
    - Feature arrays (multi-dimensional data storage)
    
    Schema hash provides deterministic ID generation via SchemaRegistry.
    """
    
    def __init__(
            self, 
            name: str,
            parameters: Parameters,
            performance: PerformanceAttributes,
            features: Features,
            default_round_digits: int = 3
            ):
        """Initialize schema from DataBlocks."""
        self.name = name
        self.parameters = parameters
        self.performance = performance
        self.features = features
        self.predicted_features = self._return_copy_with_suffix(features, "pred_")
        self.default_round_digits = default_round_digits
        self.schema_id: str = name  # Assigned via SchemaRegistry

    @classmethod
    def from_list(
        cls, 
        name: str,
        parameters: List[Parameter],
        performance_attrs: List[PerformanceAttribute],
        features: List[Feature],
        default_round_digits: int = 3
        ) -> 'DatasetSchema':
        """Reconstruct schema from dictionary."""
        parameter_block = Parameters.from_list(parameters)
        performance_block = PerformanceAttributes.from_list(performance_attrs)
        feature_block = Features.from_list(features)
        return cls(name, parameter_block, performance_block, feature_block, default_round_digits)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSchema':
        """Reconstruct schema from dictionary."""
        parameter_block = Parameters.from_dict(data["parameters"])
        performance_block = PerformanceAttributes.from_dict(data["performance_attrs"])
        feature_block = Features.from_dict(data["features"])
        
        name = data.get("schema_id", "unknown_schema")
        schema = cls(name, parameter_block, performance_block, feature_block, data.get("default_round_digits", 3))
        return schema
    
    def initialize(self, registry: 'SchemaRegistry', feature_models: List[Any]) -> str:
        """Initialize schema by computing hash and registering with SchemaRegistry."""
        # First, set dimension codes for metric arrays
        self._set_dim_codes_for_arrays(feature_models)

        # Validate dimensions before hashing
        self.parameters.validate_dimensions()
        schema_hash = self._compute_schema_hash()
        schema_struct = self.to_dict()
        
        registry.register_schema(self.name, schema_hash, schema_struct)
        return self.name

    def set_schema_id(self, schema_id: str) -> None:
        """Set the schema ID."""
        self.schema_id = schema_id
        self.name = schema_id

    def _set_dim_codes_for_arrays(self, feature_models: List[Any]) -> None:
        """Set dimension codes for all metric arrays based on dataset parameters."""
        dim_codes = self.parameters.get_dim_names()
        
        # Iterate over all feature models to set dim codes
        for model in feature_models:
            for output_code in model.outputs:
                data_array = self.features.data_objects[output_code]
                if isinstance(data_array, DataArray):
                    model_dim_codes = [code for code in model.input_parameters if code in dim_codes]
                    data_array.set_dim_codes(model_dim_codes)

    def _compute_schema_hash(self) -> str:
        """Compute deterministic hash from schema structure."""
        structure = {
            "parameters": self._block_to_hash_structure(self.parameters),
            "performance": self._block_to_hash_structure(self.performance),
            "features": self._block_to_hash_structure(self.features),
            # "default_round_digits": self.default_round_digits,
            # "calibration_weights": self.performance.calibration_weights
        }
        
        # Sort keys for determinism
        hash_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def _block_to_hash_structure(self, block) -> Dict[str, Any]:
        """Convert DataBlock to hashable structure using full object serialization."""
        return {
            name: obj.to_dict()
            for name, obj in block.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary for storage."""
        return {
            "parameters": self.parameters.to_dict(),
            "performance_attrs": self.performance.to_dict(),
            "features": self.features.to_dict(),
            "default_round_digits": self.default_round_digits,
            "schema_id": self.name,
            # "schema_hash": self._compute_schema_hash()
        }
    
    def is_compatible_with(self, other: 'DatasetSchema') -> bool:
        """Check structural compatibility with another schema."""        
        # Check all blocks using helper function
        block_checks = [
            (self.parameters, other.parameters),
            # (self.dimensions, other.dimensions),
            (self.performance, other.performance),
            (self.features, other.features)
        ]
        
        for self_block, other_block in block_checks:
            if not self_block._is_identical(other_block):
                raise ValueError(
                    f"Schema block '{self_block.__class__.__name__}' is not identical "
                    f"to {other_block.__class__.__name__}."
                )        
        return True
    
    def _return_copy_with_suffix(self, data_block: DataBlock, suffix: str) -> DataBlock:
        data_block_suffix = Features()
        for feat in data_block.data_objects.values():
            # Create a new feature instance with the modified code
            new_feat = copy.deepcopy(feat)
            new_feat.code = f"{suffix}{feat.code}"
            data_block_suffix.add(new_feat.code, new_feat)
        return data_block_suffix

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
    
    def register_schema(self, name: str, schema_hash: str, schema_struct: Dict[str, Any]) -> None:
        """Register schema with a specific name, validating against existing entries."""
        
        # 1. Check if name already exists
        existing_hash_for_name = self.get_hash_by_id(name)
        
        if existing_hash_for_name:
            # Name exists. Check if hash matches.
            if existing_hash_for_name != schema_hash:
                raise ValueError(
                    f"Schema name '{name}' is already registered with a different structure. "
                    f"Existing hash: {existing_hash_for_name}, New hash: {schema_hash}. "
                    "Please use a different name or ensure the schema structure is identical."
                )
            # If matches, we are good. Ensure registry is consistent (it should be).
            return

        # 2. Check if hash already exists (under a different name)
        if schema_hash in self.registry:
            existing_name = self.registry[schema_hash]["schema_id"]
            if existing_name != name:
                 raise ValueError(
                    f"This schema structure is already registered under the name '{existing_name}'. "
                    f"Cannot register the same structure as '{name}' to avoid ambiguity."
                )
        
        # 3. Register new
        self.registry[schema_hash] = {
            "schema_id": name,
            "created": datetime.now().isoformat(),
            "structure": schema_struct
        }
        self._save_registry()
    
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
