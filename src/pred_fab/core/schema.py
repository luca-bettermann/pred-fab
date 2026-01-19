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

from pred_fab.utils import LocalData, PfabLogger

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
            root_folder: str, 
            name: str,
            parameters: Parameters,
            features: Features,
            performance: PerformanceAttributes
            ):
        """Initialize schema from DataBlocks."""
        
        self.name = name
        self.parameters = parameters
        self.features = features
        self.performance = performance
        self.predicted_features = self._return_copy_with_suffix(features, "pred_")

        # Initialize local data handler and logger
        self.local_data = LocalData(root_folder)
        self.logger = PfabLogger.get_logger(self.local_data.get_log_folder('logs'))

        # Initialize schema
        self._initialize()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], root_folder: str) -> 'DatasetSchema':
        """Reconstruct schema from dictionary."""
        parameter_block = Parameters.from_dict(data["parameters"])
        feature_block = Features.from_dict(data["features"])
        performance_block = PerformanceAttributes.from_dict(data["performance_attrs"])
        
        name = data.get("schema_id")
        if name is None:
            raise ValueError("Schema dictionary must contain 'schema_id' field.")
        
        schema = cls(root_folder, name, parameter_block, feature_block, performance_block)
        return schema
    
    def _initialize(self) -> None:
        """Initialize schema by computing hash and registering with SchemaRegistry."""
        self.registry = SchemaRegistry(self.local_data.local_folder)

        # Validate dimensions before hashing
        self.parameters.validate_dimensions()
        schema_hash = self._compute_schema_hash()
        schema_struct = self.to_dict()
        
        self.registry.register_schema(self.name, schema_hash, schema_struct)
        self.local_data.set_schema(self.name)
        self.logger.console_success(f"Successfully initialized schema with ID: {self.name}.")


    def _compute_schema_hash(self) -> str:
        """Compute deterministic hash from schema structure."""
        structure = {
            "parameters": self._block_to_hash_structure(self.parameters),
            "performance": self._block_to_hash_structure(self.performance),
            "features": self._block_to_hash_structure(self.features)
        }
        
        # Sort keys for determinism
        hash_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def _block_to_hash_structure(self, block: DataBlock) -> Dict[str, Any]:
        """Convert DataBlock to hashable structure using full object serialization."""
        return {
            name: obj.to_dict()
            for name, obj in block.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary for storage."""
        return {
            "parameters": self.parameters.to_dict(),
            "features": self.features.to_dict(),
            "performance_attrs": self.performance.to_dict(),
            "schema_id": self.name
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
                    "Please use a different name or ensure the schema structure is identical."
                    "If you want to overwrite, please delete the existing entry in the schema_registry.json file."
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
    
    def get_hash_by_id(self, schema_id: str) -> Optional[str]:
        """Get schema hash by schema_id."""
        for schema_hash, entry in self.registry.items():
            if entry["schema_id"] == schema_id:
                return schema_hash
        return None
    
    # def _generate_next_id(self) -> str:
    #     """Generate next sequential schema ID (schema_001, schema_002, ...)."""
    #     if not self.registry:
    #         return "schema_001"
        
    #     # Find max existing number
    #     max_num = 0
    #     for entry in self.registry.values():
    #         schema_id = entry["schema_id"]
    #         if schema_id.startswith("schema_"):
    #             try:
    #                 num = int(schema_id.split("_")[1])
    #                 max_num = max(max_num, num)
    #             except (IndexError, ValueError):
    #                 continue
        
    #     return f"schema_{(max_num + 1):03d}"
    
    # def get_schema_by_id(self, schema_id: str) -> Optional[Dict[str, Any]]:
    #     """Get schema structure by schema_id."""
    #     for entry in self.registry.values():
    #         if entry["schema_id"] == schema_id:
    #             return entry["structure"]
    #     return None
    
    # def list_schemas(self) -> Dict[str, str]:
    #     """List all registered schemas as dict mapping schema_id to creation timestamp."""
    #     return {
    #         entry["schema_id"]: entry["created"]
    #         for entry in self.registry.values()
    #     }
    
    # def export(self) -> Dict[str, Any]:
    #     """Export complete registry dictionary for portability."""
    #     return self.registry.copy()
    
    # def import_registry(self, data: Dict[str, Any]) -> None:
    #     """Import and merge registry data with existing entries."""
    #     self.registry.update(data)
    #     self._save_registry()
