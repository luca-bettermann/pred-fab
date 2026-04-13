"""DatasetSchema and SchemaRegistry — define dataset structure and map hashes to human-readable IDs."""

import os
import json
import hashlib
from datetime import datetime
from typing import Any

from pred_fab.utils import LocalData, PfabLogger

import copy
from .data_blocks import (
    DataBlock,
    Parameters,
    PerformanceAttributes,
    Features,
    Domains,
)
from .data_objects import DataArray, DataDomainAxis, Domain
from ..utils.enum import Roles

class DatasetSchema:
    """Structural definition of a dataset (parameters, features, performance, domains); hash registered via SchemaRegistry."""

    def __init__(
            self,
            root_folder: str,
            name: str,
            parameters: Parameters,
            features: Features,
            performance: PerformanceAttributes,
            domains: Domains = Domains()
            ):
        """Initialize schema from DataBlocks."""
        self.name = name
        self.parameters = parameters
        self.features = features
        self.performance_attrs = performance
        self.domains = domains

        # Initialize local data handler and logger
        self.local_data = LocalData(root_folder)
        self.logger = PfabLogger.get_logger(self.local_data.get_log_folder('logs'))

        # Initialize schema
        self._initialize()

    @classmethod
    def from_dict(cls, data: dict[str, Any], root_folder: str) -> 'DatasetSchema':
        """Reconstruct schema from dictionary."""
        parameter_block = Parameters.from_dict(data["parameters"])
        feature_block = Features.from_dict(data["features"])
        performance_block = PerformanceAttributes.from_dict(data["performance_attrs"])
        domains = Domains.from_dict(data.get("domains", {}))

        name = data.get("schema_id")
        if name is None:
            raise ValueError("Schema dictionary must contain 'schema_id' field.")

        schema = cls(root_folder, name, parameter_block, feature_block, performance_block, domains)
        return schema

    def _initialize(self) -> None:
        """Initialize schema: register domain axis params into Parameters, set feature columns, compute hash, register."""
        # Register domain axis params into Parameters (auto-created DataDomainAxis objects)
        for domain in self.domains.values():
            for axis_param in domain.create_axis_params(Roles.PARAMETER):
                if not self.parameters.has(axis_param.code):
                    self.parameters.add(axis_param)

        # Set column names on feature DataArrays from domain definitions
        self._initialize_feature_columns()

        self.registry = SchemaRegistry(self.local_data.local_folder)
        schema_hash = self._compute_schema_hash()
        schema_struct = self.to_dict()

        self.registry.register_schema(self.name, schema_hash, schema_struct)
        self.local_data.set_schema(self.name)
        self.logger.console_success(f"Successfully initialized schema with ID: {self.name}.")

    def state_report(self) -> None:
        """Print schema overview to console."""
        _B = "\033[1m"
        _D = "\033[2m"
        _R = "\033[0m"

        lines = [f"\n  {_B}Schema: {self.name}{_R}"]

        # Parameters (exclude domain axis params)
        lines.append(f"\n  {_D}Parameters{_R}")
        for code, obj in self.parameters.items():
            if isinstance(obj, DataDomainAxis):
                continue
            c = obj.constraints
            lo, hi = c.get("min", ""), c.get("max", "")
            bounds = f"[{lo}, {hi}]" if lo != "" else ""
            if obj.runtime_adjustable:
                ptype = "runtime"
            else:
                ptype = ""
            lines.append(f"    {code:<20s} {bounds:<15s} {_D}{ptype}{_R}")

        # Domains
        if self.domains and len(list(self.domains.keys())) > 0:
            lines.append(f"\n  {_D}Domains{_R}")
            for domain_code, domain in self.domains.items():
                lines.append(f"    {domain_code}")
                for axis in domain.axes:
                    lo, hi = axis.min_val, axis.max_val
                    bounds = f"[{lo}, {hi}]"
                    fixed = "fixed" if (hi is not None and lo == hi) else ""
                    lines.append(f"      {axis.code:<18s} {bounds:<15s} {_D}{fixed}{_R}")

        # Features
        lines.append(f"\n  {_D}Features{_R}")
        for code, obj in self.features.items():
            if hasattr(obj, 'is_recursive') and obj.is_recursive:
                ftype = "recursive"
            elif hasattr(obj, 'context') and obj.context:
                ftype = "context"
            else:
                ftype = ""
            lines.append(f"    {code:<20s} {_D}{ftype}{_R}")

        # Performance attributes
        lines.append(f"\n  {_D}Performance{_R}")
        for code in self.performance_attrs.keys():
            lines.append(f"    {code}")

        self.logger.console_summary("\n".join(lines))
        self.logger.console_new_line()

    def _initialize_feature_columns(self) -> None:
        """Set iterator column names on each feature DataArray from domain definitions."""
        for feat_code, feat_obj in self.features.items():
            if not isinstance(feat_obj, DataArray):
                continue
            if feat_obj.columns:
                # Already explicitly set (e.g. in builders or from_dict); skip.
                continue
            domain_code = feat_obj.domain_code
            if domain_code is None:
                feat_obj.set_columns([feat_code])
            else:
                if not self.domains.has(domain_code):
                    raise ValueError(
                        f"Feature '{feat_code}' references domain '{domain_code}' which is not registered."
                    )
                domain = self.domains.get(domain_code)
                depth = feat_obj.feature_depth
                axes = domain.axes if depth is None else domain.axes[:depth]
                col_names = [ax.iterator_code for ax in axes] + [feat_code]
                feat_obj.set_columns(col_names)


    def _compute_schema_hash(self) -> str:
        """Compute deterministic hash from schema structure."""
        structure = {
            "parameters": self._block_to_hash_structure(self.parameters),
            "performance": self._block_to_hash_structure(self.performance_attrs),
            "features": self._block_to_hash_structure(self.features),
            "domains": self.domains.to_hash_dict(),
        }

        # Sort keys for determinism
        hash_str = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def _block_to_hash_structure(self, block: DataBlock) -> dict[str, Any]:
        """Convert DataBlock to hashable structure using to_hash_dict() (excludes operational metadata).

        Recursive features are excluded — they derive from existing tensors and
        don't change storage structure.
        """
        result: dict[str, Any] = {}
        for name, obj in block.items():
            if isinstance(obj, DataArray) and obj.is_recursive:
                continue
            result[name] = obj.to_hash_dict()
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize schema to dictionary for storage."""
        return {
            "parameters": self.parameters.to_dict(),
            "features": self.features.to_dict(),
            "performance_attrs": self.performance_attrs.to_dict(),
            "domains": self.domains.to_dict(),
            "schema_id": self.name
        }

    def get_domain_for_feature(self, feature_code: str) -> Domain | None:
        """Return the Domain for a feature, or None for scalar features."""
        feat_obj = self.features.data_objects.get(feature_code)
        if feat_obj is None or not isinstance(feat_obj, DataArray):
            return None
        domain_code = feat_obj.domain_code
        if domain_code is None:
            return None
        return self.domains.get(domain_code) if self.domains.has(domain_code) else None

    def is_compatible_with(self, other: 'DatasetSchema') -> bool:
        """Check structural compatibility with another schema."""
        # Check all blocks using helper function
        block_checks = [
            (self.parameters, other.parameters),
            (self.performance_attrs, other.performance_attrs),
            (self.features, other.features)
        ]

        for self_block, other_block in block_checks:
            if not self_block.is_compatible(other_block):
                raise ValueError(
                    f"Schema block '{self_block.__class__.__name__}' is not identical "
                    f"to {other_block.__class__.__name__}."
                )
        return True


class SchemaRegistry:
    """Persists schema hash → schema_id mappings as JSON in the local folder."""

    def __init__(self, local_folder: str):
        self.local_folder = local_folder
        self.registry_path = os.path.join(self.local_folder, "schema_registry.json")
        self.registry: dict[str, dict[str, Any]] = {}

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

    def register_schema(self, name: str, schema_hash: str, schema_struct: dict[str, Any]) -> None:
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

    def get_hash_by_id(self, schema_id: str) -> str | None:
        """Get schema hash by schema_id."""
        for schema_hash, entry in self.registry.items():
            if entry["schema_id"] == schema_id:
                return schema_hash
        return None
