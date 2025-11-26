"""
SchemaRegistry for mapping schema hashes to human-readable IDs.

Provides deterministic schema_id generation based on structural hash,
ensuring same schema structure → same folder path across runs.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


class SchemaRegistry:
    """
    Registry mapping schema hashes to human-readable schema IDs.
    
    Stored as JSON file in {local_folder}/.lbp/schema_registry.json
    
    Format:
    {
        "hash_abc123...": {
            "schema_id": "schema_001",
            "created": "2025-11-21T10:30:00",
            "structure": {...}  # Full schema dict for debugging
        }
    }
    
    Note: Assumes single-user access for simplicity.
    """
    
    def __init__(self, local_folder: str):
        """
        Initialize registry.
        
        Args:
            local_folder: Base local data folder
        """
        self.local_folder = local_folder
        self.lbp_folder = os.path.join(local_folder, ".lbp")
        self.registry_path = os.path.join(self.lbp_folder, "schema_registry.json")
        self.registry: Dict[str, Dict[str, Any]] = {}
        
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from file, create if doesn't exist."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            # Create .lbp folder if needed
            os.makedirs(self.lbp_folder, exist_ok=True)
            self.registry = {}
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        os.makedirs(self.lbp_folder, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get_or_create_schema_id(
        self, 
        schema_hash: str, 
        schema_struct: Dict[str, Any],
        preferred_name: Optional[str] = None
    ) -> str:
        """
        Get existing schema_id or create new one.
        
        Deterministic: same hash always returns same ID.
        
        Args:
            schema_hash: Hash from DatasetSchema._compute_schema_hash()
            schema_struct: Full schema dictionary from to_dict()
            preferred_name: Optional preferred schema_id (if available)
            
        Returns:
            schema_id (e.g., "schema_001" or user's preferred name)
        """
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
        """
        Get schema structure by schema_id.
        
        Args:
            schema_id: Schema ID to look up
            
        Returns:
            Schema structure dict or None if not found
        """
        for entry in self.registry.values():
            if entry["schema_id"] == schema_id:
                return entry["structure"]
        return None
    
    def get_hash_by_id(self, schema_id: str) -> Optional[str]:
        """
        Get schema hash by schema_id.
        
        Args:
            schema_id: Schema ID to look up
            
        Returns:
            Schema hash or None if not found
        """
        for schema_hash, entry in self.registry.items():
            if entry["schema_id"] == schema_id:
                return schema_hash
        return None
    
    def list_schemas(self) -> Dict[str, str]:
        """
        List all registered schemas.
        
        Returns:
            Dict mapping schema_id to creation timestamp
        """
        return {
            entry["schema_id"]: entry["created"]
            for entry in self.registry.values()
        }
    
    def export(self) -> Dict[str, Any]:
        """
        Export registry for portability.
        
        Returns:
            Complete registry dictionary
        """
        return self.registry.copy()
    
    def import_registry(self, data: Dict[str, Any]) -> None:
        """
        Import registry data (merge with existing).
        
        Args:
            data: Registry dictionary to merge
        """
        self.registry.update(data)
        self._save_registry()
