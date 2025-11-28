from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, final
import numpy as np

class IExternalData(ABC):
    """Abstract class for accessing and writing experiment metadata to an external source."""

    def __init__(self, client: Any = None) -> None:
        """Initialize with optional client for data access."""
        self.client = client

    # === ABSTRACT METHODS ===
    @abstractmethod
    def pull_exp_records(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load experiment records in batch from external source.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            missing_exp_codes: List of experiment codes that were not found
            exp_records_dict: Dict mapping experiment codes to their records
        """
        ...

    # === OPTIONAL METHODS ===        
    def pull_aggr_metrics(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load aggregated metrics from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            missing_exp_codes: List of experiment codes that were not found
            aggr_metrics_dict: Dict mapping experiment codes to their aggregated metrics
        """
        missing_exp_codes = exp_codes
        aggr_metrics_dict = {}

        # Default implementation returns all as missing
        return missing_exp_codes, aggr_metrics_dict

    def pull_metrics_arrays(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, np.ndarray]]:
        """
        Load metrics arrays from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            missing_exp_codes: List of experiment codes that were not found
            metrics_arrays_dict: Dict mapping experiment codes to their metrics arrays
        """
        missing_exp_codes = exp_codes
        metrics_arrays_dict = {}

        # Default implementation returns all as missing
        return missing_exp_codes, metrics_arrays_dict

    def push_exp_records(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """
        Save experiment records to external source.
        
        Args:
            exp_codes: List of experiment codes to save
            data: Dict mapping experiment codes to experiment record data
            recompute: If False, only push if data doesn't exist. If True, push/overwrite regardless.
            **kwargs: Additional arguments for implementation-specific options
            
        Returns:
            True if data was actually written/overwritten, False otherwise
        """
        # Default implementation - override in subclasses
        return False

    def push_aggr_metrics(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """
        Save aggregated metrics to external source.
        
        Args:
            exp_codes: List of experiment codes to save
            data: Dict mapping experiment codes to aggregated metrics data
            recompute: If False, only push if data doesn't exist. If True, push/overwrite regardless.
            **kwargs: Additional arguments for implementation-specific options
            
        Returns:
            True if data was actually written/overwritten, False otherwise
        """
        # Default implementation - override in subclasses
        return False

    def push_metrics_arrays(self, exp_codes: List[str], data: Dict[str, np.ndarray], recompute: bool, **kwargs) -> bool:
        """
        Save metrics arrays to external source.
        
        Args:
            exp_codes: List of experiment codes to save
            data: Dict mapping experiment codes to metrics arrays data
            recompute: If False, only push if data doesn't exist. If True, push/overwrite regardless.
            **kwargs: Additional arguments for implementation-specific options
            
        Returns:
            True if data was actually written/overwritten, False otherwise
        """
        # Default implementation - override in subclasses
        return False
    
    def push_schema(self, schema_id: str, schema_data: Dict[str, Any]) -> bool:
        """
        Save dataset schema to external source.
        
        Args:
            schema_id: Unique identifier for the schema (typically hash)
            schema_data: Serialized schema dictionary
            
        Returns:
            True if schema was successfully saved, False otherwise
        """
        # Default implementation - override in subclasses
        return False
    
    def pull_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve dataset schema from external source.
        
        Args:
            schema_id: Unique identifier for the schema
            
        Returns:
            Schema dictionary if found, None otherwise
        """
        # Default implementation - override in subclasses
        return None
    
    # === PUBLIC API METHODS ===
    @final
    def pull_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """
        Retrieve experiment record by experiment code.
        
        Args:
            exp_code: Unique experiment identifier

        Returns:
            Dict of exp_record with "id", "Code" and "Parameters" keys
        """
        missing, records = self.pull_exp_records([exp_code])
        if exp_code in missing or exp_code not in records:
            raise KeyError(f"Experiment record not found: {exp_code}")
        return records[exp_code]

    # === PRIVATE METHODS ===
    @final
    def _client_check(self) -> None:
        """Validate that client is properly initialized."""
        if not self.client:
            raise ValueError("Client not initialized. Provide a valid client instance to the DataInterface.")
