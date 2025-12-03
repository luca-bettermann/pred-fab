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
    def pull_parameters(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load experiment parameters in batch from external source.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            missing_exp_codes: List of experiment codes that were not found
            parameters_dict: Dict mapping experiment codes to their parameters
        """
        ...

    # === OPTIONAL METHODS ===        
    def pull_performance(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load performance metrics from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            missing_exp_codes: List of experiment codes that were not found
            performance_dict: Dict mapping experiment codes to their performance metrics
        """
        missing_exp_codes = exp_codes
        performance_dict = {}

        # Default implementation returns all as missing
        return missing_exp_codes, performance_dict

    def pull_features(self, exp_codes: List[str], feature_name: str = "default", **kwargs) -> tuple[List[str], Dict[str, np.ndarray]]:
        """
        Load feature arrays from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            feature_name: Identifier for the feature type
            
        Returns:
            missing_exp_codes: List of experiment codes that were not found
            features_dict: Dict mapping experiment codes to their feature arrays
        """
        missing_exp_codes = exp_codes
        features_dict = {}

        # Default implementation returns all as missing
        return missing_exp_codes, features_dict

    def push_parameters(self, exp_codes: List[str], parameters: Dict[str, Dict[str, Any]], recompute: bool = False) -> bool:
        """
        Save experiment parameters to external source.
        
        Args:
            exp_codes: List of experiment codes to save
            parameters: Dict mapping experiment codes to their parameters
            recompute: Whether to overwrite existing records
            
        Returns:
            success: True if all records were saved successfully
        """
        return False

    def push_performance(self, exp_codes: List[str], performance: Dict[str, Dict[str, Any]], recompute: bool = False) -> bool:
        """
        Save performance metrics to external source.
        
        Args:
            exp_codes: List of experiment codes to save
            performance: Dict mapping experiment codes to their metrics
            recompute: Whether to overwrite existing metrics
            
        Returns:
            success: True if all metrics were saved successfully
        """
        return False

    def push_features(self, exp_codes: List[str], features: Dict[str, np.ndarray], recompute: bool = False, feature_name: str = "default", **kwargs) -> bool:
        """
        Save feature arrays to external source.
        
        Args:
            exp_codes: List of experiment codes to save
            features: Dict mapping experiment codes to their arrays
            recompute: Whether to overwrite existing arrays
            feature_name: Identifier for the feature type
            
        Returns:
            success: True if all arrays were saved successfully
        """
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
    def pull_exp_parameters(self, exp_code: str) -> Dict[str, Any]:
        """
        Retrieve experiment record by experiment code.
        
        Args:
            exp_code: Unique experiment identifier

        Returns:
            Dict of exp_record with "id", "Code" and "Parameters" keys
        """
        missing, records = self.pull_parameters([exp_code])
        if exp_code in missing or exp_code not in records:
            raise KeyError(f"Experiment record not found: {exp_code}")
        return records[exp_code]

    # === PRIVATE METHODS ===
    @final
    def _client_check(self) -> None:
        """Validate that client is properly initialized."""
        if not self.client:
            raise ValueError("Client not initialized. Provide a valid client instance to the DataInterface.")
