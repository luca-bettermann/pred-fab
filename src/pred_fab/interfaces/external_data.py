from abc import ABC, abstractmethod
from typing import Any, final
import numpy as np

class IExternalData(ABC):
    """Abstract class for accessing and writing experiment metadata to an external source."""

    def __init__(self, client: Any = None) -> None:
        """Initialize with optional client for data access."""
        self.client = client

    # === ABSTRACT METHODS ===
    @abstractmethod
    def pull_parameters(self, exp_codes: list[str]) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Fetch parameters for given experiment codes; returns (missing_codes, code→params dict)."""
        ...

    # === OPTIONAL METHODS ===        
    def pull_performance(self, exp_codes: list[str]) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Fetch performance metrics for given codes; returns (missing_codes, code→metrics dict). Default: all missing."""
        missing_exp_codes = exp_codes
        performance_dict = {}

        # Default implementation returns all as missing
        return missing_exp_codes, performance_dict

    def pull_features(self, exp_codes: list[str], feature_name: str = "default", **kwargs) -> tuple[list[str], dict[str, np.ndarray]]:
        """Fetch feature arrays for given codes; returns (missing_codes, code→array dict). Default: all missing."""
        missing_exp_codes = exp_codes
        features_dict = {}

        # Default implementation returns all as missing
        return missing_exp_codes, features_dict

    def push_parameters(self, exp_codes: list[str], parameters: dict[str, dict[str, Any]], recompute: bool = False) -> bool:
        """Push parameters to external source; returns True on success. Default: no-op (False)."""
        return False

    def push_performance(self, exp_codes: list[str], performance: dict[str, dict[str, Any]], recompute: bool = False) -> bool:
        """Push performance metrics to external source; returns True on success. Default: no-op (False)."""
        return False

    def push_features(self, exp_codes: list[str], features: dict[str, np.ndarray], recompute: bool = False, feature_name: str = "default", **kwargs) -> bool:
        """Push feature arrays to external source; returns True on success. Default: no-op (False)."""
        return False
    
    def push_schema(self, schema_id: str, schema_data: dict[str, Any]) -> bool:
        """Push schema to external source; returns True on success. Default: no-op (False)."""
        return False

    def pull_schema(self, schema_id: str) -> dict[str, Any] | None:
        """Fetch schema dict by ID from external source; returns None if not found. Default: no-op."""
        return None

    # === PUBLIC API METHODS ===
    @final
    def pull_exp_parameters(self, exp_code: str) -> dict[str, Any]:
        """Fetch parameters for a single experiment code; raises KeyError if not found."""
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
