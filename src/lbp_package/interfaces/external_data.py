"""
Data Interface for LBP Framework

Standardized interface for accessing structured study and experiment metadata from various data sources.

DATA RESPONSIBILITY:
====================
- **Handles:** Structured metadata, study/experiment parameters, performance configurations
- **Does NOT handle:** Unstructured data (geometry files, sensor streams, raw experimental data)
- **Boundary:** FeatureModel._load_data() handles domain-specific unstructured data

DATA LOADING PRINCIPLES:
========================
- Hierarchical approach to loading and storing structured data.
- First, check if data is available in memory (current session cache).
- Second, load data from local JSON/CSV files if available.
- Third, query external source for any missing data. Return error if not available.
- Once retrieved from external source, automatically store as local files.
- Data is always stored locally - there is no "database-only" mode.
- recompute_flag=True forces loading from external source, bypassing local cache.

CORE METHODS:
=============
- get_study_record/parameters(): Study-level data (constant across experiments)
- get_exp_record/variables(): Experiment-level data (varies between experiments)  
- get_performance_records(): Performance metric configurations
- push_to_database(): Store results

PARAMETER INTEGRATION:
======================
- get_study_parameters() → @study_parameter fields
- get_exp_variables() → @exp_parameter fields
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class IExternalData(ABC):
    """Abstract base class for accessing structured study and experiment metadata."""

    def __init__(self, client: Any = None):
        """Initialize with optional client for data access."""
        self.client = client

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def pull_study_record(self, study_code: str) -> Dict[str, Any]:
        """
        Retrieve study metadata by study code.
        
        Args:
            study_code: Unique study identifier

        Returns:
            Study record with "id", "Code" and "Parameters" and "Performance" keys
        """
        ...

    @abstractmethod
    def pull_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """
        Retrieve experiment metadata by experiment code.
        
        Args:
            exp_code: Unique experiment identifier

        Returns:
            Experiment record with "id", "Code" and "Parameters" keys
        """
        ...

    # === OPTIONAL METHODS ===        
    def pull_aggr_metrics(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load aggregated metrics from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            Tuple of (missing_exp_codes, aggr_metrics_dict)
        """
        # Default implementation returns all as missing
        return exp_codes, {}

    def pull_metrics_arrays(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, np.ndarray]]:
        """
        Load metrics arrays from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            Tuple of (missing_exp_codes, metrics_arrays_dict)
        """
        # Default implementation returns all as missing
        return exp_codes, {}
            
    def push_study_records(self, study_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """
        Save study records to external source.
        
        Args:
            study_codes: List of study codes to save
            data: Dict mapping study codes to study record data
            recompute: If False, only push if data doesn't exist. If True, push/overwrite regardless.
            **kwargs: Additional arguments for implementation-specific options
            
        Returns:
            True if data was actually written/overwritten, False otherwise
        """
        # Default implementation - override in subclasses
        return False

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
    
    # === PUBLIC API METHODS (Called externally) ===
    def pull_study_records(self, study_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load study records from external source.
        
        Args:
            study_codes: List of study codes to load
            
        Returns:
            Tuple of (missing_study_codes, study_records_dict)
        """
        study_records_dict = {}
        missing_study_codes = []
        
        for study_code in study_codes:
            try:
                study_records_dict[study_code] = self.pull_study_record(study_code)
            except:
                missing_study_codes.append(study_code)
                
        return missing_study_codes, study_records_dict

    def pull_exp_records(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load experiment records from external source.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            Tuple of (missing_exp_codes, exp_records_dict)
        """
        exp_records_dict = {}
        missing_exp_codes = []
        
        for exp_code in exp_codes:
            try:
                exp_records_dict[exp_code] = self.pull_exp_record(exp_code)
            except:
                missing_exp_codes.append(exp_code)
                
        return missing_exp_codes, exp_records_dict

    # === OPTIONAL METHODS ===
    def update_system_performance(self, study_record: Dict[str, Any]) -> None:
        """
        Update aggregated system performance metrics.
        
        Args:
            study_record: Study record for performance update
        """
        pass

    def _client_check(self) -> None:
        """Validate that client is properly initialized."""
        if not self.client:
            raise ValueError("Client not initialized. Provide a valid client instance to the DataInterface.")
