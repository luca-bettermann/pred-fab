"""
Data Interface for LBP Framework

Standardized interface for accessing structured study and experiment metadata from various data sources.

DATA RESPONSIBILITY:
====================
- **Handles:** Structured metadata, study/experiment parameters, performance configurations
- **Does NOT handle:** Unstructured data (geometry files, sensor streams, raw experimental data)
- **Boundary:** FeatureModel._load_data() handles domain-specific unstructured data

DATA LOADING PRINCIPLES:
=============
- Hierarchical approach to the loading and storing of training data.
- First, check if data is available in memory, i.e. stored in the EvaluationSystem.
- Second, load the data from local files.
- Third, query the database for any missing data. Return error if not available.
- Once the data is retrieved, load it in memory and store it as local files.
- This ensures that the database is only queried once.

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

class ExternalDataInterface(ABC):
    """Abstract base class for accessing structured study and experiment metadata."""

    def __init__(self, client: Any = None):
        """Initialize with optional client for data access."""
        self.client = client

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def get_study_record(self, study_code: str) -> Dict[str, Any]:
        """
        Retrieve study metadata by study code.
        
        Args:
            study_code: Unique study identifier

        Returns:
            Study record with "id", "Code" and "Parameters" keys
        """
        ...

    @abstractmethod
    def get_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """
        Retrieve experiment metadata by experiment code.
        
        Args:
            exp_code: Unique experiment identifier

        Returns:
            Experiment record with "id", "Code" and "Parameters" keys
        """
        ...

    @abstractmethod
    def get_study_parameters(self, study_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract study parameters for @study_parameter fields.
        
        Args:
            study_record: Study record from get_study_record()

        Returns:
            Dictionary of study parameters {param_name: value}
        """
        ...

    @abstractmethod
    def get_performance_records(self, study_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get performance metric configurations for a study.
        
        Args:
            study_record: Study record from get_study_record()

        Returns:
            List of performance records with "Code" field
        """
        ...

    @abstractmethod
    def get_exp_variables(self, exp_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract experiment parameters for @exp_parameter fields.
        
        Args:
            exp_record: Experiment record from get_exp_record()

        Returns:
            Dictionary of experiment variables {param_name: value}
        """
        ...

    @abstractmethod
    def get_study_dataset(self, study_record: Dict[str, Any], restrict_to_exp_codes: List[str] = []) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve complete structured dataset for a study.
        
        Args:
            study_record: Study record from get_study_record()
            restrict_to_exp_codes: Filter to specific experiments ([] = all)

        Returns:
            Dict[exp_code, {param_name: value, performance_code: value}]
        """
        ...

    # === OPTIONAL METHODS ===
    def load_aggr_metrics(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load aggregated metrics from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            Tuple of (missing_exp_codes, aggr_metrics_dict)
        """
        # Default implementation returns all as missing
        return exp_codes, {}

    def load_metrics_arrays(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
        """
        Load metrics arrays from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            Tuple of (missing_exp_codes, metrics_arrays_dict)
        """
        # Default implementation returns all as missing
        return exp_codes, {}

    def load_exp_params(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Load experiment parameters from external source for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            Tuple of (missing_exp_codes, exp_params_dict)
        """
        # Default implementation returns all as missing
        return exp_codes, {}

    def load_study_records(self, study_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
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
                study_records_dict[study_code] = self.get_study_record(study_code)
            except:
                missing_study_codes.append(study_code)
                
        return missing_study_codes, study_records_dict

    def load_exp_records(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
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
                exp_records_dict[exp_code] = self.get_exp_record(exp_code)
            except:
                missing_exp_codes.append(exp_code)
                
        return missing_exp_codes, exp_records_dict

    def save_aggr_metrics(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]]) -> None:
        """Save aggregated metrics to external source."""
        # Default implementation - override in subclasses
        pass

    def save_metrics_arrays(self, exp_codes: List[str], data: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Save metrics arrays to external source."""
        # Default implementation - override in subclasses
        pass

    def save_exp_params(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]]) -> None:
        """Save experiment parameters to external source."""
        # Default implementation - override in subclasses
        pass

    def save_study_records(self, study_codes: List[str], data: Dict[str, Dict[str, Any]]) -> None:
        """Save study records to external source."""
        # Default implementation - override in subclasses
        pass

    def save_exp_records(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]]) -> None:
        """Save experiment records to external source."""
        # Default implementation - override in subclasses
        pass

    def load_experiments_from_database(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
        """
        Load experiment data from database for multiple experiments.
        
        Args:
            exp_codes: List of experiment codes to load
            
        Returns:
            Tuple of (missing_exp_codes, aggr_metrics_dict, metrics_arrays_dict, exp_params_dict)
        """
        # Default implementation returns all as missing
        return exp_codes, {}, {}, {}

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
