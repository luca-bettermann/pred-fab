from abc import ABC, abstractmethod
from typing import Any, Dict, List

class DataInterface(ABC):
    """
    Abstract base class for data interfaces.
    
    Provides a standard interface for accessing structured study and experiment metadata
    from various sources (databases, APIs, files, etc.).
    
    Data Responsibility Boundary:
    - DataInterface: Manages structured, universal metadata (study parameters, experiment
      conditions, performance configurations) that follows standardized formats
    - FeatureModel/PredictionModel: Handle domain-specific, unstructured data (geometry
      files, sensor streams, images, proprietary formats) via their _load_data methods
    """

    def __init__(self, client: Any = None):
        """
        Initialize the data interface with an optional client.
        
        Args:
            client: Optional client object for data access (database, API client, etc.)
        """
        self.client = client

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def get_study_record(self, study_code: str) -> Dict[str, Any]:
        """
        Retrieve structured study metadata by study code.
        
        Data Boundary: Handles universal study metadata that follows standardized formats.
        Does NOT handle domain-specific data files (geometry, sensor data, etc.).
        
        Args:
            study_code: Unique identifier for the study
            
        Returns:
            Dictionary containing study record with fields:
            - "id": Study identifier  
            - "fields": {"Code": study_code, "Name": study_name, ...}
        """
        ...

    @abstractmethod
    def get_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """
        Retrieve structured experiment metadata by experiment code.
        
        Data Boundary: Handles universal experiment metadata, not raw experimental data
        (measurements, images, etc.) which are handled by model._load_data methods.
        
        Args:
            exp_code: Unique identifier for the experiment
            
        Returns:
            Dictionary containing experiment record with fields:
            - "id": Experiment identifier
            - "fields": {"Code": exp_code, "Name": exp_name, ...}
        """
        ...

    @abstractmethod
    def get_study_parameters(self, study_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured study parameters from a study record.
        
        Data Boundary: Returns standardized parameters (target values, tolerances,
        equipment settings) that can be handled by the parameter handling system.
        
        Args:
            study_record: Study record dictionary from get_study_record()
            
        Returns:
            Dictionary of study parameters {param_name: value}
            These parameters are used by model/exp_parameter() decorators
        """
        ...

    @abstractmethod
    def get_performance_records(self, study_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get structured performance metric configurations for a study.
        
        Data Boundary: Returns metadata about what performance metrics to evaluate,
        not the actual performance data itself.
        
        Args:
            study_record: Study record dictionary from get_study_record()
            
        Returns:
            List of performance record dictionaries, each containing:
            - 'Code': Performance metric identifier (required)
            - Additional metadata fields as needed
        """
        ...

    @abstractmethod
    def get_exp_variables(self, exp_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured experiment variables from an experiment record.
        
        Data Boundary: Returns standardized experimental conditions and parameters
        that can be handled by the parameter handling system.
        
        Args:
            exp_record: Experiment record dictionary from get_exp_record()
            
        Returns:
            Dictionary of experiment variables {param_name: value}
            These variables are used by exp_parameter() decorators
        """
        ...

    @abstractmethod
    def get_study_dataset(self, study_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve the complete dataset for a study.
        
        Data Boundary: Returns structured data that can be used for analysis,
        not raw experimental data files or proprietary formats.
        
        Returns:
            Dictionary containing experiment dicts with all relevant parameters
            and performances.
            Dict[exp_code_1, {param_name: value, performance_code: value}]
        """
        ...

    # === OPTIONAL METHODS ===
    def push_to_database(self, exp_record: Dict[str, Any], value_dict: Dict[str, Any]) -> None:
        """
        Push structured performance results to the database.
        
        Data Boundary: Stores standardized performance metrics and metadata,
        not raw experimental data or complex analysis results.
        
        Args:
            exp_record: Experiment record dictionary
            value_dict: Dictionary containing performance values and metadata
        """
        pass

    def update_system_performance(self, study_record: Dict[str, Any]) -> None:
        """
        Update aggregated system performance metrics.
        
        Data Boundary: Updates structured, aggregated performance data for
        system-wide tracking and optimization.
        
        Args:
            study_record: Study record dictionary
        """
        pass

    def _client_check(self) -> None:
        """Validate that client is properly initialized."""
        if not self.client:
            raise ValueError("Client not initialized. Provide a valid client instance to the DataInterface.")
