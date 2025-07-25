from abc import ABC, abstractmethod
from typing import Any, Dict, List

class DataInterface(ABC):
    """
    Abstract base class for data interfaces.
    
    Provides a standard interface for accessing study and experiment data
    from various sources (databases, APIs, files, etc.).
    """

    def __init__(self, client: Any = None):
        """
        Initialize the data interface with an optional client.
        
        Args:
            client: Optional client object for data access
        """
        self.client = client

    # === ABSTRACT METHODS (Must be implemented by subclasses) ===
    @abstractmethod
    def get_study_record(self, study_code: str) -> Dict[str, Any]:
        """
        Retrieve a study record by study code.
        
        Args:
            study_code: Unique identifier for the study
            
        Returns:
            Dictionary containing study record data
        """
        ...

    @abstractmethod
    def get_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """
        Retrieve an experiment record by experiment code.
        
        Args:
            exp_code: Unique identifier for the experiment
            
        Returns:
            Dictionary containing experiment record data
        """
        ...

    @abstractmethod
    def get_study_parameters(self, study_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract study parameters from a study record.
        
        Args:
            study_record: Study record dictionary
            
        Returns:
            Dictionary of study parameters
        """
        ...

    @abstractmethod
    def get_performance_records(self, study_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get performance records associated with a study.
        
        Args:
            study_record: Study record dictionary
            
        Returns:
            List of performance record dictionaries, each containing a 'Code' key
        """
        ...

    @abstractmethod
    def get_exp_variables(self, exp_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract experiment variables from an experiment record.
        
        Args:
            exp_record: Experiment record dictionary
            
        Returns:
            Dictionary of experiment variables
        """
        ...

    # === OPTIONAL METHODS ===
    def push_to_database(self, exp_record: Dict[str, Any], performance_code: str, value_dict: Dict[str, Any]) -> None:
        """
        Push performance data to the database.
        
        Args:
            exp_record: Experiment record dictionary
            performance_code: Code identifying the performance metric
            value_dict: Dictionary containing performance values
        """
        pass

    def update_system_performance(self, study_record: Dict[str, Any]) -> None:
        """
        Update aggregated system performance metrics.
        
        Args:
            study_record: Study record dictionary
        """
        pass

    def _client_check(self) -> None:
        """Validate that client is properly initialized."""
        if not self.client:
            raise ValueError("Client not initialized. Provide a valid client instance to the DataInterface.")
