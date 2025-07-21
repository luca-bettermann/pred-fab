from typing import Dict, Any, List
from lbp_package.lbp_package.data_interface import DataInterface

class ExampleDataInterface(DataInterface):
    """Example data interface for demonstration and testing."""
    
    def __init__(self, test_data_dir: str, study_params: Dict[str, Any], experiment_data: Dict[str, Any]):
        super().__init__(None)
        self.test_data_dir = test_data_dir
        self.study_params = study_params
        self.experiment_data = experiment_data
        self.study_record = None
        self.exp_record = None
        
    def get_study_record(self, study_code: str) -> Dict[str, Any]:
        """Return mock study record."""
        self.study_record = {
            "id": "mock_study_001",
            "fields": {
                "Code": study_code,
                "Name": f"Test Study {study_code}"
            }
        }
        return self.study_record
    
    def get_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """Return mock experiment record."""
        self.exp_record = {
            "id": "mock_exp_001",
            "fields": {
                "Code": exp_code,
                "Name": f"Test Experiment {exp_code}"
            }
        }
        return self.exp_record
    
    def get_study_parameters(self, study_record: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock study parameters."""
        return self.study_params.copy()
    
    def get_performance_records(self, study_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return mock performance records."""
        return [
            {"Code": "path_deviation", "Active": True},
            {"Code": "energy_consumption", "Active": True}
        ]
    
    def get_exp_variables(self, exp_record: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock experiment variables."""
        return self.experiment_data.copy()
    
    def push_to_database(self, exp_record: Dict[str, Any], performance_code: str, value_dict: Dict[str, Any]) -> None:
        """Mock database push - just store for verification."""
        if not hasattr(self, 'pushed_data'):
            self.pushed_data = {}
        self.pushed_data[performance_code] = value_dict
    
    def update_system_performance(self, study_record: Dict[str, Any]) -> None:
        """Mock system performance update."""
        self.system_performance_updated = True
