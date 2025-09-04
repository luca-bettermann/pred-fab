import os
from typing import Dict, Any, List
from lbp_package import ExternalDataInterface


class MockDataInterface(ExternalDataInterface):
    """
    Example data interface that mocks an external data source by returning hardcoded JSON data.
    """
    
    def __init__(self, local_folder: str):
        super().__init__(None)
        self.local_folder = local_folder
        
    def pull_study_record(self, study_code: str) -> Dict[str, Any]:
        """Mock pulling of study record from external source by returning hardcoded data."""

        # hardcoded study data
        study_data = {
            "id": 0,
            "Code": study_code,
            "Parameters": {        
                "target_deviation": 0.0,
                "max_deviation": 1.0,
                "target_energy": 0.0,
                "max_energy": 10000.0,
                "power_rating": 50.0
            },
            "Performance": ["path_deviation", "energy_consumption"]
        }
        return study_data
    
    def pull_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """Mock pulling of experiment record from external source by returning hardcoded data."""

        # Check if the experiment code is valid
        implemented_codes = ['test_001', 'test_002', 'test_003']
        if exp_code not in implemented_codes:
            raise ValueError(f"Unknown experiment code: {exp_code}. Implemented codes are: {implemented_codes}")

        # hardcoded parameters
        params = {
            'test_001': [2, 2, 30.0, 0.2],
            'test_002': [3, 3, 40.0, 0.3],
            'test_003': [4, 4, 50.0, 0.4],
        }

        exp_data = {
            "id": int(''.join(filter(str.isdigit, exp_code))),
            "Code": exp_code,
            "Parameters": {
            "n_layers": params[exp_code][0],
            "n_segments": params[exp_code][1],
            "layerTime": params[exp_code][2],
            "layerHeight": params[exp_code][3]
            }
        }
        return exp_data
    
    def pull_study_dataset(self, study_record: Dict[str, Any], restrict_to_exp_codes: List[str] = []) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve experiment codes for the study. 
        Note: This method only returns experiment structure - actual data should be loaded via hierarchical pattern in LBPManager.
        """
        study_code = study_record["Code"]
        study_dir = os.path.join(self.local_folder, study_code)
        
        dataset = {}
        
        # Find all experiment directories
        if not os.path.exists(study_dir):
            return dataset
            
        for item in os.listdir(study_dir):
            item_path = os.path.join(study_dir, item)
            
            # Skip files, only process directories that look like experiments
            if not os.path.isdir(item_path) or not item.startswith(study_code):
                continue
                
            exp_code = item
            
            # Skip if experiment code is not in the restricted list and list not empty
            if restrict_to_exp_codes and exp_code not in restrict_to_exp_codes:
                continue
                
            # Return minimal structure - actual data loading handled by LBPManager hierarchical pattern
            dataset[exp_code] = {"exp_code": exp_code}
        
        return dataset

