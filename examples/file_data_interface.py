import json
import os
from typing import Dict, Any, List
from lbp_package import ExternalDataInterface


class FileDataInterface(ExternalDataInterface):
    """
    Example data interface that reads source records from local JSON files.
    
    This implementation focuses on source record retrieval (study/experiment metadata)
    while hierarchical data loading is handled by the base DataInterface class.
    """
    
    def __init__(self, local_folder: str):
        super().__init__(None)
        self.local_folder = local_folder
        
    def get_study_record(self, study_code: str) -> Dict[str, Any]:
        """Load study record from study_params.json."""
        study_params_path = os.path.join(self.local_folder, study_code, "study_params.json")
        
        try:
            with open(study_params_path, 'r') as f:
                study_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Study parameters file not found: {study_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in study parameters file {study_params_path}: {e}")
        
        return {
            "id": f"study_{study_code}",
            "fields": {
                "Code": study_code,
                "Name": study_data.get("study_name", f"Study {study_code}")
            }
        }
    
    def get_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """Load experiment record from exp_params.json."""
        # Extract study code and exp number from exp_code (e.g., "test_001")
        parts = exp_code.split('_')
        study_code = '_'.join(parts[:-1])
        
        exp_params_path = os.path.join(self.local_folder, study_code, exp_code, "exp_params.json")
        
        try:
            with open(exp_params_path, 'r') as f:
                exp_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Experiment parameters file not found: {exp_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in experiment parameters file {exp_params_path}: {e}")
        
        return {
            "id": f"exp_{exp_code}",
            "fields": {
                "Code": exp_code,
                "Name": exp_data.get("exp_name", f"Experiment {exp_code}")
            }
        }

    def get_study_parameters(self, study_record: Dict[str, Any]) -> Dict[str, Any]:
        """Load study parameters from study_params.json."""
        study_code = study_record["fields"]["Code"]
        study_params_path = os.path.join(self.local_folder, study_code, "study_params.json")
        
        try:
            with open(study_params_path, 'r') as f:
                study_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Study parameters file not found: {study_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in study parameters file {study_params_path}: {e}")
        
        parameters = study_data.get("Parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"Study parameters must be a dictionary in {study_params_path}")
        
        return parameters
    
    def get_performance_records(self, study_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load performance records from performance_records.json."""
        study_code = study_record["fields"]["Code"]
        perf_records_path = os.path.join(self.local_folder, study_code, "performance_records.json")
        
        with open(perf_records_path, 'r') as f:
            perf_data = json.load(f)
        
        return perf_data.get("records", [])
    
    def get_exp_variables(self, exp_record: Dict[str, Any]) -> Dict[str, Any]:
        """Load experiment variables from exp_params.json."""
        exp_code = exp_record["fields"]["Code"]
        parts = exp_code.split('_')
        study_code = '_'.join(parts[:-1])
        
        exp_params_path = os.path.join(self.local_folder, study_code, exp_code, "exp_params.json")
        
        with open(exp_params_path, 'r') as f:
            exp_data = json.load(f)
        
        return exp_data.get("Parameters", {})
    
    def get_study_dataset(self, study_record: Dict[str, Any], restrict_to_exp_codes: List[str] = []) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve experiment codes for the study. 
        Note: This method only returns experiment structure - actual data should be loaded via hierarchical pattern in LBPManager.
        """
        study_code = study_record["fields"]["Code"]
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
