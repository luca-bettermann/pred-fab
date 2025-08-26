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
        
    def pull_study_record(self, study_code: str) -> Dict[str, Any]:
        """Load study record from study_record.json."""
        study_params_path = os.path.join(self.local_folder, "study_record.json")
        
        try:
            with open(study_params_path, 'r') as f:
                study_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Study parameters file not found: {study_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in study parameters file {study_params_path}: {e}")
        
        # Return the study data directly from the JSON file
        # The JSON file should already contain the required fields: id, Code, Parameters
        return study_data
    
    def pull_exp_record(self, exp_code: str) -> Dict[str, Any]:
        """Load experiment record from exp_record.json."""
        exp_params_path = os.path.join(self.local_folder, exp_code, "exp_record.json")
        
        try:
            with open(exp_params_path, 'r') as f:
                exp_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Experiment parameters file not found: {exp_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in experiment parameters file {exp_params_path}: {e}")
        
        # Return the experiment data directly from the JSON file
        # The JSON file should already contain the required fields: id, Code, Parameters
        return exp_data
    
    def pull_performance_records(self, study_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load performance records from performance_records.json."""
        study_code = study_record["Code"]
        perf_records_path = os.path.join(self.local_folder, study_code, "performance_records.json")
        
        with open(perf_records_path, 'r') as f:
            perf_data = json.load(f)
        
        return perf_data.get("records", [])
    
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
