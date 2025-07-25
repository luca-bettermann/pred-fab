import json
import os
from pathlib import Path
from typing import Dict, Any, List
from src.lbp_package.data_interface import DataInterface


class FileDataInterface(DataInterface):
    """Example data interface that reads from local JSON files."""
    
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
        
        parameters = study_data.get("parameters", {})
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
        
        return exp_data.get("parameters", {})
    
    def push_to_database(self, exp_record: Dict[str, Any], performance_code: str, value_dict: Dict[str, Any]) -> None:
        """Mock database push - just store for verification."""
        if not hasattr(self, 'pushed_data'):
            self.pushed_data = {}
        self.pushed_data[performance_code] = value_dict
    
    def update_system_performance(self, study_record: Dict[str, Any]) -> None:
        """Mock system performance update."""
        self.system_performance_updated = True

