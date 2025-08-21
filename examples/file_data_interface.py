import json
import os
import datetime
from typing import Dict, Any, List
from lbp_package import DataInterface


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
    
    def get_study_dataset(self, study_record: Dict[str, Any], restrict_to_exp_codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve the complete dataset for a study by reading all experiment files.
        
        Returns:
            Dictionary containing experiment data with parameters and performances.
            Format: {exp_code: {param_name: value, performance_code: value}}
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
            exp_data = {}

            # Skip if experiment code is not in the restricted list and list not empty
            if restrict_to_exp_codes and exp_code not in restrict_to_exp_codes:
                continue

            try:
                # Load experiment parameters
                exp_params_path = os.path.join(item_path, "exp_params.json")
                if os.path.exists(exp_params_path):
                    with open(exp_params_path, 'r') as f:
                        exp_params = json.load(f).get("Parameters", {})
                        exp_data.update(exp_params)

                # Load performance json file if it exists
                results_dir = os.path.join(item_path, "results")
                assert os.path.exists(results_dir), f"Results directory does not exist: {results_dir}. Run evaluation first"

                performance_file = os.path.join(results_dir, f"{exp_code}_performance.json")
                assert os.path.exists(performance_file), f"Performance file does not exist: {performance_file}. Run evaluation first"

                with open(performance_file, 'r') as f:
                    perf_data = json.load(f)
                    # Extract the aggregated Value from performance metrics
                    if "Value" in perf_data:
                        exp_data["Value"] = perf_data["Value"]

                # Look for performance result files
                for result_file in os.listdir(results_dir):
                    if result_file.endswith("_performance.json"):
                            performance_code = result_file.replace(f"{exp_code}_", "").replace("_performance.json", "")
                            
                            with open(os.path.join(results_dir, result_file), 'r') as f:
                                perf_data = json.load(f)
                                # Extract the aggregated Value from performance metrics
                                if "Value" in perf_data:
                                    exp_data[performance_code] = perf_data["Value"]
                
                # Only add experiment if we have some data
                if exp_data:
                    dataset[exp_code] = exp_data
                    
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # Skip experiments with missing or corrupted data
                print(f"Warning: Could not load data for experiment {exp_code}: {e}")
                continue
        
        return dataset

    def push_to_database(self, exp_record: Dict[str, Any], performance_metrics: Dict[str, Dict[str, Any]]) -> None:
        """
        Save performance results to local JSON files in the experiment's results directory.
        
        Args:
            exp_record: Experiment record dictionary
            value_dict: Dictionary containing performance values and metadata
                       Expected to contain 'Value' key with the aggregated performance
        """
        exp_code = exp_record["fields"]["Code"]
        parts = exp_code.split('_')
        study_code = '_'.join(parts[:-1])
        
        # Create results directory
        exp_dir = os.path.join(self.local_folder, study_code, exp_code)
        results_dir = os.path.join(exp_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Create performance json file
        perf_file = os.path.join(results_dir, f"{exp_code}_performance.json")
        with open(perf_file, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        return datetime.datetime.now().isoformat()
