# folder_navigator.py
import os
from typing import List, Optional, Dict, Any, Tuple
import shutil
import json
import pandas as pd
import numpy as np


class LocalDataInterface:
    """
    Manages local file system navigation and operations for LBP framework.
    
    Handles all local data operations including JSON/CSV file reading/writing,
    path management, and local caching of experiment data.
    """

    def __init__(
            self, 
            root_folder: str,
            local_folder: str, 
            server_folder: Optional[str] = None, 
            study_code: Optional[str] = None
            ) -> None:
        """
        Initialize folder navigator with base paths.

        Args:
            local_folder: Path to local data storage
            server_folder: Path to server data storage
            study_code: Optional study code for immediate initialization
        """
        self.root_folder: str = root_folder
        self.local_folder: str = local_folder
        self.server_folder: Optional[str] = server_folder
        self.study_code: Optional[str] = study_code
        self.study_folder: Optional[str] = None

        if study_code is not None:
            self.set_study_code(study_code)

    def set_study_code(self, study_code: str) -> None:
        """
        Set or change the study code.

        Args:
            study_code: Unique identifier for the study
        """
        assert isinstance(study_code, str) and study_code, "Study code must be a non-empty string"
        self.study_code = study_code
        self.study_folder = os.path.join(self.local_folder, self.study_code)

    def get_experiment_code(self, exp_nr: int) -> str:
        """
        Generate experiment code from experiment number.

        Args:
            exp_nr: Experiment number

        Returns:
            Formatted experiment code (e.g., "STUDY_001")
        """
        assert isinstance(exp_nr, int) and exp_nr >= 0, "Experiment number must be a non-negative integer"
        return f"{self.study_code}_{str(exp_nr).zfill(3)}"

    def get_experiment_folder(self, exp_code: str) -> str:
        """
        Get full path to local experiment folder.

        Args:
            exp_nr: Experiment number

        Returns:
            Full path to experiment folder
        """
        assert isinstance(exp_code, str) and exp_code, "Experiment code must be a non-empty string"
        if self.study_folder is None:
            raise ValueError("Study code must be set before getting experiment folder")
        return os.path.join(self.study_folder, exp_code)

    def get_server_experiment_folder(self, exp_code: str) -> str:
        """
        Get full path to server experiment folder.

        Args:
            exp_code: Experiment code

        Returns:
            Full path to server experiment folder
        """
        if self.server_folder is None:
            raise ValueError("Server folder must be set before getting server experiment folder")
        if self.study_code is None:
            raise ValueError("Study code must be set before getting experiment folder")
        assert isinstance(exp_code, str) and exp_code, "Experiment code must be a non-empty string"
        return os.path.join(self.server_folder, self.study_code, exp_code)

    def get_experiment_file_path(self, exp_code: str, filename: str) -> str:
        """
        Get full path to a file within an experiment folder.

        Args:
            exp_code: Experiment code
            filename: Name of the file (can include subdirectory, e.g., "results/performance.json")

        Returns:
            Full path to the file
        """
        assert isinstance(exp_code, str) and exp_code, "Experiment code must be a non-empty string"
        assert isinstance(filename, str) and filename, "Filename must be a non-empty string"
        if self.study_folder is None:
            raise ValueError("Study code must be set before getting experiment file path")
        return os.path.join(self.study_folder, exp_code, filename)

    def get_experiment_results_folder(self, exp_code: str) -> str:
        """
        Get full path to experiment results folder.

        Args:
            exp_code: Experiment code

        Returns:
            Full path to experiment results folder
        """
        assert isinstance(exp_code, str) and exp_code, "Experiment code must be a non-empty string"
        if self.study_folder is None:
            raise ValueError("Study code must be set before getting experiment results folder")
        return os.path.join(self.study_folder, exp_code, "results")

    def list_experiments(self) -> List[str]:
        """
        List all experiment folder names within the study folder.

        Returns:
            List of experiment folder names
        """
        if self.study_folder is None:
            raise ValueError("Study code must be set before listing experiments")
        
        return [
            d
            for d in os.listdir(self.study_folder)
            if os.path.isdir(os.path.join(self.study_folder, d))
        ]

    def copy_to_folder(self, src_path: str, target_folder: str) -> str:
        """
        Copy file or folder to target directory.

        Args:
            src_path: Source file or folder path
            target_folder: Target directory path

        Returns:
            Path to copied file or folder

        Raises:
            FileNotFoundError: If source path doesn't exist
            NotADirectoryError: If target is not a directory
            RuntimeError: If copy operation fails
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source path {src_path} does not exist.")

        if not os.path.isdir(target_folder):
            raise NotADirectoryError(f"Target folder {target_folder} does not exist or is not a directory.")

        base_name = os.path.basename(src_path.rstrip(os.sep))
        dst_path = os.path.join(target_folder, base_name)

        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)

            return dst_path

        except Exception as e:
            raise RuntimeError(f"Failed to copy {src_path}: {str(e)}")

    def check_folder_access(self, folder_path: str) -> bool:
        """
        Verify folder accessibility.

        Returns:
            True if folder is accessible

        Raises:
            ConnectionError: If folder is not accessible
        """
        if folder_path is None:
            raise ValueError("Folder path must be set to check access.")

        is_connected = os.path.exists(folder_path)
        if not is_connected:
            raise ConnectionError(f"Folder access {folder_path}: FAILED")

        return is_connected

    # === DATA LOADING METHODS ===
    def load_aggr_metrics(self, exp_codes: List[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load aggregated metrics from local files for multiple experiments."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}", "results"],
            filename="{code}_performance",
            require_study_code=True
        )

    def load_metrics_arrays(self, exp_codes: List[str]) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]]]:
        """Load metrics arrays from local CSV files for multiple experiments."""
        if not self.study_code:
            raise ValueError("Study code must be set before loading experiments")
            
        missing_exp_codes = []
        metrics_arrays_dict = {}
        
        for exp_code in exp_codes:
            try:
                # Construct file paths
                results_dir = os.path.join(self.local_folder, self.study_code, exp_code, "results")
                
                # Load metrics arrays from CSV files
                metrics_arrays_dict[exp_code] = {}
                if os.path.exists(results_dir):
                    found_csv = False
                    for filename in os.listdir(results_dir):
                        if filename.endswith('.csv') and filename.startswith(exp_code):
                            found_csv = True
                            # Extract performance code from filename
                            performance_code = filename.replace(f"{exp_code}_", "").replace(".csv", "")
                            csv_path = os.path.join(results_dir, filename)
                            
                            # Load CSV as numpy array
                            df = pd.read_csv(csv_path)
                            metrics_arrays_dict[exp_code][performance_code] = df.values
                    
                    if not found_csv:
                        missing_exp_codes.append(exp_code)
                else:
                    missing_exp_codes.append(exp_code)
                            
            except Exception as e:
                print(f"Failed to load metrics arrays for experiment {exp_code}: {e}")
                missing_exp_codes.append(exp_code)
                
        return missing_exp_codes, metrics_arrays_dict

    def load_exp_params(self, exp_codes: List[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load experiment parameters from local files for multiple experiments."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}"],
            filename="exp_params",
            require_study_code=True,
            extract_key="Parameters"
        )

    def load_study_records(self, study_codes: List[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load study records from local files."""
        return self._load_files_generic(
            codes=study_codes,
            subdirs=["{code}"],
            filename="study_params",
            require_study_code=False
        )

    def load_exp_records(self, exp_codes: List[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load experiment records from local files."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}"],
            filename="exp_record",
            require_study_code=True
        )

    def _load_files_generic(self, 
                            codes: List[str], 
                            subdirs: List[str],
                            filename: str,
                            require_study_code: bool = True,
                            extract_key: Optional[str] = None) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Simple load function for single JSON files."""
        if require_study_code and not self.study_code:
            raise ValueError("Study code must be set before loading experiments")
            
        missing_codes = []
        result_dict = {}
        
        for code in codes:
            try:
                # Build file path
                if require_study_code:
                    path_parts = [self.local_folder, self.study_code] + subdirs
                else:
                    path_parts = [self.local_folder] + subdirs
                
                path_parts = [part.replace("{code}", code) for part in path_parts]
                file_name = filename.replace("{code}", code)
                file_path = os.path.join(*path_parts, f"{file_name}.json")
                
                # Load file
                if not os.path.exists(file_path):
                    missing_codes.append(code)
                    continue
                    
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if extract_key:
                        data = data.get(extract_key, {})
                    result_dict[code] = data
                    
            except Exception as e:
                print(f"Failed to load data for {code}: {e}")
                missing_codes.append(code)
                
        return missing_codes, result_dict

    # === DATA SAVING METHODS ===
    def save_aggr_metrics(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]]) -> None:
        """Save aggregated metrics to local files (single summary file per experiment)."""
        self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}", "results"],
            filename="{code}_performance",
            wrap_in_parameters=False
        )

    def save_metrics_arrays(self, exp_codes: List[str], data: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Save metrics arrays to local CSV files."""
        if not self.study_code:
            raise ValueError("Study code must be set before saving")
            
        for exp_code in exp_codes:
            if exp_code in data:
                # Build directory path
                results_dir = os.path.join(self.local_folder, self.study_code, exp_code, "results")
                os.makedirs(results_dir, exist_ok=True)
                
                # Handle nested data (iterate over performance codes)
                for perf_code, array in data[exp_code].items():
                    file_name = f"{exp_code}_{perf_code}.csv"
                    file_path = os.path.join(results_dir, file_name)
                    df = pd.DataFrame(array)
                    df.to_csv(file_path, index=False)

    def save_exp_params(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]]) -> None:
        """Save experiment parameters to local files."""
        self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}"],
            filename="exp_params",
            wrap_in_parameters=True
        )

    def save_study_records(self, study_codes: List[str], data: Dict[str, Dict[str, Any]]) -> None:
        """Save study records to local files."""
        self._save_files_generic(
            codes=study_codes,
            data=data,
            subdirs=["{code}"],
            filename="study_record",
        )

    def save_exp_records(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]]) -> None:
        """Save experiment records to local files."""
        self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}"],
            filename="exp_record",
        )

    def _save_files_generic(self, 
                            codes: List[str], 
                            data: Dict[str, Dict[str, Any]], 
                            subdirs: List[str],
                            filename: str,
                            wrap_in_parameters: bool = False) -> None:
        """Simple save function for non-nested data."""
        if not self.study_code:
            raise ValueError("Study code must be set before saving")
            
        for code in codes:
            if code in data:
                # Build directory path
                dir_parts = [self.local_folder, self.study_code] + subdirs
                dir_parts = [part.replace("{code}", code) for part in dir_parts]
                dir_path = os.path.join(*dir_parts)
                os.makedirs(dir_path, exist_ok=True)
                
                # Handle simple data
                code_data = data[code]
                if wrap_in_parameters:
                    code_data = {"Parameters": code_data}
                
                file_name = filename.replace("{code}", code)
                file_path = os.path.join(dir_path, f"{file_name}.json")
                
                with open(file_path, 'w') as f:
                    json.dump(code_data, f, indent=2)