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
        if not isinstance(study_code, str) or not study_code:
            raise ValueError("Study code must be a non-empty string")
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
        if not isinstance(exp_nr, int) or exp_nr < 0:
            raise ValueError("Experiment number must be a non-negative integer")
        return f"{self.study_code}_{str(exp_nr).zfill(3)}"

    def get_experiment_folder(self, exp_code: str) -> str:
        """
        Get full path to local experiment folder.

        Args:
            exp_nr: Experiment number

        Returns:
            Full path to experiment folder
        """
        if not isinstance(exp_code, str) or not exp_code:
            raise ValueError("Experiment code must be a non-empty string")
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
    def load_study_records(self, study_codes: List[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load study records from local files."""
        return self._load_files_generic(
            codes=study_codes,
            subdirs=["{code}"],
            filename="study_record",
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
        
    def load_aggr_metrics(self, exp_codes: List[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load aggregated metrics from local files for multiple experiments."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}", "results"],
            filename="performance",
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
                        if filename.endswith('.csv'):
                            found_csv = True
                            # Extract performance code from filename
                            performance_code = filename.replace(".csv", "")
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

    # === DATA SAVING METHODS ===
    def save_aggr_metrics(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """Save aggregated metrics to local files (single summary file per experiment)."""
        return self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}", "results"],
            filename="performance",
            recompute=recompute,
            wrap_in_parameters=False
        )

    def save_metrics_arrays(
            self, 
            exp_codes: List[str], 
            metrics_array: Dict[str, Dict[str, np.ndarray]],
            recompute: bool,
            **kwargs
            ) -> bool:
        """Save metrics arrays to local CSV files with custom column names.
        
        Args:
            exp_codes: List of experiment codes to save
            metrics_array: Nested dictionary {exp_code: {perf_code: np.ndarray}}
            **kwargs: Additional parameters including:
                - metric_names: Dict mapping performance codes to column names
                - dim_combinations: Dict with dimensional combination data
                - dim_iterators: Dict with dimensional iterator names
        """
        if not self.study_code:
            raise ValueError("Study code must be set before saving")
        if not all(key in kwargs for key in ['metric_names', 'dim_combinations', 'dim_iterators']):
            raise ValueError("Missing required keyword arguments")

        # Extract parameters from kwargs
        metric_names = kwargs.get('metric_names', {})
        dim_combinations = kwargs.get('dim_combinations', {})
        dim_iterators = kwargs.get('dim_iterators', {})
                    
        saved = False
        for exp_code in exp_codes:
            if exp_code in metrics_array:
                metric_array = metrics_array[exp_code]

                if not (metric_array.keys() == metric_names.keys() == dim_combinations.keys() == dim_iterators.keys()):
                    raise ValueError(f"Incoherent performance codes in dictionaries for experiment {exp_code}")

                # Build directory path
                results_dir = os.path.join(self.local_folder, self.study_code, exp_code, "results")
                os.makedirs(results_dir, exist_ok=True)
                
                # Handle nested data (iterate over performance codes)
                for perf_code, array in metric_array.items():
                    file_name = f"{perf_code}.csv"
                    file_path = os.path.join(results_dir, file_name)

                    # Only save if file doesn't exist or recompute is True
                    if not os.path.exists(file_path) or recompute:
                        # unpack
                        names = metric_names[perf_code]
                        dim_comb = dim_combinations[perf_code][exp_code]
                        dim_iter = dim_iterators[perf_code]

                        # build matrix
                        values = array.reshape(-1, len(names))
                        matrix = np.empty((len(values), len(dim_iter + names)))
                        matrix[:, len(dim_iter):] = values
                        if dim_comb.size:
                            matrix[:, :(len(dim_iter))] = dim_comb.reshape(-1, len(dim_iter))

                        # save as csv
                        df = pd.DataFrame(matrix, columns=dim_iter + names)
                        df.to_csv(file_path, index=False)
                        if not saved:
                            saved = True

            else:
                raise ValueError(f"No data found for experiment code {exp_code} to save metrics arrays.")
        # Return whether any files were saved
        return saved
    
    def save_study_records(self, study_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """Save study records to local files."""
        return self._save_files_generic(
            codes=study_codes,
            data=data,
            subdirs=[],
            filename="study_record",
            recompute=recompute,
        )

    def save_exp_records(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """Save experiment records to local files."""
        return self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}"],
            filename="exp_record",
            recompute=recompute,
        )

    # === INTERNAL METHODS ===
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

    def _save_files_generic(self, 
                            codes: List[str], 
                            data: Dict[str, Dict[str, Any]], 
                            subdirs: List[str],
                            filename: str,
                            recompute: bool,
                            wrap_in_parameters: bool = False) -> bool:
        """Simple save function for non-nested data."""
        if not self.study_code:
            raise ValueError("Study code must be set before saving")
            
        saved = False
        for code in codes:
            if code in data:
                # Build directory path
                dir_parts = [self.local_folder, self.study_code] + subdirs
                dir_parts = [part.replace("{code}", code) for part in dir_parts]
                dir_path = os.path.join(*dir_parts)
                os.makedirs(dir_path, exist_ok=True)
                
                file_name = filename.replace("{code}", code)
                file_path = os.path.join(dir_path, f"{file_name}.json")
                
                # Only save if file doesn't exist or recompute is True
                if not os.path.exists(file_path) or recompute:
                    # Handle simple data
                    code_data = data[code]
                    if wrap_in_parameters:
                        code_data = {"Parameters": code_data}
                    
                    with open(file_path, 'w') as f:
                        json.dump(code_data, f, indent=2)
                    if not saved:
                        saved = True
            else:
                raise ValueError(f"No data found for code {code} to save.")
        # return wheter or not files actually were saved
        return saved
