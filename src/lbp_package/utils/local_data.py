# folder_navigator.py
import os
from typing import List, Optional, Dict, Any, Tuple
import shutil
import json
import pandas as pd
import numpy as np


class LocalData:
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
            schema_id: Optional[str] = None
            ) -> None:
        """
        Initialize folder navigator with base paths.

        Args:
            root_folder: Project root folder
            local_folder: Path to local data storage
            server_folder: Path to server data storage
            schema_id: Optional schema ID for immediate initialization
        """
        self.root_folder: str = root_folder
        self.local_folder: str = local_folder
        self.server_folder: Optional[str] = server_folder
        self.schema_id: Optional[str] = schema_id
        self.schema_folder: Optional[str] = None

        if schema_id is not None:
            self.set_schema_id(schema_id)

    # === PUBLIC API METHODS ===
    def set_schema_id(self, schema_id: str) -> None:
        """
        Set or change the schema ID.
        
        Args:
            schema_id: Schema identifier (e.g., 'schema_001')
        """
        if not isinstance(schema_id, str) or not schema_id:
            raise ValueError("Schema ID must be a non-empty string")
        
        self.schema_id = schema_id
        self.schema_folder = os.path.join(self.local_folder, self.schema_id)

    def get_experiment_code(self, exp_nr: int) -> str:
        """Generate experiment code from experiment number."""
        if not isinstance(exp_nr, int) or exp_nr < 0:
            raise ValueError("Experiment number must be a non-negative integer")
        return f"{self.schema_id}_{str(exp_nr).zfill(3)}"

    def get_experiment_folder(self, exp_code: str) -> str:
        """Get full path to local experiment folder."""
        if not isinstance(exp_code, str) or not exp_code:
            raise ValueError("Experiment code must be a non-empty string")
        if self.schema_folder is None:
            raise ValueError("Schema ID must be set before getting experiment folder")
        return os.path.join(self.schema_folder, exp_code)

    def get_server_experiment_folder(self, exp_code: str) -> str:
        """Get full path to server experiment folder."""
        if self.server_folder is None:
            raise ValueError("Server folder must be set before getting server experiment folder")
        if self.schema_id is None:
            raise ValueError("Schema ID must be set before getting experiment folder")
        assert isinstance(exp_code, str) and exp_code, "Experiment code must be a non-empty string"
        return os.path.join(self.server_folder, self.schema_id, exp_code)

    def get_experiment_file_path(self, exp_code: str, filename: str) -> str:
        """Get full path to a file within an experiment folder."""
        assert isinstance(exp_code, str) and exp_code, "Experiment code must be a non-empty string"
        assert isinstance(filename, str) and filename, "Filename must be a non-empty string"
        if self.schema_folder is None:
            raise ValueError("Schema ID must be set before getting experiment file path")
        return os.path.join(self.schema_folder, exp_code, filename)

    def list_experiments(self) -> List[str]:
        """List all experiment folder names within the schema folder."""
        if self.schema_folder is None:
            raise ValueError("Schema ID must be set before listing experiments")
        
        return [
            d
            for d in os.listdir(self.schema_folder)
            if os.path.isdir(os.path.join(self.schema_folder, d))
        ]

    def copy_to_folder(self, src_path: str, target_folder: str) -> str:
        """Copy file or folder to target directory."""
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

    def check_folder_access(self, folder_path: str):
        """Verify folder accessibility."""
        if folder_path is None:
            raise ValueError("Folder path must be set to check access.")

        is_connected = os.path.exists(folder_path)
        if not is_connected:
            raise ConnectionError(f"Folder access {folder_path}: FAILED")

    def check_availability(self, code: str, memory: Dict[str, Any]) -> None:
        if code not in memory:
            raise ValueError(f"{code} is not available in memory.")
        
    # === DATA LOADING METHODS ===
    def load_exp_records(self, exp_codes: List[str], **kwargs) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load experiment records from local files."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=[str(self.schema_id), "{code}"],
            filename="exp_record",
            contains_columns=False
        )
        
    def load_aggr_metrics(self, exp_codes: List[str], **kwargs) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load aggregated metrics from local files for multiple experiments."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=[str(self.schema_id), "{code}", "arrays"],
            filename="performance",
            contains_columns=False
        )

    def load_metrics_arrays(self, exp_codes: List[str], **kwargs) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load metrics arrays from local files for multiple experiments."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=[str(self.schema_id), "{code}", "arrays"],
            filename=f"{kwargs.get('perf_code')}",
            contains_columns=True
        )

    # === DATA SAVING METHODS ===
    def save_exp_records(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """Save experiment records to local files."""
        return self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}"],
            filename="exp_record",
            recompute=recompute,
        )

    def save_aggr_metrics(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """Save aggregated metrics to local files (single summary file per experiment)."""
        return self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}", "arrays"],
            filename="performance",
            recompute=recompute
        )

    def save_metrics_arrays(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], recompute: bool, **kwargs) -> bool:
        """Save aggregated metrics to local files (single summary file per experiment)."""
        return self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}", "arrays"],
            filename=f"{kwargs.get('perf_code')}",
            recompute=recompute,
            column_names=kwargs.get('column_names')
        )
    
    def save_schema(self, schema_dict: Dict[str, Any], recompute: bool = False) -> bool:
        """
        Save schema JSON to local storage.
        
        Args:
            schema_dict: Schema dictionary to save
            recompute: Overwrite if exists
            
        Returns:
            True if saved, False if skipped
        """
        if not self.schema_folder:
            raise ValueError("Schema folder not configured")
        
        schema_file = os.path.join(self.schema_folder, "schema.json")
        os.makedirs(self.schema_folder, exist_ok=True)
        
        # Only save if doesn't exist or recompute is True
        if not os.path.exists(schema_file) or recompute:
            with open(schema_file, 'w') as f:
                json.dump(schema_dict, f, indent=2)
            return True
        return False
    
    # === PRIVATE METHODS ===
    def _load_files_generic(self, 
                            codes: List[str], 
                            subdirs: List[str],
                            filename: str,
                            contains_columns: bool = False) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Simple load function for single JSON files."""
        if not self.schema_id:
            raise ValueError("Schema ID must be set before loading experiments")
            
        missing_codes = []
        result_dict = {}
        file_type = "csv" if contains_columns else "json"

        for code in codes:
            try:
                # Build file path
                path_parts = [self.local_folder] + subdirs
                path_parts = [part.replace("{code}", code) for part in path_parts]
                file_name = filename.replace("{code}", code)
                file_path = os.path.join(*path_parts, f"{file_name}.{file_type}")
                
                # Check if file exists
                if not os.path.exists(file_path):
                    missing_codes.append(code)
                    continue

                # Load file
                if file_type == "csv":
                    df = pd.read_csv(file_path)
                    result_dict[code] = df.values
                else:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        result_dict[code] = data

            except Exception as e:
                print(f"Failed to load data for {code}: {e}")
                missing_codes.append(code)
                
        return missing_codes, result_dict

    def _save_files_generic(self, 
                            codes: List[str], 
                            data: Dict[str, Any], 
                            subdirs: List[str],
                            filename: str,
                            recompute: bool,
                            column_names: Optional[List[str]] = None) -> bool:
        """Simple save function for non-nested data."""
        if not self.schema_id:
            raise ValueError("Schema ID must be set before saving")
            
        saved = False
        file_type = "csv" if column_names is not None else "json"
        for code in codes:
            if code in data:
                # Build directory path
                dir_parts = [self.local_folder, self.schema_id] + subdirs
                dir_parts = [part.replace("{code}", code) for part in dir_parts]
                dir_path = os.path.join(*dir_parts)
                os.makedirs(dir_path, exist_ok=True)
                
                file_name = filename.replace("{code}", code)
                file_path = os.path.join(dir_path, f"{file_name}.{file_type}")

                # Only save if file doesn't exist or recompute is True
                if not os.path.exists(file_path) or recompute:
                    # Handle simple data
                    code_data = data[code]

                    # csv file if it has column names
                    if file_type == "csv" and column_names is not None:
                        # reshape if we actually have dimensions
                        if len(code_data) > 1:
                            code_data = code_data.reshape(-1, len(column_names))
                        df = pd.DataFrame(code_data, columns=column_names)
                        df.to_csv(file_path, index=False)
                    else:
                        with open(file_path, 'w') as f:
                            json.dump(code_data, f, indent=2)

                    # flag that at least one file was saved
                    if not saved:
                        saved = True
            else:
                raise ValueError(f"No data found for code {code} to save.")
        # return whether or not files actually were saved
        return saved
