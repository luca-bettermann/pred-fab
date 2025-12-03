# folder_navigator.py
import os
from typing import List, Optional, Dict, Any, Tuple
import shutil
import json
import pandas as pd
import numpy as np


class LocalData:
    """
    Manages local file system operations for LBP framework.
    
    - JSON/CSV file reading/writing
    - Path management and folder navigation
    - Local caching of experiment data
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
        """Set or change the schema ID."""
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

    def check_folder_access(self, folder_path: str) -> None:
        """Verify folder accessibility."""
        if folder_path is None:
            raise ValueError("Folder path must be set to check access.")

        is_connected = os.path.exists(folder_path)
        if not is_connected:
            raise ConnectionError(f"Folder access {folder_path}: FAILED")

    def check_availability(self, code: str, memory: Dict[str, Any]) -> None:
        """Check if code exists in memory."""
        if code not in memory:
            raise ValueError(f"Code not available in memory: {code}")
        
    # === DATA LOADING METHODS ===
    def load_parameters(self, exp_codes: List[str], **kwargs) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load experiment parameters from local files."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=[str(self.schema_id), "{code}"],
            filename="parameters",
            file_format="json"
        )
        
    def load_performance(self, exp_codes: List[str], **kwargs) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load performance metrics from local files."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=[str(self.schema_id), "{code}"],
            filename="performance",
            file_format="json"
        )

    def load_features(self, exp_codes: List[str], **kwargs) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Load feature arrays from local files."""
        feature_name = kwargs.get('feature_name')
        if not feature_name:
            raise ValueError("feature_name required in kwargs for load_features")
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=[str(self.schema_id), "{code}"],
            filename=feature_name,
            file_format="csv"
        )

    # === DATA SAVING METHODS ===
    def save_parameters(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], 
                        recompute: bool, **kwargs) -> bool:
        """Save experiment parameters to local files."""
        return self._save_files_generic(
            codes=exp_codes, data=data,
            subdirs=["{code}"], filename="parameters", recompute=recompute,
            file_format="json"
        )

    def save_performance(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], 
                         recompute: bool, **kwargs) -> bool:
        """Save performance metrics to local files."""
        return self._save_files_generic(
            codes=exp_codes, data=data,
            subdirs=["{code}"], filename="performance", recompute=recompute,
            file_format="json"
        )

    def save_features(self, exp_codes: List[str], data: Dict[str, Dict[str, Any]], 
                      recompute: bool, feature_name: str, column_names: Optional[List[str]], **kwargs) -> bool:
        """Save feature arrays to local files."""
        if not feature_name:
            raise ValueError("feature_name required in kwargs for save_features")
        return self._save_files_generic(
            codes=exp_codes, data=data,
            subdirs=["{code}"], filename=feature_name, 
            recompute=recompute, column_names=column_names,
            file_format="csv"
        )
    
    def save_schema(self, schema_dict: Dict[str, Any], recompute: bool = False) -> bool:
        """Save schema JSON to local storage."""
        if not self.schema_folder:
            raise ValueError("Schema folder not configured")
        
        # Build path and ensure directory exists
        schema_file = os.path.join(self.schema_folder, "schema.json")
        os.makedirs(self.schema_folder, exist_ok=True)
        
        # Save if doesn't exist or recompute is True
        if not os.path.exists(schema_file) or recompute:
            with open(schema_file, 'w') as f:
                json.dump(schema_dict, f, indent=2)
            return True
        return False
    
    # === PRIVATE METHODS ===
    def _load_files_generic(
        self, codes: List[str], subdirs: List[str], filename: str, 
        contains_columns: bool = False, file_format: str = "json"
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """Generic file loader for JSON/CSV files."""
        if not self.schema_id:
            raise ValueError("Schema ID must be set")
            
        missing_codes = []
        result_dict = {}
        
        # Determine file type
        file_type = file_format

        for code in codes:
            try:
                # Build file path
                path_parts = [self.local_folder] + subdirs
                path_parts = [part.replace("{code}", code) for part in path_parts]
                file_name = filename.replace("{code}", code)
                file_path = os.path.join(*path_parts, f"{file_name}.{file_type}")
                
                # Check existence and load
                if not os.path.exists(file_path):
                    missing_codes.append(code)
                    continue

                if file_type == "csv":
                    df = pd.read_csv(file_path)
                    result_dict[code] = df.values
                else:
                    with open(file_path, 'r') as f:
                        result_dict[code] = json.load(f)
            except Exception as e:
                print(f"Failed to load {code}: {e}")
                missing_codes.append(code)
                
        return missing_codes, result_dict

    def _save_files_generic(
        self, codes: List[str], data: Dict[str, Any], subdirs: List[str], 
        filename: str, recompute: bool, column_names: Optional[List[str]] = None,
        file_format: str = "json"
    ) -> bool:
        """Generic file saver for JSON/CSV files."""
        if not self.schema_id:
            raise ValueError("Schema ID must be set")
            
        saved = False
        file_type = file_format
        
        for code in codes:
            if code not in data:
                raise ValueError(f"No data found for code: {code}")
            
            # Build directory path
            dir_parts = [self.local_folder, self.schema_id] + subdirs
            dir_parts = [part.replace("{code}", code) for part in dir_parts]
            dir_path = os.path.join(*dir_parts)
            os.makedirs(dir_path, exist_ok=True)
            
            # Build file path
            file_name = filename.replace("{code}", code)
            file_path = os.path.join(dir_path, f"{file_name}.{file_type}")

            # Save if doesn't exist or recompute is True
            if not os.path.exists(file_path) or recompute:
                code_data = data[code]

                if file_type == "csv":
                    # Reshape and save as CSV
                    # If column_names is None, we create default ones or save without header?
                    # Pandas to_csv needs columns if we want header.
                    # If column_names is None, we can save with header=False
                    
                    if column_names is not None:
                        if len(code_data) > 1 and len(code_data.shape) > 1:
                             # Ensure dimensions match
                             pass
                        df = pd.DataFrame(code_data, columns=column_names)
                        df.to_csv(file_path, index=False)
                    else:
                        df = pd.DataFrame(code_data)
                        df.to_csv(file_path, index=False, header=False)
                else:
                    # Save as JSON
                    with open(file_path, 'w') as f:
                        json.dump(code_data, f, indent=2)

                saved = True
                
        return saved
