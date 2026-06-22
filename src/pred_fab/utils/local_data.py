# folder_navigator.py
import os
import logging
from typing import Any, Callable
import shutil
import json

from ..utils.enum import BlockType, FileFormat
# pandas is imported lazily in the CSV read/write methods so this module (on the
# torch/pandas-free model import path via Dataset) stays pandas-free at import.


class LocalData:
    """Local filesystem handler for JSON/CSV experiment data — path management, reading, and writing."""

    def __init__(
            self,
            root_folder: str,
            local_folder_name: str = "local"
            ) -> None:
        self.root_folder: str = root_folder
        self.local_folder: str = os.path.join(root_folder, local_folder_name)

        self.schema_id: str | None = None
        self.schema_folder: str | None = None

    # === PUBLIC API METHODS ===

    def get_log_folder(self, name: str) -> str:
        """Set or change the log folder within local storage."""
        if not isinstance(name, str) or not name:
            raise ValueError("Log folder name must be a non-empty string")
        
        log_folder = os.path.join(self.local_folder, name)
        os.makedirs(log_folder, exist_ok=True)
        return log_folder

    def set_schema(self, schema_id: str) -> None:
        """Set or change the schema ID."""
        if not isinstance(schema_id, str) or not schema_id:
            raise ValueError("Schema ID must be a non-empty string")
        
        self.schema_id = schema_id
        self.schema_folder = os.path.join(self.local_folder, self.schema_id)
        os.makedirs(self.schema_folder, exist_ok=True)

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

    def get_experiment_file_path(self, exp_code: str, filename: str) -> str:
        """Get full path to a file within an experiment folder."""
        if not isinstance(exp_code, str) or not exp_code:
            raise TypeError("Experiment code must be a non-empty string")
        if not isinstance(filename, str) or not filename:
            raise TypeError("Filename must be a non-empty string")
        if self.schema_folder is None:
            raise ValueError("Schema ID must be set before getting experiment file path")
        return os.path.join(self.schema_folder, exp_code, filename)

    def list_experiments(self) -> list[str]:
        """List all experiment codes within the schema folder.

        Walks nested directories to find leaf folders (those containing
        files or no subdirectories), supporting codes like
        ``discovery/000`` alongside flat codes like ``exp_001``.

        Returned codes are relative to the schema folder
        (e.g. ``discovery/000``).
        """
        if self.schema_folder is None:
            raise ValueError("Schema ID must be set before listing experiments")
        if not os.path.exists(self.schema_folder):
            return []

        experiments = []
        for root, dirs, files in os.walk(self.schema_folder):
            if not dirs or files:
                rel = os.path.relpath(root, self.schema_folder)
                if rel != ".":
                    experiments.append(rel)
        return sorted(experiments)

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

    def check_availability(self, code: str, memory: dict[str, Any]) -> None:
        """Check if code exists in memory."""
        if code not in memory:
            raise ValueError(f"Code not available in memory: {code}")
        
    # === DATA LOADING METHODS ===

    def load_parameters(self, exp_codes: list[str], **kwargs) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Load experiment parameters from local files."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}"],
            filename=BlockType.PARAMETERS.value,
            file_format=FileFormat.JSON
        )
        
    def load_performance(self, exp_codes: list[str], **kwargs) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Load performance metrics from local files."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}"],
            filename=BlockType.PERF_ATTRS.value,
            file_format=FileFormat.JSON
        )

    def load_parameter_updates(self, exp_codes: list[str], **kwargs) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Load parameter update event logs from local files."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}"],
            filename=BlockType.PARAM_UPDATES.value,
            file_format=FileFormat.JSON
        )

    def load_metadata(self, exp_codes: list[str], **kwargs) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Load experiment metadata (e.g. ``dataset_code``) from local files."""
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}"],
            filename=BlockType.METADATA.value,
            file_format=FileFormat.JSON
        )

    def load_features(self, exp_codes: list[str], **kwargs) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Load feature arrays from local files."""
        feature_name = kwargs.get('feature_name')
        if not feature_name:
            raise ValueError("feature_name required in kwargs for load_features")
        return self._load_files_generic(
            codes=exp_codes,
            subdirs=["{code}"],
            filename=feature_name,
            file_format=FileFormat.CSV
        )

    # === DATA SAVING METHODS ===
    
    def save_schema(self, schema_dict: dict[str, Any], recompute: bool = False) -> bool:
        """Save schema JSON to local storage."""
        if not self.schema_folder:
            raise ValueError("Schema folder not configured")
        
        # Build path and ensure directory exists
        schema_file = os.path.join(self.schema_folder, "schema." + FileFormat.JSON.value)
        os.makedirs(self.local_folder, exist_ok=True)
        os.makedirs(self.schema_folder, exist_ok=True)
        
        # Save if doesn't exist or recompute is True
        if not os.path.exists(schema_file) or recompute:
            with open(schema_file, 'w') as f:
                json.dump(schema_dict, f, indent=2)
            return True
        return False

    def save_experiment_sets(self, sets: list[dict[str, Any]]) -> bool:
        """Save serialized ExperimentSet definitions to ``experiment_sets.json`` (schema root)."""
        if not self.schema_folder:
            raise ValueError("Schema folder not configured")
        os.makedirs(self.schema_folder, exist_ok=True)
        path = os.path.join(self.schema_folder, "experiment_sets." + FileFormat.JSON.value)
        with open(path, 'w') as f:
            json.dump(sets, f, indent=2)
        return True

    def load_experiment_sets(self) -> list[dict[str, Any]]:
        """Load serialized ExperimentSet definitions; ``[]`` if none saved yet."""
        if not self.schema_folder:
            raise ValueError("Schema folder not configured")
        path = os.path.join(self.schema_folder, "experiment_sets." + FileFormat.JSON.value)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f)

    def save_parameters(self, exp_codes: list[str], data: dict[str, dict[str, Any]],
                        recompute: bool, **kwargs) -> bool:
        """Save experiment parameters to local files."""
        return self._save_files_generic(
            codes=exp_codes, 
            data=data,
            subdirs=["{code}"], 
            filename=BlockType.PARAMETERS.value, 
            recompute=recompute,
            file_format=FileFormat.JSON
        )

    def save_performance(self, exp_codes: list[str], data: dict[str, dict[str, Any]], 
                         recompute: bool, **kwargs) -> bool:
        """Save performance metrics to local files."""
        return self._save_files_generic(
            codes=exp_codes, 
            data=data,
            subdirs=["{code}"], 
            filename=BlockType.PERF_ATTRS.value, 
            recompute=recompute,
            file_format=FileFormat.JSON
        )

    def save_parameter_updates(self, exp_codes: list[str], data: dict[str, dict[str, Any]],
                               recompute: bool, **kwargs) -> bool:
        """Save parameter update event logs to local files."""
        return self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}"],
            filename=BlockType.PARAM_UPDATES.value,
            recompute=recompute,
            file_format=FileFormat.JSON
        )

    def save_metadata(self, exp_codes: list[str], data: dict[str, dict[str, Any]],
                      recompute: bool, **kwargs) -> bool:
        """Save experiment metadata (e.g. ``dataset_code``) to local files."""
        return self._save_files_generic(
            codes=exp_codes,
            data=data,
            subdirs=["{code}"],
            filename=BlockType.METADATA.value,
            recompute=recompute,
            file_format=FileFormat.JSON
        )
    
    def save_features(self, exp_codes: list[str], data: dict[str, dict[str, Any]], 
                      recompute: bool, feature_name: str, column_names: list[str], **kwargs) -> bool:
        """Save feature arrays to local files."""
        if not feature_name:
            raise ValueError("feature_name required in kwargs for save_features")
        
        # Compute column names for 
        if not column_names:
            raise ValueError("column_names_getter function must be provided for saving feature arrays")

        return self._save_files_generic(
            codes=exp_codes, 
            data=data,
            subdirs=["{code}"], 
            filename=feature_name, 
            recompute=recompute, 
            file_format=FileFormat.CSV,
            column_names=column_names,
        )
    
    # === PRIVATE METHODS ===

    def _load_files_generic(
            self, 
            codes: list[str], 
            subdirs: list[str], 
            filename: str, 
            file_format: FileFormat
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Generic file loader for JSON/CSV files."""
        if not self.schema_id:
            raise ValueError("Schema ID must be set")
        if not self.schema_folder or not os.path.exists(self.schema_folder):
            raise ValueError("Schema folder must be set and exist")
            
        missing_codes = []
        result_dict = {}

        for code in codes:
            # Build file path
            path_parts = [self.schema_folder] + subdirs
            path_parts = [part.replace("{code}", code) for part in path_parts]
            file_name = filename.replace("{code}", code)
            file_path = os.path.join(*path_parts, f"{file_name}.{file_format.value}")

            # Check existence and load
            if not os.path.exists(file_path):
                missing_codes.append(code)
                continue

            try:
                if file_format == FileFormat.CSV:
                    import pandas as pd  # local: CSV feature I/O (ML path)
                    df = pd.read_csv(file_path)
                    result_dict[code] = df
                elif file_format == FileFormat.JSON:
                    with open(file_path, 'r') as f:
                        result_dict[code] = json.load(f)
                else:
                    raise ValueError(f"Unknown file format {file_format.value}. Check enum.")
            except FileNotFoundError:
                # File vanished between the existence check and the read (TOCTOU);
                # treat as genuinely absent.
                missing_codes.append(code)
            except Exception:
                # Any other failure (corrupt/locked/unparseable file) is a real
                # error, not a missing file — surface it instead of silently
                # dropping data.
                logging.getLogger(__name__).exception(
                    "Failed to load %s for code '%s' from %s", file_format.value, code, file_path
                )
                raise

        return missing_codes, result_dict

    def _save_files_generic(
        self, 
        codes: list[str], 
        data: dict[str, Any], 
        subdirs: list[str], 
        filename: str, 
        recompute: bool, 
        file_format: FileFormat,
        column_names: list[str] | None = None
    ) -> bool:
        """Generic file saver for JSON/CSV files."""
        if not self.schema_id:
            raise ValueError("Schema ID must be set")
        if not self.schema_folder or not os.path.exists(self.schema_folder):
            raise ValueError("Schema folder must be set and exist")
            
        saved = False        
        for code in codes:
            if code not in data:
                raise ValueError(f"No data found for code: {code}")
            
            # Build directory path
            dir_parts = [self.schema_folder] + subdirs
            dir_parts = [part.replace("{code}", code) for part in dir_parts]
            dir_path = os.path.join(*dir_parts)
            os.makedirs(dir_path, exist_ok=True)
            
            # Build file path
            file_name = filename.replace("{code}", code)
            file_path = os.path.join(dir_path, f"{file_name}.{file_format.value}")

            # Save if doesn't exist or recompute is True
            if not os.path.exists(file_path) or recompute:
                code_data = data[code]

                # Save as CSV
                if file_format == FileFormat.CSV:
                    import pandas as pd  # local: CSV feature I/O (ML path)
                    df = pd.DataFrame(code_data, columns=column_names)  # type: ignore[call-overload]
                    df.to_csv(file_path, index=False)
                # Save as JSON
                elif file_format == FileFormat.JSON:
                    with open(file_path, 'w') as f:
                        json.dump(code_data, f, indent=2)
                else:
                    raise ValueError(f"Unknown file format {file_format}. Check enum.")
                saved = True
        return saved
