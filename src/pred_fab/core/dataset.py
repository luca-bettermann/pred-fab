"""
Dataset class for data container with schema validation.

Dataset is an independent entity holding experiment data and validating
against a DatasetSchema. It does NOT handle persistence (that's LocalData's job).
"""

import numpy as np
import pandas as pd
import os
from typing import Callable, Dict, Any, Optional, List, Tuple, Literal, Type
import functools

from .schema import DatasetSchema
from ..core import DataBlock, Parameters, Features, PerformanceAttributes, DataDimension

from ..interfaces.external_data import IExternalData
from ..utils import LocalData, PfabLogger

BlockType = Literal['parameters', 'performance', 'feature', 'predicted_feature']

class ExperimentData:
    """
    Complete data for a single experiment.
    
    - Parameters
    - Performance
    - Features
    - Predicted Features
    """

    def __init__(self, 
                 exp_code: str, 
                 parameters: Parameters, 
                 performance: PerformanceAttributes, 
                 features: Features, 
                 predicted_features: Features
                 ):
        self.exp_code = exp_code
        self.parameters = parameters
        self.performance = performance
        self.features = features
        self.predicted_features = predicted_features
    
    # === Helper Methods for Validation ===

    def is_valid(self, schema: 'DatasetSchema') -> bool:
        """Check structural compatibility of exp with schema."""
        # Check all blocks using helper function
        block_checks = [
            (self.parameters, schema.parameters),
            (self.performance, schema.performance),
            (self.features, schema.features),
            (self.predicted_features, schema.predicted_features)
        ]
        
        for self_block, other_block in block_checks:
            if not self_block.is_compatible(other_block):
                raise ValueError(
                    f"Schema block '{self_block.__class__.__name__}' is not identical "
                    f"to {other_block.__class__.__name__}."
                )        
        return True
    
    def is_complete(self, feature_code: str, evaluate_from: int, evaluate_to: Optional[int]) -> bool:
        """Check if feature array is non-empty in specified range."""
        if not self.features.has(feature_code):
            raise KeyError(f"Feature code '{feature_code}' not found in experiment '{self.exp_code}'")

        array = self.features.get_value(feature_code)
        end_index = evaluate_to if evaluate_to is not None else array.shape[0]
        
        # Check if all values in the specified range are NaN
        if np.all(~np.isnan(array[evaluate_from:end_index])):
            return True
        return False

    # === Helper Methods for Data Access ===

    def set_data(self, values: Dict[str, Any], block_type: BlockType) -> None:
        """Set values for a specific data type."""
        if block_type == "parameters":
            self.parameters.set_values(values)
        elif block_type == "performance":
            self.performance.set_values(values)
        elif block_type == "feature":
            self.features.set_values(values)
        elif block_type == "predicted_feature":
            self.predicted_features.set_values(values)
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
    def get_data_dict(self, block_type: BlockType) -> Dict[str, Any]:
        """Get values as dict for a specific data type."""
        if block_type == "parameters":
            return self.parameters.get_values_dict()
        elif block_type == "performance":
            return self.performance.get_values_dict()
        elif block_type == "feature":
            return self.features.get_values_dict()
        elif block_type == "predicted_feature":
            return self.predicted_features.get_values_dict()
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
    def has_data(self, block_type: BlockType) -> bool:
        """Check if values are set for a specific data type."""
        if block_type == "parameters":
            return bool(self.parameters.get_values_dict())
        elif block_type == "performance":
            return bool(self.performance.get_values_dict())
        elif block_type == "feature":
            return bool(self.features.get_values_dict())
        elif block_type == "predicted_feature":
            return bool(self.predicted_features.get_values_dict())
        else:
            raise ValueError(f"Unknown block type: {block_type}")
            
    def is_feature_populated(self, feature_name: str) -> bool:
        """Check if a specific feature is populated (not just initialized)."""
        return self.features.is_populated(feature_name)
        
    def is_predicted_feature_populated(self, feature_name: str) -> bool:
        """Check if a specific predicted feature is populated (not just initialized)."""
        return self.predicted_features.is_populated(feature_name)

class Dataset:
    """
    Data container with schema validation.
    
    - Schema reference and static parameter values
    - Experiment records with hierarchical load/save
    - Feature memoization cache for IFeatureModel efficiency
    """
    
    def __init__(self, 
                 schema: DatasetSchema, 
                 external_data: Optional[IExternalData] = None,
                 debug_flag: bool = False):
        """
        Initialize Dataset.
        
        Args:
            schema: DatasetSchema defining structure
            external_data: IExternalData instance for external storage
            debug_flag: Skip external operations if True (local-only mode)
        """
        self.schema = schema
        self.local_data = schema.local_data
        self.external_data = external_data
        self.debug_flag = debug_flag

        # Initialize local data handler and logger
        self.logger = PfabLogger.get_logger(schema.local_data.get_log_folder('logs'))
        
        # Master storage using ExperimentData
        self._experiments: Dict[str, ExperimentData] = {}  # exp_code → ExperimentData

    def get_experiment(self, exp_code: str) -> ExperimentData:
        """Get complete ExperimentData for an exp_code."""
        if exp_code not in self._experiments:
            raise KeyError(f"Experiment {exp_code} not found")
        return self._experiments[exp_code]
    
    # === Create ExperimentData Objects ===

    def _init_from_schema(self, block_class: Any, schema_dict: DataBlock, suffix: str = '') -> Any:
        block = block_class()
        for name, data_obj in schema_dict.items():
            block.add(suffix + name, data_obj)
        return block
        
    def _create_experiment_shell(self, exp_code: str) -> ExperimentData:
        """Create new empty experiment shell with all blocks initialized."""
        params_block = self._init_from_schema(Parameters, self.schema.parameters)
        perf_block = self._init_from_schema(PerformanceAttributes, self.schema.performance)
        arrays_block = self._init_from_schema(Features, self.schema.features)
        pred_block = self._init_from_schema(Features, self.schema.features, suffix='pred_')
        
        return ExperimentData(
            exp_code=exp_code,
            parameters=params_block,
            performance=perf_block,
            features=arrays_block,
            predicted_features=pred_block
        )
    
    def _build_experiment_data(
        self,
        exp_code: str,
        parameters: Dict[str, Any],
        performance: Optional[Dict[str, Any]],
        metric_arrays: Optional[Dict[str, np.ndarray]],
        predicted_arrays: Optional[Dict[str, np.ndarray]] = None
    ) -> ExperimentData:
        """Build ExperimentData from loaded components."""
        # 1. Create shell with schema structure
        exp_data = self._create_experiment_shell(exp_code)
        
        # 2. Set parameters and dimensions
        exp_data.parameters.set_values(parameters)

        # 3. Initialize feature arrays based on parameters
        exp_data.features.initialize_arrays(exp_data.parameters)
        exp_data.predicted_features.initialize_arrays(exp_data.parameters)
        
        # 4. Set optional blocks
        if performance:
            exp_data.set_data(performance, block_type="performance")

        if metric_arrays:
            exp_data.set_data(metric_arrays, block_type="feature")
            
        if predicted_arrays:
            exp_data.set_data(predicted_arrays, block_type="predicted_feature")

        # 5. Validate against schema
        exp_data.is_valid(self.schema)

        return exp_data
    
    def create_experiment(
        self,
        exp_code: str,
        parameters: Dict[str, Any],
        performance: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, np.ndarray]] = None,
        recompute: bool = False
    ) -> ExperimentData:
        """
        Create a new experiment manually.
        
        Args:
            exp_code: Unique experiment code
            parameters: Dictionary of parameter values (Mandatory)
            performance: Optional performance metrics
            metric_arrays: Optional feature arrays
            recompute: If True, overwrite existing experiment in memory
            
        Raises:
            ValueError: If experiment already exists and recompute is False
        """
        # Check memory
        if exp_code in self._experiments and not recompute:
            raise ValueError(f"Experiment {exp_code} already exists in memory")
            
        # Check local storage
        if not recompute:
            # Check if folder exists
            exp_folder = self.local_data.get_experiment_folder(exp_code)
            if os.path.exists(exp_folder):
                 raise ValueError(f"Experiment {exp_code} already exists locally")

        # Build and store
        exp_data = self._build_experiment_data(exp_code, parameters, performance, features)
        self._experiments[exp_code] = exp_data
        return exp_data
    
    def add_experiment(self, exp_data: ExperimentData, recompute: bool = False) -> None:
        """Manually add an existing ExperimentData to the dataset."""        
        # Check memory
        if exp_data.exp_code in self._experiments and not recompute:
            raise ValueError(f"Experiment {exp_data.exp_code} already exists in memory")
        
        # Validate against schema
        exp_data.is_valid(self.schema)
        
        # Store
        self._experiments[exp_data.exp_code] = exp_data
    
    def get_experiment_codes(self) -> List[str]:
        """Get list of all experiment codes in dataset."""
        return list(self._experiments.keys())
    
    def get_experiment_params(self, exp_code: str) -> Dict[str, Any]:
        """Get experiment parameters as dictionary."""
        exp_data = self.get_experiment(exp_code)
        params = {}
        for name in exp_data.parameters.keys():
            if exp_data.parameters.has_value(name):
                params[name] = exp_data.parameters.get_value(name)
        return params
    
    def has_experiment(self, exp_code: str) -> bool:
        """Check if experiment exists in dataset."""
        return exp_code in self._experiments
    
    # === Helper Methods for Hierarchical Loading/Saving ===

    def _set_exp_data(self, code: str, data: Any, block_type: BlockType) -> None:
        if code in self._experiments:
            exp = self._experiments[code]
            exp.set_data(data, block_type)

    def _has_exp_data(self, code: str, block_type: BlockType) -> bool:
        if code not in self._experiments:
            return False
        exp = self._experiments[code]
        return exp.has_data(block_type)
    
    def _get_exp_data(self, code: str, block_type: BlockType) -> Any:
        if code not in self._experiments:
            return None
        exp = self._experiments[code]
        return exp.get_data_dict(block_type)
    
    # === Hierarchical Load/Save Methods ===
    
    def populate(self, source: str = "local") -> int:
        """Load all experiments from storage hierarchically by scanning dataset folder."""
        if source != "local":
            raise NotImplementedError("Only 'local' source is currently supported")
        
        # Scan local folders for experiment codes
        exp_codes = self.local_data.list_experiments()
        
        # Use batch loading
        missing = self.load_experiments(exp_codes)
        loaded_count = len(exp_codes) - len(missing)
        
        return loaded_count

    def load_experiment(self, exp_code: str) -> ExperimentData:
        """Add experiment by loading it hierarchically."""
        # 1. Hierarchical load
        missing = self.load_experiments([exp_code])
        
        if exp_code in missing:
            raise KeyError(f"Experiment {exp_code} could not be loaded")
        
        return self.get_experiment(exp_code)
    
    def load_experiments(self, exp_codes: List[str], recompute: bool = False) -> List[str]:
        """Load multiple experiments using hierarchical pattern with progress tracking."""
        
        # 1. Ensure shells exist
        for code in exp_codes:
            if code not in self._experiments:
                self._experiments[code] = self._create_experiment_shell(code)

        # 2. Load Experiment Parameters
        missing_params = self._hierarchical_load(
            "experiment parameters",
            exp_codes,
            loader=self.local_data.load_parameters,
            setter=functools.partial(self._set_exp_data, block_type="parameters"),
            in_memory=functools.partial(self._has_exp_data, block_type="parameters"),
            external_loader=self.external_data.pull_parameters if self.external_data else None,
            recompute=recompute
        )
        
        if missing_params and len(self.schema.parameters.data_objects):
            raise ValueError(f"No parameters found for any of the following experiments: {missing_params}")

        # Filter codes that were actually found and validate (parameters are mandatory)
        for code in exp_codes:
            exp_data = self.get_experiment(code)
            exp_data.is_valid(self.schema)

            # Initialize feature arrays based on loaded parameters
            exp_data.features.initialize_arrays(exp_data.parameters)
            exp_data.predicted_features.initialize_arrays(exp_data.parameters)
        
        # 3. Load Features
        missing_features_union = set()
        for metric_name in self.schema.features.keys():
            missing_features = self._hierarchical_load(
                f"feature '{metric_name}'",
                exp_codes,
                loader=self.local_data.load_features,
                setter=functools.partial(self._set_exp_data, block_type="feature"),
                in_memory=lambda code: self.get_experiment(code).is_feature_populated(metric_name),
                external_loader=self.external_data.pull_features if self.external_data else None,
                recompute=recompute,
                feature_name=metric_name # Passed to kwargs
            )
            missing_features_union.update(missing_features)
        
        # 4. Load Performance Metrics
        missing_performance = self._hierarchical_load(
            "performance metrics",
            exp_codes,
            loader=self.local_data.load_performance,
            setter=functools.partial(self._set_exp_data, block_type="performance"),
            in_memory=functools.partial(self._has_exp_data, block_type="performance"),
            external_loader=self.external_data.pull_performance if self.external_data else None,
            recompute=recompute
        )

        # 5. Load Predicted Features
        missing_pred_features_union = set()
        for metric_name in self.schema.features.keys():
            pred_name = f"predicted_{metric_name}"
            missing_pred_features = self._hierarchical_load(
                f"predicted feature '{metric_name}'",
                exp_codes,
                loader=self.local_data.load_features,
                setter=functools.partial(self._set_exp_data, block_type="predicted_feature"),
                in_memory=lambda code: self.get_experiment(code).is_predicted_feature_populated(metric_name),
                external_loader=self.external_data.pull_features if self.external_data else None,
                recompute=recompute,
                feature_name=pred_name # Passed to kwargs
            )
            missing_pred_features_union.update(missing_pred_features)

        # Summary
        total = len(exp_codes)
        summary = [
            "\n===== Populate Dataset =====",
            f"\nExperiments: \t\t{total}",
            f"  - Parameters: \t{total - len(missing_params)}/{total}",
            f"  - Features: \t\t{total - len(missing_features_union)}/{total}",
            f"  - Performance: \t{total - len(missing_performance)}/{total}",
            f"  - Predicted Features: {total - len(missing_pred_features_union)}/{total}",
        ]
        
        self.logger.console_info("\n".join(summary))
        self.logger.console_new_line()
        self.logger.console_success(f"Successfully loaded {len(exp_codes)} experiments.")
        return missing_params
    
    def save_all(self, recompute: bool = False) -> None:
        """Save all experiments currently in memory."""
        exp_codes = list(self._experiments.keys())
        self.save_experiments(exp_codes, recompute=recompute)
    
    def save_experiments(self, exp_codes: List[str], recompute: bool = False) -> None:
        """Save multiple experiments hierarchically with progress tracking."""
        # Override external flag if in debug mode
        if self.debug_flag:
            external = False
        
        # Filter to experiments that exist in dataset
        codes_to_save = [code for code in exp_codes if self.has_experiment(code)]
        
        if not codes_to_save:
            if self.logger:
                self.logger.warning(f"No experiments to save from {exp_codes}")
            return
        
        # 1. Save Schema
        self.save_schema(recompute=recompute)
        
        # 2. Save Parameters
        self._hierarchical_save(
            "experiment parameters", codes_to_save,
            getter=functools.partial(self._get_exp_data, block_type="parameters"),
            saver=self.local_data.save_parameters,
            external_saver=self.external_data.push_parameters if self.external_data else None,
            recompute=recompute
        )
        
        # 3. Save Performance
        self._hierarchical_save(
            "performance metrics", codes_to_save,
            getter=functools.partial(self._get_exp_data, block_type="performance"),
            saver=self.local_data.save_performance,
            external_saver=self.external_data.push_performance if self.external_data else None,
            recompute=recompute
        )
            
        # 4. Save Features
        for name in self.schema.features.keys():
            # Iterate over both feature and predicted feature
            for block_type in ["feature", "predicted_feature"]:

                self._hierarchical_save(
                    f"{name}_{block_type}", codes_to_save,
                    getter=functools.partial(self._get_exp_data, block_type="feature"),
                    saver=self.local_data.save_features,
                    external_saver=self.external_data.push_features if self.external_data else None,
                    recompute=recompute,
                    feature_name=name,
                    column_names=self._infer_column_names(name)
                )
        
    def save_schema(self, recompute: bool = False) -> None:
        """Save schema hierarchically."""        
        # 1. Save locally
        if self.local_data.save_schema(self.schema.to_dict(), recompute=recompute):
            self.logger.console_info(f"Saved dataset schema '{self.schema.name}' as local file.")
        # 2. Save externally
        if self.external_data and not self.debug_flag:
             self.external_data.push_schema(self.schema.name, self.schema.to_dict())

    def _infer_column_names(self, array_name: str) -> Optional[List[str]]:
        """Infer column names for metric array based on dimensions."""
        # For now, return None (no columns)
        # In the future, could use dimension info to generate column names
        return None
    
    def _check_for_retrieved_codes(self, target_pre: List[str], target_post: List[str], dtype: str, source: str, console_output: bool = False) -> List[str]:
        """Check which codes were successfully retrieved and log to console."""
        retrieved_codes = [code for code in target_pre if code not in target_post]
        if retrieved_codes and self.logger:
            message = f"Retrieved {dtype} {retrieved_codes} from {source}."
            if console_output:
                self.logger.console_info(message)
            else:
                self.logger.info(message)
        return retrieved_codes
    
    def _hierarchical_load(self, 
                        dtype: str,
                        target_codes: List[str],
                        loader: Callable[..., Tuple[List[str], Dict[str, Any]]],
                        setter: Callable[[str, Any], None],
                        in_memory: Callable[[str], bool],
                        external_loader: Optional[Callable[..., Tuple[List[str], Any]]] = None,
                        recompute: bool = False,
                        console_output: bool = False,
                        **kwargs) -> List[str]:
        """Universal hierarchical data loading: Memory → Local Files → External Source"""
        # 1. Check memory
        missing_memory = [code for code in target_codes if not in_memory(code)]
        self._check_for_retrieved_codes(target_codes, missing_memory, dtype, "memory", console_output=console_output)

        if not missing_memory:
            return []
        
        # 2. Load from local files
        if not recompute:
            missing_local, local_data = loader(missing_memory, **kwargs)
            # directly store retrieved data in ExpData object
            for code, data in local_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(missing_memory, missing_local, dtype, "local files", console_output=console_output)
        else:
            missing_local = missing_memory
            self.logger.info(f"Recompute mode: Skipping loading {dtype} from local files")

        if not missing_local:
            return []

        # 3. Load from external sources
        if not self.debug_flag and external_loader:
            missing_external, external_data = external_loader(missing_local, **kwargs)
            # directly store retrieved data in ExpData object
            for code, data in external_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(missing_local, missing_external, dtype, "external source", console_output=console_output)
        elif self.debug_flag:
            missing_external = missing_local 
            self.logger.info(f"Debug mode: Skipping loading {dtype} from external source")
        else:
            missing_external = missing_local
            self.logger.warning(f"No external data interface provided: Skipping loading {dtype} from external source")

        return missing_external

    def _hierarchical_save(self, 
                           dtype: str,
                           target_codes: List[str],
                           getter: Callable[[str], Any],
                           saver: Callable[..., bool],
                           external_saver: Optional[Callable[..., bool]] = None,
                           recompute: bool = False,
                           **kwargs) -> None:
        """Universal hierarchical data saving: Memory → Local Files → External Source"""
        # 1. Filter to codes that exist in memory (and have data)
        data_to_save = {}
        for code in target_codes:
            val = getter(code)
            if val: # Assuming empty dict/None means no data
                data_to_save[code] = val
        
        if not data_to_save:
            self.logger.console_warning(f"{dtype} {target_codes} found in memory without data.")
            return
        
        codes_to_save = list(data_to_save.keys())
        
        # 2. Save to local files
        saved = saver(codes_to_save, data_to_save, recompute, **kwargs)
        if saved:
            self.logger.console_info(f"Saved {dtype} {codes_to_save} as local files.")
        else:
            self.logger.info(f"{dtype.capitalize()} {codes_to_save} already exist as local files.")

        # 3. Save to external source (skip if in debug mode)
        if not self.debug_flag and external_saver:
            pushed = external_saver(codes_to_save, data_to_save, recompute, **kwargs)
            if pushed:
                self.logger.console_info(f"Pushed {dtype} {codes_to_save} to external source.")
            else:
                self.logger.info(f"{dtype} {codes_to_save} already exists in external source.")
        elif self.debug_flag:
            self.logger.info(f"Debug mode: Skipped pushing {dtype} {codes_to_save} to external source.")
        else:
            self.logger.warning(f"No external data interface provided: Skipped pushing {dtype} {codes_to_save} to external source.")

    def export_to_dataframe(self, experiment_codes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export experiment data to flattened DataFrames (X, y).
        Uses Parameters block logic for dimension iteration.
        """
        if not experiment_codes:
            return pd.DataFrame(), pd.DataFrame()
        
        X_rows = []
        y_rows = []
        
        for code in experiment_codes:
            exp_data = self.get_experiment(code)
            if exp_data.features is None:
                continue
            
            # Get parameter info
            all_params = exp_data.parameters.get_values_dict()
            dim_names = exp_data.parameters.get_dim_names()
            
            # Static params (X base)
            static_params = {k: v for k, v in all_params.items() if k not in dim_names}
            
            if not dim_names:
                # Case 1: No dimensions (Scalar experiment)
                y_dict = {}
                for feature_name in exp_data.features.keys():
                    value = exp_data.features.get_value(feature_name)
                    # Handle scalar or 1-element array
                    if isinstance(value, np.ndarray):
                        y_dict[feature_name] = float(value.flat[0])
                    else:
                        y_dict[feature_name] = float(value)
                
                X_rows.append(static_params)
                y_rows.append(y_dict)
                continue
            
            # Case 2: Multi-dimensional experiment
            # Get all index combinations
            dim_combinations = exp_data.parameters.get_dim_combinations(dim_names)
            iterator_map = exp_data.parameters.get_dim_iterator_names()
            
            # Pre-fetch feature arrays
            feature_arrays = {
                name: exp_data.features.get_value(name) 
                for name in exp_data.features.keys()
            }
            
            for idx_tuple in dim_combinations:
                # Build X row (Static + Iterators)
                row_dict = static_params.copy()
                for i, dim_name in enumerate(dim_names):
                    iterator_name = iterator_map[dim_name]
                    row_dict[iterator_name] = idx_tuple[i]
                
                X_rows.append(row_dict)
                
                # Build y row (Features at index)
                y_dict = {}
                for feature_name, array in feature_arrays.items():
                    if isinstance(array, np.ndarray):
                        val = array[idx_tuple]
                        if not np.isnan(val):
                            y_dict[feature_name] = float(val)
                
                y_rows.append(y_dict)
        
        if not X_rows:
            return pd.DataFrame(), pd.DataFrame()
        
        return pd.DataFrame(X_rows), pd.DataFrame(y_rows)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Dataset(schema_id='{self.schema.name}', "
            f"experiments={len(self._experiments)})"
        )

