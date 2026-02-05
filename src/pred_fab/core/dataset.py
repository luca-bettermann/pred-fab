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
# from ..utils.enum import BlockType, PRED_SUFFIX, Loaders
from ..utils.enum import BlockType, Loaders

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
                #  predicted_features: Features
                 ):
        self.code = exp_code
        self.parameters = parameters
        self.performance = performance
        self.features = features
        # self.predicted_features = predicted_features

    # === Helper Methods for Validation ===

    def is_valid(self, schema: 'DatasetSchema') -> bool:
        """Check structural compatibility of exp with schema."""
        # Check all blocks using helper function
        block_checks = [
            (self.parameters, schema.parameters),
            (self.performance, schema.performance_attrs),
            (self.features, schema.features),
            # (self.predicted_features, schema.predicted_features)
        ]
        
        for self_block, other_block in block_checks:
            if not self_block.is_compatible(other_block):
                raise ValueError(
                    f"Schema block {self_block.__class__.__name__} is not identical "
                    f"to {other_block.__class__.__name__}."
                )        
        return True
    
    def is_complete(self, feature_code: str, evaluate_from: int, evaluate_to: Optional[int]) -> bool:
        """Check if feature array is non-empty in specified range."""
        if not self.features.has(feature_code):
            raise KeyError(f"Feature code '{feature_code}' not found in experiment '{self.code}'")

        array = self.features.get_value(feature_code)
        end_index = evaluate_to if evaluate_to is not None else array.shape[0]
        
        # Check if all values in the specified range are NaN
        if np.all(~np.isnan(array[evaluate_from:end_index])):
            return True
        return False

    # === Helper Methods for Data Access ===

    def set_data(self, values: Any, block_type: BlockType, logger: PfabLogger) -> None:
        """Set values for a specific data type."""
        if block_type == BlockType.PARAMETERS:
            self.parameters.set_values_from_dict(values, logger)
        elif block_type == BlockType.FEATURES:
            self.features.set_values_from_df(values, logger)
        elif block_type == BlockType.PERF_ATTRS:
            self.performance.set_values_from_dict(values, logger)
        # elif block_type == BlockType.FEATURES_PRED:
        #     self.predicted_features.set_values_from_df(values, logger)
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
    def get_data_dict(self, block_type: str) -> Dict[str, Any]:
        """Get values as dict for a specific data type."""
        if block_type == BlockType.PARAMETERS:
            return self.parameters.get_values_dict()
        elif block_type == BlockType.PERF_ATTRS:
            return self.performance.get_values_dict()
        elif block_type == BlockType.FEATURES:
            return self.features.get_values_dict()
        # elif block_type == BlockType.FEATURES_PRED:
        #     return self.predicted_features.get_values_dict()
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
    def has_data(self, block_type: BlockType) -> bool:
        """Check if values are set for a specific data type."""
        if block_type == BlockType.PARAMETERS:
            return bool(self.parameters.get_values_dict())
        elif block_type == BlockType.PERF_ATTRS:
            return bool(self.performance.get_values_dict())
        elif block_type == BlockType.FEATURES:
            return bool(self.features.get_values_dict())
        # elif block_type == BlockType.FEATURES_PRED:
        #     return bool(self.predicted_features.get_values_dict())
        else:
            raise ValueError(f"Unknown block type: {block_type}")
            
    def is_feature_populated(self, feature_name: str) -> bool:
        """Check if a specific feature is populated (not just initialized)."""
        return self.features.is_populated(feature_name)
        
    # def is_predicted_feature_populated(self, feature_name: str) -> bool:
    #     """Check if a specific predicted feature is populated (not just initialized)."""
    #     return self.predicted_features.is_populated(feature_name)

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
        
        # Feature column names
        # self.feature_columns: Optional[Dict[str, List[str]]] = None

    def get_experiment(self, exp_code: str) -> ExperimentData:
        """Get complete ExperimentData for an exp_code."""
        if exp_code not in self._experiments:
            raise KeyError(f"Experiment {exp_code} not found")
        return self._experiments[exp_code]

    def get_all_experiments(self) -> List[ExperimentData]:
        """Get list of all ExperimentData objects."""
        return list(self._experiments.values())
    
    # === Create ExperimentData Objects ===

    def _init_from_schema(self, block_class: Any, schema_dict: DataBlock, suffix: str = '') -> Any:
        block = block_class()
        for name, data_obj in schema_dict.items():
            block.add(suffix + name, data_obj)
        return block
        
    def _create_experiment_shell(self, exp_code: str) -> ExperimentData:
        """Create new empty experiment shell with all blocks initialized."""
        params_block = self._init_from_schema(Parameters, self.schema.parameters)
        perf_block = self._init_from_schema(PerformanceAttributes, self.schema.performance_attrs)
        arrays_block = self._init_from_schema(Features, self.schema.features)
        # pred_block = self._init_from_schema(Features, self.schema.features, suffix=PRED_SUFFIX)
        
        return ExperimentData(
            exp_code=exp_code,
            parameters=params_block,
            performance=perf_block,
            features=arrays_block,
            # predicted_features=pred_block
        )
    
    def _build_experiment_data(
        self,
        exp_code: str,
        parameters: Dict[str, Any],
        performance: Optional[Dict[str, Any]],
        metric_arrays: Optional[Dict[str, np.ndarray]],
        # predicted_arrays: Optional[Dict[str, np.ndarray]] = None
    ) -> ExperimentData:
        """Build ExperimentData from loaded components."""
        # 1. Create shell with schema structure
        exp_data = self._create_experiment_shell(exp_code)
        
        # 2. Set parameters and dimensions
        exp_data.parameters.set_values_from_dict(parameters, self.logger)

        # 3. Initialize feature arrays based on parameters
        exp_data.features.initialize_arrays(exp_data.parameters)
        # exp_data.predicted_features.initialize_arrays(exp_data.parameters)
        
        # 4. Set optional blocks
        if performance:
            exp_data.set_data(performance, BlockType.PERF_ATTRS, self.logger)

        if metric_arrays:
            exp_data.set_data(metric_arrays, BlockType.FEATURES, self.logger)
            
        # if predicted_arrays:
        #     exp_data.set_data(predicted_arrays, BlockType.FEATURES_PRED, self.logger)

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
        if exp_data.code in self._experiments and not recompute:
            raise ValueError(f"Experiment {exp_data.code} already exists in memory")
        
        # Validate against schema
        exp_data.is_valid(self.schema)
        
        # Store
        self._experiments[exp_data.code] = exp_data
    
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

    def get_populated_experiment_codes(self) -> List[str]:
        """Get list of experiments that have all features in schema populated."""
        feature_names = list(self.schema.features.keys())
        return [
            code for code in self.get_experiment_codes()
            if all(self.get_experiment(code).is_feature_populated(f) for f in feature_names)
        ]

    def state_report(self) -> None:
        """Log an overview of the current dataset to the console."""
        exp_codes = self.get_experiment_codes()
        total = len(exp_codes)
        feature_names = list(self.schema.features.keys())
        
        # Count parameter and performance presence
        count_params = sum(1 for c in exp_codes if self.get_experiment(c).has_data(BlockType.PARAMETERS))
        count_perf = sum(1 for c in exp_codes if self.get_experiment(c).has_data(BlockType.PERF_ATTRS))
        
        # Count completely populated features (using existing helper)
        count_features = len(self.get_populated_experiment_codes())

        # # Count completely populated predicted features
        # count_pred = sum(1 for c in exp_codes if all(
        #     self.get_experiment(c).is_predicted_feature_populated(name) for name in feature_names
        # ))

        summary = [
            f"===== 'Dataset State Report' =====",
            f"\nSchema: \t\t{self.schema.name}",
            f"Experiments: \t\t{total}",
            f"  - Parameters: \t{count_params}/{total}",
            f"  - Features: \t\t{count_features}/{total}",
            f"  - Performance: \t{count_perf}/{total}",
            # f"  - Predicted Features: {count_pred}/{total}",
        ]

        self.logger.console_new_line()
        self.logger.console_info("\n".join(summary))
        self.logger.console_new_line()
    
    # === Helper Methods for Hierarchical Loading/Saving ===

    def _set_exp_data(self, code: str, data: Any, block_type: BlockType) -> None:
        if code in self._experiments:
            exp = self._experiments[code]
            exp.set_data(data, block_type, self.logger)

    def _has_exp_data(self, code: str, block_type: BlockType) -> bool:
        if code not in self._experiments:
            return False
        exp = self._experiments[code]
        return exp.has_data(block_type)
    
    def _get_exp_data(self, code: str, block_type: str) -> Any:
        if code not in self._experiments:
            return None
        exp = self._experiments[code]
        return exp.get_data_dict(block_type)
    
    def _get_exp_feature_array(self, code: str, feature_name: str, block_type: BlockType) -> Optional[np.ndarray]:
        features = self._get_exp_data(code, block_type=block_type)
        if feature_name not in features:
            raise KeyError(f"{block_type} '{feature_name}' not found for experiment '{code}'")
        return features[feature_name]
    
    def _get_array_column_names(self, feature_name: str) -> List[str]:
        """Get column names for a specific feature array."""
        if feature_name not in self.schema.features.data_objects:
            raise KeyError(f"Feature '{feature_name}' not found in schema")
        
        return self.schema.features.get(feature_name).columns # type: ignore

    # === Hierarchical Load/Save Methods ===
    
    def populate(self, source: Loaders = Loaders.LOCAL, verbose_flag: bool = False) -> int:
        """Load all experiments from storage hierarchically by scanning dataset folder."""
        if source != Loaders.LOCAL:
            raise NotImplementedError(f"Only {source.value} source is currently supported")
        
        # Scan local folders for experiment codes
        exp_codes = self.local_data.list_experiments()
        
        # Use batch loading
        missing = self.load_experiments(exp_codes, verbose=verbose_flag)
        loaded_count = len(exp_codes) - len(missing)
        
        return loaded_count

    def load_experiment(self, exp_code: str, verbose: bool = False) -> ExperimentData:
        """Add experiment by loading it hierarchically."""
        # 1. Hierarchical load
        missing = self.load_experiments([exp_code], verbose=verbose)
        
        if exp_code in missing:
            raise KeyError(f"Experiment {exp_code} could not be loaded")
        
        return self.get_experiment(exp_code)
    
    def load_experiments(self, exp_codes: List[str], recompute_flag: bool = False, verbose: bool = False) -> List[str]:
        """Load multiple experiments using hierarchical pattern with progress tracking."""
        
        self.logger.console_new_line()
        self._logging(f"Loading experiments {exp_codes}...", self.logger.console_execute, verbose)

        # 1. Ensure shells exist
        for code in exp_codes:
            if code not in self._experiments:
                self._experiments[code] = self._create_experiment_shell(code)

        # 2. Load Experiment Parameters
        missing_params = self._hierarchical_load(
            BlockType.PARAMETERS,
            exp_codes,
            loader=self.local_data.load_parameters,
            setter=functools.partial(self._set_exp_data, block_type=BlockType.PARAMETERS),
            in_memory=functools.partial(self._has_exp_data, block_type=BlockType.PARAMETERS),
            external_loader=self.external_data.pull_parameters if self.external_data else None,
            recompute_flag=recompute_flag,
            verbose=verbose
        )
        
        if missing_params and len(self.schema.parameters.data_objects):
            raise ValueError(f"No parameters found for any of the following experiments: {missing_params}")

        # Filter codes that were actually found and validate (parameters are mandatory)
        for code in exp_codes:
            exp_data = self.get_experiment(code)
            exp_data.is_valid(self.schema)

            # Initialize feature arrays based on loaded parameters
            exp_data.features.initialize_arrays(exp_data.parameters, recompute_flag)
            # exp_data.predicted_features.initialize_arrays(exp_data.parameters, recompute_flag)

        # 3. Load Performance Metrics
        missing_performance = self._hierarchical_load(
            BlockType.PERF_ATTRS, exp_codes,
            loader=self.local_data.load_performance,
            setter=functools.partial(self._set_exp_data, block_type=BlockType.PERF_ATTRS),
            in_memory=functools.partial(self._has_exp_data, block_type=BlockType.PERF_ATTRS),
            external_loader=self.external_data.pull_performance if self.external_data else None,
            recompute_flag=recompute_flag,
            verbose=verbose
        )

        # 4. Load Features
        missing_features_union = set()
        for name in self.schema.features.keys():
            missing_features = self._hierarchical_load(
                name, exp_codes,
                loader=self.local_data.load_features,
                setter=functools.partial(self._set_exp_data, block_type=BlockType.FEATURES),
                in_memory=lambda code: self.get_experiment(code).is_feature_populated(name),
                external_loader=self.external_data.pull_features if self.external_data else None,
                recompute_flag=recompute_flag,
                verbose=verbose,
                feature_name=name # Passed to kwargs
            )
            missing_features_union.update(missing_features)

        # # 5. Load Predicted Features
        # missing_pred_features_union = set()
        # for name in self.schema.features.keys():
        #     missing_pred_features = self._hierarchical_load(
        #         PRED_SUFFIX + name, exp_codes,
        #         loader=self.local_data.load_features,
        #         setter=functools.partial(self._set_exp_data, block_type=BlockType.FEATURES_PRED),
        #         in_memory=lambda code: self.get_experiment(code).is_predicted_feature_populated(name),
        #         external_loader=self.external_data.pull_features if self.external_data else None,
        #         recompute_flag=recompute_flag,
        #         verbose=verbose,
        #         feature_name=PRED_SUFFIX + name # Passed to kwargs
        #     )
        #     missing_pred_features_union.update(missing_pred_features)

        self.logger.console_success(f"Successfully loaded {len(exp_codes)} experiments.")
        self.logger.console_new_line()
        return missing_params
    
    def save_all(self, recompute_flag: bool = False, verbose_flag: bool = False) -> None:
        """Save all experiments currently in memory."""
        exp_codes = list(self._experiments.keys())
        self.save_experiments(exp_codes, recompute=recompute_flag, verbose=verbose_flag)

    def save_experiment(self, exp_code: str, recompute: bool = False, verbose: bool = False) -> None:
        """Save a single experiment hierarchically."""
        self.save_experiments([exp_code], recompute=recompute, verbose=verbose)
    
    def save_experiments(self, exp_codes: List[str], recompute: bool = False, verbose=False) -> None:
        """Save multiple experiments hierarchically with progress tracking."""        
        # Filter to experiments that exist in dataset
        codes_to_save = [code for code in exp_codes if self.has_experiment(code)]

        self.logger.console_new_line()
        self._logging(f"Saving experiments {codes_to_save}...", self.logger.console_execute, verbose)
        
        if not codes_to_save:
            self.logger.console_warning(f"None of {exp_codes} exist in dataset - skipping save operation")
            return
        elif exp_codes != codes_to_save:
            missing = set(exp_codes) - set(codes_to_save)
            self.logger.console_warning(f"Experiments {missing} do not exist in dataset - skipping save for these")
        
        # 1. Save Schema
        self.save_schema(recompute=recompute, verbose=verbose)
        
        # 2. Save Parameters
        self._hierarchical_save(
            BlockType.PARAMETERS, codes_to_save,
            getter=functools.partial(self._get_exp_data, block_type=BlockType.PARAMETERS),
            saver=self.local_data.save_parameters,
            external_saver=self.external_data.push_parameters if self.external_data else None,
            recompute=recompute,
            verbose=verbose
        )

        # 3. Save Performance
        self._hierarchical_save(
            BlockType.PERF_ATTRS, codes_to_save,
            getter=functools.partial(self._get_exp_data, block_type=BlockType.PERF_ATTRS),
            saver=self.local_data.save_performance,
            external_saver=self.external_data.push_performance if self.external_data else None,
            recompute=recompute,
            verbose=verbose
        )

        # 4. Save Features
        for name in self.schema.features.keys():
            self._hierarchical_save(
                name, codes_to_save,
                getter=functools.partial(self._get_exp_feature_array, feature_name=name, block_type=BlockType.FEATURES),
                saver=self.local_data.save_features,
                external_saver=self.external_data.push_features if self.external_data else None,
                recompute=recompute,
                column_names=self._get_array_column_names(name),
                verbose=verbose,
                feature_name=name # pass to kwargs
            )

        # # 5. Save Predicted Features
        # for name in self.schema.features.keys():
        #     self._hierarchical_save(
        #         PRED_SUFFIX + name, codes_to_save,
        #         getter=functools.partial(self._get_exp_feature_array, feature_name=PRED_SUFFIX + name, block_type=BlockType.FEATURES_PRED),
        #         saver=self.local_data.save_features,
        #         external_saver=self.external_data.push_features if self.external_data else None,
        #         recompute=recompute,
        #         column_names=self._get_array_column_names(name),
        #         verbose=verbose,
        #         feature_name=PRED_SUFFIX + name # pass to kwargs
        #     )

        self.logger.console_success(f"Successfully saved experiments {codes_to_save}.")
        self.logger.console_new_line()
        
    def save_schema(self, recompute: bool = False, verbose: bool = True) -> None:
        """Save schema hierarchically."""        
        # 1. Save locally
        saved = self.local_data.save_schema(self.schema.to_dict(), recompute=recompute)
        if saved:
            self._logging(f"Saved dataset schema '{self.schema.name}' as local file.", self.logger.console_saved, verbose)
        else:
            self.logger.info(f"Dataset schema '{self.schema.name}' already exists as local file.")

        # 2. Save externally
        if self.external_data and not self.debug_flag:
            pushed = self.external_data.push_schema(self.schema.name, self.schema.to_dict())
            if pushed:
                self._logging(f"Pushed dataset schema '{self.schema.name}' to external source.", self.logger.console_pushed, verbose)
            else:
                self.logger.info(f"Skipped pushing schema to external source (check ExternalData logic).")
        elif verbose:
            self.logger.info(f"Skipped external push for schema '{self.schema.name}' (debug={self.debug_flag}, has_ext_data={self.external_data is not None}).")
    
    def _check_for_retrieved_codes(self, target_pre: List[str], target_post: List[str], dtype: str, source: Loaders, verbose: bool) -> List[str]:
        """Check which codes were successfully retrieved and log to console."""
        retrieved_codes = [code for code in target_pre if code not in target_post]
        if retrieved_codes:
            message = f"Retrieved from {source.value}: {dtype} for {len(retrieved_codes)} experiments."
            if source == Loaders.MEMORY:
                self.logger.info(message)
            elif source == Loaders.LOCAL:
                self._logging(message, self.logger.console_loaded, verbose)
            elif source == Loaders.EXTERNAL:
                self._logging(message, self.logger.console_pulled, verbose)
            else:
                raise ValueError(f"Unknown Loaders source '{source}', check enum.")
        return retrieved_codes
    
    def _hierarchical_load(self, 
                        dtype: str,
                        target_codes: List[str],
                        loader: Callable[..., Tuple[List[str], Dict[str, Any]]],
                        setter: Callable[[str, Any], None],
                        in_memory: Callable[[str], bool],
                        external_loader: Optional[Callable[..., Tuple[List[str], Any]]] = None,
                        recompute_flag: bool = False,
                        verbose: bool = False,
                        **kwargs) -> List[str]:
        """Universal hierarchical data loading: Memory → Local Files → External Source"""
        # 1. Check memory
        memory_missing = [code for code in target_codes if not in_memory(code)]
        self._check_for_retrieved_codes(target_codes, memory_missing, dtype, Loaders.MEMORY, verbose)

        if not memory_missing:
            return []
        
        # 2. Load from local files
        if not recompute_flag:
            local_missing, local_data = loader(memory_missing, **kwargs)
            # directly store retrieved data in ExpData object
            for code, data in local_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(memory_missing, local_missing, dtype, Loaders.LOCAL, verbose)
        else:
            local_missing = memory_missing
            self.logger.info(f"Recompute mode: Skipping loading {dtype} from local files")

        if not local_missing:
            return []

        # 3. Load from external sources
        if not self.debug_flag and external_loader:
            external_missing, external_data = external_loader(local_missing, **kwargs)
            # directly store retrieved data in ExpData object
            for code, data in external_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(local_missing, external_missing, dtype, Loaders.EXTERNAL, verbose)
        elif self.debug_flag:
            external_missing = local_missing 
            self.logger.info(f"Debug mode: Skipping loading {dtype} from external source")
        else:
            external_missing = local_missing
            self.logger.warning(f"No external data interface provided: Skipping loading {dtype} from external source")

        return external_missing

    def _hierarchical_save(self, 
                           dtype: str,
                           target_codes: List[str],
                           getter: Callable[[str], Any],
                           saver: Callable[..., bool],
                           external_saver: Optional[Callable[..., bool]] = None,
                           column_names: Optional[List[str]] = None,
                           recompute: bool = False,
                           verbose: bool = True,
                           **kwargs) -> None:
        """Universal hierarchical data saving: Memory → Local Files → External Source"""
        # 1. Filter to codes that exist in memory (and have data)
        data_to_save = {}
        for code in target_codes:
            val = getter(code)
            # Check for non-empty data (handling dicts and arrays)
            if isinstance(val, (dict, list)) and len(val) > 0:
                data_to_save[code] = val
            elif isinstance(val, np.ndarray) and val.size > 0 and not np.all(np.isnan(val)):
                data_to_save[code] = val
            elif not isinstance(val, (dict, list, np.ndarray)):
                raise ValueError(f"Unsupported data type for saving: {type(val)}")
            else:
                self.logger.warning(f"No data to save for {dtype} '{code}'")
                    
        if not data_to_save:
            self.logger.console_warning(f"No data in memory ({len(target_codes)} exps): {dtype}")
            return
        
        codes_to_save = list(data_to_save.keys())
        
        # 2. Save to local files
        saved = saver(codes_to_save, data_to_save, recompute, column_names=column_names, **kwargs)
        if saved:
            self._logging(f"Saved to local files: {dtype} for {len(codes_to_save)} experiments.", self.logger.console_saved, verbose)
        else:
            self.logger.info(f"{dtype.capitalize()} already exist as local files.")

        # 3. Save to external source (skip if in debug mode)
        if not self.debug_flag and external_saver:
            pushed = external_saver(codes_to_save, data_to_save, recompute, **kwargs)
            if pushed:
                self._logging(f"Pushed to external source: {dtype} for {len(codes_to_save)} experiments.", self.logger.console_pushed, verbose)
            else:
                self.logger.info(f"Skipped pushing {dtype} to external source due to missing implementation in ExternalData.")
        elif self.debug_flag:
            self.logger.info(f"Skipped pushing {dtype} to external source due to debug mode.")
        else:
            self.logger.warning(f"Skipped pushing {dtype} to external source due missing ExternalData source.")

    def export_to_dataframe(self, experiment_codes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export experiment data to DataFrames (X, y).
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
            iterator_map = exp_data.parameters.get_dim_iterator_codes()
            
            # Pre-fetch feature arrays
            feature_arrays = {
                name: exp_data.features.get_value(name) 
                for name in exp_data.features.keys()
            }
            
            for idx_tuple in dim_combinations:
                # Build X row (Static + Iterators)
                row_dict = static_params.copy()
                for i, dim_name in enumerate(dim_names):
                    iterator_name = exp_data.parameters.get_dim_iterator_codes(codes=[dim_name])[0]
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

    def _logging(self, msg: str, verbose_func: Callable[[str], None], verbose: bool):
        if verbose:
            verbose_func(msg)
        else:
            self.logger.info(msg)

