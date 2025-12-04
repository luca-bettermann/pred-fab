"""
Dataset class for data container with schema validation.

Dataset is an independent entity holding experiment data and validating
against a DatasetSchema. It does NOT handle persistence (that's LocalData's job).
"""

import numpy as np
import os
from typing import Callable, Dict, Any, Optional, List, Tuple, Literal, Type
from dataclasses import dataclass
import functools

from .schema import DatasetSchema
from .data_blocks import DataBlock, Parameters, Dimensions, MetricArrays, PerformanceAttributes
from .data_objects import DataDimension

from ..interfaces.external_data import IExternalData
from ..utils.local_data import LocalData
from ..utils.logger import LBPLogger

BlockType = Literal['parameters', 'performance', 'feature', 'predicted_feature']

@dataclass
class ExperimentData:
    """
    Complete data for a single experiment.
    
    - Parameters
    - Dimensions
    - Performance
    - Features
    - Predicted Features
    """
    exp_code: str
    parameters: Parameters
    dimensions: Dimensions
    performance: PerformanceAttributes
    features: MetricArrays
    predicted_features: MetricArrays
    
    # @property
    # def dimensions(self) -> Dict[str, Any]:
    #     """View into parameters for only dimensional parameters (those with '.' in name)."""
    #     dims = {}
    #     for name, data_obj in self.parameters.items():
    #         if isinstance(data_obj, DataDimension):
    #             if self.parameters.has_value(name):
    #                 dims[name] = self.parameters.get_value(name)
    #     return dims
    
    def is_valid(self, schema: 'DatasetSchema') -> bool:
        """Check structural compatibility of exp with schema."""
        # Check all blocks using helper function
        block_checks = [
            (self.parameters, schema.parameters),
            (self.dimensions, schema.dimensions),
            (self.performance, schema.performance_attrs),
            (self.features, schema.features),
            (self.predicted_features, schema.features)
        ]
        
        for self_block, other_block in block_checks:
            if not self_block.is_compatible(other_block):
                raise ValueError(
                    f"Schema block '{self_block.__class__.__name__}' is not identical "
                    f"to {other_block.__class__.__name__}."
                )        
        return True

    # === Helper Methods for Data Access ===

    def set_data(self, values: Dict[str, Any], block_type: BlockType) -> None:
        """Set values for a specific data type."""
        if block_type == "parameters":
            self.parameters.set_values(values)
        elif block_type == "dimensions":
            self.dimensions.set_values(values)
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
        elif block_type == "dimensions":
            return self.dimensions.get_values_dict()
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
        elif block_type == "dimensions":
            return bool(self.dimensions.get_values_dict())
        elif block_type == "performance":
            return bool(self.performance.get_values_dict())
        elif block_type == "feature":
            return bool(self.features.get_values_dict())
        elif block_type == "predicted_feature":
            return bool(self.predicted_features.get_values_dict())
        else:
            raise ValueError(f"Unknown block type: {block_type}")

class Dataset:
    """
    Data container with schema validation.
    
    - Schema reference and static parameter values
    - Experiment records with hierarchical load/save
    - Feature memoization cache for IFeatureModel efficiency
    """
    
    def __init__(self, schema: DatasetSchema, schema_id: str,
                 local_data: LocalData, 
                 logger: LBPLogger,
                 external_data: Optional[IExternalData] = None,
                 debug_mode: bool = False):
        """
        Initialize Dataset.
        
        Args:
            schema: DatasetSchema defining structure
            schema_id: Schema ID from SchemaRegistry
            local_data: LocalData instance for file operations
            external_data: IExternalData instance for external storage
            logger: LBPLogger for progress tracking
            debug_mode: Skip external operations if True (local-only mode)
        """
        self.schema = schema
        self.schema_id = schema_id
        self.local_data = local_data
        self.logger = logger
        self.external_data = external_data
        self.debug_mode = debug_mode
        
        # Master storage using ExperimentData
        self._experiments: Dict[str, ExperimentData] = {}  # exp_code → ExperimentData
        
        # Feature memoization for IFeatureModel efficiency
        self._feature_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}  # param_tuple → feature_dict

    def get_experiment(self, exp_code: str) -> ExperimentData:
        """Get complete ExperimentData for an exp_code."""
        if exp_code not in self._experiments:
            raise KeyError(f"Experiment {exp_code} not found")
        return self._experiments[exp_code]
    
    # === Create ExperimentData Objects ===

    def _init_from_schema(self, block_class: Any, schema_dict: DataBlock) -> Any:
        block = block_class()
        for name, data_obj in schema_dict.items():
            block.add(name, data_obj)
        return block
        
    def _create_experiment_shell(self, exp_code: str) -> ExperimentData:
        """Create new empty experiment shell with all blocks initialized."""
        params_block = self._init_from_schema(Parameters, self.schema.parameters)
        dim_block = self._init_from_schema(Dimensions, self.schema.dimensions)
        perf_block = self._init_from_schema(PerformanceAttributes, self.schema.performance_attrs)
        arrays_block = self._init_from_schema(MetricArrays, self.schema.features)
        pred_block = self._init_from_schema(MetricArrays, self.schema.features)
        
        return ExperimentData(
            exp_code=exp_code,
            parameters=params_block,
            dimensions=dim_block,
            performance=perf_block,
            features=arrays_block,
            predicted_features=pred_block
        )
    
    def _build_experiment_data(
        self,
        exp_code: str,
        parameters: Dict[str, Any],
        dimensions: Dict[str, Any],
        performance: Optional[Dict[str, Any]],
        metric_arrays: Optional[Dict[str, np.ndarray]],
        predicted_arrays: Optional[Dict[str, np.ndarray]] = None
    ) -> ExperimentData:
        """Build ExperimentData from loaded components."""
        # 1. Create shell with schema structure
        exp_data = self._create_experiment_shell(exp_code)
        
        # 2. Set parameters and dimensions
        exp_data.parameters.set_values(parameters)
        exp_data.dimensions.set_values(dimensions)
        
        # 3. Set optional blocks
        if performance:
            exp_data.set_data(performance, block_type="performance")

        if metric_arrays:
            exp_data.set_data(metric_arrays, block_type="feature")
            
        if predicted_arrays:
            exp_data.set_data(predicted_arrays, block_type="predicted_feature")

        # 4. Validate against schema
        exp_data.is_valid(self.schema)

        return exp_data
    
    def create_experiment(
        self,
        exp_code: str,
        parameters: Dict[str, Any],
        dimensions: Dict[str, Any],
        performance: Optional[Dict[str, Any]] = None,
        metric_arrays: Optional[Dict[str, np.ndarray]] = None,
        recompute: bool = False
    ) -> ExperimentData:
        """
        Create a new experiment manually.
        
        Args:
            exp_code: Unique experiment code
            parameters: Dictionary of parameter values (Mandatory)
            dimensions: Dictionary of dimension values (Mandatory)
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
        exp_data = self._build_experiment_data(exp_code, parameters, dimensions, performance, metric_arrays)
        self._experiments[exp_code] = exp_data
        return exp_data
    
    # === Feature Memoization for IFeatureModel ===
    
    def has_cached_features_at(self, **dim_values) -> bool:
        """Check if features are cached for specific parameter values."""
        param_tuple = self._make_param_tuple(dim_values)
        return param_tuple in self._feature_cache
    
    def get_cached_feature_value(self, feature_name: str, **dim_values) -> Any:
        """Get cached feature value for specific parameters."""
        param_tuple = self._make_param_tuple(dim_values)
        if param_tuple not in self._feature_cache:
            raise KeyError(f"No features cached for parameters: {dim_values}")
        
        if feature_name not in self._feature_cache[param_tuple]:
            raise KeyError(f"Feature '{feature_name}' not found in cache")
        
        return self._feature_cache[param_tuple][feature_name]
    
    def cache_feature_value(self, feature_name: str, value: Any, **dim_values) -> None:
        """Cache feature value for specific parameters."""
        param_tuple = self._make_param_tuple(dim_values)
        if param_tuple not in self._feature_cache:
            self._feature_cache[param_tuple] = {}
        self._feature_cache[param_tuple][feature_name] = value
    
    def clear_feature_cache(self) -> None:
        """Clear all cached feature values. Used for recomputation."""
        self._feature_cache.clear()
    
    def _make_param_tuple(self, param_dict: Dict[str, Any]) -> Tuple[str, ...]:
        """Create hashable tuple from parameter dict for cache keys."""
        items = sorted(param_dict.items())
        return tuple(f"{name}={value}" for name, value in items)
    
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
        
        if missing_params:
            raise ValueError(f"No parameters found for any of the following experiments: {missing_params}")
        
        # Filter codes that were actually found and validate (parameters are mandatory)
        found_codes = [code for code in exp_codes if code not in missing_params]
        for code in found_codes:
            exp_data = self.get_experiment(code)
            exp_data.is_valid(self.schema)

        # 3. Load Performance Metrics
        missing_performance = self._hierarchical_load(
            "performance metrics",
            found_codes,
            loader=self.local_data.load_performance,
            setter=functools.partial(self._set_exp_data, block_type="performance"),
            in_memory=functools.partial(self._has_exp_data, block_type="performance"),
            external_loader=self.external_data.pull_performance if self.external_data else None,
            recompute=recompute
        )
        self.logger.console_info(f"Performance metrics loaded for {len(found_codes)-len(missing_performance)}/{len(found_codes)} experiments.")
        
        # 4. Load Features
        for metric_name in self.schema.features.keys():
            missing_features = self._hierarchical_load(
                f"feature '{metric_name}'",
                found_codes,
                loader=self.local_data.load_features,
                setter=functools.partial(self._set_exp_data, block_type="feature"),
                in_memory=functools.partial(self._has_exp_data, block_type="feature"),
                external_loader=self.external_data.pull_features if self.external_data else None,
                recompute=recompute,
                feature_name=metric_name # Passed to kwargs
            )
            self.logger.console_info(f"Feature '{metric_name}' loaded for {len(found_codes)-len(missing_features)}/{len(found_codes)} experiments.")

        # 5. Load Predicted Features
        for metric_name in self.schema.features.keys():
            pred_name = f"predicted_{metric_name}"
            missing_pred_features = self._hierarchical_load(
                f"predicted feature '{metric_name}'",
                found_codes,
                loader=self.local_data.load_features,
                setter=functools.partial(self._set_exp_data, block_type="predicted_feature"),
                in_memory=functools.partial(self._has_exp_data, block_type="predicted_feature"),
                external_loader=self.external_data.pull_features if self.external_data else None,
                recompute=recompute,
                feature_name=pred_name # Passed to kwargs
            )
            self.logger.console_info(f"Predicted feature '{metric_name}' loaded for {len(found_codes)-len(missing_pred_features)}/{len(found_codes)} experiments.")
        return missing_params
    
    def save_all(self, recompute: bool = False) -> None:
        """Save all experiments currently in memory."""
        exp_codes = list(self._experiments.keys())
        self.save_experiments(exp_codes, recompute=recompute)
    
    def save_experiments(self, exp_codes: List[str], recompute: bool = False) -> None:
        """Save multiple experiments hierarchically with progress tracking."""
        # Override external flag if in debug mode
        if self.debug_mode:
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
            self.logger.console_info(f"Saved dataset schema '{self.schema_id}' as local file.")
        # 2. Save externally
        if self.external_data and not self.debug_mode:
             self.external_data.push_schema(self.schema_id, self.schema.to_dict())

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
                        **kwargs) -> List[str]:
        """Universal hierarchical data loading: Memory → Local Files → External Source"""
        # 1. Check memory
        missing_memory = [code for code in target_codes if not in_memory(code)]
        self._check_for_retrieved_codes(target_codes, missing_memory, dtype, "memory")

        if not missing_memory:
            return []
        
        # 2. Load from local files
        if not recompute:
            missing_local, local_data = loader(missing_memory, **kwargs)
            # directly store retrieved data in ExpData object
            for code, data in local_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(missing_memory, missing_local, dtype, "local files", console_output=True)
        else:
            missing_local = missing_memory
            self.logger.info(f"Recompute mode: Skipping loading {dtype} from local files")

        if not missing_local:
            return []

        # 3. Load from external sources
        if not self.debug_mode and external_loader:
            missing_external, external_data = external_loader(missing_local, **kwargs)
            # directly store retrieved data in ExpData object
            for code, data in external_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(missing_local, missing_external, dtype, "external source", console_output=True)
        elif self.debug_mode:
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
        if not self.debug_mode and external_saver:
            pushed = external_saver(codes_to_save, data_to_save, recompute, **kwargs)
            if pushed:
                self.logger.console_info(f"Pushed {dtype} {codes_to_save} to external source.")
            else:
                self.logger.info(f"{dtype} {codes_to_save} already exists in external source.")
        elif self.debug_mode:
            self.logger.info(f"Debug mode: Skipped pushing {dtype} {codes_to_save} to external source.")
        else:
            self.logger.warning(f"No external data interface provided: Skipped pushing {dtype} {codes_to_save} to external source.")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Dataset(schema_id='{self.schema_id}', "
            f"experiments={len(self._experiments)})"
        )

