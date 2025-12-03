"""
Dataset class for data container with schema validation.

Dataset is an independent entity holding experiment data and validating
against a DatasetSchema. It does NOT handle persistence (that's LocalData's job).
"""

import numpy as np
import os
from typing import Callable, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .schema import DatasetSchema
from .data_blocks import DataBlock, Parameters, MetricArrays, PerformanceAttributes
from .data_objects import DataDimension

from ..interfaces.external_data import IExternalData
from ..utils.local_data import LocalData
from ..utils.logger import LBPLogger


@dataclass
class ExperimentData:
    """
    Complete data for a single experiment.
    
    - Parameters (static + dynamic + dimensional)
    - Performance, metric arrays, and predicted metric arrays (always initialized to empty blocks)
    - Features (optional, deprecated)
    """
    exp_code: str
    parameters: Parameters
    performance: PerformanceAttributes
    features: MetricArrays
    predicted_features: MetricArrays
    
    @property
    def dimensions(self) -> Dict[str, Any]:
        """View into parameters for only dimensional parameters (those with '.' in name)."""
        dims = {}
        for name, data_obj in self.parameters.items():
            if isinstance(data_obj, DataDimension):
                if self.parameters.has_value(name):
                    dims[name] = self.parameters.get_value(name)
        return dims


class Dataset:
    """
    Data container with schema validation.
    
    - Schema reference and static parameter values
    - Experiment records with hierarchical load/save
    - Feature memoization cache for IFeatureModel efficiency
    """
    
    def __init__(self, name: str, schema: DatasetSchema, schema_id: str,
                 local_data: LocalData, 
                 external_data: IExternalData,
                 logger: LBPLogger,
                 debug_mode: bool = False):
        """
        Initialize Dataset.
        
        Args:
            name: Dataset name
            schema: DatasetSchema defining structure
            schema_id: Schema ID from SchemaRegistry
            local_data: LocalData instance for file operations
            external_data: IExternalData instance for external storage
            logger: LBPLogger for progress tracking
            debug_mode: Skip external operations if True (local-only mode)
        """
        self.name = name
        self.schema = schema
        self.schema_id = schema_id
        self.local_data = local_data
        self.external_data = external_data
        self.debug_mode = debug_mode
        self.logger = logger
        
        # Master storage using ExperimentData
        self._experiments: Dict[str, ExperimentData] = {}  # exp_code → ExperimentData
        
        # Feature memoization for IFeatureModel efficiency
        self._feature_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}  # param_tuple → feature_dict

    def _create_experiment_shell(self, exp_code: str) -> ExperimentData:
        """Create new empty experiment shell with all blocks initialized."""
        params_block = Parameters()
        for name, data_obj in self.schema.parameters.items():
            params_block.add(name, data_obj)
        
        perf_block = PerformanceAttributes()
        for name, data_obj in self.schema.performance_attrs.items():
            perf_block.add(name, data_obj)
        
        arrays_block = MetricArrays()
        for name, data_obj in self.schema.features.items():
            arrays_block.add(name, data_obj)
            
        pred_block = MetricArrays()
        for name, data_obj in self.schema.features.items():
            pred_block.add(name, data_obj)
        
        return ExperimentData(
            exp_code=exp_code,
            parameters=params_block,
            performance=perf_block,
            features=arrays_block,
            predicted_features=pred_block
        )
    
    def create_experiment(
        self,
        exp_code: str,
        exp_params: Dict[str, Any],
        performance: Optional[Dict[str, Any]] = None,
        metric_arrays: Optional[Dict[str, np.ndarray]] = None,
        recompute: bool = False
    ) -> ExperimentData:
        """
        Create a new experiment manually.
        
        Args:
            exp_code: Unique experiment code
            exp_params: Dictionary of parameter values (Mandatory)
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
        exp_data = self._build_experiment_data(exp_code, exp_params, performance, metric_arrays)
        self._experiments[exp_code] = exp_data
        return exp_data

    def add_experiment(self, exp_code: str) -> ExperimentData:
        """
        Add experiment by loading it hierarchically.
        
        Args:
            exp_code: Experiment code to load
            
        Returns:
            ExperimentData object
            
        Raises:
            KeyError: If experiment cannot be found/loaded
        """
        # 2. Hierarchical load
        missing = self.load_experiments([exp_code])
        
        if exp_code in missing:
            raise KeyError(f"Experiment {exp_code} could not be loaded")
            
        return self._experiments[exp_code]
    
    def get_experiment(self, exp_code: str) -> ExperimentData:
        """Get complete ExperimentData for an exp_code."""
        if exp_code not in self._experiments:
            raise KeyError(f"Experiment {exp_code} not found")
        return self._experiments[exp_code]
    
    # === Feature Memoization for IFeatureModel ===
    
    def has_features_at(self, **param_values) -> bool:
        """Check if features are cached for specific parameter values."""
        param_tuple = self._make_param_tuple(param_values)
        return param_tuple in self._feature_cache
    
    def get_feature_value(self, feature_name: str, **param_values) -> Any:
        """Get cached feature value for specific parameters."""
        param_tuple = self._make_param_tuple(param_values)
        if param_tuple not in self._feature_cache:
            raise KeyError(f"No features cached for parameters: {param_values}")
        
        if feature_name not in self._feature_cache[param_tuple]:
            raise KeyError(f"Feature '{feature_name}' not found in cache")
        
        return self._feature_cache[param_tuple][feature_name]
    
    def set_feature_value(self, feature_name: str, value: Any, **param_values) -> None:
        """Cache feature value for specific parameters."""
        param_tuple = self._make_param_tuple(param_values)
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
            setter=lambda c, d: self._experiments[c].parameters.set_values(d),
            is_loaded=lambda c: bool(self._experiments[c].parameters.get_values_dict()),
            external_loader=self.external_data.pull_parameters,
            recompute=recompute
        )
        
        # Filter codes that were actually found (parameters are mandatory)
        found_codes = [code for code in exp_codes if code not in missing_params]
        
        if not found_codes:
            if self.logger:
                self.logger.warning(f"No parameters found for any of {len(exp_codes)} experiments.")
            return exp_codes

        # 3. Load Performance Metrics
        self._hierarchical_load(
            "performance metrics",
            found_codes,
            loader=self.local_data.load_performance,
            setter=lambda c, d: self._experiments[c].performance.set_values(d),
            is_loaded=lambda c: bool(self._experiments[c].performance.get_values_dict()),
            external_loader=self.external_data.pull_performance,
            recompute=recompute
        )
        
        # 4. Load Features
        for metric_name in self.schema.features.keys():
            self._hierarchical_load(
                f"feature '{metric_name}'",
                found_codes,
                loader=self.local_data.load_features,
                setter=lambda c, d: self._experiments[c].features.set_value(metric_name, d),
                is_loaded=lambda c: self._experiments[c].features.has_value(metric_name),
                external_loader=self.external_data.pull_features,
                recompute=recompute,
                feature_name=metric_name # Passed to kwargs
            )

        # 5. Load Predicted Features
        for metric_name in self.schema.features.keys():
            pred_name = f"predicted_{metric_name}"
            self._hierarchical_load(
                f"predicted feature '{metric_name}'",
                found_codes,
                loader=self.local_data.load_features,
                setter=lambda c, d: self._experiments[c].predicted_features.set_value(metric_name, d),
                is_loaded=lambda c: self._experiments[c].predicted_features.has_value(metric_name),
                external_loader=self.external_data.pull_features,
                recompute=recompute,
                feature_name=pred_name # Passed to kwargs
            )

        return missing_params
    
    def _build_experiment_data(
        self,
        exp_code: str,
        exp_params: Dict[str, Any],
        performance: Optional[Dict[str, Any]],
        metric_arrays: Optional[Dict[str, np.ndarray]],
        predicted_arrays: Optional[Dict[str, np.ndarray]] = None
    ) -> ExperimentData:
        """Build ExperimentData from loaded components."""
        # Create parameter block
        params_block = Parameters()
        for name, data_obj in self.schema.parameters.items():
            params_block.add(name, data_obj)
        
        # Add dynamic values
        for name, value in exp_params.items():
            if not params_block.has(name):
                # Strict schema enforcement
                raise ValueError(f"Unknown parameter: {name}")
            
            params_block.set_value(name, value)
        
        # Create performance block
        perf_block = PerformanceAttributes()
        for name, data_obj in self.schema.performance_attrs.items():
            perf_block.add(name, data_obj)
        if performance:
            for name, value in performance.items():
                if perf_block.has(name):
                    perf_block.set_value(name, value)
        
        # Create metric arrays block
        arrays_block = MetricArrays()
        for name, data_obj in self.schema.features.items():
            arrays_block.add(name, data_obj)
        if metric_arrays:
            for name, array in metric_arrays.items():
                if arrays_block.has(name):
                    arrays_block.set_value(name, array)
                    
        # Create predicted arrays block
        pred_block = MetricArrays()
        for name, data_obj in self.schema.features.items():
            pred_block.add(name, data_obj)
        if predicted_arrays:
            for name, array in predicted_arrays.items():
                if pred_block.has(name):
                    pred_block.set_value(name, array)
        
        return ExperimentData(
            exp_code=exp_code,
            parameters=params_block,
            performance=perf_block,
            features=arrays_block,
            predicted_features=pred_block
        )
    
    def _create_new_experiment(self, exp_code: str) -> ExperimentData:
        """Create new empty experiment shell."""
        # This method is deprecated/removed as per user request to not create empty shells
        # But kept for internal consistency if needed, though logic moved to create_experiment
        raise NotImplementedError("Use create_experiment with parameters instead")
    
    def save_all(self, local: bool = True, external: bool = False, recompute: bool = False) -> Dict[str, int]:
        """Save all experiments currently in memory."""
        exp_codes = list(self._experiments.keys())
        return self.save_experiments(exp_codes, local, external, recompute)
    
    def save_experiments(
        self,
        exp_codes: List[str],
        local: bool = True,
        external: bool = False,
        recompute: bool = False
    ) -> Dict[str, int]:
        """Save multiple experiments hierarchically with progress tracking."""
        # Override external flag if in debug mode
        if self.debug_mode:
            external = False
        
        # Filter to experiments that exist in dataset
        codes_to_save = [code for code in exp_codes if self.has_experiment(code)]
        
        if not codes_to_save:
            if self.logger:
                self.logger.warning(f"No experiments to save from {exp_codes}")
            return {"local": 0, "external": 0}
        
        # 1. Save Schema
        self.save_schema(local, external, recompute)
        
        # 2. Save Parameters
        self._hierarchical_save(
            "experiment parameters", codes_to_save,
            getter=lambda c: self._experiments[c].parameters.get_values_dict(),
            saver=self.local_data.save_parameters if local else lambda *a, **k: False,
            external_saver=self.external_data.push_parameters if external else None,
            recompute=recompute
        )
        
        # 3. Save Performance
        self._hierarchical_save(
            "performance metrics", codes_to_save,
            getter=lambda c: self._experiments[c].performance.get_values_dict(),
            saver=self.local_data.save_performance if local else lambda *a, **k: False,
            external_saver=self.external_data.push_performance if external else None,
            recompute=recompute
        )
            
        # 4. Save Features
        for name in self.schema.features.keys():
            def make_local_saver(metric_name):
                return lambda c, d, r, **k: self.local_data.save_features(
                    c, d, r, feature_name=metric_name, column_names=self._infer_column_names(metric_name)
                )

            def make_external_saver(metric_name):
                return lambda c, d, r, **k: self.external_data.push_features(
                    c, d, r, feature_name=metric_name
                )

            self._hierarchical_save(
                f"feature '{name}'", codes_to_save,
                getter=lambda c: self._experiments[c].features.get_value(name) if self._experiments[c].features.has_value(name) else None,
                saver=make_local_saver(name) if local else lambda *a, **k: False,
                external_saver=make_external_saver(name) if external else None,
                recompute=recompute
            )
            
        # 5. Save Predicted Features
        for name in self.schema.features.keys():
            pred_name = f"predicted_{name}"
            
            def make_local_saver_pred(metric_name):
                return lambda c, d, r, **k: self.local_data.save_features(
                    c, d, r, feature_name=metric_name, column_names=self._infer_column_names(metric_name)
                )

            def make_external_saver_pred(metric_name):
                return lambda c, d, r, **k: self.external_data.push_features(
                    c, d, r, feature_name=metric_name
                )

            self._hierarchical_save(
                f"predicted feature '{name}'", codes_to_save,
                getter=lambda c: self._experiments[c].predicted_features.get_value(name) if self._experiments[c].predicted_features.has_value(name) else None,
                saver=make_local_saver_pred(pred_name) if local else lambda *a, **k: False,
                external_saver=make_external_saver_pred(pred_name) if external else None,
                recompute=recompute
            )
        
        return {"local": len(codes_to_save), "external": len(codes_to_save) if external else 0}
    
    def save_schema(self, local: bool = True, external: bool = False, recompute: bool = False) -> None:
        """Save schema hierarchically."""
        if self.debug_mode: external = False
        
        schema_data = {"schema": self.schema.to_dict()}
        codes = ["schema"]
        
        def local_saver(c, d, r, **k):
            return self.local_data.save_schema(d["schema"], recompute=r)
            
        def external_saver(c, d, r, **k):
            return self.external_data.push_schema(self.schema_id, d["schema"])
            
        local_s = local_saver if local else lambda *args, **kwargs: False
        external_s = external_saver if external else None
        
        def schema_getter(code):
            return schema_data
            
        self._hierarchical_save(
            "schema", codes,
            getter=schema_getter,
            saver=local_s,
            external_saver=external_s,
            recompute=recompute
        )
    
    def _infer_column_names(self, array_name: str) -> Optional[List[str]]:
        """Infer column names for metric array based on dimensions."""
        # For now, return None (no columns)
        # In the future, could use dimension info to generate column names
        return None
    
    def _hierarchical_load(self, 
                        dtype: str,
                        target_codes: List[str],
                        loader: Callable[..., Tuple[List[str], Dict[str, Any]]],
                        setter: Callable[[str, Any], None],
                        is_loaded: Callable[[str], bool],
                        external_loader: Optional[Callable[..., Tuple[List[str], Any]]] = None,
                        recompute: bool = False,
                        **kwargs) -> List[str]:
        """Universal hierarchical data loading: Memory → Local Files → External Source"""
        # 1. Check memory
        missing_memory = [code for code in target_codes if not is_loaded(code)]
        self._check_for_retrieved_codes(target_codes, missing_memory, dtype, "memory")

        if not missing_memory:
            return []
        
        # 2. Load from local files
        if not recompute:
            missing_local, local_data = loader(missing_memory, **kwargs)
            for code, data in local_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(missing_memory, missing_local, dtype, "local files", console_output=True)
        else:
            missing_local = missing_memory
            if self.logger:
                self.logger.info(f"Recompute mode: Skipping loading {dtype} from local files")

        if not missing_local:
            return []

        # 3. Load from external sources
        if not self.debug_mode and external_loader:
            missing_external, external_data = external_loader(missing_local, **kwargs)
            for code, data in external_data.items():
                setter(code, data)
            self._check_for_retrieved_codes(missing_local, missing_external, dtype, "external source", console_output=True)
        elif self.debug_mode:
            missing_external = missing_local 
            if self.logger:
                self.logger.info(f"Debug mode: Skipping loading {dtype} from external source")
        else:
            missing_external = missing_local
            if self.logger:
                self.logger.warning(f"No external data interface provided: Skipping loading {dtype} from external source")

        return missing_external

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
            if self.logger:
                self.logger.debug(f"{dtype} {target_codes} found in memory.")
            return
        
        codes_to_save = list(data_to_save.keys())
        
        # 2. Save to local files
        saved = saver(codes_to_save, data_to_save, recompute, **kwargs)
        if saved and self.logger:
            self.logger.console_info(f"Saved {dtype} {codes_to_save} as local files.")
        elif self.logger:
            self.logger.info(f"{dtype.capitalize()} {codes_to_save} already exist as local files.")

        # 3. Save to external source (skip if in debug mode)
        if not self.debug_mode and external_saver:
            pushed = external_saver(codes_to_save, data_to_save, recompute, **kwargs)
            if pushed and self.logger:
                self.logger.console_info(f"Pushed {dtype} {codes_to_save} to external source.")
            elif self.logger:
                self.logger.info(f"{dtype} {codes_to_save} already exists in external source.")
        elif self.debug_mode and self.logger:
            self.logger.info(f"Debug mode: Skipped pushing {dtype} {codes_to_save} to external source.")
        elif self.logger:
            self.logger.warning(f"No external data interface provided: Skipped pushing {dtype} {codes_to_save} to external source.")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Dataset(name='{self.name}', schema_id='{self.schema_id}', "
            f"experiments={len(self._experiments)})"
        )

