"""
Dataset class for data container with schema validation.

Dataset is an independent entity holding experiment data and validating
against a DatasetSchema. It does NOT handle persistence (that's LocalData's job).
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from .schema import DatasetSchema
from .data_blocks import DataBlock, Parameters, MetricArrays, PerformanceAttributes
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
    parameters: DataBlock
    performance: DataBlock = None  # type: ignore[assignment]  # Auto-initialized in __post_init__
    metric_arrays: DataBlock = None  # type: ignore[assignment]  # Auto-initialized in __post_init__
    predicted_metric_arrays: DataBlock = None  # type: ignore[assignment]  # Auto-initialized in __post_init__
    features: Optional[DataBlock] = None
    
    def __post_init__(self):
        """Auto-initialize data blocks if not provided."""
        from .data_blocks import PerformanceAttributes, MetricArrays
        
        if self.performance is None:
            self.performance = PerformanceAttributes()
        if self.metric_arrays is None:
            self.metric_arrays = MetricArrays()
        if self.predicted_metric_arrays is None:
            self.predicted_metric_arrays = MetricArrays()

    
    @property
    def dimensions(self) -> Dict[str, Any]:
        """View into parameters for only dimensional parameters (those with '.' in name)."""
        from .data_objects import DataDimension
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
                 local_data: Optional[LocalData] = None, 
                 external_data: Optional[IExternalData] = None,
                 logger: Optional[LBPLogger] = None,
                 debug_mode: bool = False):
        """
        Initialize Dataset.
        
        Args:
            name: Dataset name
            schema: DatasetSchema defining structure
            schema_id: Schema ID from SchemaRegistry
            local_data: Optional LocalData instance for file operations
            external_data: Optional IExternalData instance for external storage
            logger: Optional LBPLogger for progress tracking
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
        
        # Static values stored in DataBlock matching schema
        self._static_values = DataBlock()
        for name, data_obj in self.schema.parameters.items():
            self._static_values.add(name, data_obj)
        
        # Feature memoization for IFeatureModel efficiency
        self._feature_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}  # param_tuple → feature_dict
    
    def set_static_values(self, values: Dict[str, Any]) -> None:
        """Set static parameter values shared across all experiments."""
        # Validate and set each static value
        for name, value in values.items():
            if not self._static_values.has(name):
                raise ValueError(f"Unknown parameter: {name}")
            self._static_values.set_value(name, value)
    
    def add_experiment(
        self,
        exp_code: str,
        exp_params: Optional[Dict[str, Any]] = None,
        performance: Optional[Dict[str, Any]] = None,
        metric_arrays: Optional[Dict[str, np.ndarray]] = None
    ) -> ExperimentData:
        """Add experiment using hierarchical load (if exp_params=None) or manual creation."""
        if exp_params is None:
            # Hierarchical load: Memory → Local → External → Create
            # 1. Check memory
            if exp_code in self._experiments:
                return self._experiments[exp_code]
            
            # 2. Try local load
            missing = self._load_from_local_batch([exp_code])
            if exp_code not in missing:
                return self._experiments[exp_code]
            
            # 3. Try external load (unless debug mode)
            if not self.debug_mode and self.external_data:
                missing = self._load_from_external_batch([exp_code])
                if exp_code not in missing:
                    return self._experiments[exp_code]
            
            # 4. Create new (empty shell with just exp_code)
            exp_data = self._create_new_experiment(exp_code)
            self._experiments[exp_code] = exp_data
            return exp_data
        else:
            # Manual creation
            exp_data = self._build_experiment_data(exp_code, exp_params, performance, metric_arrays)
            self._experiments[exp_code] = exp_data
            return exp_data
    
    def get_experiment(self, exp_code: str) -> ExperimentData:
        """Get complete ExperimentData for an experiment."""
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
    
    def get_static_value(self, param_name: str) -> Any:
        """Get static parameter value."""
        if not self._static_values.has(param_name):
            raise KeyError(f"Static parameter {param_name} not found")
        return self._static_values.get_value(param_name)
    
    # === Hierarchical Load/Save Methods ===
    
    def populate(self, source: str = "local") -> int:
        """Load all experiments from storage hierarchically by scanning dataset folder."""
        if source != "local":
            raise NotImplementedError("Only 'local' source is currently supported")
        
        if not self.local_data:
            raise ValueError("LocalData instance required for populate()")
        
        # Scan local folders for experiment codes
        exp_codes = self.local_data.list_experiments()
        
        # Use batch loading
        missing = self.load_experiments(exp_codes)
        loaded_count = len(exp_codes) - len(missing)
        
        return loaded_count
    
    def load_experiments(self, exp_codes: List[str], recompute: bool = False) -> List[str]:
        """Load multiple experiments using hierarchical pattern with progress tracking."""
        # 1. Check memory - filter already loaded experiments
        missing_memory = [
            code for code in exp_codes
            if not self.has_experiment(code)
        ]
        
        if not missing_memory:
            if self.logger:
                self.logger.info(f"All {len(exp_codes)} experiments already in memory")
            return []
        
        # 2. Load from local files (unless recompute)
        missing_local = missing_memory
        if not recompute:
            missing_local = self._load_from_local_batch(missing_memory)
            loaded_count = len(missing_memory) - len(missing_local)
            if loaded_count > 0 and self.logger:
                self.logger.console_info(f"Loaded {loaded_count} experiments from local files")
        else:
            if self.logger:
                self.logger.info("Recompute mode: Skipping local file loading")
        
        if not missing_local:
            return []
        
        # 3. Load from external source (unless debug)
        missing_external = missing_local
        if not self.debug_mode and self.external_data is not None:
            missing_external = self._load_from_external_batch(missing_local)
            loaded_count = len(missing_local) - len(missing_external)
            if loaded_count > 0 and self.logger:
                self.logger.console_info(f"Loaded {loaded_count} experiments from external source")
        elif self.debug_mode and self.logger:
            self.logger.info("Debug mode: Skipping external data loading")
        elif self.external_data is None and self.logger:
            self.logger.warning("No external interface: Skipping external loading")
        
        # 4. Create empty shells for any remaining
        for exp_code in missing_external:
            exp_data = self._create_new_experiment(exp_code)
            self._experiments[exp_code] = exp_data
        
        return missing_external
    
    def _load_from_local_batch(self, exp_codes: List[str]) -> List[str]:
        """Load multiple experiments from local storage."""
        if not self.local_data:
            return exp_codes
        
        missing = []
        
        for exp_code in exp_codes:
            try:
                # Load exp_record
                missing_exp, exp_records = self.local_data.load_exp_records([exp_code])
                if exp_code in missing_exp:
                    missing.append(exp_code)
                    continue
                
                exp_params = exp_records[exp_code].get("Parameters", {})
                
                # Load performance (returns None if missing)
                missing_perf, perf_data = self.local_data.load_aggr_metrics([exp_code])
                performance = perf_data.get(exp_code) if exp_code not in missing_perf else None
                
                # Load metric arrays for each metric in schema
                metric_arrays = {}
                for perf_code in self.schema.metric_arrays.keys():
                    missing_arrays, arrays_data = self.local_data.load_metrics_arrays(
                        [exp_code], perf_code=perf_code
                    )
                    if exp_code not in missing_arrays:
                        metric_arrays[perf_code] = arrays_data[exp_code]
                
                # Build and store experiment data
                exp_data = self._build_experiment_data(exp_code, exp_params, performance, metric_arrays)
                self._experiments[exp_code] = exp_data
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load {exp_code} from local: {e}")
                else:
                    print(f"Warning: Failed to load {exp_code} from local: {e}")
                missing.append(exp_code)
        
        return missing
    
    def _load_from_external_batch(self, exp_codes: List[str]) -> List[str]:
        """Load multiple experiments from external storage."""
        if not self.external_data:
            return exp_codes
        
        missing = []
        
        for exp_code in exp_codes:
            try:
                # Load exp_record
                missing_exp, exp_records = self.external_data.pull_exp_records([exp_code])
                if exp_code in missing_exp:
                    missing.append(exp_code)
                    continue
                
                exp_params = exp_records[exp_code].get("Parameters", {})
                
                # Load performance
                missing_perf, perf_data = self.external_data.pull_aggr_metrics([exp_code])
                performance = None if exp_code in missing_perf else perf_data[exp_code]
                
                # Load metric arrays
                missing_arrays, arrays_data = self.external_data.pull_metrics_arrays([exp_code])
                metric_arrays = None if exp_code in missing_arrays else arrays_data
                
                # Build and store experiment data
                exp_data = self._build_experiment_data(exp_code, exp_params, performance, metric_arrays)
                self._experiments[exp_code] = exp_data
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load {exp_code} from external: {e}")
                else:
                    print(f"Warning: Failed to load {exp_code} from external: {e}")
                missing.append(exp_code)
        
        return missing
    
    def _build_experiment_data(
        self,
        exp_code: str,
        exp_params: Dict[str, Any],
        performance: Optional[Dict[str, Any]],
        metric_arrays: Optional[Dict[str, np.ndarray]]
    ) -> ExperimentData:
        """Build ExperimentData from loaded components."""
        # Create parameter block
        params_block = DataBlock()
        for name, data_obj in self.schema.parameters.items():
            params_block.add(name, data_obj)
        
        # Copy static values
        for name in self._static_values.keys():
            if self._static_values.has_value(name):
                params_block.set_value(name, self._static_values.get_value(name))
        
        # Add dynamic values
        for name, value in exp_params.items():
            if not params_block.has(name):
                raise ValueError(f"Unknown parameter: {name}")
            
            if name in self._static_values.keys() and self.logger:
                self.logger.warning(f"Parameter '{name}' is static but also provided in exp_params for '{exp_code}'")
                self.logger.warning(f"Parameter '{name}' is overwritting {self._static_values.get_value(name)} with {value} for '{exp_code}'")
            params_block.set_value(name, value)
        
        # Create performance block (always initialized)
        perf_block = PerformanceAttributes()
        for name, data_obj in self.schema.performance_attrs.items():
            perf_block.add(name, data_obj)
        if performance:
            for name, value in performance.items():
                if perf_block.has(name):
                    perf_block.set_value(name, value)
        
        # Create metric arrays block (always initialized)
        arrays_block = MetricArrays()
        for name, data_obj in self.schema.metric_arrays.items():
            arrays_block.add(name, data_obj)
        if metric_arrays:
            for name, array in metric_arrays.items():
                if arrays_block.has(name):
                    arrays_block.set_value(name, array)
        
        return ExperimentData(
            exp_code=exp_code,
            parameters=params_block,
            performance=perf_block,
            metric_arrays=arrays_block,
            predicted_metric_arrays=MetricArrays()  # Always initialized
        )
    
    def _create_new_experiment(self, exp_code: str) -> ExperimentData:
        """Create new empty experiment shell."""
        params_block = DataBlock()
        for name, data_obj in self.schema.parameters.items():
            params_block.add(name, data_obj)
        
        # Copy static values
        for name in self._static_values.keys():
            if self._static_values.has_value(name):
                params_block.set_value(name, self._static_values.get_value(name))
        
        # Initialize all data blocks eagerly
        perf_block = PerformanceAttributes()
        for name, data_obj in self.schema.performance_attrs.items():
            perf_block.add(name, data_obj)
        
        arrays_block = MetricArrays()
        for name, data_obj in self.schema.metric_arrays.items():
            arrays_block.add(name, data_obj)
        
        return ExperimentData(
            exp_code=exp_code,
            parameters=params_block,
            performance=perf_block,
            metric_arrays=arrays_block,
            predicted_metric_arrays=MetricArrays()
        )
    
    def save(
        self,
        local: bool = True,
        external: bool = False,
        recompute: bool = False
    ) -> Dict[str, int]:
        """Save all experiments hierarchically to local and/or external storage."""
        # Override external flag if in debug mode
        if self.debug_mode:
            external = False
        
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
        
        results = {"local": 0, "external": 0}
        
        # 1. Save schema first
        if local:
            self._save_schema_local()
        if external and self.external_data:
            self._save_schema_external()
        
        # 2. Save all experiments
        for exp_code in codes_to_save:
            saved = self.save_experiment(exp_code, local, external, recompute)
            if saved["local"]:
                results["local"] += 1
            if saved["external"]:
                results["external"] += 1
        
        # 3. Log progress
        if results["local"] > 0 and self.logger:
            self.logger.console_info(f"Saved {results['local']} experiments to local files")
        if results["external"] > 0 and self.logger:
            self.logger.console_info(f"Saved {results['external']} experiments to external source")
        elif external and self.debug_mode and self.logger:
            self.logger.info("Debug mode: Skipping external save")
        
        return results
    
    def save_experiment(
        self,
        exp_code: str,
        local: bool = True,
        external: bool = False,
        recompute: bool = False
    ) -> Dict[str, bool]:
        """Save single experiment hierarchically."""
        if exp_code not in self._experiments:
            raise KeyError(f"Experiment {exp_code} not found")
        
        exp_data = self._experiments[exp_code]
        results = {"local": False, "external": False}
        
        # Prepare data structures
        exp_record = {
            "id": exp_code,
            "Code": exp_code,
            "Parameters": self.get_experiment_params(exp_code)
        }
        
        # Save to local
        if local and self.local_data:
            # Save exp_record
            self.local_data.save_exp_records([exp_code], {exp_code: exp_record}, recompute)
            
            # Save performance if has values
            perf_dict = exp_data.performance.get_values_dict()
            if perf_dict:
                self.local_data.save_aggr_metrics([exp_code], {exp_code: perf_dict}, recompute)
            
            # Save metric arrays if has values
            for name in exp_data.metric_arrays.keys():
                if exp_data.metric_arrays.has_value(name):
                    array = exp_data.metric_arrays.get_value(name)
                    column_names = self._infer_column_names(name)
                    self.local_data.save_metrics_arrays(
                        [exp_code],
                        {exp_code: array},
                        recompute,
                        perf_code=name,
                        column_names=column_names
                    )
            
            results["local"] = True
        
        # Save to external
        if external and self.external_data:
            # Save exp_record
            self.external_data.push_exp_records([exp_code], {exp_code: exp_record}, recompute)
            
            # Save performance if has values
            perf_dict = exp_data.performance.get_values_dict()
            if perf_dict:
                self.external_data.push_aggr_metrics([exp_code], {exp_code: perf_dict}, recompute)
            
            # Save metric arrays if has values
            arrays_dict = exp_data.metric_arrays.get_values_dict()
            if arrays_dict:
                self.external_data.push_metrics_arrays([exp_code], arrays_dict, recompute)
            
            results["external"] = True
        
        return results
    
    def _save_schema_local(self) -> bool:
        """Save schema to local storage using LocalData."""
        if not self.local_data:
            return False
        
        return self.local_data.save_schema(self.schema.to_dict(), recompute=True)
    
    def _save_schema_external(self) -> bool:
        """Push schema to external storage."""
        if not self.external_data:
            return False
        
        schema_data = self.schema.to_dict()
        return self.external_data.push_schema(self.schema_id, schema_data)
    
    def _infer_column_names(self, array_name: str) -> Optional[List[str]]:
        """Infer column names for metric array based on dimensions."""
        # For now, return None (no columns)
        # In the future, could use dimension info to generate column names
        return None
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Dataset(name='{self.name}', schema_id='{self.schema_id}', "
            f"experiments={len(self._experiments)})"
        )
