"""
Dataset class for data container with schema validation.

Dataset is an independent entity holding experiment data and validating
against a DatasetSchema. It does NOT handle persistence (that's LocalData's job).
"""

import numpy as np
from typing import Dict, Any, Optional, List
from .schema import DatasetSchema


class Dataset:
    """
    Data container with schema validation.
    
    Dataset holds:
    - Reference to DatasetSchema (structure definition)
    - Static parameter values (shared across experiments)
    - Experiment records and performance data
    
    Dataset is a data container only - persistence is handled by
    LocalData through the hierarchical load/save pattern in LBPAgent.
    """
    
    def __init__(self, name: str, schema: DatasetSchema, schema_id: str):
        """
        Initialize Dataset.
        
        Args:
            name: Dataset name
            schema: DatasetSchema defining structure
            schema_id: Schema ID from SchemaRegistry
        """
        self.name = name
        self.schema = schema
        self.schema_id = schema_id
        
        # Master storage - Systems get references to these via attach_dataset()
        self._aggr_metrics: Dict[str, Dict[str, Dict]] = {}  # exp_code → perf_code → metrics
        self._metric_arrays: Dict[str, Dict[str, np.ndarray]] = {}  # perf_code → exp_code → array
        self._exp_records: Dict[str, Dict[str, Any]] = {}  # exp_code → record
        
        # Static values set once from study_params
        self._static_values: Dict[str, Any] = {}
    
    def set_static_values(self, values: Dict[str, Any]) -> None:
        """
        Set static parameter values (from study_params).
        
        Args:
            values: Dictionary of static parameter values
            
        Raises:
            ValueError: If validation fails
        """
        # Validate all static params
        for name in self.schema.static_params.keys():
            if name not in values:
                raise ValueError(f"Missing static parameter: {name}")
            self.schema.static_params.validate_value(name, values[name])
        
        self._static_values = values.copy()
    
    def add_experiment(
        self,
        exp_code: str,
        exp_params: Dict[str, Any],
        aggr_metrics: Optional[Dict[str, Dict]] = None,
        metric_arrays: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Add experiment with validation.
        
        Validates that exp_params contains all required parameters
        and that dimensional parameters are positive integers.
        
        Args:
            exp_code: Experiment code
            exp_params: Experiment parameters
            aggr_metrics: Optional aggregated metrics by performance code
            metric_arrays: Optional metric arrays by performance code
            
        Raises:
            ValueError: If required parameters missing
            TypeError: If parameter has wrong type
        """
        # Check required dynamic params present
        required = set(self.schema.dynamic_params.keys())
        provided = set(exp_params.keys())
        missing = required - provided
        
        if missing:
            raise ValueError(
                f"Experiment {exp_code} missing required parameters: {missing}"
            )
        
        # Validate dimensional params are positive integers
        for dim_param_name in self.schema.dimensional_params.keys():
            if dim_param_name not in exp_params:
                raise ValueError(
                    f"Experiment {exp_code} missing dimensional parameter: {dim_param_name}"
                )
            
            value = exp_params[dim_param_name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"Dimensional parameter {dim_param_name} must be int, "
                    f"got {type(value).__name__} for experiment {exp_code}"
                )
            
            if value < 1:
                raise ValueError(
                    f"Dimensional parameter {dim_param_name}={value} must be positive "
                    f"for experiment {exp_code}"
                )
        
        # Type validation for all dynamic params
        for name, value in exp_params.items():
            if name in self.schema.dynamic_params.data_objects:
                try:
                    self.schema.dynamic_params.validate_value(name, value)
                except (ValueError, TypeError) as e:
                    raise type(e)(
                        f"Validation failed for parameter '{name}' in experiment {exp_code}: {str(e)}"
                    ) from e
        
        # Store experiment record
        self._exp_records[exp_code] = {
            "Code": exp_code,
            "Parameters": exp_params
        }
        
        # Store metrics if provided (with validation)
        if aggr_metrics is not None:
            self._validate_performance_metrics(exp_code, aggr_metrics)
            self._aggr_metrics[exp_code] = aggr_metrics
        
        if metric_arrays is not None:
            for perf_code, array in metric_arrays.items():
                if perf_code not in self._metric_arrays:
                    self._metric_arrays[perf_code] = {}
                self._metric_arrays[perf_code][exp_code] = array
    
    def _validate_performance_metrics(
        self,
        exp_code: str,
        aggr_metrics: Dict[str, Dict]
    ) -> None:
        """
        Validate performance metrics against schema.
        
        Args:
            exp_code: Experiment code
            aggr_metrics: Aggregated metrics by performance code
            
        Raises:
            ValueError: If unexpected performance codes found
        """
        provided_perf = set(aggr_metrics.keys())
        expected_perf = set(self.schema.performance_attrs.keys())
        
        unexpected = provided_perf - expected_perf
        if unexpected:
            raise ValueError(
                f"Experiment {exp_code} has unexpected performance codes: {unexpected}. "
                f"Expected: {expected_perf}"
            )
    
    def get_experiment_codes(self) -> List[str]:
        """Get list of all experiment codes in dataset."""
        return list(self._exp_records.keys())
    
    def get_experiment_params(self, exp_code: str) -> Dict[str, Any]:
        """
        Get experiment parameters.
        
        Args:
            exp_code: Experiment code
            
        Returns:
            Parameter dictionary
            
        Raises:
            KeyError: If experiment not found
        """
        if exp_code not in self._exp_records:
            raise KeyError(f"Experiment {exp_code} not found in dataset")
        
        return self._exp_records[exp_code]["Parameters"]
    
    def has_experiment(self, exp_code: str) -> bool:
        """Check if experiment exists in dataset."""
        return exp_code in self._exp_records
    
    def get_static_value(self, param_name: str) -> Any:
        """
        Get static parameter value.
        
        Args:
            param_name: Static parameter name
            
        Returns:
            Parameter value
            
        Raises:
            KeyError: If parameter not found
        """
        if param_name not in self._static_values:
            raise KeyError(f"Static parameter {param_name} not found")
        
        return self._static_values[param_name]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Dataset(name='{self.name}', schema_id='{self.schema_id}', "
            f"experiments={len(self._exp_records)})"
        )
