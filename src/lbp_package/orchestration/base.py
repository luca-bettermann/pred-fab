"""
Base class for orchestration systems.

Provides shared functionality for EvaluationSystem and PredictionSystem.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import fields, is_dataclass

from ..core.dataset import Dataset, ExperimentData
from ..utils import LBPLogger


class BaseOrchestrationSystem(ABC):
    """
    Base class for orchestration systems (Evaluation, Prediction).
    
    Provides shared functionality:
    - Dataset and logger initialization
    - DataObject extraction from model fields
    - Parameter extraction from ExperimentData
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize orchestration system with dataset and logger."""
        self.dataset = dataset
        self.logger = logger
    
    @abstractmethod
    def get_models(self) -> Any:
        """
        Return registered models in implementation-specific structure.
        
        EvaluationSystem: Dict[str, IEvaluationModel]
        PredictionSystem: List[IPredictionModel]
        """
        pass
    
    def get_model_specs(self) -> Dict[str, Any]:
        """Extract input/output DataObject specifications from registered models."""
        from ..core.data_objects import DataReal, DataInt, DataBool, DataCategorical, DataDimension
        
        specs = {"inputs": {}}  # param_name -> DataObject
        
        # Get models in implementation-specific structure
        models = self.get_models()
        
        # Handle both dict (EvaluationSystem) and list (PredictionSystem) structures
        model_list = models.values() if isinstance(models, dict) else models
        
        for model in model_list:
            # Verify model is a dataclass
            if not is_dataclass(model):
                continue
            
            # Extract input parameters from model fields
            model_fields = fields(model)
            for field in model_fields:
                # Skip special fields
                if field.name in ('logger', 'feature_model', 'feature_models', 'dataset'):
                    continue
                
                # Check if field default is a DataObject
                if field.default is not field.default_factory:  # type: ignore
                    default_val = field.default
                    data_object_types = (DataReal, DataInt, DataBool, DataCategorical, DataDimension)
                    
                    if isinstance(default_val, data_object_types):
                        param_name = field.name
                        
                        # Check for conflicts
                        if param_name in specs["inputs"]:
                            existing = specs["inputs"][param_name]
                            if existing.to_dict() != default_val.to_dict():
                                raise ValueError(
                                    f"Parameter '{param_name}' has conflicting definitions:\n"
                                    f"  Existing: {existing.to_dict()}\n"
                                    f"  New: {default_val.to_dict()}"
                                )
                        else:
                            specs["inputs"][param_name] = default_val
        
        return specs
    
    def _extract_params_from_exp_data(self, exp_data: ExperimentData) -> Dict[str, Any]:
        """Extract all parameters from exp_data into a flat dict."""
        params = {}
        for name in self.dataset.schema.parameters.keys():
            if exp_data.parameters.has_value(name):
                params[name] = exp_data.parameters.get_value(name)
        return params
