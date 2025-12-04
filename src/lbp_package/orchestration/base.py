"""
Base class for orchestration systems.

Provides shared functionality for EvaluationSystem and PredictionSystem.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

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
        
        EvaluationSystem: List[IEvaluationModel]
        PredictionSystem: List[IPredictionModel]
        """
        pass
    
    def get_model_specs(self) -> Dict[str, Any]:
        """Extract input/output DataObject specifications from registered models."""        
        specs = {"inputs": {}}  # param_name -> DataObject
        
        # Get models in implementation-specific structure
        models = self.get_models()
        
        for model in models:
            # Use DataclassMixin to get schema objects
            if hasattr(model, 'get_schema_objects'):
                schema_objects = model.get_schema_objects()
                
                for param_name, data_obj in schema_objects.items():
                    # Check for conflicts
                    if param_name in specs["inputs"]:
                        existing = specs["inputs"][param_name]
                        if existing.to_dict() != data_obj.to_dict():
                            raise ValueError(
                                f"Parameter '{param_name}' has conflicting definitions:\n"
                                f"  Existing: {existing.to_dict()}\n"
                                f"  New: {data_obj.to_dict()}"
                            )
                    else:
                        specs["inputs"][param_name] = data_obj
        return specs
    
    def _get_params_from_exp_data(self, exp_data: ExperimentData) -> Dict[str, Any]:
        """Extract all parameters from exp_data into a flat dict."""
        return exp_data.parameters.get_values_dict()
