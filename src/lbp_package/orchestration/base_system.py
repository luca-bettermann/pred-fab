"""
Base class for orchestration systems.

Provides shared functionality for EvaluationSystem and PredictionSystem.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional

from lbp_package.core.data_objects import DataObject

from ..core.dataset import Dataset, ExperimentData
from ..interfaces import BaseInterface
from ..utils import LBPLogger


class BaseOrchestrationSystem(ABC):
    """
    Base class for orchestration systems (Evaluation, Prediction).
    
    Provides shared functionality:
    - Dataset and logger initialization
    - DataObject extraction from model fields
    - Parameter extraction from ExperimentData
    """
    
    def __init__(self, logger: LBPLogger):
        """Initialize orchestration system with dataset and logger."""
        self.logger: LBPLogger = logger
        self.active: bool = True
        self.models: List[Any] = []
    
    def get_models(self) -> List[Any]:
        """Return registered models in implementation-specific structure."""
        return self.models
    
    def get_model_specs(self) -> Dict[str, List[str]]:
        """Extract input/output DataObject specifications from registered models."""        
        specs = {
            "input_parameters": [],
            "input_features": [],
            "outputs": [],
            }
        
        # Get models in implementation-specific structure        
        for model in self.get_models():
            specs["input_parameters"].extend(model.input_parameters)
            specs["input_features"].extend(model.input_features)
            
            for output in model.outputs:
                if output not in specs["outputs"]:
                    specs["outputs"].append(output)
                else:
                    raise ValueError(
                        f"Output '{output}' is produced by multiple models."
                    )
        return specs
    
    def deactivate(self) -> None:
        """Deactivate the orchestration system."""
        self.active = False

    def activate(self) -> None:
        """Activate the orchestration system."""
        self.active = True
