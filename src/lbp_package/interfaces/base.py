from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..core import Dataset, DataObject, DataDimension
from ..utils import LBPLogger


class BaseInterface(ABC):
    """
    Base class for interface models (Evaluation, Prediction).
    
    Provides shared functionality:
    - Dataset and logger initialization
    - DataObject extraction from model fields
    - Parameter extraction from ExperimentData
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize interface model with dataset and logger."""
        self.dataset = dataset
        self.logger = logger

    @property
    @abstractmethod
    def required_parameters(self) -> List[DataObject]:
        """
        Define the parameters and dimensions this model needs from the experiment.
        
        Returns:
            List of DataObjects defining the schema.
            Example: [Parameter.real("speed", ...), Dimension.int("layers", ...)]
        """
        ...

    def _get_required_dimensions(self) -> List[DataDimension]:
        """Get list of required dimension names from required_parameters."""
        return [obj for obj in self.required_parameters if isinstance(obj, DataDimension)]