from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, final

from ..core import Dataset, DataObject, DataDimension, Parameter, Feature, PerformanceAttribute
from ..utils import PfabLogger


class BaseInterface(ABC):
    """Shared base for interface models — logger setup, DataObject references, and property validation."""
    
    def __init__(self, logger: PfabLogger):
        self.logger = logger

        # reference to DataObjects from schema
        self._ref_parameters: Dict[str, Parameter] = {}
        self._ref_features: Dict[str, Feature] = {}
        self._ref_performance_attrs: Dict[str, DataObject] = {}

        # Validate user implemented properties
        self._validate_properties()

    @property
    @abstractmethod
    def input_parameters(self) -> List[str]:
        """Parameter codes required as model input (must match schema DataObject codes)."""
        ...

    @property
    @abstractmethod
    def input_features(self) -> List[str]:
        """Feature codes required as model input (must match schema DataObject codes)."""
        ...

    @property
    @abstractmethod
    def outputs(self) -> List[str]:
        """Feature or performance attribute codes produced by this model."""
        ...

    @final
    def _validate_properties(self) -> None:
        """Validate user implemented properties."""
        if not isinstance(self.input_parameters, list):
            raise TypeError(f"input_parameters of model {self} must be List[str], got {type(self.input_parameters).__name__}")
        if not isinstance(self.input_features, list):
            raise TypeError(f"input_features of model {self} must be List[str], got {type(self.input_features).__name__}")
        if not isinstance(self.outputs, list):
            raise TypeError(f"outputs of model {self} must be List[str], got {type(self.outputs).__name__}")

    @final
    def set_ref_parameters(self, parameters: List[Parameter]) -> None:
        """Set reference to Parameter DataObjects used by this model."""
        self.set_reference(parameters, self.input_parameters, self._ref_parameters)

    @final
    def set_ref_features(self, features: List[Feature]) -> None:
        """Set reference to Feature DataObjects used by this model."""
        self.set_reference(features, self.input_features + self.outputs, self._ref_features)

    @final
    def set_ref_performance_attrs(self, performance_attrs: List[DataObject]) -> None:
        """Set reference to performance attribute DataObjects used by this model."""
        self.set_reference(performance_attrs, self.outputs, self._ref_performance_attrs)

    @final
    def set_reference(self, parameters: List[Any], ref_property: List[str], location: Dict[str, Any]) -> None:
        """Set reference to Parameter DataObjects used by this model."""
        for param in parameters:
            if param.code in ref_property:
                location[param.code] = param
                
    @final
    def get_input_dimensions(self) -> List[DataDimension]:
        """Get list of required dimension names from input_parameters."""
        return [obj for obj in self._ref_parameters.values() if isinstance(obj, DataDimension)]
    
