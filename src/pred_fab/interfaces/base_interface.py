from abc import ABC, abstractmethod
from typing import Any, final

from ..core import Dataset, DataObject, Parameter, Feature, PerformanceAttribute
from ..utils import PfabLogger


class BaseInterface(ABC):
    """Shared base for interface models — logger setup, DataObject references, and property validation."""

    def __init__(self, logger: PfabLogger):
        self.logger = logger

        # reference to DataObjects from schema
        self._ref_parameters: dict[str, Parameter] = {}
        self._ref_features: dict[str, Feature] = {}
        self._ref_performance_attrs: dict[str, DataObject] = {}

        # Validate user implemented properties
        self._validate_properties()

    @property
    @abstractmethod
    def input_parameters(self) -> list[str]:
        """Parameter codes required as model input (must match schema DataObject codes)."""
        ...

    @property
    @abstractmethod
    def input_features(self) -> list[str]:
        """Feature codes required as model input (must match schema DataObject codes)."""
        ...

    @property
    @abstractmethod
    def outputs(self) -> list[str]:
        """Feature or performance attribute codes produced by this model."""
        ...

    @final
    def _validate_properties(self) -> None:
        """Validate user implemented properties."""
        if not isinstance(self.input_parameters, list):
            raise TypeError(f"input_parameters of model {self} must be list[str], got {type(self.input_parameters).__name__}")
        if not isinstance(self.input_features, list):
            raise TypeError(f"input_features of model {self} must be list[str], got {type(self.input_features).__name__}")
        if not isinstance(self.outputs, list):
            raise TypeError(f"outputs of model {self} must be list[str], got {type(self.outputs).__name__}")

    @final
    def set_ref_parameters(self, parameters: list[Parameter]) -> None:
        """Set reference to Parameter DataObjects used by this model."""
        self.set_reference(parameters, self.input_parameters, self._ref_parameters)

    @final
    def set_ref_features(self, features: list[Feature]) -> None:
        """Set reference to Feature DataObjects used by this model."""
        self.set_reference(features, self.input_features + self.outputs, self._ref_features)

    @final
    def set_ref_performance_attrs(self, performance_attrs: list[DataObject]) -> None:
        """Set reference to performance attribute DataObjects used by this model."""
        self.set_reference(performance_attrs, self.outputs, self._ref_performance_attrs)

    @final
    def set_reference(self, parameters: list[Any], ref_property: list[str], location: dict[str, Any]) -> None:
        """Set reference to Parameter DataObjects used by this model."""
        for param in parameters:
            if param.code in ref_property:
                location[param.code] = param
                
    @property
    def input_domain(self) -> str | None:
        """Domain code this model operates in; None for experiment-level (scalar) models."""
        return None

