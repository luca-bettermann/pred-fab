"""Shared base class for orchestration systems."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from pred_fab.core.data_objects import DataObject

from ..core.dataset import Dataset, ExperimentData, DatasetSchema
from ..interfaces import BaseInterface
from ..utils import PfabLogger


class BaseOrchestrationSystem(ABC):
    """Base class providing shared model-registry and schema-validation helpers."""

    def __init__(self, logger: PfabLogger, random_seed: int | None = None):
        self.logger: PfabLogger = logger
        self.models: list[Any] = []
        self._random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    @property
    def random_seed(self) -> int | None:
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int | None) -> None:
        self._random_seed = value
        self.rng = np.random.RandomState(value)
    
    def get_models(self) -> list[Any]:
        """Return registered models."""
        return self.models
    
    def get_model_specs(self) -> dict[str, list[str]]:
        """Aggregate input/output specs across all registered models."""
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
    
    def set_ref_objects(self, schema: DatasetSchema) -> None:
        for model in self.get_models():
            model.set_ref_parameters(schema.parameters.data_objects.values())
            model.set_ref_features(schema.features.data_objects.values())
            model.set_ref_performance_attrs(schema.performance_attrs.data_objects.values())
    