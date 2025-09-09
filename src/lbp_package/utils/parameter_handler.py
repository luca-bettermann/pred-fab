from dataclasses import dataclass, field, fields, Field
from abc import ABC
from typing import Any

# === PARAMETER DECORATORS ===
def study_parameter(default=None, **kwargs) -> Any:
    """Mark a dataclass field as a study parameter (constant across experiments)."""
    return field(default=default, metadata={'param_type': 'model'}, **kwargs)

def exp_parameter(default=None, **kwargs) -> Any:
    """Mark a dataclass field as an experiment parameter (varies between experiments)."""
    return field(default=default, metadata={'param_type': 'experiment'}, **kwargs)

def dim_parameter() -> Any:
    """Mark a dataclass field as a dimensional parameter (changes during execution)."""
    return field(default=None, metadata={'param_type': 'dimension'})

@dataclass
class ParameterHandling(ABC):
    """Base class providing automatic parameter management for the three parameter types."""
    
    # === PUBLIC API METHODS ===
    def set_study_parameters(self, **kwargs) -> None:
        """Set study parameters (fields marked with @study_parameter)."""
        model_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'model'}
        for key, value in kwargs.items():
            if key in model_param_names:
                setattr(self, key, value)

    def set_exp_parameters(self, **kwargs) -> None:
        """Set experiment parameters (fields marked with @exp_parameter)."""
        experiment_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'experiment'}
        for key, value in kwargs.items():
            if key in experiment_param_names:
                setattr(self, key, value)

    def set_dim_parameters(self, **kwargs) -> None:
        """Set dimensional parameters (fields marked with @dim_parameter)."""
        runtime_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'dimension'}
        for key, value in kwargs.items():
            if key in runtime_param_names:
                setattr(self, key, value)

    def get_dim_parameters(self) -> dict:
        """Get current dimensional parameters as a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.metadata.get('param_type') == 'dimension'}

