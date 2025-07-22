from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
from typing import List

def model_parameter(default=None, **kwargs):
    """
    Mark a dataclass field as a model parameter.

    Model parameters:
    - Define the model's configuration
    - Are typically set during initialization and remain constant

    Args:
        default: Default value for the parameter
        **kwargs: Additional keyword arguments for dataclasses.field

    Returns:
        A dataclass field with model parameter metadata
    """
    return field(default=default, metadata={'param_type': 'model'}, **kwargs)


def exp_parameter(default=None, **kwargs):
    """
    Mark a dataclass field as an experiment parameter.

    Experiment parameters:
    - Are specific to a particular experiment
    - Can be used to distinguish between different experimental setups

    Args:
        default: Default value for the parameter
        **kwargs: Additional keyword arguments for dataclasses.field

    Returns:
        A dataclass field with experiment parameter metadata
    """
    return field(default=default, metadata={'param_type': 'experiment'}, **kwargs)


def runtime_parameter():
    """
    Mark a dataclass field as a runtime parameter.

    Runtime parameters:
    - Default to None (must be provided at runtime)
    - Can change between execution runs
    - Are typically experiment-specific values

    Returns:
        A dataclass field with runtime parameter metadata and default None
    """
    return field(default=None, metadata={'param_type': 'runtime'})

@dataclass
class ParameterHandling(ABC):
    """
    Elegant parameter handling system using dataclasses with three parameter types.
    
    This system provides clean parameter management:
    - Model parameters: Fields marked with @model_parameter() decorator
    - Experiment parameters: Fields marked with @exp_parameter() decorator  
    - Runtime parameters: Fields marked with @runtime_parameter() decorator
    - Automatic parameter filtering using **kwargs unpacking
    - Extra parameters are automatically ignored
    - Full type safety and IDE support
    """
    
    def set_model_parameters(self, **kwargs) -> None:
        """
        Set model parameters (fields marked with @model_parameter()).
        
        Model parameters define the model's configuration and are typically
        set during initialization.
        
        Args:
            **kwargs: Model parameters to set
        """
        model_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'model'}
        for key, value in kwargs.items():
            if key in model_param_names:
                setattr(self, key, value)

    def set_experiment_parameters(self, **kwargs) -> None:
        """
        Set experiment parameters (fields marked with @exp_parameter()).
        
        Experiment parameters are specific to a particular experiment and
        can be used to distinguish between different experimental setups.
        
        Args:
            **kwargs: Experiment parameters to set
        """
        experiment_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'experiment'}
        for key, value in kwargs.items():
            if key in experiment_param_names:
                setattr(self, key, value)

    def set_runtime_parameters(self, **kwargs) -> None:
        """
        Set runtime parameters (fields marked with @runtime_parameter()).

        Runtime parameters can change between runs and are typically
        experiment-specific values passed during execution.

        Args:
            **kwargs: Runtime parameters to set
        """
        runtime_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'runtime'}
        for key, value in kwargs.items():
            if key in runtime_param_names:
                setattr(self, key, value)

    @abstractmethod
    def _validate_parameters(self) -> None:
        """
        Validate parameter values after they have been set.
        
        Subclasses can override this method to implement custom validation logic
        for their specific parameters. This method is called after parameter initialization.
        """
        ...
