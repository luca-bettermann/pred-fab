"""
Parameter Handling System for LBP Framework

This module provides an elegant parameter handling system using dataclasses with three distinct parameter types:

PARAMETER TYPES:
================

1. **study_parameter**: 
   - Defined on study level and stay static for different experiments
   - Set by study data that is pulled from the data interface
   - Remain constant throughout all experiments within a study
   - Examples: model configuration, equipment settings, material properties

2. **exp_parameter**: 
   - Defined on experiment level and vary between different experiments  
   - Typically defined as a study variable and set as parameter on experiment level
   - Specific to a particular experiment within a study
   - Examples: layer_time, temperature, printing_speed

3. **dim_parameter**: 
   - Define the segmentation/dimensionality of an experiment and change throughout execution
   - Give information about the current dimensionality index during processing
   - Set dynamically during experiment execution to track current position
   - Examples: layer_id, segment_id, measurement_point_id

USAGE EXAMPLE:
==============

    @dataclass
    class MyModel(ParameterHandling):
        # Study-level parameters (constant across experiments)
        power_rating: float = study_parameter(50.0)
        equipment_type: str = study_parameter("printer_v2")
        
        # Experiment-level parameters (vary between experiments)
        layer_time: float = exp_parameter()
        temperature: float = exp_parameter(200.0)
        
        # Dimensional parameters (change during execution)
        layer_id: int = dim_parameter()
        segment_id: int = dim_parameter()

    # Usage
    model = MyModel()
    model.set_study_parameters(power_rating=75.0, equipment_type="printer_v3")
    model.set_exp_parameters(layer_time=30.0, temperature=220.0)
    model.set_dim_parameters(layer_id=5, segment_id=2)

"""

from dataclasses import dataclass, field, fields
from abc import ABC

def study_parameter(default=None, **kwargs):
    """Mark a dataclass field as a study parameter (constant across experiments)."""
    return field(default=default, metadata={'param_type': 'model'}, **kwargs)


def exp_parameter(default=None, **kwargs):
    """Mark a dataclass field as an experiment parameter (varies between experiments)."""
    return field(default=default, metadata={'param_type': 'experiment'}, **kwargs)


def dim_parameter():
    """Mark a dataclass field as a dimensional parameter (changes during execution)."""
    return field(default=None, metadata={'param_type': 'runtime'})

@dataclass
class ParameterHandling(ABC):
    """Base class providing automatic parameter management for the three parameter types."""
    
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
        runtime_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'runtime'}
        for key, value in kwargs.items():
            if key in runtime_param_names:
                setattr(self, key, value)

