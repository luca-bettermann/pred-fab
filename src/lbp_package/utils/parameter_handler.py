from dataclasses import dataclass, field, fields, Field
from abc import ABC
from typing import Any, Set

# === PARAMETER DECORATORS ===
# AIXD-aligned naming
def parameter_static(default=None, **kwargs) -> Any:
    """Mark a dataclass field as a static parameter (constant across experiments)."""
    return field(default=default, metadata={'param_type': 'model'}, **kwargs)

def parameter_dynamic(default=None, **kwargs) -> Any:
    """Mark a dataclass field as a dynamic parameter (varies between experiments)."""
    return field(default=default, metadata={'param_type': 'experiment'}, **kwargs)

def parameter_dimensional() -> Any:
    """Mark a dataclass field as a dimensional parameter (runtime iteration)."""
    return field(default=None, metadata={'param_type': 'dimension'})

# Legacy aliases for backward compatibility
study_parameter = parameter_static
exp_parameter = parameter_dynamic
dim_parameter = parameter_dimensional

@dataclass
class ParameterHandling(ABC):
    """Base class providing automatic parameter management for the three parameter types."""
    
    # === AIXD-ALIGNED API ===
    def set_parameters_static(self, **kwargs) -> None:
        """Set static parameters (fields marked with @parameter_static)."""
        static_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'model'}
        for key, value in kwargs.items():
            if key in static_param_names:
                setattr(self, key, value)
    
    def set_parameters_dynamic(self, **kwargs) -> None:
        """Set dynamic parameters (fields marked with @parameter_dynamic)."""
        dynamic_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'experiment'}
        for key, value in kwargs.items():
            if key in dynamic_param_names:
                setattr(self, key, value)
    
    def set_parameters_dimensional(self, **kwargs) -> None:
        """Set dimensional parameters (fields marked with @parameter_dimensional)."""
        dimensional_param_names = {f.name for f in fields(self) if f.metadata.get('param_type') == 'dimension'}
        for key, value in kwargs.items():
            if key in dimensional_param_names:
                setattr(self, key, value)
    
    # === HELPER METHODS FOR SCHEMA GENERATION ===
    def get_param_names_by_type(self, param_type: str) -> Set[str]:
        """
        Extract parameter names by decorator type.
        
        Args:
            param_type: One of 'model', 'experiment', 'dimension'
            
        Returns:
            Set of parameter names
        """
        return {f.name for f in fields(self) if f.metadata.get('param_type') == param_type}
    
    def get_param_field(self, param_name: str) -> Field:
        """
        Get full field metadata for a parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Field instance with metadata
            
        Raises:
            ValueError: If parameter not found
        """
        for f in fields(self):
            if f.name == param_name:
                return f
        raise ValueError(f"Parameter '{param_name}' not found in {self.__class__.__name__}")
    
    def get_dim_parameters(self) -> dict:
        """Get current dimensional parameters as a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.metadata.get('param_type') == 'dimension'}
    
    # === LEGACY API (backward compatibility) ===
    def set_study_parameters(self, **kwargs) -> None:
        """Legacy alias for set_parameters_static."""
        self.set_parameters_static(**kwargs)
    
    def set_exp_parameters(self, **kwargs) -> None:
        """Legacy alias for set_parameters_dynamic."""
        self.set_parameters_dynamic(**kwargs)
    
    def set_dim_parameters(self, **kwargs) -> None:
        """Legacy alias for set_parameters_dimensional."""
        self.set_parameters_dimensional(**kwargs)


