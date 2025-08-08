from dataclasses import dataclass
from lbp_package.utils import (
    ParameterHandling, model_parameter, exp_parameter, runtime_parameter
)


@dataclass
class MockParameterClass(ParameterHandling):
    """Test class for parameter handling."""
    
    model_param1: float = model_parameter(1.0) # type: ignore
    model_param2: str = model_parameter("default") # type: ignore
    exp_param1: int = exp_parameter(10) # type: ignore
    exp_param2: bool = exp_parameter(False) # type: ignore
    runtime_param1: str = runtime_parameter() # type: ignore
    runtime_param2: int = runtime_parameter() # type: ignore


class TestParameterHandler:
    """Test parameter handling functionality."""
    
    def test_model_parameter_setting(self):
        """Test setting model parameters."""
        test_obj = MockParameterClass()
        
        # Set model parameters
        test_obj.set_model_parameters(
            model_param1=2.5,
            model_param2="updated",
            exp_param1=999,  # Should be ignored
            invalid_param="ignored"  # Should be ignored
        )
        
        # Verify model parameters were set
        assert test_obj.model_param1 == 2.5
        assert test_obj.model_param2 == "updated"
        
        # Verify other parameters unchanged
        assert test_obj.exp_param1 == 10
        assert test_obj.exp_param2 == False
    
    def test_experiment_parameter_setting(self):
        """Test setting experiment parameters."""
        test_obj = MockParameterClass()
        
        # Set experiment parameters
        test_obj.set_experiment_parameters(
            exp_param1=20,
            exp_param2=True,
            model_param1=999,  # Should be ignored
            invalid_param="ignored"  # Should be ignored
        )
        
        # Verify experiment parameters were set
        assert test_obj.exp_param1 == 20
        assert test_obj.exp_param2 == True
        
        # Verify other parameters unchanged
        assert test_obj.model_param1 == 1.0
        assert test_obj.model_param2 == "default"
    
    def test_runtime_parameter_setting(self):
        """Test setting runtime parameters."""
        test_obj = MockParameterClass()
        
        # Set runtime parameters
        test_obj.set_runtime_parameters(
            runtime_param1="runtime_value",
            runtime_param2=42,
            model_param1=999,  # Should be ignored
            invalid_param="ignored"  # Should be ignored
        )
        
        # Verify runtime parameters were set
        assert test_obj.runtime_param1 == "runtime_value"
        assert test_obj.runtime_param2 == 42
        
        # Verify other parameters unchanged
        assert test_obj.model_param1 == 1.0
        assert test_obj.exp_param1 == 10
    
    def test_parameter_filtering(self):
        """Test that parameter filtering works correctly."""
        test_obj = MockParameterClass()
        
        # Set mixed parameters
        all_params = {
            "model_param1": 3.0,
            "exp_param1": 30,
            "runtime_param1": "mixed_test",
            "invalid_param": "should_be_ignored"
        }
        
        test_obj.set_model_parameters(**all_params)
        test_obj.set_experiment_parameters(**all_params)
        test_obj.set_runtime_parameters(**all_params)
        
        # Verify correct parameters were set
        assert test_obj.model_param1 == 3.0
        assert test_obj.exp_param1 == 30
        assert test_obj.runtime_param1 == "mixed_test"
        
        # Verify defaults for unset parameters
        assert test_obj.model_param2 == "default"
        assert test_obj.exp_param2 == False
        assert test_obj.runtime_param2 is None
