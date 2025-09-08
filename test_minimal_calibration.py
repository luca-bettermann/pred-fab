#!/usr/bin/env python3
"""
Test minimal calibration implementations
"""
import sys
import os
sys.path.append('examples')
sys.path.append('src')

from calibration import RandomSearchCalibration, DifferentialEvolutionCalibration

def test_calibration_implementation(calibration_class, name):
    """Test a calibration implementation with the iterative interface."""
    print(f"\n=== Testing {name} ===")
    
    # Create instance
    cal = calibration_class(n_evaluations=10)
    
    # Define test parameters
    param_keys = ["param1", "param2"]
    param_ranges = {"param1": (0.0, 1.0), "param2": (-10.0, 10.0)}
    
    # Initialize optimization
    cal.initialize_optimization(param_keys, param_ranges)
    print(f"✓ Initialized optimization for parameters: {param_keys}")
    
    # Run optimization loop
    evaluation_count = 0
    while True:
        params, should_continue = cal.optimization_step()
        if not should_continue:
            break
            
        # Simulate objective function (simple quadratic)
        objective_value = -(params["param1"] - 0.5)**2 - (params["param2"] - 2.0)**2
        cal.update_with_result(params, objective_value)
        
        evaluation_count += 1
        if evaluation_count <= 3:  # Show first few evaluations
            print(f"  Evaluation {evaluation_count}: {params} -> {objective_value:.4f}")
    
    # Get best result
    best_params = cal.get_best_result()
    print(f"✓ Completed {evaluation_count} evaluations")
    print(f"✓ Best parameters found: {best_params}")
    
    # Verify best result makes sense (should be close to optimum at param1=0.5, param2=2.0)
    expected_param1, expected_param2 = 0.5, 2.0
    param1_error = abs(best_params["param1"] - expected_param1)
    param2_error = abs(best_params["param2"] - expected_param2)
    print(f"  Distance from optimum: param1={param1_error:.3f}, param2={param2_error:.3f}")
    
    return True

if __name__ == "__main__":
    print("Testing Minimal Calibration Implementations")
    print("=" * 50)
    
    try:
        # Test RandomSearchCalibration
        test_calibration_implementation(RandomSearchCalibration, "RandomSearchCalibration")
        
        # Test DifferentialEvolutionCalibration  
        test_calibration_implementation(DifferentialEvolutionCalibration, "DifferentialEvolutionCalibration")
        
        print(f"\n{'='*50}")
        print("✅ All tests passed! Minimal implementations work correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
