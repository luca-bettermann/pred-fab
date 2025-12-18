"""
Metrics utility for calculating performance metrics.
"""

import numpy as np
from typing import Dict, Union, Optional

class Metrics:
    """Static class for calculating regression metrics."""
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate MAE, RMSE, and R² for regression results.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing 'mae', 'rmse', 'r2', and 'n_samples'
        """
        # Ensure numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
            
        if len(y_true) == 0:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'r2': 0.0,
                'n_samples': 0
            }
            
        # Calculate metrics
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        # Avoid division by zero for constant target
        if ss_tot < 1e-8:
            r2 = 0.0 if ss_res > 1e-8 else 1.0
        else:
            r2 = float(1 - (ss_res / ss_tot))
            
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_samples': len(y_true)
        }
