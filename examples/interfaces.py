"""
Mockup interfaces for the example workflow.
"""

import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from lbp_package.interfaces import (
    IFeatureModel, 
    IEvaluationModel, 
    IPredictionModel, 
    IExternalData
)
from lbp_package.core import DataObject, DataReal, DataInt, DataDimension, DataArray, Dataset
from lbp_package.utils import LBPLogger

# --- Feature Models ---

class MockFeatureModelA(IFeatureModel):
    """Mock feature model A."""
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        super().__init__(dataset, logger)
        
    @property
    def input_parameters(self) -> List[str]:
        return ["param_1", "param_2"]
        
    @property
    def input_features(self) -> List[str]:
        return []
        
    @property
    def outputs(self) -> List[str]:
        return ["feature_1", "feature_2"]

    def _load_data(self, params: Dict, **dimensions) -> Any:
        return {"base_val": params.get("param_1", 0) * params.get("param_2", 1)}

    def _compute_feature_logic(
        self, 
        data: Any, 
        params: Dict, 
        visualize: bool = False,
        **dimensions
        ) -> Dict[str, float]:
        
        base = data["base_val"]
        d1 = dimensions.get("d1", 0)
        d2 = dimensions.get("d2", 0)
        
        f1 = base + d1 * 0.1 + d2 * 0.01
        f2 = base * 0.5 - d1 * 0.05
        
        return {
            "feature_1": f1,
            "feature_2": f2
        }

class MockFeatureModelB(IFeatureModel):
    """Mock feature model B."""
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        super().__init__(dataset, logger)
        
    @property
    def input_parameters(self) -> List[str]:
        return []
        
    @property
    def input_features(self) -> List[str]:
        return []
        
    @property
    def outputs(self) -> List[str]:
        return ["feature_3"]

    def _load_data(self, params: Dict, **dimensions) -> Any:
        return {}

    def _compute_feature_logic(
        self, 
        data: Any, 
        params: Dict, 
        visualize: bool = False,
        **dimensions
        ) -> Dict[str, float]:
        
        # Just random noise or constant
        return {
            "feature_3": np.random.random()
        }

# --- Evaluation Models ---

class MockEvaluationModelA(IEvaluationModel):
    """Mock evaluation model A."""
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        super().__init__(dataset, logger)
        
    @property
    def input_parameters(self) -> List[str]:
        return ["param_1"]
        
    @property
    def input_features(self) -> List[str]:
        return ["feature_1"]
        
    @property
    def outputs(self) -> List[str]:
        return ["performance_1"]

    def input_feature(self) -> str:
        return "feature_1"

    def output_performance(self) -> str:
        return "performance_1"

    def _compute_target_value(self, params: Dict, **dimensions) -> float:
        return params.get("param_1", 5.0) * 2.0

class MockEvaluationModelB(IEvaluationModel):
    """Mock evaluation model B."""
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        super().__init__(dataset, logger)
        
    @property
    def input_parameters(self) -> List[str]:
        return []
        
    @property
    def input_features(self) -> List[str]:
        return ["feature_2"]
        
    @property
    def outputs(self) -> List[str]:
        return ["performance_2"]

    def input_feature(self) -> str:
        return "feature_2"

    def output_performance(self) -> str:
        return "performance_2"

    def _compute_target_value(self, params: Dict, **dimensions) -> float:
        return 10.0 # Constant target

# --- Prediction Model ---

class MockPredictionModel(IPredictionModel):
    """Mock prediction model."""
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        super().__init__(dataset, logger)
        # Inputs: param_1, param_2, dim_1, dim_2, feature_3
        # Outputs: feature_1, feature_2
        self.weights = np.random.rand(5, 2) 
        
    @property
    def input_parameters(self) -> List[str]:
        return ["param_1", "param_2", "dim_1", "dim_2"]
        
    @property
    def input_features(self) -> List[str]:
        return ["feature_3"]
        
    @property
    def outputs(self) -> List[str]:
        return ["feature_1", "feature_2"]

    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.weights.shape[0]:
             return np.dot(X[:, :self.weights.shape[0]], self.weights)
        return np.dot(X, self.weights)

    def train(self, train_batches: List[Tuple[np.ndarray, np.ndarray]], val_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        self.logger.info("Mock training...")
        self.weights += 0.01

# --- External Data Source ---

class MockExternalData(IExternalData):
    """Mock external data source."""
    
    def pull_parameters(self, exp_codes: List[str]) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        found_params = {}
        missing = []
        
        for i, code in enumerate(exp_codes):
            found_params[code] = {
                "param_1": 5.0 * i/3,
                "param_2": 2 + i/2,
                "dim_1": 2 + i,
                "dim_2": 3
            }
            
        return missing, found_params
