from typing import Any, Dict, Tuple, Type, Optional, List
import numpy as np

from ..utils import LBPLogger
from ..core import Dataset, ExperimentData, DataReal, Parameters
from .base import BaseOrchestrationSystem


class EvaluationSystem(BaseOrchestrationSystem):
    """
    Orchestrates multiple evaluation models using Dataset.
    
    Manages evaluation model execution and stores results in ExperimentData.
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize evaluation system."""
        super().__init__(dataset, logger)

    # === EVALUATION ===

    def compute_exp_evaluation(
        self,
        exp_data: ExperimentData,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        recompute: bool = False
    ) -> Dict[str, Optional[float]]:
        """Execute all evaluations for an experiment and mutate exp_data with results."""
        
        # Prepare feature dict slice
        features_dict = exp_data.features.get_values_dict()

        # Check if the there are any incomplete feature arrays
        incomplete_features = {code: not exp_data.is_complete(code, evaluate_from, evaluate_to) 
                               for code in exp_data.features.keys()}
        
        # Determine which performance codes to skip based on existing values
        skip_for_code = {code: exp_data.performance.has_value(code)
                         for code in exp_data.performance.keys() if not recompute}

        # Compute performance from features
        performance_dict = self._evaluate_feature_dict(
            features_dict=features_dict,
            parameters=exp_data.parameters,
            incomplete_features=incomplete_features,
            skip_for_code=skip_for_code
        )

        # Update exp_data with results
        # We only set performance if the full experiment was evaluated
        if evaluate_from == 0 and evaluate_to is None:
            exp_data.performance.set_values(performance_dict)
        else:
            self.logger.info("Partial evaluation detected; not updating ExperimentData performance values.")
            self.logger.info(f"{performance_dict}")

        return performance_dict

    def _evaluate_feature_dict(
        self,
        features_dict: Dict[str, np.ndarray],
        parameters: Parameters,
        incomplete_features: Dict[str, bool] = {},
        skip_for_code: Dict[str, bool] = {}
    ) -> Dict[str, Optional[float]]:
        """Core evaluation logic from raw parameters."""
        
        # Prepare result dictionaries
        performance_dict: Dict[str, Optional[float]] = {}

        # Run evaluation for each performance code
        for eval_model in self.models:
            # Skip if already loaded
            if skip_for_code.get(eval_model.outputs, False):
                self.logger.info(f"Skipping evaluation for '{eval_model.outputs}' as performance already complete.")
                continue
            # Skip if the feature array is incomplete -> we only evaluate on complete feature arrays
            elif incomplete_features.get(eval_model.input_features, False):
                self.logger.info(f"Skipping evaluation for '{eval_model.input_features}' as feature array incomplete.'")
                continue

            # Run evaluation (we only keep the average performance))
            avg_performance, _ = eval_model.compute_performance(
                feature_array=features_dict[eval_model.input_features],
                parameters=parameters
                )

            # Collect results
            performance_dict[eval_model.outputs] = avg_performance
            self.logger.info(f"Computed performance '{eval_model.outputs}': {avg_performance}")
        
        return performance_dict