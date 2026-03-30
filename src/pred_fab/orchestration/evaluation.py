from typing import Any, Dict, Tuple, Type, Optional, List
import numpy as np

from ..utils import PfabLogger
from ..core import Dataset, ExperimentData, DataReal, Parameters
from ..interfaces import IEvaluationModel
from .base_system import BaseOrchestrationSystem


class EvaluationSystem(BaseOrchestrationSystem):
    """Orchestrates performance evaluation across all registered evaluation models."""

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)
        self.models: List[IEvaluationModel] = []

    # === EVALUATION ===

    def run_evaluation(
        self,
        exp_data: ExperimentData,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        recompute: bool = False
    ) -> Dict[str, Optional[float]]:
        """Execute all evaluations for an experiment and mutate exp_data with results."""

        # Prepare feature dict: convert N-D tensors to 2-D tables [iter..., value]
        # so that evaluation models can iterate rows uniformly regardless of feature depth.
        features_dict: Dict[str, np.ndarray] = {}
        for code, tensor in exp_data.features.get_values_dict().items():
            features_dict[code] = exp_data.features.tensor_to_table(code, tensor, exp_data.parameters)

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
            exp_data.performance.set_values_from_dict(performance_dict, self.logger)
        else:
            self.logger.info("Partial evaluation detected; not updating ExperimentData performance values.")
            self.logger.info(f"{performance_dict}")
        # COMMENT: is it even possible to have partial performance values? in online adaptation, we tune the prediction model, but we do not re-evaluate the performance of the features, right? As a matter of fact, we never use the evaluated values for anything, only the functions to evaluate predictions, correct? It makes sense, just want to double-check the logic.
        # COMMENT: this means, we should never run_evaluation with partial data? and we only do this to document how well an experiment performed, but we never use these values again.
        return performance_dict

    def _evaluate_feature_dict(
        self,
        features_dict: Dict[str, np.ndarray],
        parameters: Parameters,
        incomplete_features: Dict[str, bool] = {},
        skip_for_code: Dict[str, bool] = {}
    ) -> Dict[str, Optional[float]]:
        """Run all evaluation models and return {perf_code: value} dict."""
        
        # Prepare result dictionaries
        performance_dict: Dict[str, Optional[float]] = {}

        # Run evaluation for each performance code
        for eval_model in self.models:
            # Skip if already loaded
            if skip_for_code.get(eval_model.output_performance, False):
                self.logger.info(f"Skipping evaluation for '{eval_model.output_performance}' as performance already complete.")
                continue
            # Skip if the feature array is incomplete -> we only evaluate on complete feature arrays
            elif incomplete_features.get(eval_model.input_feature, False):
                self.logger.info(f"Skipping evaluation for '{eval_model.input_feature}' as feature array incomplete.'")
                continue

            # Run evaluation (we only keep the average performance)
            avg_performance, _, _ = eval_model.compute_performance(
                feature_array=features_dict[eval_model.input_feature],
                parameters=parameters,
            )

            # Collect results
            performance_dict[eval_model.output_performance] = avg_performance
            self.logger.info(f"Computed performance '{eval_model.output_performance}': {avg_performance}")
        
        return performance_dict
    
    def get_models(self) -> List[IEvaluationModel]:
        """Return registered evaluation models."""
        return self.models