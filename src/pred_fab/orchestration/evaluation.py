from typing import Any
import numpy as np

from ..utils import PfabLogger
from ..core import Dataset, ExperimentData, DataReal, Parameters
from ..interfaces import IEvaluationModel
from .base_system import BaseOrchestrationSystem


class EvaluationSystem(BaseOrchestrationSystem):
    """Orchestrates performance evaluation across all registered evaluation models."""

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)
        self.models: list[IEvaluationModel] = []

    # === EVALUATION ===

    def run_evaluation(
        self,
        exp_data: ExperimentData,
        recompute: bool = False
    ) -> dict[str, float | None]:
        """Score all features for a completed experiment and store results in exp_data.

        This is post-experiment documentation only — the resulting performance values are
        not consumed by any calibration or prediction logic. Calibration uses
        _evaluate_feature_dict directly via a closure for its internal scoring.
        """
        # Prepare feature dict: convert N-D tensors to 2-D tables [iter..., value]
        # so that evaluation models can iterate rows uniformly regardless of feature depth.
        features_dict: dict[str, np.ndarray] = {}
        for code, tensor in exp_data.features.get_values_dict().items():
            features_dict[code] = exp_data.features.tensor_to_table(code, tensor, exp_data.parameters)

        # Mark features that were never computed so their eval models can be skipped.
        incomplete_features = {code: not exp_data.is_complete(code, 0, None)
                               for code in exp_data.features.keys()}

        # Determine which performance codes to skip based on existing values
        skip_for_code = {code: exp_data.performance.has_value(code)
                         for code in exp_data.performance.keys() if not recompute}

        # Compute and store performance
        performance_dict = self._evaluate_feature_dict(
            features_dict=features_dict,
            parameters=exp_data.parameters,
            incomplete_features=incomplete_features,
            skip_for_code=skip_for_code
        )
        exp_data.performance.set_values_from_dict(performance_dict, self.logger)
        return performance_dict

    def _evaluate_feature_dict(
        self,
        features_dict: dict[str, np.ndarray],
        parameters: Parameters,
        incomplete_features: dict[str, bool] = {},
        skip_for_code: dict[str, bool] = {}
    ) -> dict[str, float | None]:
        """Run all evaluation models and return {perf_code: value} dict."""
        
        # Prepare result dictionaries
        performance_dict: dict[str, float | None] = {}

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
    
    def get_models(self) -> list[IEvaluationModel]:
        """Return registered evaluation models."""
        return self.models