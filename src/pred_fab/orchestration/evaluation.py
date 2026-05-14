from typing import Any
import numpy as np
import torch

from ..utils import PfabLogger, profiler
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
        recompute: bool = False,
        feature: str | None = None,
    ) -> dict[str, float | None]:
        """Score features for a completed experiment and store results in exp_data.

        When ``feature`` is provided, only evaluation models whose
        ``input_feature`` contains that string (case-insensitive) are run.
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
            skip_for_code=skip_for_code,
            feature_filter=feature,
        )
        exp_data.performance.set_values_from_dict(performance_dict, self.logger)
        return performance_dict

    def _evaluate_feature_dict(
        self,
        features_dict: dict[str, np.ndarray],
        parameters: Parameters,
        incomplete_features: dict[str, bool] = {},
        skip_for_code: dict[str, bool] = {},
        feature_filter: str | None = None,
    ) -> dict[str, float | None]:
        """Run evaluation models and return {perf_code: value} dict."""

        # Prepare result dictionaries
        performance_dict: dict[str, float | None] = {}

        # Run evaluation for each performance code
        for eval_model in self.models:
            if feature_filter is not None:
                if feature_filter.lower() not in eval_model.input_feature.lower():
                    continue
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

    def _evaluate_feature_dict_tensor(
        self,
        features_dicts_S: list[dict[str, torch.Tensor]],
        parameters_list: list[Parameters],
    ) -> dict[str, torch.Tensor]:
        """Batched tensor eval over S candidates → ``{perf_code: (S,) tensor}``.

        Per-candidate per-cell prediction tensors are flattened to ``(S, n_rows)``
        for ``compute_performance_tensor``; gradient flows from the perf scores
        back through the prediction tensors. Candidates missing a required
        feature get NaN at their slot in the output.
        """
        S = len(features_dicts_S)
        result: dict[str, torch.Tensor] = {}
        if S == 0:
            return result

        with profiler.section("eval._evaluate_feature_dict_tensor"):
            for eval_model in self.models:
                feat_code = eval_model.input_feature
                feature_values_list: list[torch.Tensor] = []
                valid_indices: list[int] = []
                for s, feat_dict in enumerate(features_dicts_S):
                    if feat_code in feat_dict:
                        # Flatten any (*feat_shape,) tensor to (n_rows,) for the eval.
                        feature_values_list.append(feat_dict[feat_code].reshape(-1))
                        valid_indices.append(s)
                if not feature_values_list:
                    continue

                # Stack into (S_valid, n_rows). Within an acquisition call all
                # candidates share feat_shape (shape group invariant from
                # _predict_from_params_tensor); n_rows is consistent.
                try:
                    feature_values_S = torch.stack(feature_values_list, dim=0)
                except RuntimeError:
                    # Heterogeneous shapes: fall back to per-candidate loop.
                    avgs_list = []
                    for fv, idx_s in zip(feature_values_list, valid_indices):
                        avgs_list.append(eval_model.compute_performance_tensor(
                            fv.unsqueeze(0), [parameters_list[idx_s]],
                        )[0])
                    avgs = torch.stack(avgs_list)
                else:
                    valid_params = [parameters_list[s] for s in valid_indices]
                    with profiler.section(f"eval.compute_performance_tensor [{eval_model.output_performance}]"):
                        avgs = eval_model.compute_performance_tensor(
                            feature_values_S, valid_params,
                        )

                # Place into result tensor of shape (S,); pad invalid candidates with NaN.
                full = torch.full((S,), float('nan'), dtype=avgs.dtype)
                for k, s in enumerate(valid_indices):
                    full[s] = avgs[k]
                result[eval_model.output_performance] = full

        return result

    def get_models(self) -> list[IEvaluationModel]:
        """Return registered evaluation models."""
        return self.models