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

    def _evaluate_feature_dict_batched(
        self,
        features_dicts_S: list[dict[str, np.ndarray]],
        parameters_list: list[Parameters],
    ) -> list[dict[str, float | None]]:
        """Batched eval over S candidates: per eval_model, dispatch one
        ``compute_performance_batched`` call processing all S candidates in
        parallel. Returns one ``{perf_code: value}`` dict per candidate.

        Models with ``TARGETS_CONSTANT=True`` benefit from vectorised numpy
        arithmetic across all ``(S, n_rows)`` cells; others fall back to a
        per-candidate scalar loop inside ``compute_performance_batched``.

        Used by the calibration hot path (``perf_fn_batched`` in PfabAgent).
        ``incomplete_features`` / ``skip_for_code`` filtering doesn't apply
        here — calibration always evaluates from in-memory predictions.
        """
        S = len(features_dicts_S)
        result: list[dict[str, float | None]] = [{} for _ in range(S)]
        if S == 0:
            return result

        with profiler.section("eval._evaluate_feature_dict_batched"):
            for eval_model in self.models:
                feat_code = eval_model.input_feature
                feature_arrays: list[np.ndarray] = []
                valid_indices: list[int] = []
                for s, feat_dict in enumerate(features_dicts_S):
                    if feat_code in feat_dict:
                        feature_arrays.append(feat_dict[feat_code])
                        valid_indices.append(s)
                if not feature_arrays:
                    continue

                valid_params = [parameters_list[s] for s in valid_indices]
                with profiler.section(f"eval.compute_performance_batched [{eval_model.output_performance}]"):
                    avgs = eval_model.compute_performance_batched(feature_arrays, valid_params)
                for k, s in enumerate(valid_indices):
                    result[s][eval_model.output_performance] = avgs[k]

        return result

    def _evaluate_feature_dict_tensor(
        self,
        features_dicts_S: list[dict[str, torch.Tensor]],
        parameters_list: list[Parameters],
    ) -> dict[str, torch.Tensor]:
        """Tensor-typed batched eval. Returns ``{perf_code: torch.Tensor (S,)}``.

        Mirrors ``_evaluate_feature_dict_batched`` but routes per eval_model
        through ``compute_performance_tensor`` (gradient-traversable) instead
        of ``compute_performance_batched`` (numpy). Returns a flat dict
        keyed by performance code (not a per-candidate list of dicts), since
        gradient acquisition treats the per-candidate axis as the batch dim
        and reduces over it via mean/sum elsewhere.

        Each ``features_dicts_S[s][feat_code]`` is the per-candidate
        per-cell prediction tensor of shape ``(*feat_shape,)``. Internally
        flattened to ``(S, n_rows)`` for the eval model. Gradient flows
        from the output back through the per-feat tensor inputs to whatever
        leaf produced them (typically a params tensor in Strategy D commit 5).

        Candidates missing a required feature are excluded from that
        eval_model's batch — their entry in the output is filled with NaN
        (the gradient-aware equivalent of "no perf score").
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