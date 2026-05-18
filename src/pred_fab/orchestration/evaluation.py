from typing import Any
import numpy as np
import torch

from ..core import ExperimentData, Parameters
from ..core.data_objects import Features
from ..interfaces.evaluation import IEvaluationModel
from ..utils import PfabLogger, profiler
from .base_system import BaseOrchestrationSystem


class EvaluationSystem(BaseOrchestrationSystem):
    """Aggregates dimensional features into scalar experiment-level performance."""

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)
        self.models: list[IEvaluationModel] = []

    def run_evaluation(
        self,
        exp_data: ExperimentData,
        recompute: bool = False,
        feature: str | None = None,
    ) -> dict[str, float | None]:
        """Score features for a completed experiment and store results in exp_data."""
        features_dict: dict[str, np.ndarray] = {}
        for code, tensor in exp_data.features.get_values_dict().items():
            features_dict[code] = exp_data.features.tensor_to_table(code, tensor, exp_data.parameters)

        incomplete_features = {code: not exp_data.is_complete(code, 0, None)
                               for code in exp_data.features.keys()}

        skip_for_code = {code: exp_data.performance.has_value(code)
                         for code in exp_data.performance.keys() if not recompute}

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
        performance_dict: dict[str, float | None] = {}

        for eval_model in self.models:
            if feature_filter is not None:
                if not any(feature_filter.lower() in f.lower() for f in eval_model.input_features):
                    continue

            if skip_for_code.get(eval_model.output_performance, False):
                self.logger.info(f"Skipping evaluation for '{eval_model.output_performance}' — already complete.")
                continue

            if any(incomplete_features.get(f, False) for f in eval_model.input_features):
                self.logger.info(f"Skipping evaluation for '{eval_model.output_performance}' — input feature incomplete.")
                continue

            if not all(f in features_dict for f in eval_model.input_features):
                self.logger.warning(f"Skipping '{eval_model.output_performance}' — missing input features.")
                continue

            model_features = {f: features_dict[f] for f in eval_model.input_features}
            avg_performance, _ = eval_model.compute_performance(model_features, parameters)
            performance_dict[eval_model.output_performance] = avg_performance
            self.logger.info(f"Computed performance '{eval_model.output_performance}': {avg_performance}")

        return performance_dict

    def _evaluate_feature_dict_tensor(
        self,
        features_dicts_S: list[dict[str, torch.Tensor]],
        parameters_list: list[Parameters],
    ) -> dict[str, torch.Tensor]:
        """Batched tensor eval over S candidates → ``{perf_code: (S,) tensor}``."""
        S = len(features_dicts_S)
        result: dict[str, torch.Tensor] = {}
        if S == 0:
            return result

        with profiler.section("eval._evaluate_feature_dict_tensor"):
            for eval_model in self.models:
                feat_codes = eval_model.input_features

                valid_indices: list[int] = []
                for s, feat_dict in enumerate(features_dicts_S):
                    if all(f in feat_dict for f in feat_codes):
                        valid_indices.append(s)
                if not valid_indices:
                    continue

                model_tensors: dict[str, torch.Tensor] = {}
                try:
                    for f in feat_codes:
                        stacked = torch.stack(
                            [features_dicts_S[s][f].reshape(-1) for s in valid_indices], dim=0,
                        )
                        model_tensors[f] = stacked
                except RuntimeError:
                    avgs_list = []
                    for s in valid_indices:
                        single = {f: features_dicts_S[s][f].reshape(-1).unsqueeze(0) for f in feat_codes}
                        avgs_list.append(eval_model.compute_performance_tensor(
                            single, [parameters_list[s]],
                        )[0])
                    avgs = torch.stack(avgs_list)
                else:
                    valid_params = [parameters_list[s] for s in valid_indices]
                    with profiler.section(f"eval.compute_performance_tensor [{eval_model.output_performance}]"):
                        avgs = eval_model.compute_performance_tensor(model_tensors, valid_params)

                full = torch.full((S,), float('nan'), dtype=avgs.dtype)
                for k, s in enumerate(valid_indices):
                    full[s] = avgs[k]
                result[eval_model.output_performance] = full

        return result

    def get_models(self) -> list[IEvaluationModel]:
        return self.models

    def get_model_specs(self) -> dict[str, list[str]]:
        return {m.output_performance: m.input_features for m in self.models}
