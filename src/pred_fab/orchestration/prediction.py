"""
Prediction System for managing prediction models.

Coordinates prediction model training and inference within the AIXD architecture.
Prediction models predict features, which can then be evaluated for performance.
Integrates with DataModule for normalization and batching.
"""

from dataclasses import dataclass, field
from typing import Any
import copy
import pandas as pd
import numpy as np
import pickle

from ..core import Dataset, ExperimentData, DataModule, DatasetSchema
from ..core.data_objects import DataDomainAxis, DataArray
from ..interfaces.prediction import IPredictionModel, IDeterministicModel
from ..interfaces.tuning import IResidualModel, MLPResidualModel
from ..utils import PfabLogger, Metrics, LocalData, SplitType
from ..utils.enum import BlockType
from .base_system import BaseOrchestrationSystem


@dataclass
class _ModelKDE:
    """Per-model KDE state for NatPN-light uncertainty estimation."""
    model: IPredictionModel
    latent_points: np.ndarray       # (n_configs, n_active_dims)
    weights: np.ndarray             # (n_configs,) normalized
    q_max: float
    active_mask: np.ndarray         # bool mask over latent dims
    n_active_dims: int
    weight: float = 1.0             # performance weight for aggregation


class PredictionSystem(BaseOrchestrationSystem):
    """
    Orchestrates prediction model operations with DataModule integration.
    
    - Manages prediction model registry and feature-to-model mapping
    - Coordinates training with normalization and batching via DataModule
    - Handles feature prediction with automatic denormalization
    """
    
    def __init__(self, logger: PfabLogger, schema: DatasetSchema, local_data: LocalData, res_model: IResidualModel | None = None):
        """Initialize prediction system."""
        super().__init__(logger)
        self.models: list[IPredictionModel] = []
        self.residual_model: IResidualModel = MLPResidualModel(logger) if res_model is None else res_model

        self.schema: DatasetSchema = schema
        self.local_data: LocalData = local_data
        self.datamodule: DataModule | None = None  # Stored after training

        # Domain map populated during train(): model id → derived domain code
        self._model_domain_map: dict[int, str | None] = {}

        # Per-model KDE state for NatPN-light uncertainty estimation (populated after training).
        # Maps model id(model) → _ModelKDE. Deterministic models are excluded.
        self._model_kdes: dict[int, _ModelKDE] = {}
        self._n_exp: int = 0
        self._exploration_radius: float = 0.5              # c: bubble radius at N=1

        # Performance-based weights for uncertainty aggregation.
        # Maps feature name → weight. Set via set_uncertainty_weights(); defaults to equal.
        self._uncertainty_weights: dict[str, float] = {}

    def _assert_trained(self) -> DataModule:
        """Raise if the system has not been trained yet; return the active DataModule."""
        if self.datamodule is None or not self.datamodule._is_fitted:
            raise RuntimeError("PredictionSystem not trained. Call train() first.")
        return self.datamodule

    def get_system_input_parameters(self) -> list[str]:
        """Get the parameter codes of all model inputs."""
        return self._get_unique_values('input_parameters')

    def get_system_input_features(self) -> list[str]:
        """Get the feature codes of all model inputs."""
        return self._get_unique_values('input_features')
    
    def get_system_outputs(self) -> list[str]:
        """Get the model outputs."""
        return self._get_unique_values('outputs')
    
    def _get_unique_values(self, attr: str) -> list[str]:
        codes = []
        for model in self.models:
            for code in getattr(model, attr):
                if code in codes:
                    continue
                codes.append(code)
        return codes
    
    def _filter_batches_for_model(self, batches: list[tuple[np.ndarray, np.ndarray]], model: IPredictionModel) -> list[tuple[np.ndarray, np.ndarray]]:
        """Filter batch X/y to the columns required by this model, expanding categoricals to one-hot."""
        dm = self._assert_trained()
        input_indices = dm.get_input_indices(model.input_parameters + model.input_features)
        output_indices = [dm.output_columns.index(f) for f in model.outputs]
        return [(X[:, input_indices], y[:, output_indices]) for X, y in batches]

    def train(self, datamodule: DataModule, **kwargs) -> None:
        """Train all prediction models using DataModule configuration."""
        # Store a copy to prevent mutation after training
        self.datamodule = datamodule
        
        # Check if training split is empty
        split_sizes = self.datamodule.get_split_sizes()
        if split_sizes['train'] == 0:
            raise ValueError(
                "Cannot train on empty training set. All data is in test/val splits. "
                "Reduce test_size and/or val_size in DataModule configuration."
            )
        
        # Validate dimensional coherence and derive domain codes for all registered models
        for model in self.models:
            domain_code = model.validate_dimensional_coherence(self.schema)
            self._model_domain_map[id(model)] = domain_code

        self.logger.info("Starting prediction model training...")

        # Fit normalization
        self.logger.info("Fitting normalization on training data...")
        self.datamodule.fit_normalization(SplitType.TRAIN)

        # Provide normalization context to deterministic models
        for model in self.models:
            if isinstance(model, IDeterministicModel):
                norm_state = self.datamodule.get_normalization_state()
                model.set_normalization_context(
                    parameter_stats=norm_state.get('parameter_stats', {}),
                    feature_stats=norm_state.get('feature_stats', {}),
                    categorical_mappings=norm_state.get('categorical_mappings', {}),
                )

        # Get batches
        train_batches = self.datamodule.get_batches(SplitType.TRAIN)
        val_batches = self.datamodule.get_batches(SplitType.VAL)

        # Train each registered model with inline progress
        total = len(self.models)
        trained_count = 0
        for model in self.models:
            model_train_batches = self._filter_batches_for_model(train_batches, model)
            model_val_batches = self._filter_batches_for_model(val_batches, model)

            self.logger.info(f"Training model for features {model.outputs}...")
            model.train(model_train_batches, model_val_batches, **kwargs)
            trained_count += 1

            primary_feature = model.outputs[0] if model.outputs else "unknown"
            self.logger.info(f"Trained model for '{primary_feature}'")

            # Inline progress: [====    ] 2/3
            if self.logger._console_output_enabled:
                bar_len = 12
                filled = int(bar_len * trained_count / total)
                bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
                end = "\n" if trained_count == total else "\r"
                print(f"  Training [{bar}] {trained_count}/{total}", end=end, flush=True)

        self.logger.console_success(
            f"Training complete: {trained_count}/{total} models trained"
        )

        # Fit KDE on latent representations of training configs (NatPN-light)
        self._fit_kde(datamodule)
    
    # === UNCERTAINTY ESTIMATION (NatPN-light, per-model KDE) ===

    def set_uncertainty_weights(self, weights: dict[str, float]) -> None:
        """Set performance-based weights for per-model uncertainty aggregation.

        Maps feature names (prediction model outputs) to their importance weight.
        Used to compute a weighted average of per-model uncertainties.
        If not set, all non-deterministic models contribute equally.
        """
        self._uncertainty_weights = dict(weights)

    def _get_model_weight(self, model: IPredictionModel) -> float:
        """Resolve the aggregation weight for a model from its output features."""
        if not self._uncertainty_weights:
            return 1.0
        total = 0.0
        for feat in model.outputs:
            total += self._uncertainty_weights.get(feat, 0.0)
        return total if total > 0 else 1.0

    def _fit_kde(self, datamodule: DataModule) -> None:
        """Fit one KDE per non-deterministic model on latent representations of training configs.

        Each model gets its own KDE in its own latent space. Deterministic models are excluded
        (zero epistemic uncertainty by definition). The per-model uncertainties are aggregated
        into a single scalar by ``uncertainty()`` using performance-based weights.

        One latent point per unique effective parameter configuration per model.
        Non-trajectory experiments contribute 1 point (weight = sqrt(total_rows)).
        Trajectory experiments contribute K points (weight_k = sqrt(segment_rows)).
        """
        self._model_kdes = {}
        kde_models = [m for m in self.models if not isinstance(m, IDeterministicModel)]
        if not kde_models:
            self.logger.info("All models are deterministic — uncertainty defaults to 0.0.")
            return

        # Collect per-experiment config data (shared across models).
        exp_configs: list[tuple[dict[str, Any], float]] = []  # (params, weight)
        n_exp = 0
        for code in datamodule.get_split_codes(SplitType.TRAIN):
            exp = datamodule.dataset.get_experiment(code)
            n_exp += 1
            n_rows = exp.get_num_rows()

            if not exp.parameter_updates:
                params = exp.parameters.get_values_dict().copy()
                exp_configs.append((params, float(np.sqrt(max(n_rows, 1)))))
            else:
                events = sorted(exp.parameter_updates, key=lambda e: exp._event_start_index(e))
                seg_start = 0
                for event in events:
                    seg_end = exp._event_start_index(event)
                    seg_rows = seg_end - seg_start
                    if seg_rows > 0:
                        params = exp.get_effective_parameters_for_row(seg_start)
                        exp_configs.append((params, float(np.sqrt(max(seg_rows, 1)))))
                    seg_start = seg_end
                seg_rows = n_rows - seg_start
                if seg_rows > 0:
                    params = exp.get_effective_parameters_for_row(seg_start)
                    exp_configs.append((params, float(np.sqrt(max(seg_rows, 1)))))

        self._n_exp = n_exp
        if not exp_configs:
            self.logger.info("No training configs for evidence model — uncertainty defaults to 1.0.")
            return

        # Fit one KDE per non-deterministic model.
        for model in kde_models:
            latent_points: list[np.ndarray] = []
            point_weights: list[float] = []

            for params, w in exp_configs:
                z = self._encode_params_for_model(model, params, datamodule)
                if z is not None:
                    latent_points.append(z)
                    point_weights.append(w)

            if not latent_points:
                continue

            latent_array = np.array(latent_points)
            weights_array = np.array(point_weights)
            weights_array = weights_array / weights_array.sum()

            # Drop constant dimensions.
            if latent_array.shape[0] > 1:
                per_dim_std = np.std(latent_array, axis=0)
                active_mask = per_dim_std > 1e-8
                if not np.any(active_mask):
                    continue
            else:
                active_mask = np.ones(latent_array.shape[1], dtype=bool)

            projected = latent_array[:, active_mask]
            n_active_dims = projected.shape[1]
            h = self._exploration_radius * np.sqrt(float(n_active_dims)) / np.sqrt(float(n_exp))
            q_max = self._compute_q_max(projected, weights_array, h)

            self._model_kdes[id(model)] = _ModelKDE(
                model=model,
                latent_points=projected,
                weights=weights_array,
                q_max=q_max,
                active_mask=active_mask,
                n_active_dims=n_active_dims,
                weight=self._get_model_weight(model),
            )

            self.logger.info(
                f"Evidence KDE for {model.__class__.__name__}: "
                f"{len(latent_points)} configs, {n_active_dims}/{latent_array.shape[1]} active dims, "
                f"h={h:.4f}, weight={self._model_kdes[id(model)].weight:.2f}."
            )

        if self._model_kdes:
            total_w = sum(k.weight for k in self._model_kdes.values())
            self.logger.info(
                f"Evidence model: {len(self._model_kdes)} per-model KDEs from {n_exp} experiments "
                f"(c={self._exploration_radius}, total_weight={total_w:.2f})."
            )

    @staticmethod
    def _compute_q_max(points: np.ndarray, weights: np.ndarray, h: float) -> float:
        """Max weighted KDE density over all training latent points."""
        dists_sq = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)  # (N, N)
        q_vals = np.exp(-dists_sq / (2.0 * h ** 2)) @ weights                       # (N,)
        return float(np.max(q_vals))

    def configure_exploration(self, exploration_radius: float) -> None:
        """Set the exploration radius c that governs KDE bandwidth and sharpness.

        h = c·√d / √N  (dimension-aware bubble radius, d = n_active_latent_dims)
        γ = max(1, c·√N)  (edge steepness increases as data accumulates)

        Call before train() to take effect immediately, or after train() to
        update all per-model evidence KDEs in place with the new radius.
        """
        self._exploration_radius = exploration_radius
        if self._model_kdes and self._n_exp > 0:
            for kde in self._model_kdes.values():
                d = float(kde.n_active_dims)
                h = exploration_radius * np.sqrt(d) / np.sqrt(float(self._n_exp))
                kde.q_max = self._compute_q_max(kde.latent_points, kde.weights, h)
            self.logger.info(
                f"Evidence model updated: exploration_radius={exploration_radius}, "
                f"{len(self._model_kdes)} KDEs, N={self._n_exp}."
            )

    def _encode_params_for_model(
        self, model: IPredictionModel, params: dict[str, Any], datamodule: DataModule,
    ) -> np.ndarray | None:
        """Encode a params dict to latent representation via a specific model's encode()."""
        try:
            X_norm = datamodule.params_to_array(params)
            return self._encode_from_norm_array_for_model(model, X_norm)
        except Exception:
            return None

    def _encode_from_norm_array_for_model(
        self, model: IPredictionModel, X_norm: np.ndarray,
    ) -> np.ndarray:
        """Encode a 1-D normalized parameter array via a specific model's encode()."""
        if self.datamodule is None:
            return X_norm
        input_cols = model.input_parameters + model.input_features
        input_indices = self.datamodule.get_input_indices(input_cols, skip_missing=True)
        if not input_indices:
            return X_norm
        X_model = X_norm[input_indices].reshape(1, -1)
        return model.encode(X_model)[0]

    def _compute_model_uncertainty(self, kde: _ModelKDE, X_norm: np.ndarray) -> float:
        """Compute uncertainty for a single model's KDE at a normalized parameter vector."""
        z = self._encode_from_norm_array_for_model(kde.model, X_norm.reshape(-1))
        z = z[kde.active_mask]

        d     = float(kde.n_active_dims)
        h     = self._exploration_radius * np.sqrt(d) / np.sqrt(float(self._n_exp))
        gamma = max(1.0, self._exploration_radius * np.sqrt(float(self._n_exp)))

        dists_sq = np.sum((kde.latent_points - z) ** 2, axis=1)
        q = float(np.dot(kde.weights, np.exp(-dists_sq / (2.0 * h ** 2))))

        if kde.q_max <= 0:
            return 1.0
        ratio  = float(np.clip(q / kde.q_max, 0.0, 1.0))
        n_post = float(self._n_exp) * (ratio ** gamma)
        u      = 1.0 / (1.0 + n_post)
        u_min  = 1.0 / (1.0 + float(self._n_exp))
        return float(np.clip((u - u_min) / (1.0 - u_min + 1e-12), 0.0, 1.0))

    def _compute_model_similarity(self, kde: _ModelKDE, X1: np.ndarray, X2: np.ndarray) -> float:
        """Compute kernel similarity for a single model's latent space."""
        z1 = self._encode_from_norm_array_for_model(kde.model, X1.reshape(-1))
        z2 = self._encode_from_norm_array_for_model(kde.model, X2.reshape(-1))
        z1 = z1[kde.active_mask]
        z2 = z2[kde.active_mask]
        d = float(kde.n_active_dims)
        h = self._exploration_radius * np.sqrt(d) / np.sqrt(float(self._n_exp))
        if h < 1e-10:
            return 0.0
        return float(np.exp(-float(np.sum((z1 - z2) ** 2)) / (h ** 2)))

    # --- Virtual KDE points for within-trajectory spacing ---
    #
    # During trajectory optimization, the optimizer proposes one speed per layer
    # in sequence. Without virtual points, the KDE is frozen for the entire
    # trajectory — all layers see the same uncertainty landscape, so they
    # converge to similar speeds. Virtual points inject proposed-but-not-yet-
    # executed parameter configurations into the KDE after each layer, causing
    # subsequent layers to see lower uncertainty at that speed and naturally
    # spacing out across the parameter range.
    #
    # Virtual points are temporary: they are cleared after the trajectory call
    # completes. They do not affect the training data or the prediction model.

    def add_virtual_point(self, params: dict[str, Any], datamodule: DataModule) -> None:
        """Inject a virtual point into all per-model KDEs.

        The point is encoded into each model's latent space, appended to the
        KDE with a small weight (equivalent to a single-row observation),
        and q_max is recomputed. Call clear_virtual_points() to restore the
        original KDE state.
        """
        if not self._model_kdes:
            return

        for kde in self._model_kdes.values():
            z = self._encode_params_for_model(kde.model, params, datamodule)
            if z is None:
                continue
            z_proj = z[kde.active_mask]

            # Save original state on first virtual point addition
            if not hasattr(kde, '_original_latent_points'):
                kde._original_latent_points = kde.latent_points.copy()  # type: ignore[attr-defined]
                kde._original_weights = kde.weights.copy()              # type: ignore[attr-defined]
                kde._original_q_max = kde.q_max                        # type: ignore[attr-defined]

            # Append with weight 1.0 (single observation), then renormalize
            new_weight = 1.0
            raw_weights = kde.weights * kde.weights.sum()  # undo normalization
            raw_weights = np.append(raw_weights, new_weight)
            kde.latent_points = np.vstack([kde.latent_points, z_proj.reshape(1, -1)])
            kde.weights = raw_weights / raw_weights.sum()

            # Recompute q_max with the expanded point set
            d = float(kde.n_active_dims)
            h = self._exploration_radius * np.sqrt(d) / np.sqrt(float(self._n_exp))
            kde.q_max = self._compute_q_max(kde.latent_points, kde.weights, h)

        self.logger.debug(f"Virtual KDE point added: {list(params.keys())}")

    def clear_virtual_points(self) -> None:
        """Restore all per-model KDEs to their pre-virtual-point state."""
        if not self._model_kdes:
            return

        for kde in self._model_kdes.values():
            if hasattr(kde, '_original_latent_points'):
                kde.latent_points = kde._original_latent_points  # type: ignore[attr-defined]
                kde.weights = kde._original_weights              # type: ignore[attr-defined]
                kde.q_max = kde._original_q_max                  # type: ignore[attr-defined]
                del kde._original_latent_points                  # type: ignore[attr-defined]
                del kde._original_weights                        # type: ignore[attr-defined]
                del kde._original_q_max                          # type: ignore[attr-defined]

        self.logger.debug("Virtual KDE points cleared.")

    # --- Legacy encode helpers (kept for external callers) ---

    def _encode_params(self, params: dict[str, Any], datamodule: DataModule) -> np.ndarray | None:
        """Encode a params dict to latent representation via the primary model's encode()."""
        model = self._primary_encode_model()
        if model is None:
            return None
        return self._encode_params_for_model(model, params, datamodule)

    def _encode_from_norm_array(self, X_norm: np.ndarray) -> np.ndarray:
        """Encode a 1-D normalized parameter array via the primary model's encode()."""
        model = self._primary_encode_model()
        if model is None or self.datamodule is None:
            return X_norm
        return self._encode_from_norm_array_for_model(model, X_norm)

    def _primary_encode_model(self) -> IPredictionModel | None:
        """Return the highest-weight non-deterministic model, or first model as fallback."""
        best: IPredictionModel | None = None
        best_w = -1.0
        for m in self.models:
            if isinstance(m, IDeterministicModel):
                continue
            w = self._get_model_weight(m)
            if w > best_w:
                best, best_w = m, w
        return best if best is not None else (self.models[0] if self.models else None)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode a batch of normalized parameter arrays to latent space.

        Uses the highest-weight non-deterministic model's encode(). Falls back to
        identity if no models are registered or the system is not yet trained.
        """
        model = self._primary_encode_model()
        if model is None or self.datamodule is None:
            return X
        input_cols = model.input_parameters + model.input_features
        input_indices = self.datamodule.get_input_indices(input_cols, skip_missing=True)
        if not input_indices:
            return X
        X_model = X[:, input_indices] if X.ndim > 1 else X[input_indices].reshape(1, -1)
        return model.encode(X_model)

    # --- Aggregated public API ---

    def uncertainty(self, X: np.ndarray) -> float:
        """Compute epistemic uncertainty at a normalized parameter vector.

        Aggregates per-model uncertainties as a performance-weighted average:
            u(x) = Σ(w_i · u_i(x)) / Σ(w_i)
        where u_i is the NatPN-light uncertainty from model i's KDE.

        Deterministic models are excluded (zero epistemic uncertainty).
        Returns 1.0 before the evidence model is fitted.
        """
        if not self._model_kdes:
            return 1.0

        total_w = 0.0
        weighted_u = 0.0
        for kde in self._model_kdes.values():
            u_i = self._compute_model_uncertainty(kde, X)
            weighted_u += kde.weight * u_i
            total_w += kde.weight

        u_agg = weighted_u / total_w if total_w > 0 else 1.0
        self.logger.debug(f"uncertainty (aggregated): u={u_agg:.4f}")
        return u_agg

    def kernel_similarity(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """Performance-weighted kernel similarity between two parameter vectors.

        sim(x1, x2) = Σ(w_i · sim_i(x1, x2)) / Σ(w_i)
        where sim_i is computed in model i's latent space.

        Returns 0.0 if no KDEs are fitted.
        """
        if not self._model_kdes or self._n_exp == 0:
            return 0.0

        total_w = 0.0
        weighted_sim = 0.0
        for kde in self._model_kdes.values():
            s_i = self._compute_model_similarity(kde, X1, X2)
            weighted_sim += kde.weight * s_i
            total_w += kde.weight

        return weighted_sim / total_w if total_w > 0 else 0.0

    def predict_for_calibration(self, params: dict[str, Any]) -> tuple[dict[str, np.ndarray], Any]:
        """Predict feature arrays for all dimensional positions for calibration use.

        Runs the full dimensional prediction for the given parameter configuration
        and converts each feature tensor to a tabular array suitable for evaluation
        models (rows = [dim_iter_vals..., feature_val]).

        Args:
            params: Raw (denormalized) parameter dict for the virtual experiment.

        Returns:
            Tuple of:
                - feature_arrays: Dict mapping feature code to 2-D array
                  where each row is [dim_iter_1, ..., feature_value].
                - params_block: A copy of the schema Parameters block with values
                  set from ``params``.

        Raises:
            RuntimeError: If the system has not been trained yet.
        """
        self._assert_trained()

        predictions = self._predict_from_params(params=params, batch_size=1000)

        # Convert per-feature N-D tensors to tabular arrays: [dim_iter_vals..., feature_val]
        feature_arrays: dict[str, np.ndarray] = {}
        for feat_name, tensor in predictions.items():
            feat_shape = tensor.shape
            flat = tensor.reshape(-1)
            rows = []
            for pos, feat_val in enumerate(flat):
                idx = np.unravel_index(pos, feat_shape) if feat_shape else ()
                rows.append(list(idx) + [float(feat_val)])
            feature_arrays[feat_name] = np.array(rows, dtype=np.float64)

        # Build Parameters block with values from params
        params_block = copy.deepcopy(self.schema.parameters)
        for code, val in params.items():
            if code in params_block.data_objects:
                try:
                    params_block.set_value(code, val)
                except Exception:
                    pass

        return feature_arrays, params_block

    def tune(
            self,
            exp_data: ExperimentData,
            start: int,
            end: int | None = None,
            batch_size: int | None = None,
            **kwargs
            ) -> DataModule:
        """
        Fine-tune models with new data (online learning mode).
        
        Uses existing normalization fit from training - does NOT refit normalization.
        Only normalizes the new tuning data and calls model.tuning() for adaptation.
        
        Args:
            exp_data: ExperimentData containing tuning data
            start: Start index of new data
            end: End index of new data
            **kwargs: Additional tuning parameters passed to model.tuning()
        
        Returns:
            Temporary DataModule used for tuning
        """
        dm = self._assert_trained()

        # Create a temporary Dataset with only the tuning experiment
        self.logger.info(f"Preparing tuning data from positions {start} to {end}...")
        temp_dataset = Dataset(schema=self.schema)
        temp_dataset.add_experiment(exp_data)

        # Create a temporary DataModule for tuning data.
        # Reuse fitted normalization from the main datamodule (no refit on tuning slice).
        batch_size = end - start if batch_size is None and end is not None else batch_size
        temp_datamodule = DataModule(dataset=temp_dataset, batch_size=batch_size)
        temp_datamodule.initialize(
            input_parameters=self.get_system_input_parameters(),
            input_features=self.get_system_input_features(),
            output_columns=self.get_system_outputs(),
        )
        temp_datamodule.set_split_codes(train_codes=[exp_data.code], val_codes=[], test_codes=[])

        # Copy fitted normalization state from offline training.
        temp_datamodule.set_normalization_state(dm.get_normalization_state())

        # Export full row-wise table and enforce requested online slice.
        X_df_all, y_df_all = temp_dataset.export_to_dataframe([exp_data.code])
        if X_df_all.empty or y_df_all.empty:
            raise ValueError("Tuning dataset has no feature rows available.")

        end_index = end if end is not None else len(X_df_all)
        if start < 0 or start >= len(X_df_all):
            raise ValueError(f"Tuning start index {start} out of bounds for {len(X_df_all)} rows.")
        if end_index <= start or end_index > len(X_df_all):
            raise ValueError(f"Tuning end index {end_index} invalid for start {start} and {len(X_df_all)} rows.")

        X_df: pd.DataFrame = X_df_all.iloc[start:end_index].copy() # type: ignore
        y_df: pd.DataFrame = y_df_all.iloc[start:end_index].copy() # type: ignore

        # Prepare tune arrays with training-fitted normalization.
        X_tune = temp_datamodule.prepare_input(X_df)
        y_tune = y_df[temp_datamodule.output_columns].values.astype(np.float32)
        temp_datamodule._normalize_batch(y_tune, temp_datamodule.output_columns, temp_datamodule._feature_stats)
        
        # Initialize base predictions array
        y_pred_base = np.zeros_like(y_tune)
        
        # Get predictions from all base models
        self.logger.info("Generating base predictions for residual learning...")
        for model in self.models:
            # Get indices for this model's outputs
            output_indices = [dm.output_columns.index(f) for f in model.outputs]

            # Get model inputs (filter X columns)
            input_indices = dm.get_input_indices(model.input_parameters + model.input_features)
            X_model = X_tune[:, input_indices]
            
            # Predict (normalized)
            y_pred_model = model.forward_pass(X_model)
            
            # Place in aggregate array
            y_pred_base[:, output_indices] = y_pred_model

        # TODO: Store predicted features in exp_data.predicted_features. do we need another array?
            
        # Calculate residuals (Target - Base Prediction)
        residuals = y_tune - y_pred_base
        
        # Prepare inputs for residual model: [X, BasePredictions]
        # This allows the residual model to learn state-dependent errors (e.g. "high prediction -> high error")
        X_residual_input = np.hstack([X_tune, y_pred_base])
        
        # Train residual model
        self.logger.info(f"Training residual model on {len(X_tune)} samples...")
        self.residual_model.fit(X_residual_input, residuals)
        self.logger.info("Residual model updated")

        return temp_datamodule
    
    def validate(
        self,
        use_test: bool = False,
        performance_weights: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Validate prediction models on validation or test set.

        Returns per-feature metrics: {feature_name: {'r2': float, 'r2_adj': float, ...}}.
        When performance_weights are provided, R²_adj is computed using combined
        performance as the importance signal.
        """
        dm = self._assert_trained()

        split = SplitType.TEST if use_test else SplitType.VAL

        # Check if split is empty before trying to extract
        split_sizes = dm.get_split_sizes()
        if split_sizes[split] == 0:
            raise ValueError(
                f"Cannot validate on {split} set: split is empty. "
                f"Configure DataModule with {'test_size' if use_test else 'val_size'} > 0.0"
            )

        self.logger.console_info(f"Validating models on {split} set...")

        # Extract validation/test data
        batches = dm.get_batches(split)
        if not batches:
            self.logger.console_warning(f"No batches returned for {split} set during validation.")
            return {}

        # Concatenate batches
        X_list, y_list = zip(*batches)
        X_split = np.concatenate(X_list, axis=0)
        y_split = np.concatenate(y_list, axis=0)

        self.logger.info(f"Evaluating {len(self.models)} models on {len(X_split)} samples...")

        # Build per-row importance from experiment performance (for R²_adj)
        importance_arr: np.ndarray | None = None
        if performance_weights:
            total_w = sum(performance_weights.values())
            if total_w > 0:
                codes = dm.get_split_codes(split)
                importance_rows: list[float] = []
                for code in codes:
                    exp = dm.dataset.get_experiment(code)
                    perf = exp.performance.get_values_dict()
                    combined = sum(
                        performance_weights.get(k, 0.0) * float(v)
                        for k, v in perf.items() if v is not None
                    ) / total_w
                    # Count rows this experiment contributes
                    dim_names = exp.parameters.get_dim_names()
                    if not dim_names:
                        n_rows = 1
                    else:
                        n_rows = len(exp.parameters.get_dim_combinations(dim_names))
                    importance_rows.extend([combined] * n_rows)
                importance_arr = np.array(importance_rows, dtype=np.float64)

        # Compute per-feature metrics
        results: dict[str, dict[str, float]] = {}
        for model in self.models:
            # Get indices for this model
            input_indices = dm.get_input_indices(model.input_parameters + model.input_features)
            indices = [dm.output_columns.index(f) for f in model.outputs]

            # Get ground truth (denormalized)
            y_true_norm = y_split[:, indices]
            y_true = dm.denormalize_values(y_true_norm, model.outputs)

            # Predict from the model-specific input slice.
            y_pred_norm = model.forward_pass(X_split[:, input_indices])
            y_pred = dm.denormalize_values(y_pred_norm, model.outputs)

            self.logger.console_info(f"  Model ({', '.join(model.outputs)}):")
            self.logger.console_info(f"    {'Feature':<20} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10} | {'R²_adj':<10}")
            self.logger.console_info(f"    {'-' * 70}")

            # Per-feature metrics
            for i, feature_name in enumerate(model.outputs):
                y_true_feat = y_true[:, i]
                y_pred_feat = y_pred[:, i]

                feat_metrics = Metrics.calculate_regression_metrics(y_true_feat, y_pred_feat)

                # Add R²_adj when importance is available
                if importance_arr is not None and len(importance_arr) == len(y_true_feat):
                    adj = Metrics.calculate_adjusted_r2(
                        y_true_feat, y_pred_feat, importance_arr
                    )
                    feat_metrics['r2_adj'] = adj['r2_adj']

                results[feature_name] = feat_metrics

                r2_adj_str = f"{feat_metrics['r2_adj']:<10.4f}" if 'r2_adj' in feat_metrics else f"{'n/a':<10}"
                self.logger.console_info(
                    f"    {feature_name:<20} | "
                    f"{feat_metrics['mae']:<10.4f} | "
                    f"{feat_metrics['rmse']:<10.4f} | "
                    f"{feat_metrics['r2']:<10.4f} | "
                    f"{r2_adj_str}"
                )

        self.logger.console_success(
            f"Validation complete on {split} set ({len(X_split)} samples)"
        )

        return results

    def predict_experiment(
        self,
        exp_data: ExperimentData,
        predict_from: int = 0,
        predict_to: int | None = None,
        batch_size: int = 1000,
        overlap: int = 0
    ) -> dict[str, np.ndarray]:
        """
        Predict dimensional features and populate exp_data.predicted_metric_arrays.
        
        Args:
            exp_data: ExperimentData with parameters set (predicted_metric_arrays populated)
            predict_from: Start position index for prediction (default: 0)
            predict_to: End position index for prediction (default: None = predict all)
            batch_size: Number of dimensional positions per batch (default: 1000)
            overlap: Number of positions to overlap between consecutive batches (default: 0).
                     Useful for context-aware models (e.g., transformers) that need continuity
                     across batch boundaries. Must be < batch_size.
        """
        # Extract parameters from exp_data
        params = exp_data.parameters.get_values_dict()
        
        # Get predictions from core logic
        predictions = self._predict_from_params(
            params=params,
            predict_from=predict_from,
            predict_to=predict_to,
            batch_size=batch_size,
            overlap=overlap
        )
        
        # Store predictions in exp_data.predicted_metric_arrays
        # self._store_predictions_in_exp_data(exp_data, predictions)
        return predictions
    
    def _get_feature_shape(self, feat_code: str, params: dict[str, Any]) -> tuple[int, ...]:
        """Get the output tensor shape for a feature given current dimensional parameters."""
        feat_obj = self.schema.features.data_objects.get(feat_code)
        # data_objects stores DataArray instances; Pyright sees DataObject which lacks .columns.
        if feat_obj is None or not hasattr(feat_obj, "columns") or not feat_obj.columns:  # type: ignore[union-attr]
            return ()
        iterator_cols = feat_obj.columns[:-1]  # type: ignore[union-attr]
        if not iterator_cols:
            return ()
        iter_to_dim_code = {
            dim.iterator_code: dim.code
            for dim in self.schema.parameters._get_domain_axis_objects()
        }
        shape = []
        for ic in iterator_cols:
            size_code = iter_to_dim_code.get(ic)
            if size_code is None or size_code not in params:
                raise ValueError(
                    f"Cannot resolve size for iterator '{ic}' — "
                    f"check that '{size_code}' is present in params."
                )
            shape.append(int(params[size_code]))
        return tuple(shape)

    def _get_model_dim_info(self, model: IPredictionModel, params: dict[str, Any]) -> dict[str, Any]:
        """Build dimensional iteration structure for a specific model based on its domain and depth."""
        depth = model.depth
        if id(model) not in self._model_domain_map:
            raise RuntimeError(
                f"Model {model.__class__.__name__} has not been trained yet. "
                f"Call train() before predicting."
            )
        domain_code = self._model_domain_map[id(model)]

        dim_sizes: list[int] = []
        dim_iterators: list[str] = []
        dim_codes_ordered: list[str] = []
        dim_codes: set = set()

        if domain_code is not None and depth > 0:
            if not self.schema.domains.has(domain_code):
                raise ValueError(
                    f"Domain '{domain_code}' declared by model {model.__class__.__name__} "
                    f"is not registered in schema."
                )
            domain = self.schema.domains.get(domain_code)
            model_axes = domain.axes[:depth]

            for ax in model_axes:
                if ax.code not in params:
                    raise ValueError(
                        f"Missing domain axis parameter '{ax.code}' for model "
                        f"{model.__class__.__name__}"
                    )
                dim_sizes.append(int(params[ax.code]))
                dim_iterators.append(ax.iterator_code)
                dim_codes_ordered.append(ax.code)
                dim_codes.add(ax.code)

        shape = tuple(dim_sizes)
        total_positions = int(np.prod(shape)) if shape else 1
        param_base = {k: v for k, v in params.items() if k not in dim_codes}

        return {
            'shape': shape,
            'dim_iterators': dim_iterators,
            'dim_codes_ordered': dim_codes_ordered,
            'param_base': param_base,
            'total_positions': total_positions,
        }

    def _predict_from_params(
        self,
        params: dict[str, Any],
        predict_from: int = 0,
        predict_to: int | None = None,
        batch_size: int = 1000,
        overlap: int = 0
    ) -> dict[str, np.ndarray]:
        """Core prediction logic: per-model iteration with per-feature tensor shapes."""
        self._assert_trained()

        predictions: dict[str, np.ndarray] = {}

        for model in self.models:
            model_dim_info = self._get_model_dim_info(model, params)
            total_positions = model_dim_info['total_positions']

            p_from = max(0, predict_from)
            p_to = min(predict_to if predict_to is not None else total_positions, total_positions)
            if p_from > total_positions:
                continue

            # Initialize output arrays with per-feature shapes
            for feat in model.outputs:
                if feat not in predictions:
                    feat_shape = self._get_feature_shape(feat, params)
                    predictions[feat] = np.full(feat_shape, np.nan)

            self._execute_batched_predictions_to_dict(
                predictions=predictions,
                dim_info=model_dim_info,
                predict_from=p_from,
                predict_to=p_to,
                batch_size=batch_size,
                overlap=overlap,
                model=model,
            )

        self.logger.info(f"✓ Predicted features for {len(self.models)} model(s)")
        return predictions
    
    # def _store_predictions_in_exp_data(
    #     self, 
    #     exp_data: ExperimentData, 
    #     predictions: dict[str, np.ndarray]
    # ) -> None:
    #     """Store prediction arrays in exp_data.predicted_metric_arrays."""
    #     for feature_name, pred_array in predictions.items():
    #         if feature_name not in exp_data.predicted_features.keys():
    #             arr = DataArray(code=feature_name, role=BlockType.FEATURE)
    #             exp_data.predicted_features.add(feature_name, arr)
    #         exp_data.predicted_features.set_value(feature_name, pred_array)
    
    def _execute_batched_predictions_to_dict(
        self,
        predictions: dict[str, np.ndarray],
        dim_info: dict[str, Any],
        predict_from: int,
        predict_to: int,
        batch_size: int,
        overlap: int = 0,
        model: IPredictionModel | None = None,
    ) -> None:
        """Process positions in batches: build X, predict, denormalize, store in prediction dict. Supports overlap."""
        self.logger.info(f"Predicting positions {predict_from} to {predict_to} in batches of {batch_size} (overlap={overlap})...")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if overlap >= batch_size and predict_to - predict_from > batch_size:
            raise ValueError("overlap must be less than batch_size for multiple batches")

        batch_starts = list(range(predict_from, predict_to, batch_size))
        for i, orig_batch_start in enumerate(batch_starts):
            if i == 0:
                batch_start = orig_batch_start
                batch_end = min(batch_start + batch_size, predict_to)
            else:
                batch_start = max(orig_batch_start - overlap, predict_from)
                batch_end = min(orig_batch_start + batch_size, predict_to)

            # Build feature matrix for this batch
            X_batch, batch_indices = self._build_batch_features(
                batch_start=batch_start,
                batch_end=batch_end,
                dim_info=dim_info
            )

            # Run predictions for the specified model (or all models if None)
            self._predict_and_store_batch_to_dict(
                predictions=predictions,
                X_batch=X_batch,
                batch_indices=batch_indices,
                model=model,
            )
    
    def _build_batch_features(
        self,
        batch_start: int,
        batch_end: int,
        dim_info: dict[str, Any]
    ) -> tuple[pd.DataFrame, list[tuple[int, ...]]]:
        """Build feature matrix X with params + dimensional indices for batch positions.

        Uses dimension size-parameter codes (e.g. "dim_1") as column keys to match
        the column structure produced by Dataset.export_to_dataframe() during training.
        """
        shape = dim_info['shape']
        param_base = dim_info['param_base']
        dim_codes_ordered = dim_info['dim_codes_ordered']

        X_batch_rows = []
        batch_indices = []

        for pos in range(batch_start, batch_end):
            # Convert linear position to multi-dimensional index
            idx = np.unravel_index(pos, shape)
            batch_indices.append(idx)

            # Create feature row: non-dimensional params + dimensional indices.
            # Keys are size-parameter codes ("dim_1", "dim_2") to match training columns,
            # while values are the iterator indices (0, 1, ...) matching export_to_dataframe.
            row = param_base.copy()
            for i, dim_code in enumerate(dim_codes_ordered):
                row[dim_code] = idx[i]
            
            X_batch_rows.append(row)
        
        X_batch = pd.DataFrame(X_batch_rows)
        return X_batch, batch_indices
    
    def _predict_and_store_batch_to_dict(
        self,
        predictions: dict[str, np.ndarray],
        X_batch: pd.DataFrame,
        batch_indices: list[tuple[int, ...]],
        model: IPredictionModel | None = None,
    ) -> None:
        """Run model prediction on X_batch and store results with per-feature index truncation."""
        dm = self._assert_trained()

        # Prepare input (one-hot + normalize)
        X_norm = dm.prepare_input(X_batch)

        models_to_run = [model] if model is not None else self.models
        for m in models_to_run:
            # Filter to the columns this model was trained on (same as _filter_batches_for_model).
            input_indices = dm.get_input_indices(m.input_parameters + m.input_features)
            X_model = X_norm[:, input_indices]
            # Predict in normalized space
            y_pred_norm = m.forward_pass(X_model)

            # Denormalize to original scale
            y_pred = dm.denormalize_values(y_pred_norm, m.outputs)

            # Store predictions in arrays with per-feature index truncation
            for i, feature_name in enumerate(m.outputs):
                if feature_name not in predictions:
                    continue

                # y_pred is (batch, n_outputs)
                values = y_pred[:, i]
                feat_depth = len(predictions[feature_name].shape)

                for j, idx in enumerate(batch_indices):
                    feat_idx = idx[:feat_depth]  # truncate to this feature's depth
                    predictions[feature_name][feat_idx] = float(values[j])
    
    # === EXPORT FOR PRODUCTION INFERENCE ===
    
    def export_inference_bundle(
        self, 
        filepath: str, 
        include_evaluation: bool = False
    ) -> str:
        """Export validated inference bundle with models, normalization, and schema."""
        self._assert_trained()

        self.logger.console_info("Validating models for export...")
        
        # Validate prediction models
        for model in self.models:
            self._validate_model_export(model)
        
        self.logger.console_info("✓ All models validated successfully")
        
        # Build bundle dict
        bundle = self._create_bundle_dict(include_evaluation)
        
        # Save bundle
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)
        
        self.logger.console_success(f"✓ Exported inference bundle to: {filepath}")
        return filepath
    
    def _validate_model_export(self, model: IPredictionModel) -> None:
        """Round-trip test: export artifacts and restore to verify export support."""
        try:
            # Get artifacts
            artifacts = model._get_model_artifacts()
            
            if not isinstance(artifacts, dict):
                raise TypeError(
                    f"_get_model_artifacts() must return dict, got {type(artifacts).__name__}"
                )
            
            # Verify artifacts are picklable
            try:
                pickle.dumps(artifacts)
            except Exception as e:
                raise RuntimeError(
                    f"Artifacts not picklable: {e}. Ensure all objects in artifacts dict are picklable."
                ) from e
            
            # Create fresh instance and restore
            ModelClass = model.__class__
            fresh_model = ModelClass(logger=None)  # type: ignore[arg-type]
            fresh_model._set_model_artifacts(artifacts)
            
            # Verify feature_names match
            if fresh_model.outputs != model.outputs:
                raise ValueError(
                    f"Round-trip validation failed: feature_names mismatch. "
                    f"Original: {model.outputs}, Restored: {fresh_model.outputs}"
                )
            
            self.logger.info(f"✓ Validated {ModelClass.__name__}")
            
        except NotImplementedError as e:
            raise RuntimeError(
                f"{model.__class__.__name__} does not implement "
                f"_get_model_artifacts()/_set_model_artifacts(). "
                f"These methods are required for export."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Export validation failed for {model.__class__.__name__}: {e}"
            ) from e
    
    def _create_bundle_dict(self, include_evaluation: bool) -> dict[str, Any]:
        """Assemble bundle with models, artifacts, normalization, and schema."""
        dm = self._assert_trained()

        bundle: dict[str, Any] = {
            'prediction_models': [
                {
                    'class_path': f"{model.__class__.__module__}.{model.__class__.__name__}",
                    'feature_names': model.outputs,
                    'artifacts': model._get_model_artifacts()
                }
                for model in self.models
            ],
            'normalization': dm.get_normalization_state(),
            'schema': self.schema.to_dict()
        }
        
        # InferenceBundle can work with predictions only (evaluation happens externally)
        if include_evaluation:
            self.logger.warning(
                "Evaluation model export requested but not yet implemented. "
                "Bundle will only contain prediction models."
            )
        
        return bundle
