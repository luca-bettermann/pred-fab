"""
Prediction System for managing prediction models.

Coordinates prediction model training and inference within the AIXD architecture.
Prediction models predict features, which can then be evaluated for performance.
Integrates with DataModule for normalization and batching.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import copy
import numpy as np
import torch
import pickle

from ..core import Dataset, ExperimentData, DataModule, DatasetSchema
from ..core.data_objects import DataDomainAxis, DataArray
from ..interfaces.prediction import IPredictionModel, DeterministicModel
from ..interfaces.tuning import IResidualModel, MLPResidualModel
from ..utils import PfabLogger, ProgressBar, Metrics, LocalData, SplitType, combined_score, profiler
from ..utils.enum import BlockType
from .base_system import BaseOrchestrationSystem
from .evidence import (
    EstimatorConfig,
    EvidenceEstimator,
    KernelIndex,
    make_estimator,
)


SIGMA_MIN: float = 0.03
"""Lower bound for the Gaussian length scale. Below ~0.02 kernels become sub-cell
under any finite probe budget and the evidence model becomes over-confident about
any single point."""

RADIUS_DEFAULT: float = 0.05
"""Default kernel radius in normalised [0,1] space. σ = radius × √D."""


@dataclass
class _ModelKDE:
    """Per-model evidence state. Each latent point carries a Gaussian mass of w_j in ℝ^D."""
    model: IPredictionModel
    latent_points: np.ndarray       # (n_configs, n_active_dims)
    point_weights: np.ndarray       # (n_configs,) per-datapoint mass; stacking sets >1
    sigma: float                    # Gaussian length scale
    active_mask: np.ndarray         # bool mask over latent dims
    n_active_dims: int
    weight: float = 1.0             # performance weight for model-level aggregation


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
        # target device for tensor conversions inside
        # KDE / gradient acquisition. Set by agent.to(device); defaults to CPU.
        # KDE storage (latent_points, point_weights) remains numpy on CPU; the
        # torch estimators convert to this device at call boundaries.
        self._device: torch.device = torch.device("cpu")

        self.schema: DatasetSchema = schema
        self.local_data: LocalData = local_data
        self.datamodule: DataModule | None = None  # Stored after training

        # Domain map populated during train(): model id → derived domain code
        self._model_domain_map: dict[int, str | None] = {}

        # Per-model evidence state. Gaussian density kernel; integrated-objective acquisition.
        self._model_kdes: dict[int, _ModelKDE] = {}
        self._n_exp: int = 0
        self._radius: float = RADIUS_DEFAULT

        # When True, evidence/KDE bypasses model.encode() and operates in raw
        # normalised input space. Auto-toggled by agent.discovery_step() so the
        # initial random encoder doesn't taint discovery placement. Not user-facing.
        self._bypass_encoder: bool = False
        self._estimator_config: EstimatorConfig = EstimatorConfig()
        self._estimator: EvidenceEstimator = make_estimator(self._estimator_config)

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
    
    def _filter_batches_for_model(
        self,
        batches: list[tuple[torch.Tensor, torch.Tensor]] | list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        model: IPredictionModel,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Filter batch X/y to the model's declared input/output columns.

        accepts either 2-tuples ``(X, y)`` or
        3-tuples ``(X, y, cell_meta)``; returns 2-tuples for ``model.train``
        which doesn't need cell_meta. Cell metadata is consumed by
        PredictionSystem-level orchestrators (e.g. SS substitution).
        """
        dm = self._assert_trained()
        input_indices = dm.get_input_indices(model.input_parameters + model.input_features)
        output_indices = [dm.output_columns.index(f) for f in model.outputs]
        result: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch in batches:
            X, y = batch[0], batch[1]
            result.append((X[:, input_indices], y[:, output_indices]))
        return result

    def _build_model_dependency_graph(self) -> dict[int, set[int]]:
        """For each model, the set of OTHER models it depends on (consumes a
        recursive feature whose source is their output). Self-recursion is
        excluded; cross-model dependencies must form a DAG.
        """
        output_to_model: dict[str, IPredictionModel] = {}
        for m in self.models:
            for out in m.outputs:
                output_to_model[out] = m

        deps: dict[int, set[int]] = {id(m): set() for m in self.models}
        for m in self.models:
            for feat_code in m.input_features:
                producer = output_to_model.get(feat_code)
                if producer is not None and producer is not m:
                    deps[id(m)].add(id(producer))
        return deps

    def _topo_sort_models(self) -> list[IPredictionModel]:
        """Order models so any model's dependencies appear earlier in the list.
        Raises if the dependency graph contains a cycle.
        """
        deps = self._build_model_dependency_graph()
        in_degree = {id(m): len(deps[id(m)]) for m in self.models}
        sorted_list: list[IPredictionModel] = []
        queue = [m for m in self.models if in_degree[id(m)] == 0]
        while queue:
            m = queue.pop(0)
            sorted_list.append(m)
            for other in self.models:
                if id(m) in deps[id(other)]:
                    in_degree[id(other)] -= 1
                    if in_degree[id(other)] == 0:
                        queue.append(other)
        if len(sorted_list) != len(self.models):
            raise ValueError(
                "Cross-model recursive feature cycle detected — models cannot "
                "have mutually-dependent recursive inputs (no DAG topology)."
            )
        return sorted_list

    def _fit_single_round(
        self,
        model: IPredictionModel,
        train_batches,
        val_batches,
        **kwargs,
    ) -> None:
        """One fit call. Wrapped with progress bar by caller.

        Dispatches batch shape: TransformerModel subclasses get sequence-
        shaped (B, L, n_input) batches built from training experiments;
        flat-batched models get the standard (N, n_input) batches via
        column filtering.
        """
        from ..models.transformer import TransformerModel
        model.set_categorical_context(self._compute_model_cat_cardinalities(model))

        if isinstance(model, TransformerModel):
            model_train_batches, seq_axis_sizes, domain_axis_sizes = self._build_transformer_train_batches(
                model, SplitType.TRAIN,
            )
            model_val_batches, _, _ = self._build_transformer_train_batches(model, SplitType.VAL)
            if seq_axis_sizes:
                kwargs["seq_axis_sizes"] = seq_axis_sizes
            if domain_axis_sizes:
                kwargs["domain_axis_sizes"] = domain_axis_sizes
        else:
            model_train_batches = self._filter_batches_for_model(train_batches, model)
            model_val_batches = self._filter_batches_for_model(val_batches, model)

        self.logger.info(f"Training model for features {model.outputs}...")
        model.train(model_train_batches, model_val_batches, **kwargs)
        primary = model.outputs[0] if model.outputs else "unknown"
        self.logger.info(f"Trained model for '{primary}'")

    def _build_transformer_train_batches(
        self,
        model: IPredictionModel,
        split: SplitType,
    ) -> tuple[
        list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
        tuple[int, ...],
        tuple[int, ...],
    ]:
        """Build per-experiment ``(X_seq, y_dict)`` batches at encoder granularity.

        For each experiment in the split:
          1. Resolve dim_info (``shape``, ``dim_codes_ordered``, ``iterator_feats``).
          2. Use ``dm.build_sequence_batch`` to lay out X as ``(n_other, L, n_model_input)``
             then collapse to ``(1, L, n_model_input)`` (rows are equivalent under
             input_depth ≤ axis_depth).
          3. Pull ground-truth y for each output feature, reshape to its native
             encoder-aligned shape ``(1, L, *extra_axis_sizes_d)``.

        Returns ``(batches, seq_axis_sizes, domain_axis_sizes)`` —
        ``seq_axis_sizes`` is the per-sequence-axis size tuple shared across
        experiments; ``domain_axis_sizes`` is the schema-derived upper bound
        per domain axis (used by decoder positional embeddings).
        """
        dm = self._assert_trained()
        codes = dm.get_split_codes(split)
        if not codes:
            return [], (), ()

        input_indices = dm.get_input_indices(model.input_parameters + model.input_features)
        input_indices_t = torch.as_tensor(input_indices, dtype=torch.long)

        seq_codes: tuple[str, ...] = model.sequence_axis_code  # type: ignore[attr-defined]
        n_axes = len(seq_codes)
        first_seq_axis_sizes: tuple[int, ...] | None = None

        # Schema-derived domain axis sizes (max_val per axis) — same for all experiments.
        domain_axis_sizes: tuple[int, ...] = ()
        domain_code = self._model_domain_map.get(id(model))
        if domain_code is not None and dm.dataset.schema.domains.has(domain_code):
            domain = dm.dataset.schema.domains.get(domain_code)
            domain_axis_sizes = tuple(int(ax.max_val) for ax in domain.axes)

        # Per-feature depth lookup (feat → depth).
        feat_depths: dict[str, int] = {}
        for feat in model.outputs:
            feat_obj = model._ref_features.get(feat)  # type: ignore[attr-defined]
            cols = feat_obj.columns if (feat_obj is not None and hasattr(feat_obj, "columns")) else []  # type: ignore[union-attr]
            feat_depths[feat] = (len(cols) - 1) if cols else 0

        # Collect per-experiment batches with their natural sequence length.
        raw_batches: list[tuple[torch.Tensor, dict[str, torch.Tensor], int]] = []
        max_L = 0

        for code in codes:
            exp = dm.dataset.get_experiment(code)
            params = exp.get_effective_parameters_for_row(0)
            dim_info = self._get_model_dim_info(model, params)
            dim_codes = dim_info['dim_codes_ordered']
            shape = dim_info['shape']
            if not shape or any(c not in dim_codes for c in seq_codes):
                continue

            seq_axis_indices = [dim_codes.index(c) for c in seq_codes]
            seq_axis_idx_set = set(seq_axis_indices)
            seq_axis_sizes = [shape[i] for i in seq_axis_indices]
            seq_axis_sizes_t = tuple(int(s) for s in seq_axis_sizes)
            if first_seq_axis_sizes is None:
                first_seq_axis_sizes = seq_axis_sizes_t
            else:
                # Track max per axis for padding
                first_seq_axis_sizes = tuple(
                    max(a, b) for a, b in zip(first_seq_axis_sizes, seq_axis_sizes_t)
                )
            L = int(np.prod(seq_axis_sizes)) if seq_axis_sizes else 1
            max_L = max(max_L, L)
            other_axis_indices = [i for i in range(len(shape)) if i not in seq_axis_idx_set]
            other_sizes = [shape[i] for i in other_axis_indices]
            n_other = int(np.prod(other_sizes)) if other_sizes else 1

            # Encoder X: (n_other, L, n_input_full) → (1, L, n_input_full) → (1, L, n_model_input).
            X_seq_full = dm.build_sequence_batch(model, [params], [dim_info])
            X_seq = X_seq_full[:1].index_select(2, input_indices_t)

            exported = dm.dataset.export_to_tensor_dict(
                [code],
                x_columns=dm.input_columns,
                y_columns=dm.output_columns,
                categorical_mappings=dm.categorical_mappings,
            )
            if exported.is_empty():
                continue
            n_rows = exported.n_rows

            # Per-feature y at encoder-aligned native shape (1, L, *extra_axis_sizes_d).
            y_dict: dict[str, torch.Tensor] = {}
            for feat in model.outputs:
                col_t = exported.y.get(feat, torch.zeros(n_rows, dtype=torch.float32)).to(dtype=torch.float32)
                stats = dm._feature_stats.get(feat)
                if stats is not None:
                    col_t = dm._apply_normalization_tensor(col_t, stats)
                feat_depth = feat_depths[feat]
                extra_axis_sizes_d = tuple(shape[axis_idx] for axis_idx in range(n_axes, feat_depth))
                y_feat = torch.full((1, L) + extra_axis_sizes_d, float("nan"), dtype=torch.float32)
                for cell_flat in range(n_rows):
                    coord = np.unravel_index(cell_flat, shape)
                    seq_coord = tuple(int(coord[i]) for i in seq_axis_indices)
                    seq_pos = int(np.ravel_multi_index(seq_coord, seq_axis_sizes)) if seq_axis_sizes else 0
                    if extra_axis_sizes_d:
                        extra_coord = tuple(int(coord[axis_idx]) for axis_idx in range(n_axes, feat_depth))
                        y_feat[(0, seq_pos) + extra_coord] = col_t[cell_flat]
                    else:
                        y_feat[0, seq_pos] = col_t[cell_flat]
                y_dict[feat] = y_feat

            raw_batches.append((X_seq, y_dict, L))

        # Pad all batches to max_L. Padded X positions are zero; padded y
        # positions are NaN (ignored in loss via nanmean).
        out: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        for X_seq, y_dict, L in raw_batches:
            if L < max_L:
                pad_L = max_L - L
                X_pad = torch.zeros((1, pad_L, X_seq.shape[2]), dtype=X_seq.dtype)
                X_seq = torch.cat([X_seq, X_pad], dim=1)
                for feat in y_dict:
                    y_t = y_dict[feat]
                    pad_shape = (1, pad_L) + y_t.shape[2:]
                    y_pad = torch.full(pad_shape, float("nan"), dtype=y_t.dtype)
                    y_dict[feat] = torch.cat([y_t, y_pad], dim=1)
            out.append((X_seq, y_dict))

        return out, (first_seq_axis_sizes or ()), domain_axis_sizes

    def _compute_model_cat_cardinalities(self, model: IPredictionModel) -> dict[int, int]:
        """Translate DataModule cat cardinalities into model-relative col indices.

        For each model input column j (after ``_filter_batches_for_model``
        column selection), checks whether the corresponding DataModule
        column is categorical and, if so, records its cardinality.
        """
        dm = self._assert_trained()
        if not dm.cat_cardinalities:
            return {}
        input_cols = model.input_parameters + model.input_features
        input_indices = dm.get_input_indices(input_cols, skip_missing=True)
        result: dict[int, int] = {}
        for model_idx, dm_idx in enumerate(input_indices):
            if dm_idx in dm.cat_cardinalities:
                result[model_idx] = dm.cat_cardinalities[dm_idx]
        return result

    def train(self, datamodule: DataModule, **kwargs) -> None:
        """Train all prediction models using DataModule configuration.

        Models train in topological order so any model whose ``input_features``
        reference another model's outputs trains after that producer. Each
        model's ``train()`` runs to completion before the next; cross-model
        outputs are surfaced at prediction time via ``predictions_so_far`` in
        ``model.predict``.

        Pass ``epoch_logger=WandbLogger(...)`` to log per-epoch loss and
        post-training validation metrics to Weights & Biases.
        """
        self.datamodule = datamodule

        split_sizes = self.datamodule.get_split_sizes()
        if split_sizes['train'] == 0:
            raise ValueError(
                "Cannot train on empty training set. All data is in test/val splits. "
                "Reduce test_size and/or val_size in DataModule configuration."
            )

        for model in self.models:
            # Two-step validation: universal domain/depth coherence (final on
            # the base) followed by type-specific schema rules (overridden per
            # model class — MLP/Deterministic reject recursive features;
            # Transformer requires sequence_axis_code).
            domain_code = model.validate_dimensional_coherence(self.schema)
            model._validate_schema_compatibility(self.schema)
            self._model_domain_map[id(model)] = domain_code

        self.logger.info("Starting prediction model training...")
        self.logger.info("Fitting normalization on training data...")
        self.datamodule.fit_normalization(SplitType.TRAIN)

        for model in self.models:
            if isinstance(model, DeterministicModel):
                norm_state = self.datamodule.get_normalization_state()
                model.set_normalization_context(
                    parameter_stats=norm_state.get('parameter_stats', {}),
                    feature_stats=norm_state.get('feature_stats', {}),
                    categorical_mappings=norm_state.get('categorical_mappings', {}),
                )

        ordered_models = self._topo_sort_models()
        total = len(ordered_models)
        trained_count = 0

        for model in ordered_models:
            train_batches = self.datamodule.get_batches(SplitType.TRAIN)
            val_batches = self.datamodule.get_batches(SplitType.VAL)
            self._fit_single_round(model, train_batches, val_batches, **kwargs)
            trained_count += 1

        self.logger.info(f"Training complete: {trained_count}/{total} models trained")
        self._fit_kde(datamodule)

    # === EVIDENCE MODEL (integrated objective) ===
    #
    # ρ_j(z)  = w_j · N(z; z_j, σ²I)      normalized Gaussian density, mass w_j in ℝ^D
    # D(z)    = Σ_j ρ_j(z)                 raw evidence density, unbounded
    # E(z)    = D / (1 + D)                actual evidence, [0, 1)
    # Δ∫E     = ∫_[0,1]^D [E_new − E_old]  info gain from adding a batch
    #
    # Acquisition lives in CalibrationSystem; this module provides the math.
    # No boundary term: leakage is handled by the integration bounds [0,1]^D.

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
        """Fit one evidence model per non-deterministic model on latent representations of training configs.

        Each data point (segment) contributes 1 unit of evidence — no per-experiment
        normalization. A 7-layer experiment contributes 7 evidence units. The evidence
        sum grows with data; uncertainty u = 1/(1+E) shrinks accordingly.
        """
        self._model_kdes = {}
        kde_models = [m for m in self.models if not isinstance(m, DeterministicModel)]
        if not kde_models:
            self.logger.info("All models are deterministic — uncertainty defaults to 0.0.")
            return

        # Collect per-segment config data: each segment = 1 evidence unit.
        exp_configs: list[dict[str, Any]] = []
        n_exp = 0
        for code in datamodule.get_split_codes(SplitType.TRAIN):
            exp = datamodule.dataset.get_experiment(code)
            n_exp += 1

            if not exp.parameter_updates:
                params = exp.parameters.get_values_dict().copy()
                exp_configs.append(params)
            else:
                events = sorted(exp.parameter_updates, key=lambda e: exp._event_start_index(e))
                seg_start = 0
                for event in events:
                    seg_end = exp._event_start_index(event)
                    if seg_end > seg_start:
                        exp_configs.append(exp.get_effective_parameters_for_row(seg_start))
                    seg_start = seg_end
                n_rows = exp.get_num_rows()
                if n_rows > seg_start:
                    exp_configs.append(exp.get_effective_parameters_for_row(seg_start))

        self._n_exp = n_exp
        if not exp_configs:
            self.logger.info("No training configs for evidence model — uncertainty defaults to 1.0.")
            return

        n_params = len(datamodule.dataset.schema.parameters.data_objects)
        if n_params > n_exp:
            self.logger.console_warning(
                f"Evidence model fitted with {n_exp} experiments but {n_params} parameters — "
                f"uncertainty estimates may be unreliable."
            )

        # Fit one evidence model per non-deterministic model.
        for model in kde_models:
            latent_points: list[np.ndarray] = []

            for params in exp_configs:
                z = self._encode_params_for_model(model, params, datamodule)
                if z is not None:
                    latent_points.append(z)

            if not latent_points:
                continue

            latent_array = np.array(latent_points)

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
            sigma = self._resolve_sigma(n_active_dims)

            self._model_kdes[id(model)] = _ModelKDE(
                model=model,
                latent_points=projected,
                point_weights=np.ones(len(projected)),
                sigma=sigma,
                active_mask=active_mask,
                n_active_dims=n_active_dims,
                weight=self._get_model_weight(model),
            )

            self.logger.info(
                f"Evidence model for {model.__class__.__name__}: "
                f"{len(latent_points)} points, {n_active_dims}/{latent_array.shape[1]} active dims, "
                f"σ={sigma:.4f}, weight={self._model_kdes[id(model)].weight:.2f}."
            )

        if self._model_kdes:
            total_w = sum(k.weight for k in self._model_kdes.values())
            self.logger.info(
                f"Evidence model: {len(self._model_kdes)} models from {n_exp} experiments "
                f"(radius={self._radius}, estimator={self._estimator_config.type}, total_weight={total_w:.2f})."
            )

    def fit_empty_kde(self, datamodule: DataModule, target_n: int = 1) -> None:
        """Initialize empty evidence structures for all non-deterministic models.

        active_mask is determined by schema bounds: only dimensions with
        non-trivial range (lo < hi) are active. This ensures boundary evidence
        and σ reflect the actual optimization space, not all input columns.
        """
        self._model_kdes = {}
        self.datamodule = datamodule
        kde_models = [m for m in self.models if not isinstance(m, DeterministicModel)]
        if not kde_models:
            self.logger.info("All models are deterministic — empty evidence not needed.")
            return

        self._n_exp = target_n
        schema = datamodule.dataset.schema

        for model in kde_models:
            # Determine latent dimensionality by encoding a dummy point
            # (skipping encoder when bypass is on — latent space = raw input space).
            input_cols = model.input_parameters + model.input_features
            input_indices = datamodule.get_input_indices(input_cols, skip_missing=True)
            n_input = len(input_indices) if input_indices else len(datamodule.input_columns)
            dummy_X = np.zeros((1, n_input), dtype=np.float32)
            if self._bypass_encoder:
                z = dummy_X[0]
            else:
                z_t = model.encode(torch.from_numpy(dummy_X))
                z = z_t.detach().cpu().numpy()[0]
            n_dims = len(z)

            # Active mask from schema bounds: only dims with non-trivial range
            active_mask = np.zeros(n_dims, dtype=bool)
            dm_cols = datamodule.input_columns
            model_col_indices = input_indices if input_indices else list(range(len(dm_cols)))
            for zi, col_idx in enumerate(model_col_indices):
                if zi >= n_dims:
                    break
                col_code = dm_cols[col_idx] if col_idx < len(dm_cols) else None
                if col_code and col_code in schema.parameters.data_objects:
                    obj = schema.parameters.data_objects[col_code]
                    if hasattr(obj, 'constraints'):
                        lo = obj.constraints.get('min', None)
                        hi = obj.constraints.get('max', None)
                        if lo is not None and hi is not None and float(hi) > float(lo):
                            active_mask[zi] = True

            n_active = int(active_mask.sum())
            if n_active == 0:
                active_mask = np.ones(n_dims, dtype=bool)
                n_active = n_dims
            sigma = self._resolve_sigma(n_active)

            self._model_kdes[id(model)] = _ModelKDE(
                model=model,
                latent_points=np.empty((0, n_active)),
                point_weights=np.empty((0,)),
                sigma=sigma,
                active_mask=active_mask,
                n_active_dims=n_active,
                weight=self._get_model_weight(model),
            )

        self.logger.info(
            f"Empty evidence initialized for {len(self._model_kdes)} models "
            f"(radius={self._radius}, estimator={self._estimator_config.type})."
        )

    def _resolve_sigma(self, n_dims: int = 1) -> float:
        """σ = radius (1D bandwidth). With ANOVA kernel, no √D scaling needed."""
        sigma = float(self._radius)
        if sigma < SIGMA_MIN:
            self.logger.warning(
                f"σ={sigma:.4f} below floor; clamping to SIGMA_MIN={SIGMA_MIN}."
            )
            sigma = SIGMA_MIN
        return sigma

    def configure_exploration(
        self,
        radius: float | None = None,
        sigma: float | None = None,
    ) -> None:
        """Configure kernel radius (σ = radius × √D)."""
        if radius is not None:
            self._radius = float(radius)
        elif sigma is not None:
            self._radius = float(sigma)

        if self._model_kdes:
            for kde in self._model_kdes.values():
                kde.sigma = self._resolve_sigma(kde.n_active_dims)
            self.logger.info(
                f"Evidence model updated: radius={self._radius}, "
                f"{len(self._model_kdes)} models."
            )

    def configure_evidence(
        self,
        *,
        estimator: str | None = None,
        radii: tuple[float, ...] | None = None,
        angular_gap_deg: float | None = None,
        box: float | None = None,
        n_samples: int | None = None,
        seed: int | None = None,
        cutoff_sigmas: float | None = None,
        truncation_threshold: int | None = None,
    ) -> None:
        """Configure the evidence estimator and its tuning knobs.

        ``estimator``: ``"kernel_field"`` (deterministic shells, gradient-traversable;
        probe count grows with D) or ``"sobol_local"`` (QMC cube, fixed
        ``n_samples`` per kernel — the high-D escape hatch).

        Per-estimator knobs that don't apply to the chosen estimator are
        accepted but ignored. Rebuilds the internal estimator from the
        updated config and rebinds ``self._estimator``.
        """
        from .evidence import EstimatorConfig, make_estimator
        cur = self._estimator_config
        if estimator is not None and estimator not in ("kernel_field", "sobol_local"):
            raise ValueError(
                f"unknown estimator {estimator!r}; expected 'kernel_field' or 'sobol_local'"
            )
        new_cfg = EstimatorConfig(
            type=estimator if estimator is not None else cur.type,
            radii=radii if radii is not None else cur.radii,
            angular_gap_deg=angular_gap_deg if angular_gap_deg is not None else cur.angular_gap_deg,
            box=box if box is not None else cur.box,
            n_samples=n_samples if n_samples is not None else cur.n_samples,
            seed=seed if seed is not None else cur.seed,
            cutoff_sigmas=cutoff_sigmas if cutoff_sigmas is not None else cur.cutoff_sigmas,
            truncation_threshold=truncation_threshold if truncation_threshold is not None else cur.truncation_threshold,
        )
        self._estimator_config = new_cfg
        self._estimator = make_estimator(new_cfg)
        self.logger.info(
            f"Evidence estimator: {new_cfg.type} "
            f"({'radii=' + str(new_cfg.radii) if new_cfg.type == 'kernel_field' else 'box=' + str(new_cfg.box) + ', n_samples=' + str(new_cfg.n_samples)})"
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
        """Encode a 1-D normalized parameter array via a specific model's encode().

        When ``self._bypass_encoder`` is set (discovery mode), returns the raw
        sliced input directly so the evidence/KDE works in raw input space.
        """
        if self.datamodule is None:
            return X_norm
        input_cols = model.input_parameters + model.input_features
        input_indices = self.datamodule.get_input_indices(input_cols, skip_missing=True)
        if not input_indices:
            return X_norm
        X_model = X_norm[input_indices].reshape(1, -1).astype(np.float32)
        if self._bypass_encoder:
            return X_model[0]
        z_t = model.encode(torch.from_numpy(X_model))
        return z_t.detach().cpu().numpy()[0]

    # --- Integrated evidence math (pure numpy, per-model subspace) ---

    @staticmethod
    def _gaussian_density(z: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
        """Peak-1 Gaussian density ``(M, D) × (N, D) → (M, N)`` matching `KernelIndex`.

        KDE prediction uses ratios of weighted densities so the chosen
        normalisation cancels; the peak-1 form keeps the kernel definition
        consistent across the package.
        """
        d2 = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        return np.exp(-d2 / (2.0 * sigma ** 2))

    @classmethod
    def _raw_density(
        cls, z: np.ndarray, centers: np.ndarray, weights: np.ndarray, sigma: float,
    ) -> np.ndarray:
        """D(z) = Σ_j w_j · ρ_j(z). Returns (M,)."""
        if len(centers) == 0:
            return np.zeros(z.shape[0])
        K = cls._gaussian_density(z, centers, sigma)
        return (K * weights[None, :]).sum(axis=-1)

    def _kernel_index(self, centers: np.ndarray, weights: np.ndarray, sigma: float) -> KernelIndex:
        return KernelIndex(
            centers, weights, sigma,
            cutoff_sigmas=self._estimator_config.cutoff_sigmas,
            truncation_threshold=self._estimator_config.truncation_threshold,
        )

    def _encode_batch_from_norm_for_model_tensor(
        self, model: IPredictionModel, X_norm_batch: torch.Tensor, active_mask: np.ndarray,
    ) -> torch.Tensor:
        """Encode ``(S, D_global) → (S, n_active)`` keeping grad through ``model.encode``."""
        active_idx = torch.from_numpy(np.flatnonzero(active_mask)).long()
        if self.datamodule is None:
            return X_norm_batch.index_select(-1, active_idx)
        input_cols = model.input_parameters + model.input_features
        input_indices = self.datamodule.get_input_indices(input_cols, skip_missing=True)
        if not input_indices:
            return X_norm_batch.index_select(-1, active_idx)
        input_idx_t = torch.tensor(input_indices, dtype=torch.long)
        if self._bypass_encoder:
            return X_norm_batch.index_select(-1, input_idx_t).index_select(-1, active_idx)
        X_model = X_norm_batch.index_select(-1, input_idx_t).to(dtype=torch.float32)
        z_t = model.encode(X_model, gradient_pass=True)
        return z_t.index_select(-1, active_idx)

    def delta_integrated_evidence_batched_tensor(
        self,
        new_norm_batch_S: torch.Tensor,
        new_weights_S: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-candidate Δ∫E with gradient flowing back into ``new_norm_batch_S``.

        Routes through ``KernelFieldEstimator.integrated_evidence_perturbed_batched_joint_torch``
        with ``L=1`` (each candidate is a single added kernel). Non-KernelField
        estimators fall back to the numpy path + detach (gradient lost there).
        """
        S = int(new_norm_batch_S.shape[0])
        dtype = new_norm_batch_S.dtype
        if not self._model_kdes or S == 0:
            return torch.zeros(S, dtype=dtype)

        weights_S = (
            torch.ones(S, dtype=dtype)
            if new_weights_S is None
            else new_weights_S.to(dtype=dtype)
        )

        out = torch.zeros(S, dtype=dtype)
        total_w = 0.0
        for kde in self._model_kdes.values():
            new_centers_z_active = self._encode_batch_from_norm_for_model_tensor(
                kde.model, new_norm_batch_S, kde.active_mask,
            ).to(dtype=dtype)  # (S, n_active)

            index_old = self._kernel_index(kde.latent_points, kde.point_weights, kde.sigma)

            # Compute E_old via the same ANOVA torch path (empty index + old as "new")
            if index_old.is_empty:
                E_old = 0.0
            else:
                empty_index = self._kernel_index(
                    np.empty((0, kde.latent_points.shape[1])), np.empty(0), kde.sigma,
                )
                old_centers_t = index_old.centers.unsqueeze(0).to(dtype=dtype)  # (1, n_old, D)
                old_weights_t = index_old.weights.unsqueeze(0).to(dtype=dtype)  # (1, n_old)
                E_old = float(self._estimator.integrated_evidence_perturbed_batched_joint_torch(
                    empty_index, old_centers_t, old_weights_t,
                )[0].item())

            E_new_per_s = self._estimator.integrated_evidence_perturbed_batched_joint_torch(
                index_old,
                new_centers_z_active.unsqueeze(1),
                weights_S.unsqueeze(1),
            )
            out = out + float(kde.weight) * (E_new_per_s - E_old)
            total_w += float(kde.weight)

        return out / total_w if total_w > 0 else out

    def delta_integrated_evidence_joint_batched_tensor(
        self,
        new_norm_batch_SL: torch.Tensor,
        new_weights_SL: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-candidate joint Δ∫E (each adds L kernels jointly), gradient-traversable."""
        S = int(new_norm_batch_SL.shape[0])
        if S == 0 or new_norm_batch_SL.numel() == 0:
            return torch.zeros(S, dtype=new_norm_batch_SL.dtype)
        L = int(new_norm_batch_SL.shape[1])
        dtype = new_norm_batch_SL.dtype
        if not self._model_kdes:
            return torch.zeros(S, dtype=dtype)

        weights_SL = (
            torch.ones((S, L), dtype=dtype)
            if new_weights_SL is None
            else new_weights_SL.to(dtype=dtype)
        )

        flat_batch = new_norm_batch_SL.reshape(S * L, -1)

        out = torch.zeros(S, dtype=dtype)
        total_w = 0.0
        for kde in self._model_kdes.values():
            new_centers_flat = self._encode_batch_from_norm_for_model_tensor(
                kde.model, flat_batch, kde.active_mask,
            ).to(dtype=dtype)  # (S*L, n_active)
            new_centers_SL = new_centers_flat.reshape(S, L, -1)

            index_old = self._kernel_index(kde.latent_points, kde.point_weights, kde.sigma)

            # Compute E_old via the same ANOVA torch path
            if index_old.is_empty:
                E_old = 0.0
            else:
                empty_index = self._kernel_index(
                    np.empty((0, kde.latent_points.shape[1])), np.empty(0), kde.sigma,
                )
                old_centers_t = index_old.centers.unsqueeze(0).to(dtype=dtype)
                old_weights_t = index_old.weights.unsqueeze(0).to(dtype=dtype)
                E_old = float(self._estimator.integrated_evidence_perturbed_batched_joint_torch(
                    empty_index, old_centers_t, old_weights_t,
                )[0].item())

            E_new_per_s = self._estimator.integrated_evidence_perturbed_batched_joint_torch(
                index_old, new_centers_SL, weights_SL,
            )
            out = out + float(kde.weight) * (E_new_per_s - E_old)
            total_w += float(kde.weight)

        return out / total_w if total_w > 0 else out

    # --- Stacking / virtual evidence for sequential-phase schedule mode ---

    def push_virtual_points(
        self,
        params_list: list[dict[str, Any]],
        weights_list: list[float] | None = None,
        datamodule: DataModule | None = None,
    ) -> None:
        """Append (params, weight) pairs as temporary evidence in each KDE.

        Used by the sequential-stacked schedule mode to represent
        not-yet-scheduled experiments as stacks at their step0 with weight L_j.
        Restore via `pop_virtual_points`.
        """
        if not self._model_kdes:
            return
        dm = datamodule or self.datamodule
        if dm is None:
            return
        weights = weights_list if weights_list is not None else [1.0] * len(params_list)

        for kde in self._model_kdes.values():
            if not hasattr(kde, '_virtual_snapshot'):
                kde._virtual_snapshot = (           # type: ignore[attr-defined]
                    kde.latent_points.copy(), kde.point_weights.copy(),
                )
            new_z: list[np.ndarray] = []
            new_w: list[float] = []
            for params, w in zip(params_list, weights):
                z = self._encode_params_for_model(kde.model, params, dm)
                if z is None:
                    continue
                new_z.append(z[kde.active_mask])
                new_w.append(float(w))
            if new_z:
                kde.latent_points = np.vstack([kde.latent_points, np.stack(new_z)])
                kde.point_weights = np.concatenate([kde.point_weights, np.asarray(new_w)])

    def pop_virtual_points(self) -> None:
        """Restore each KDE's latent state to the snapshot taken by `push_virtual_points`."""
        if not self._model_kdes:
            return
        for kde in self._model_kdes.values():
            snap = getattr(kde, '_virtual_snapshot', None)
            if snap is not None:
                kde.latent_points, kde.point_weights = snap
                del kde._virtual_snapshot          # type: ignore[attr-defined]

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
            if isinstance(m, DeterministicModel):
                continue
            w = self._get_model_weight(m)
            if w > best_w:
                best, best_w = m, w
        return best if best is not None else (self.models[0] if self.models else None)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode a batch of normalized parameter arrays to latent space.

        Public API stays numpy-typed (used by callers that don't yet speak tensor);
        the tensor↔numpy boundary is internal to this method.
        """
        model = self._primary_encode_model()
        if model is None or self.datamodule is None:
            return X
        input_cols = model.input_parameters + model.input_features
        input_indices = self.datamodule.get_input_indices(input_cols, skip_missing=True)
        if not input_indices:
            return X
        X_model = X[:, input_indices] if X.ndim > 1 else X[input_indices].reshape(1, -1)
        X_model = X_model.astype(np.float32)
        if self._bypass_encoder:
            return X_model
        z_t = model.encode(torch.from_numpy(X_model))
        return z_t.detach().cpu().numpy()

    # --- Aggregated public API ---

    def uncertainty(self, X: np.ndarray) -> float:
        """Pointwise uncertainty u(z) = 1 − E(z) = 1 / (1 + D(z)).

        For visualization only — the optimizer uses `delta_integrated_evidence_aggregated`.
        Aggregated as a performance-weighted average across per-model KDEs.
        """
        if not self._model_kdes:
            return 1.0

        total_w = 0.0
        weighted_u = 0.0
        for kde in self._model_kdes.values():
            z = self._encode_from_norm_array_for_model(kde.model, X.reshape(-1))
            z_active = z[kde.active_mask].reshape(1, -1)
            D = float(self._raw_density(z_active, kde.latent_points, kde.point_weights, kde.sigma)[0])
            u_i = 1.0 / (1.0 + D)
            weighted_u += kde.weight * u_i
            total_w += kde.weight

        u = weighted_u / total_w if total_w > 0 else 1.0
        self.logger.debug(f"uncertainty (aggregated): u={u:.4f}")
        return u

    def predict_for_calibration_tensor(
        self,
        params_list: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor]]:
        """Tensor-typed mirror of ``predict_for_calibration_batched``.

        Returns predictions as ``torch.Tensor`` per feature (no numpy
        round-trip at the API boundary). Per-candidate output is a
        ``dict[feature_code, torch.Tensor]`` where each tensor has shape
        ``(*feat_shape,)`` matching the per-candidate iterator-domain
        topology.

        Gradient flow: continuous values in each ``params`` dict can be
        ``torch.Tensor`` with ``requires_grad=True`` — gradients flow
        through ``params_to_tensor`` (affine normalisation) and the autoreg
        loop's ``forward_pass(gradient_pass=True)`` calls back into the
        input tensors. Used by the gradient-based acquisition
        optimiser.

        Parameters that determine prediction *shape* (``n_layers``,
        ``n_segments``, etc.) must be plain Python ints in the params dict;
        they're used at the dim_info-resolution stage which isn't
        differentiable by construction (integer choices).
        """
        self._assert_trained()
        if not params_list:
            return []

        with profiler.section("predict.predict_for_calibration_tensor"):
            return self._predict_from_params_tensor(params_list)

    def _predict_from_params_tensor(
        self,
        params_list: list[dict[str, Any]],
    ) -> list[dict[str, torch.Tensor]]:
        """Per-candidate gradient-traversable prediction via polymorphic ``model.predict``.

        Topologically orders models so each call sees its upstream dependencies'
        outputs through ``predictions_so_far``. Each concrete model class
        (MLP/Deterministic flat-batched, Transformer sequence-batched) owns its
        own dispatch — the framework just plumbs ``params_list``, ``dm``,
        per-candidate ``dim_info_list``, and the accumulated upstream outputs.
        """
        self._assert_trained()
        S = len(params_list)
        if S == 0:
            return []

        dm = self.datamodule
        assert dm is not None  # _assert_trained guarantees

        # Per-feature → per-candidate accumulator threaded into model.predict
        # so downstream models can read upstream outputs (cross-model deps).
        accumulated: dict[str, dict[int, torch.Tensor]] = {}

        for model in self._topo_sort_models():
            dim_info_list = [self._get_model_dim_info(model, p) for p in params_list]
            per_s = model.predict(params_list, dm, dim_info_list, accumulated)
            for s, feat_dict in enumerate(per_s):
                for feat, t in feat_dict.items():
                    accumulated.setdefault(feat, {})[s] = t

        # Repackage feature → s-dict into per-candidate dicts of feature tensors.
        out: list[dict[str, torch.Tensor]] = [{} for _ in range(S)]
        for feat, per_s_dict in accumulated.items():
            for s, t in per_s_dict.items():
                out[s][feat] = t
        return out

    def tune(
            self,
            exp_data: ExperimentData,
            start: int,
            end: int | None = None,
            batch_size: int | None = None,
            **kwargs
            ) -> DataModule:
        """Fine-tune models on a slice of new experimental data, reusing the training-time normalisation."""
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

        # tensor-native tune slicing (no pandas).
        exported = temp_dataset.export_to_tensor_dict(
            [exp_data.code],
            x_columns=temp_datamodule.input_columns,
            y_columns=temp_datamodule.output_columns,
            categorical_mappings=temp_datamodule.categorical_mappings,
        )
        if exported.is_empty():
            raise ValueError("Tuning dataset has no feature rows available.")

        n_total = exported.n_rows
        end_index = end if end is not None else n_total
        if start < 0 or start >= n_total:
            raise ValueError(f"Tuning start index {start} out of bounds for {n_total} rows.")
        if end_index <= start or end_index > n_total:
            raise ValueError(f"Tuning end index {end_index} invalid for start {start} and {n_total} rows.")

        # Slice each per-column tensor in the dict.
        X_dict_sliced: dict[str, torch.Tensor] = {
            col: t[start:end_index] for col, t in exported.X.items()
        }
        X_tune = temp_datamodule.prepare_input_from_tensor_dict(X_dict_sliced)

        # Build y as (n_sliced, n_outputs) tensor + apply per-column normalisation.
        y_cols = []
        for col in temp_datamodule.output_columns:
            col_t = exported.y.get(col, torch.zeros(n_total, dtype=torch.float32))[start:end_index]
            stats = temp_datamodule._feature_stats.get(col)
            if stats is not None:
                col_t = temp_datamodule._apply_normalization_tensor(col_t, stats)
            y_cols.append(col_t.reshape(-1))
        y_tune_t = torch.stack(y_cols, dim=-1) if y_cols else torch.zeros((end_index - start, 0))
        y_tune = y_tune_t.detach().cpu().numpy()

        # Initialize base predictions array
        y_pred_base = np.zeros_like(y_tune)

        # Get predictions from all base models
        self.logger.info("Generating base predictions for residual learning...")
        for model in self.models:
            input_indices = dm.get_input_indices(model.input_parameters + model.input_features)
            input_indices_t = torch.as_tensor(input_indices, dtype=torch.long)
            X_model = X_tune.index_select(1, input_indices_t)
            y_pred_dict = model.forward_pass(X_model)
            for feat in model.outputs:
                col = dm.output_columns.index(feat)
                y_pred_base[:, col] = y_pred_dict[feat].detach().cpu().numpy()

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

    @staticmethod
    def _build_importance_weights(
        dm: DataModule,
        split: SplitType,
        performance_weights: dict[str, float] | None,
        floor: float = 0.1,
        steepness: float = 0.8,
    ) -> np.ndarray | None:
        """Build per-row importance weights via sigmoid centered at mean performance.

        weight_i = floor + (1 - floor) * sigmoid(k * (perf_i - perf_mean))

        where k = steepness / perf_std adapts to the data spread. This gives:
          - At mean performance: weight = midpoint = (1 + floor) / 2
          - Above mean: weight climbs toward 1.0 (high-performing = important)
          - Below mean: weight drops toward floor (low-performing = less important)

        The sigmoid ensures smooth transitions with no hard cutoffs. The gap
        (R²_adj - R²) is interpretable relative to the mean:
          gap > 0 → model predicts above-average experiments better
          gap < 0 → model predicts above-average experiments worse
          gap ≈ 0 → uniform prediction quality across performance range

        Returns None if no performance_weights are provided.
        """
        if not performance_weights:
            return None

        # Collect per-experiment combined scores
        codes = dm.get_split_codes(split)
        exp_scores: list[tuple[float, int]] = []  # (score, n_rows)
        for code in codes:
            exp = dm.dataset.get_experiment(code)
            perf = exp.performance.get_values_dict()
            score = combined_score(perf, performance_weights)
            dim_names = exp.parameters.get_dim_names()
            n_rows = len(exp.parameters.get_dim_combinations(dim_names)) if dim_names else 1
            exp_scores.append((score, n_rows))

        if not exp_scores:
            return None

        # Compute mean and std of performance scores
        all_scores = np.array([s for s, _ in exp_scores])
        s_mean = float(all_scores.mean())
        s_std = float(all_scores.std())

        # Sigmoid weighting centered at mean, steepness adapts to spread
        k = steepness / s_std if s_std > 1e-10 else 0.0

        importance_rows: list[float] = []
        for score, n_rows in exp_scores:
            sigmoid = 1.0 / (1.0 + np.exp(-k * (score - s_mean)))
            weight = floor + (1.0 - floor) * sigmoid
            importance_rows.extend([weight] * n_rows)

        return np.array(importance_rows, dtype=np.float64)

    def _validate_flat(
        self,
        model: IPredictionModel,
        dm: DataModule,
        X_split: torch.Tensor,
        y_split: torch.Tensor,
        importance_arr: np.ndarray | None,
    ) -> dict[str, dict[str, float]]:
        """Per-row validation for non-sequence models (MLP, deterministic)."""
        input_indices = dm.get_input_indices(model.input_parameters + model.input_features)
        input_indices_t = torch.as_tensor(input_indices, dtype=torch.long)
        indices = [dm.output_columns.index(f) for f in model.outputs]
        indices_t = torch.as_tensor(indices, dtype=torch.long)

        y_true_norm = y_split.index_select(1, indices_t)
        y_true = dm.denormalize_values(y_true_norm, model.outputs).detach().cpu().numpy()

        y_pred_dict = model.forward_pass(X_split.index_select(1, input_indices_t))
        y_pred_norm = torch.stack([y_pred_dict[f] for f in model.outputs], dim=-1)
        y_pred = dm.denormalize_values(y_pred_norm, model.outputs).detach().cpu().numpy()

        results: dict[str, dict[str, float]] = {}
        for i, feat in enumerate(model.outputs):
            feat_metrics = Metrics.calculate_regression_metrics(y_true[:, i], y_pred[:, i])
            if importance_arr is not None and len(importance_arr) == len(y_true[:, i]):
                adj = Metrics.calculate_adjusted_r2(y_true[:, i], y_pred[:, i], importance_arr)
                feat_metrics['r2_adj'] = adj['r2_adj']
            results[feat] = feat_metrics
        return results

    def _validate_transformer(
        self,
        model: IPredictionModel,
        dm: DataModule,
        split: SplitType,
        importance_arr: np.ndarray | None,
    ) -> dict[str, dict[str, float]]:
        """Sequence-aware validation for TransformerModel.

        Uses predict() with full causal context instead of per-row forward_pass().
        """
        codes = dm.get_split_codes(split)
        if not codes:
            return {}

        per_feat_true: dict[str, list[float]] = {f: [] for f in model.outputs}
        per_feat_pred: dict[str, list[float]] = {f: [] for f in model.outputs}

        for code in codes:
            exp = dm.dataset.get_experiment(code)
            params = exp.get_effective_parameters_for_row(0)
            dim_info = self._get_model_dim_info(model, params)
            shape = dim_info['shape']

            pred_list = model.predict([params], dm, [dim_info], {})
            if not pred_list:
                continue
            preds = pred_list[0]

            exported = dm.dataset.export_to_tensor_dict(
                [code],
                x_columns=dm.input_columns,
                y_columns=dm.output_columns,
                categorical_mappings=dm.categorical_mappings,
            )
            if exported.is_empty():
                continue

            for feat in model.outputs:
                y_pred_native = preds[feat].detach().cpu().numpy().ravel()
                # export_to_tensor_dict returns raw values; predict() also
                # returns denormalized values — both are in physical units.
                y_true_raw = exported.y[feat].to(dtype=torch.float32)
                y_true_flat = y_true_raw.detach().cpu().numpy().ravel()

                n = min(len(y_true_flat), len(y_pred_native))
                per_feat_true[feat].extend(y_true_flat[:n].tolist())
                per_feat_pred[feat].extend(y_pred_native[:n].tolist())

        results: dict[str, dict[str, float]] = {}
        for feat in model.outputs:
            y_t = np.array(per_feat_true[feat])
            y_p = np.array(per_feat_pred[feat])
            if len(y_t) == 0:
                continue
            feat_metrics = Metrics.calculate_regression_metrics(y_t, y_p)
            if importance_arr is not None and len(importance_arr) == len(y_t):
                adj = Metrics.calculate_adjusted_r2(y_t, y_p, importance_arr)
                feat_metrics['r2_adj'] = adj['r2_adj']
            results[feat] = feat_metrics
        return results

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

        self.logger.info(f"Validating models on {split} set...")

        # Extract validation/test data
        batches = dm.get_batches(split)
        if not batches:
            self.logger.console_warning(f"No batches returned for {split} set during validation.")
            return {}

        # Concatenate batches; drop cell_meta (third tuple element).
        X_list = [b[0] for b in batches]
        y_list = [b[1] for b in batches]
        X_split = torch.cat(X_list, dim=0)
        y_split = torch.cat(y_list, dim=0)

        self.logger.info(f"Evaluating {len(self.models)} models on {X_split.shape[0]} samples...")

        # Build per-row importance for R²_adj: experiments near the performance
        # optimum are weighted higher, so R²_adj measures prediction accuracy
        # where it matters most for calibration.
        importance_arr = self._build_importance_weights(dm, split, performance_weights)

        # Compute per-feature metrics
        results: dict[str, dict[str, float]] = {}
        for model in self.models:
            from ..models.transformer import TransformerModel
            if isinstance(model, TransformerModel):
                results.update(self._validate_transformer(model, dm, split, importance_arr))
            else:
                results.update(self._validate_flat(model, dm, X_split, y_split, importance_arr))

        # Print compact validation table with line breaks for readability
        has_adj = any('r2_adj' in m for m in results.values())
        header = f"  {'Feature':<25s}  {'R²':>8s}"
        if has_adj:
            header += f"  {'R²_adj':>8s}"
        self.logger.console_new_line()
        self.logger.console_info(header)
        self.logger.console_info(f"  {'─' * (36 if not has_adj else 46)}")
        for feat, m in results.items():
            line = f"  {feat:<25s}  {m['r2']:8.4f}"
            if has_adj and 'r2_adj' in m:
                line += f"  {m['r2_adj']:8.4f}"
            self.logger.console_info(line)
        self.logger.console_new_line()
        self.logger.console_success(
            f"Validation: {split} set, {len(X_split)} samples"
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
        """Predict dimensional features for ``exp_data`` over a position-index range.

        ``overlap`` lets context-aware models (e.g. transformers) carry continuity
        across consecutive batches; must be < ``batch_size``.
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

        shape = tuple(dim_sizes)
        total_positions = int(np.prod(shape)) if shape else 1

        # Honour the per-model input contract: X_batch carries only columns
        # this model declares via input_parameters / input_features. The X_batch
        # produced from this dim_info is consumed only by this model (callers
        # pass model=model through the predict chain), so narrowing here cannot
        # strand inputs needed by other models.
        declared_params = set(model.input_parameters)
        declared_features = set(model.input_features)
        param_base = {k: v for k, v in params.items() if k in declared_params}

        # Iterator-derived features: any input_features entry of the form
        # f"{ic}_pos" matches a Dimension's iterator_code in this model's
        # domain. Iterators are implicit on every Dimension — no schema
        # declaration. At row-build time they evaluate to idx / (size - 1).
        iterator_feats: list[tuple[str, int, int]] = []
        for feat_code in declared_features:
            if feat_code in self.schema.features.data_objects:
                continue  # real feature, not an iterator
            if not feat_code.endswith("_pos"):
                continue
            ic_target = feat_code[:-len("_pos")]
            for axis_pos, iter_code in enumerate(dim_iterators):
                if iter_code == ic_target:
                    iterator_feats.append((feat_code, axis_pos, dim_sizes[axis_pos]))
                    break

        return {
            'shape': shape,
            'dim_iterators': dim_iterators,
            'dim_codes_ordered': dim_codes_ordered,
            'param_base': param_base,
            'iterator_feats': iterator_feats,
            'total_positions': total_positions,
        }

    def _build_X_dict_flat(
        self,
        dm: "DataModule",
        dim_info_list: list[dict[str, Any]],
        cell_indices: list[tuple[int, ...]] | np.ndarray,
        iterator_feats: list[tuple[str, int, int]],
    ) -> dict[str, torch.Tensor]:
        """Build per-column tensors for the (S × n_cells, n_input_cols) batch.

        Each ``dim_info`` in ``dim_info_list`` carries a ``param_base`` dict —
        the (param_code → value) map for that S-row. ``cell_indices`` lists the
        cell-index tuples within the dimension. Output is keyed by
        ``dm.input_columns`` and ready for ``prepare_input_from_tensor_dict``.
        Categoricals emit cat-index ``int64`` tensors; iterator features emit
        normalised positions; numeric params/features emit float scalars.
        """
        S = len(dim_info_list)
        n_cells = len(cell_indices)
        iter_feat_lookup = {fc: (axis_pos, size) for fc, axis_pos, size in iterator_feats}
        X_dict_flat: dict[str, torch.Tensor] = {}
        for col_name in dm.input_columns:
            if col_name in dm.categorical_mappings:
                cats = dm.categorical_mappings[col_name]
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                idx_vals: list[int] = []
                for s in range(S):
                    param_base = dim_info_list[s]['param_base']
                    v = param_base.get(col_name, cats[0] if cats else None)
                    cell_val = cat_to_idx.get(v, 0)
                    idx_vals.extend([cell_val] * n_cells)
                X_dict_flat[col_name] = torch.tensor(idx_vals, dtype=torch.long)
            elif col_name in iter_feat_lookup:
                axis_pos, size = iter_feat_lookup[col_name]
                vals: list[float] = []
                for _s in range(S):
                    for idx in cell_indices:
                        vals.append(float(idx[axis_pos]) / max(size - 1, 1))
                X_dict_flat[col_name] = torch.tensor(vals, dtype=torch.float32)
            else:
                num_vals: list[float] = []
                for s in range(S):
                    param_base = dim_info_list[s]['param_base']
                    v = float(param_base.get(col_name, 0.0))
                    num_vals.extend([v] * n_cells)
                X_dict_flat[col_name] = torch.tensor(num_vals, dtype=torch.float32)
        return X_dict_flat

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
        """Process positions in batches: build X, predict, denormalize, store in prediction dict. Supports overlap.

        ``model`` (optional) restricts prediction to that model's outputs.
        Recursive autoreg dispatch was removed alongside the cell-loop
        machinery — sequence-aware prediction now lives in
        ``TransformerModel.predict``; this helper feeds the numpy /
        non-recursive path only.
        """
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
    ) -> tuple[dict[str, torch.Tensor], list[tuple[int, ...]]]:
        """Build a per-column tensor-dict feature batch for ``prepare_input_from_tensor_dict``.

        Categoricals are emitted as long-index, numerics + iterator-derived
        features as float — same semantics as ``Dataset.export_to_tensor_dict``.
        """
        shape = dim_info['shape']
        param_base = dim_info['param_base']
        iterator_feats = dim_info['iterator_feats']
        dm = self._assert_trained()

        batch_indices: list[tuple[int, ...]] = [
            tuple(int(c) for c in np.unravel_index(pos, shape))
            for pos in range(batch_start, batch_end)
        ]
        n_cells = len(batch_indices)
        iter_feat_lookup = {fc: (axis_pos, size) for fc, axis_pos, size in iterator_feats}

        X_dict: dict[str, torch.Tensor] = {}
        for col_name in dm.input_columns:
            if col_name in dm.categorical_mappings:
                cats = dm.categorical_mappings[col_name]
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                v = param_base.get(col_name, cats[0] if cats else None)
                cell_val = cat_to_idx.get(v, 0)
                X_dict[col_name] = torch.full((n_cells,), cell_val, dtype=torch.long)
            elif col_name in iter_feat_lookup:
                axis_pos, size = iter_feat_lookup[col_name]
                vals = [float(idx[axis_pos]) / max(size - 1, 1) for idx in batch_indices]
                X_dict[col_name] = torch.tensor(vals, dtype=torch.float32)
            else:
                v_f = float(param_base.get(col_name, 0.0))
                X_dict[col_name] = torch.full((n_cells,), v_f, dtype=torch.float32)

        return X_dict, batch_indices

    def _predict_and_store_batch_to_dict(
        self,
        predictions: dict[str, np.ndarray],
        X_batch: dict[str, torch.Tensor],
        batch_indices: list[tuple[int, ...]],
        model: IPredictionModel | None = None,
    ) -> None:
        """Run model prediction on X_batch tensor dict and store results.

        ``X_batch`` is now a tensor dict (was DataFrame);
        prep goes through ``prepare_input_from_tensor_dict``.
        """
        dm = self._assert_trained()
        X_norm = dm.prepare_input_from_tensor_dict(X_batch)

        models_to_run = [model] if model is not None else self.models
        for m in models_to_run:
            input_indices = dm.get_input_indices(m.input_parameters + m.input_features)
            input_indices_t = torch.as_tensor(input_indices, dtype=torch.long)
            X_model = X_norm.index_select(1, input_indices_t)
            y_pred_dict = m.forward_pass(X_model)
            y_pred_norm = torch.stack([y_pred_dict[f] for f in m.outputs], dim=-1)
            y_pred = dm.denormalize_values(y_pred_norm, m.outputs)

            # y_pred: (batch, n_outputs) tensor
            y_np = y_pred.detach().cpu().numpy()
            for i, feature_name in enumerate(m.outputs):
                if feature_name not in predictions:
                    continue
                values = y_np[:, i]
                feat_depth = len(predictions[feature_name].shape)
                for j, idx in enumerate(batch_indices):
                    feat_idx = idx[:feat_depth]
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
