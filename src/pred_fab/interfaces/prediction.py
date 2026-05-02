"""Abstract interface for prediction models that learn parameter→feature mappings."""

from abc import ABC, abstractmethod
from typing import Any, final
import copy
import numpy as np
import torch

from .base_interface import BaseInterface
from ..utils.logger import PfabLogger
from ..utils.enum import NormMethod
from ..core import DataObject, Dataset


class IPredictionModel(BaseInterface):
    """Abstract base for prediction models: train on experiments, predict features, support export/import.

    Domain is derived from the schema during PredictionSystem initialization; all outputs must
    share the same domain_code and feature_depth. Do not declare input_domain — it is inferred
    from the output features registered in the schema.

    Three concrete subclasses cover the dominant fab modelling architectures:

    - ``DeterministicModel`` — closed-form formulas (no training).
    - ``MLPModel`` — feed-forward MLP for tabular / non-sequential mappings.
    - ``TransformerModel`` — encoder-only transformer with causal attention
      for sequential / autoregressive mappings.

    Each subclass implements its own ``predict`` (the interface defaults to flat-
    batched dispatch, suitable for MLP/Deterministic; Transformer overrides with
    sequence dispatch). End users plug one concrete model per domain;
    ``PredictionSystem`` orchestrates the mix via topological sort over
    cross-model dependencies.
    """

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

    @property
    def depth(self) -> int:
        """Max iterator depth across output features; 0 for scalar outputs. Requires set_ref_features() called first."""
        max_depth = 0
        for code in self.outputs:
            feat = self._ref_features.get(code)
            # _ref_features stores DataArray instances (Feature() factory output);
            # Pyright sees Feature (factory class) which lacks .columns — type: ignore needed.
            if feat is not None and hasattr(feat, "columns") and feat.columns:  # type: ignore[union-attr]
                max_depth = max(max_depth, len(feat.columns) - 1)  # type: ignore[union-attr]
        return max_depth

    def validate_dimensional_coherence(self, schema: Any) -> str | None:
        """Per-class structural rules for the model's domain/depth declarations.

        Returns the derived domain code (single named domain shared across outputs,
        or None for scalar models).

        Each concrete subclass overrides to enforce its own rules:

        - ``MLPModel`` — single domain + depth uniformity + input-depth ≤ op-depth.
        - ``TransformerModel`` — single domain + axis-depth ≤ min-output-depth +
          input-depth ≤ op-depth.
        - ``DeterministicModel`` — no restrictions; derives domain best-effort.

        Helpers ``_derive_single_domain``, ``_assert_uniform_output_depth``,
        ``_assert_input_depth_within_op_depth`` cover the recurring rule fragments.
        """
        return self._derive_single_domain(schema)

    # ------------------------------------------------------------------
    # Validation helpers — building blocks for per-class implementations
    # ------------------------------------------------------------------

    def _output_depths(self) -> dict[str, int]:
        """Map each output feature code to its iterator depth (0 for scalar)."""
        depths: dict[str, int] = {}
        for code in self.outputs:
            feat = self._ref_features.get(code)
            cols = feat.columns if (feat is not None and hasattr(feat, "columns")) else []  # type: ignore[union-attr]
            depths[code] = (len(cols) - 1) if cols else 0
        return depths

    def _derive_single_domain(self, schema: Any) -> str | None:
        """Derive the single named output domain; raise if outputs span multiple named domains.

        Returns the single named domain code, or None for scalar-only models.
        """
        name = self.__class__.__name__
        named_domains: set[str] = set()
        for code in self.outputs:
            feat_obj = schema.features.data_objects.get(code)
            domain_code = feat_obj.domain_code if (feat_obj is not None and hasattr(feat_obj, "domain_code")) else None  # type: ignore[union-attr]
            if domain_code is not None:
                named_domains.add(domain_code)
        if len(named_domains) > 1:
            raise ValueError(
                f"{name}: output features span multiple named domains {named_domains}. "
                f"A prediction model must operate within a single domain."
            )
        return next(iter(named_domains)) if named_domains else None

    def _assert_uniform_output_depth(self) -> None:
        """Raise if outputs mix depths."""
        name = self.__class__.__name__
        depths = self._output_depths()
        if len(set(depths.values())) > 1:
            raise ValueError(
                f"{name}: output features have mixed depths {depths}. "
                f"This model class requires all outputs to share the same depth."
            )

    def _assert_input_depth_within_op_depth(self) -> None:
        """Raise if any input feature depth exceeds the model's operational depth."""
        name = self.__class__.__name__
        op_depth = self.depth
        for code in self.input_features:
            feat = self._ref_features.get(code)
            feat_cols = feat.columns if (feat is not None and hasattr(feat, "columns")) else []  # type: ignore[union-attr]
            if feat_cols:
                input_feat_depth = len(feat_cols) - 1
                if input_feat_depth > op_depth:
                    raise ValueError(
                        f"{name}: input feature '{code}' has depth {input_feat_depth}, "
                        f"which exceeds the model's operational depth {op_depth}. A "
                        f"model cannot consume inputs at finer granularity than its outputs."
                    )

    # === ABSTRACT METHODS ===

    # abstract methods from BaseInterface:
    # - input_parameters
    # - input_features
    # - outputs

    @abstractmethod
    def forward_pass(
        self,
        X: torch.Tensor,
        gradient_pass: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run model inference on normalized X (batch, n_inputs) → ``dict[feat_code, tensor]``.

        Returns one entry per ``self.outputs`` feature, each tensor in
        normalized output space at the feature's natural per-cell shape (for
        the flat-batched case, a 1-D ``(batch,)`` tensor of scalars). Multi-
        depth models may return tensors with extra trailing axes (Phase B —
        not yet wired).

        ``gradient_pass=False`` (default): inference under ``torch.no_grad()``.
        ``gradient_pass=True``: gradients flow through the network. The
        framework guarantees X arrives as a CPU float32 tensor in normalized
        input space.
        """
        pass

    @abstractmethod
    def train(
        self,
        train_batches: list[tuple[torch.Tensor, torch.Tensor]],
        val_batches: list[tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ) -> None:
        """Train the model on (X, y) tensor batch tuples."""
        pass

    # === POLYMORPHIC PREDICT + TYPE-SPECIFIC SCHEMA CHECK ===
    #
    # The framework dispatches per-candidate prediction by calling ``predict``
    # on each model in topological order; each model class (MLP, Transformer,
    # Deterministic) owns its own implementation. This replaces the framework-
    # side cell loop / sequence dispatch branching.
    #
    # ``_validate_schema_compatibility`` is the type-specific complement to
    # ``validate_dimensional_coherence`` — the latter is universal and ``@final``;
    # this one each subclass overrides to enforce its own rules (e.g. MLP
    # rejects recursive features; Transformer requires ``sequence_axis_code``
    # to resolve to a real domain axis).

    def predict(
        self,
        params_list: list[dict[str, Any]],
        dm: Any,
        dim_info_list: list[dict[str, Any]],
        predictions_so_far: dict[str, dict[int, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        """Per-candidate prediction → ``list[dict[feat_code, (*feat_shape) tensor]]``.

        Default: flat-batched dispatch. Builds ``(sum_s n_cells_s, n_input)``
        via ``dm.build_flat_batch``, runs one ``forward_pass``, and
        de-multiplexes per-(s, cell) into per-feature ``(*feat_shape)``
        tensors via ``torch.stack`` so the autograd graph stays connected.

        Suitable for any per-cell-independent mapping (MLP, deterministic
        formula). ``TransformerModel`` overrides with sequence dispatch.

        All outputs of a flat-batched model share the same iterator depth
        (rule 1 of ``validate_dimensional_coherence``), so each output's
        per-candidate ``feat_shape`` equals ``dim_info['shape']``.

        ``predictions_so_far`` is accepted for interface conformance but
        not consumed yet — cross-model predicted features aren't supported
        on the gradient path today either.
        """
        del predictions_so_far  # cross-model deps not consumed here yet
        S = len(params_list)
        if S == 0:
            return []
        if len(dim_info_list) != S:
            raise ValueError(
                f"predict: dim_info_list length {len(dim_info_list)} "
                f"does not match params_list length {S}.",
            )

        X_flat, row_map = dm.build_flat_batch(params_list, dim_info_list)
        if X_flat.shape[0] == 0:
            return [{feat: torch.zeros(()) for feat in self.outputs} for _ in range(S)]

        input_indices = dm.get_input_indices(self.input_parameters + self.input_features)
        input_indices_t = torch.as_tensor(input_indices, dtype=torch.long)
        X_model = X_flat.index_select(1, input_indices_t)

        y_norm_dict = self.forward_pass(X_model, gradient_pass=True)
        # Stack for normalization/denormalization (per-feature stats), then split.
        y_norm_stacked = torch.stack(
            [y_norm_dict[feat] for feat in self.outputs], dim=-1,
        )  # (n_rows, n_outputs)
        y_denorm_stacked = dm.denormalize_values(y_norm_stacked, self.outputs)

        accum: list[dict[str, dict[int, torch.Tensor]]] = [
            {feat: {} for feat in self.outputs} for _ in range(S)
        ]
        for row_idx, (s, cell_flat) in enumerate(row_map):
            for f_idx, feat in enumerate(self.outputs):
                accum[s][feat][cell_flat] = y_denorm_stacked[row_idx, f_idx]

        out: list[dict[str, torch.Tensor]] = [{} for _ in range(S)]
        for s in range(S):
            feat_shape = dim_info_list[s]['shape']
            n_cells_s = int(np.prod(feat_shape)) if feat_shape else 1
            for feat in self.outputs:
                slots = [accum[s][feat][c] for c in range(n_cells_s)]
                if feat_shape:
                    out[s][feat] = torch.stack(slots, dim=0).reshape(feat_shape)
                else:
                    out[s][feat] = slots[0]

        return out

    def _validate_schema_compatibility(self, schema: Any) -> None:
        """Type-specific schema check, run after ``validate_dimensional_coherence``.

        Default: no-op. Concrete model classes override to enforce class-
        specific rules (e.g. ``TransformerModel`` requires
        ``sequence_axis_code`` to resolve to a real domain axis).
        """
        del schema

    # === LATENT ENCODING ===

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Map normalized parameters to latent space; default is identity.

        ``gradient_pass=True`` instructs the implementation to keep autograd
        graph live through the encoder (skip ``torch.no_grad()``). Used by
        the gradient acquisition path.
        """
        del gradient_pass  # base impl is identity, gradient flows naturally
        return X

    # === CATEGORICAL CONTEXT ===

    def set_categorical_context(self, col_to_cardinality: dict[int, int]) -> None:
        """Inform the model which of its input columns are categorical (cat-index).

        ``col_to_cardinality`` maps **model-relative column index** → number of
        categories for each categorical column in the model's input space
        (after column selection in ``_filter_batches_for_model``).

        Default: no-op. Override in models that consume categorical inputs
        (typically ``MLPModel`` subclasses use this to size internal
        ``F.one_hot`` / ``nn.Embedding`` expansion).
        """
        del col_to_cardinality  # base impl ignores

    # === ONLINE LEARNING ===

    def tuning(self, tune_batches: list[tuple[torch.Tensor, torch.Tensor]], **kwargs) -> None:
        """Fine-tune with new measurements during fabrication; override to enable online learning."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tuning. "
            f"Override tuning() method to enable online learning."
        )
    
    # === EXPORT/IMPORT SUPPORT ===
    
    def _get_model_artifacts(self) -> dict[str, Any]:
        """Serialize trained model state for InferenceBundle export; override to enable. All values must be picklable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support export. "
            f"Override _get_model_artifacts() and _set_model_artifacts() to enable export."
        )
    
    def _set_model_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Restore trained model state from artifacts dict; must exactly reverse _get_model_artifacts()."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support import. "
            f"Override _get_model_artifacts() and _set_model_artifacts() to enable import."
        )


class DeterministicModel(IPredictionModel):
    """Prediction model backed by a known analytical formula, not learned from data.

    Subclasses implement ``formula(X_raw)`` which receives denormalized inputs
    (physical values, categoricals as integer indices) and returns raw output values.

    ``forward_pass`` is pre-defined: it denormalizes inputs, calls ``formula``,
    and renormalizes outputs — preserving the normalized-in/normalized-out contract
    of ``IPredictionModel`` without the user needing to handle normalization.

    ``train()`` is a no-op. ``encode()`` returns identity (no learned latent space).
    """

    def __init__(self, logger: PfabLogger) -> None:
        self._norm_parameter_stats: dict[str, dict[str, Any]] = {}
        self._norm_feature_stats: dict[str, dict[str, Any]] = {}
        self._norm_categorical_mappings: dict[str, list[str]] = {}
        self._norm_context_set = False
        super().__init__(logger)

    # === NORMALIZATION CONTEXT ===

    @final
    def set_normalization_context(
        self,
        parameter_stats: dict[str, dict[str, Any]],
        feature_stats: dict[str, dict[str, Any]],
        categorical_mappings: dict[str, list[str]],
    ) -> None:
        """Store normalization statistics so forward_pass can denormalize inputs and renormalize outputs.

        Called by PredictionSystem after DataModule normalization is fitted.
        """
        self._norm_parameter_stats = copy.deepcopy(parameter_stats)
        self._norm_feature_stats = copy.deepcopy(feature_stats)
        self._norm_categorical_mappings = copy.deepcopy(categorical_mappings)
        self._norm_context_set = True

    @property
    def categorical_mappings(self) -> dict[str, list[str]]:
        """Mapping of categorical parameter names to their sorted category lists."""
        return self._norm_categorical_mappings

    # === ABSTRACT: USER IMPLEMENTS THIS ===

    @abstractmethod
    def formula(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Compute output values from raw (denormalized) input values.

        X has shape (batch, n_raw_inputs) with columns ordered by ``input_parameters``.
        Real parameters are in original scale; categorical parameters are integer-encoded
        (index into the sorted category list available via ``self.categorical_mappings``).

        Must return ``dict[feat_code, np.ndarray]`` — one entry per ``self.outputs``,
        each a 1-D ``(batch,)`` array in original (physical) scale.
        """
        ...

    # === PRE-DEFINED PIPELINE ===

    @final
    def forward_pass(
        self,
        X: torch.Tensor,
        gradient_pass: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Denormalize inputs → formula → renormalize outputs (per feature).

        Internal numpy boundary: ``formula(X_raw)`` operates on numpy arrays
        in physical units; the tensor↔numpy round-trip is a once-per-call
        boundary cost (cheap relative to the formula itself).

        ``gradient_pass`` is accepted for IPredictionModel API conformance
        but **has no effect** — the deterministic numpy formula path is
        intrinsically not gradient-traversable. Callers needing autograd
        should use a learnable ``IPredictionModel`` subclass.
        """
        del gradient_pass  # unused by design
        if not self._norm_context_set:
            raise RuntimeError(
                f"{self.__class__.__name__}.forward_pass() called before "
                f"set_normalization_context(). This is set automatically by "
                f"PredictionSystem during training."
            )
        X_np = X.detach().cpu().numpy()
        X_raw = self._denormalize_inputs(X_np)
        y_raw_dict = self.formula(X_raw)
        return {
            feat: torch.from_numpy(self._normalize_output_feature(y_raw_dict[feat], feat).astype(np.float32))
            for feat in self.outputs
        }

    @final
    def train(
        self,
        train_batches: list[tuple[torch.Tensor, torch.Tensor]],
        val_batches: list[tuple[torch.Tensor, torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        """No-op — deterministic models have no learned parameters."""
        pass

    def encode(self, X: torch.Tensor, gradient_pass: bool = False) -> torch.Tensor:
        """Identity — no learned latent space for analytical models."""
        del gradient_pass
        return X

    # === INTERNAL NORMALIZATION HELPERS ===

    def _denormalize_inputs(self, X_norm: np.ndarray) -> np.ndarray:
        """Reverse normalization and collapse one-hot categoricals to integer indices.

        Output columns are ordered by ``input_parameters``: one column per parameter,
        with categoricals collapsed from N one-hot columns to a single integer index.
        """
        batch_size = X_norm.shape[0]
        raw_cols: list[np.ndarray] = []
        col_idx = 0

        for param in self.input_parameters:
            if param in self._norm_categorical_mappings:
                n_cats = len(self._norm_categorical_mappings[param])
                onehot = X_norm[:, col_idx:col_idx + n_cats]
                cat_idx = np.argmax(onehot, axis=1).astype(np.float64)
                raw_cols.append(cat_idx.reshape(-1, 1))
                col_idx += n_cats
            else:
                vals = X_norm[:, col_idx:col_idx + 1].copy()
                if param in self._norm_parameter_stats:
                    stats = self._norm_parameter_stats[param]
                    vals = self._reverse_normalization(vals, stats)
                raw_cols.append(vals)
                col_idx += 1

        return np.hstack(raw_cols) if raw_cols else np.empty((batch_size, 0))

    def _normalize_output_feature(self, y_raw: np.ndarray, feat_code: str) -> np.ndarray:
        """Apply normalization to a single feature's raw output values."""
        stats = self._norm_feature_stats.get(feat_code)
        if stats is None:
            return y_raw
        return self._apply_normalization(y_raw, stats)

    @staticmethod
    def _reverse_normalization(data: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        """Reverse normalization for a data array using pre-computed stats."""
        method = stats['method']
        if method == NormMethod.NONE:
            return data
        elif method == NormMethod.STANDARD:
            return data * stats['std'] + stats['mean']
        elif method == NormMethod.MIN_MAX:
            denom = stats['max'] - stats['min']
            if abs(denom) < 1e-12:
                return np.full_like(data, fill_value=stats['min'], dtype=np.float64)
            return data * (stats['max'] - stats['min']) + stats['min']
        elif method == NormMethod.ROBUST:
            iqr = stats['q3'] - stats['q1']
            return data * iqr + stats['median']
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def _apply_normalization(data: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        """Apply normalization to a data array using pre-computed stats."""
        method = stats['method']
        if method == NormMethod.NONE:
            return data
        elif method == NormMethod.STANDARD:
            return (data - stats['mean']) / (stats['std'] + 1e-8)
        elif method == NormMethod.MIN_MAX:
            denom = stats['max'] - stats['min']
            if abs(denom) < 1e-12:
                return np.zeros_like(data, dtype=np.float64)
            return (data - stats['min']) / (stats['max'] - stats['min'] + 1e-8)
        elif method == NormMethod.ROBUST:
            iqr = stats['q3'] - stats['q1']
            return (data - stats['median']) / (iqr + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

