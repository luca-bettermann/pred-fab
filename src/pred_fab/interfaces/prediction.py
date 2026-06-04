"""Abstract interface for prediction models that learn parameter→feature mappings."""

from abc import ABC, abstractmethod
from typing import Any, final
import copy
import numpy as np
import torch

from .base_interface import BaseInterface
from ..utils.logger import PfabLogger
from ..core.normalisers import normaliser_from_dict
from ..core import DataObject, Dataset


_UNRESOLVED = object()  # sentinel for "feature code couldn't be resolved against the schema"


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
    @abstractmethod
    def domain_spec(self) -> tuple[str | None, int | list[int]]:
        """Declare the model's structural envelope.

        Returns ``(domain_code, accepted_depths)``:

        - ``domain_code`` — the single schema domain this model operates in
          (``None`` for scalar-only models that touch no domain).
        - ``accepted_depths`` — ``int`` (single depth) or ``list[int]`` (multi).
          Every declared output's schema-depth must lie in this set;
          input-feature depth rules vary per class.

        Validation cross-checks declared outputs and input_features against
        this envelope and raises ``ValueError`` on mismatch.
        """
        ...

    def _accepted_depths(self) -> list[int]:
        """Normalised list view of the depth(s) declared in ``domain_spec``."""
        _, d = self.domain_spec
        return [int(d)] if isinstance(d, int) else [int(x) for x in d]

    @property
    def depth(self) -> int:
        """Max iterator depth declared in ``domain_spec`` (0 for scalar-only models)."""
        accepted = self._accepted_depths()
        return max(accepted) if accepted else 0

    def validate_dimensional_coherence(self, schema: Any) -> str | None:
        """Cross-check declared ``domain_spec`` against the schema.

        Base rules (apply to all classes):

        1. ``domain_spec`` is implemented (subclasses must override).
        2. Every declared output exists in the schema and lives in
           ``domain_spec[0]``.
        3. Every declared output's schema-depth is in ``_accepted_depths()``.
        4. Every declared input feature exists in the schema and lives in
           ``domain_spec[0]``.

        Returns ``domain_spec[0]`` so orchestration can register the model's
        domain. Subclasses override to add class-specific structural rules
        (uniform depth, axis-depth bounds, input-depth caps).
        """
        try:
            declared_domain, _ = self.domain_spec
        except NotImplementedError as e:
            raise ValueError(
                f"{self.__class__.__name__} must implement the `domain_spec` "
                f"property to declare its structural envelope.",
            ) from e

        self._assert_features_in_declared_domain(schema, self.outputs, kind="output")
        self._assert_features_in_declared_domain(schema, self.input_features, kind="input feature")
        self._assert_output_depths_in_accepted_set(schema)
        return declared_domain

    # ------------------------------------------------------------------
    # Validation helpers — used by per-class overrides
    # ------------------------------------------------------------------

    def _assert_features_in_declared_domain(
        self,
        schema: Any,
        codes: list[str],
        kind: str,
    ) -> None:
        """Each code must resolve in schema (declared Feature or iterator-input-code)
        with a domain matching ``domain_spec[0]``.
        """
        name = self.__class__.__name__
        declared_domain, _ = self.domain_spec
        for code in codes:
            feat_domain = self._resolve_feature_domain(schema, code)
            if feat_domain is _UNRESOLVED:
                raise ValueError(
                    f"{name}: declared {kind} '{code}' is not in the schema "
                    f"(neither a declared Feature nor a domain iterator-input-code).",
                )
            if feat_domain != declared_domain:
                raise ValueError(
                    f"{name}: {kind} '{code}' has schema domain {feat_domain!r}, "
                    f"but model declares domain {declared_domain!r}.",
                )

    @staticmethod
    def _resolve_feature_domain(schema: Any, code: str) -> Any:
        """Return the domain code for a feature ``code`` looked up against the schema.

        Resolution order: declared Feature → domain iterator-input-code → unresolved.
        Returns the sentinel ``_UNRESOLVED`` if the code is neither.
        """
        feat_obj = schema.features.data_objects.get(code)
        if feat_obj is not None:
            return getattr(feat_obj, "domain_code", None)
        for d_code in schema.domains.keys():
            domain = schema.domains.get(d_code)
            if code in domain.iterator_input_codes:
                return d_code
        return _UNRESOLVED

    def _schema_feature_depth(self, schema: Any, code: str) -> int:
        """Schema-declared depth for a feature code.

        Resolution order:
        - Declared Feature: depth = len(columns) - 1.
        - Iterator-input-code: depth = axis index + 1 (e.g. ``layer_idx_pos`` on the
          first axis of a 2-axis domain has depth 1).
        - Unknown: 0 (caller's responsibility to guard via `_assert_features_in_declared_domain`).
        """
        feat_obj = schema.features.data_objects.get(code)
        if feat_obj is not None:
            cols = getattr(feat_obj, "columns", None) or []
            return (len(cols) - 1) if cols else 0
        for d_code in schema.domains.keys():
            domain = schema.domains.get(d_code)
            for axis_idx, ax in enumerate(domain.axes):
                if f"{ax.iterator_code}_pos" == code:
                    return axis_idx + 1
        return 0

    def _assert_output_depths_in_accepted_set(self, schema: Any) -> None:
        """Each declared output's schema-depth must be in ``_accepted_depths()``."""
        name = self.__class__.__name__
        accepted = self._accepted_depths()
        for code in self.outputs:
            feat_depth = self._schema_feature_depth(schema, code)
            if feat_depth not in accepted:
                raise ValueError(
                    f"{name}: output '{code}' has schema depth {feat_depth}, "
                    f"not in the model's accepted depths {accepted}.",
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
        train_batches: list[tuple[torch.Tensor, Any]],
        val_batches: list[tuple[torch.Tensor, Any]],
        **kwargs,
    ) -> None:
        """Train the model on ``(X, y)`` batch tuples.

        The y element's type depends on the concrete subclass:

        - ``MLPModel`` / ``DeterministicModel``: ``torch.Tensor`` ``(B, n_outputs)``.
        - ``TransformerModel``: ``dict[feat_code, tensor]`` at native shape per feature.
        """
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

    def build_training_batches(
        self, system: Any, train_batches: Any, val_batches: Any, kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        """Build this model's (train, val) batches — flat column-filter default.

        Sequence models override to build sequence-shaped batches. ``system`` is
        the PredictionSystem, passed in for its batch-building helpers.
        """
        return (
            system._filter_batches_for_model(train_batches, self),
            system._filter_batches_for_model(val_batches, self),
        )

    def validate_split(
        self, system: Any, dm: Any, split: Any, x_split: Any, y_split: Any,
        cell_meta: Any, importance_dict: Any,
    ) -> dict[str, dict[str, float]]:
        """Validate this model on a split — flat default; sequence models override."""
        return system._validate_flat(self, dm, x_split, y_split, cell_meta, importance_dict)

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
        """Reverse normalization via the canonical NormaliserModule."""
        return normaliser_from_dict(stats).reverse(data)

    @staticmethod
    def _apply_normalization(data: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
        """Apply normalization via the canonical NormaliserModule."""
        return normaliser_from_dict(stats).forward(data)

