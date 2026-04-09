"""Abstract interface for prediction models that learn parameter→feature mappings."""

from abc import ABC, abstractmethod
from typing import Any, final
import copy
import numpy as np

from .base_interface import BaseInterface
from ..utils.logger import PfabLogger
from ..utils.enum import NormMethod
from ..core import DataObject, Dataset


class IPredictionModel(BaseInterface):
    """Abstract base for prediction models: train on experiments, predict features, support export/import.

    Domain is derived from the schema during PredictionSystem initialization; all outputs must
    share the same domain_code and feature_depth. Do not declare input_domain — it is inferred
    from the output features registered in the schema.
    """

    def __init__(self, logger: PfabLogger):
        super().__init__(logger)

    @property
    def depth(self) -> int:
        """Max iterator depth across output features; 0 for scalar outputs. Requires set_ref_features() called first."""
        max_depth = 0
        for code in self.outputs:
            feat = self._ref_features.get(code)
            # _ref_features stores DataArray instances (Feature.array() factory output);
            # Pyright sees Feature (factory class) which lacks .columns — type: ignore needed.
            if feat is not None and hasattr(feat, "columns") and feat.columns:  # type: ignore[union-attr]
                max_depth = max(max_depth, len(feat.columns) - 1)  # type: ignore[union-attr]
        return max_depth

    def validate_dimensional_coherence(self, schema: Any) -> str | None:
        """Enforce structural rules on the model's domain declarations and derive the domain code.

        1. Output features may not mix depths (error).
        2. All output features must share the same named domain (error). This is also the
           derivation step: the returned domain code is the single named domain, or None for
           scalar models.
        3. Input features may not exceed the model's operational depth (error).

        Returns the derived domain code (single named domain, or None for scalar models).
        """
        name = self.__class__.__name__
        op_depth = self.depth

        # Rule 1: mixed output depths are an error
        if self.outputs:
            output_depths = {}
            for code in self.outputs:
                feat = self._ref_features.get(code)
                cols = feat.columns if (feat is not None and hasattr(feat, "columns")) else []  # type: ignore[union-attr]
                d = (len(cols) - 1) if cols else 0
                output_depths[code] = d
            if len(set(output_depths.values())) > 1:
                raise ValueError(
                    f"{name}: output features have mixed depths {output_depths}. "
                    f"The model will iterate at depth {op_depth}. Shallower outputs "
                    f"will be overwritten on each deeper iteration step."
                )

        # Rule 2: all output features must share the same named domain (None = scalar, allowed alongside any domain).
        # The single named domain is also the derived domain_code returned to the caller.
        output_domains = set()
        for code in self.outputs:
            feat_obj = schema.features.data_objects.get(code)
            domain_code = feat_obj.domain_code if (feat_obj is not None and hasattr(feat_obj, "domain_code")) else None  # type: ignore[union-attr]
            output_domains.add(domain_code)
        named_domains = {d for d in output_domains if d is not None}
        if len(named_domains) > 1:
            raise ValueError(
                f"{name}: output features span multiple named domains {named_domains}. "
                f"A prediction model must operate within a single domain."
            )

        derived_domain = next(iter(named_domains)) if named_domains else None

        # Rule 3: input features must not exceed operational depth
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

        return derived_domain

    # === ABSTRACT METHODS ===

    # abstract methods from BaseInterface:
    # - input_parameters
    # - input_features
    # - outputs

    @abstractmethod
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Run model inference on normalized X (batch, n_params) → normalized y (batch, n_features)."""
        pass

    @abstractmethod
    def train(self, train_batches: list[tuple[np.ndarray, np.ndarray]], val_batches: list[tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        """Train the model on (X, y) batch tuples."""
        pass

    # === LATENT ENCODING ===

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Map normalized parameters to latent space; default is identity. Override for custom latent encoding."""
        return X

    # === ONLINE LEARNING ===

    def tuning(self, tune_batches: list[tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
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


class IDeterministicModel(IPredictionModel):
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
    def formula(self, X: np.ndarray) -> np.ndarray:
        """Compute output values from raw (denormalized) input values.

        X has shape (batch, n_raw_inputs) with columns ordered by ``input_parameters``.
        Real parameters are in original scale; categorical parameters are integer-encoded
        (index into the sorted category list available via ``self.categorical_mappings``).

        Must return shape (batch, n_outputs) in original (physical) scale.
        """
        ...

    # === PRE-DEFINED PIPELINE ===

    @final
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Denormalize inputs → formula → renormalize outputs."""
        if not self._norm_context_set:
            raise RuntimeError(
                f"{self.__class__.__name__}.forward_pass() called before "
                f"set_normalization_context(). This is set automatically by "
                f"PredictionSystem during training."
            )
        X_raw = self._denormalize_inputs(X)
        y_raw = self.formula(X_raw)
        y_norm = self._normalize_outputs(y_raw)
        return y_norm

    @final
    def train(
        self,
        train_batches: list[tuple[np.ndarray, np.ndarray]],
        val_batches: list[tuple[np.ndarray, np.ndarray]],
        **kwargs: Any,
    ) -> None:
        """No-op — deterministic models have no learned parameters."""
        pass

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Identity — no learned latent space for analytical models."""
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

    def _normalize_outputs(self, y_raw: np.ndarray) -> np.ndarray:
        """Apply normalization to raw output values."""
        y_norm = y_raw.copy()
        for i, feat in enumerate(self.outputs):
            if feat in self._norm_feature_stats:
                stats = self._norm_feature_stats[feat]
                y_norm[:, i] = self._apply_normalization(y_norm[:, i], stats)
        return y_norm

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

