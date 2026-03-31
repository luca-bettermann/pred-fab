"""Abstract interface for prediction models that learn parameter→feature mappings."""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional, Any, final, Tuple
import numpy as np

from .base_interface import BaseInterface
from ..utils.logger import PfabLogger
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

    def validate_dimensional_coherence(self, schema: Any) -> Optional[str]:
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
    def train(self, train_batches: List[Tuple[np.ndarray, np.ndarray]], val_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        """Train the model on (X, y) batch tuples."""
        pass

    # === LATENT ENCODING ===

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Map normalized parameters to latent space; default is identity. Override for custom latent encoding."""
        return X

    # === ONLINE LEARNING ===

    def tuning(self, tune_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        """Fine-tune with new measurements during fabrication; override to enable online learning."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tuning. "
            f"Override tuning() method to enable online learning."
        )
    
    # === EXPORT/IMPORT SUPPORT ===
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        """Serialize trained model state for InferenceBundle export; override to enable. All values must be picklable."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support export. "
            f"Override _get_model_artifacts() and _set_model_artifacts() to enable export."
        )
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Restore trained model state from artifacts dict; must exactly reverse _get_model_artifacts()."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support import. "
            f"Override _get_model_artifacts() and _set_model_artifacts() to enable import."
        )

