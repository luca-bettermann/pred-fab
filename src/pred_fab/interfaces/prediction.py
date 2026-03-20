"""Abstract interface for prediction models that learn parameter→feature mappings."""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional, Any, final, Tuple
import numpy as np

from .base_interface import BaseInterface
from ..utils.logger import PfabLogger
from ..core import DataObject, Dataset


class IPredictionModel(BaseInterface):
    """Abstract base for prediction models: train on experiments, predict features, support export/import."""

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

    @property
    @abstractmethod
    def input_domain(self) -> Optional[str]:
        """Domain code this model predicts features for; None for scalar models."""
        ...

    def validate_dimensional_coherence(self, schema: Any) -> None:
        """Enforce structural rules on the model's domain declarations.

        1. Output features may not mix depths (warning).
        2. All output features must share the same domain (error).
        3. input_domain must match the domain of output features (error).
        4. Input features may not exceed the model's operational depth (error).
        """
        name = self.__class__.__name__
        op_depth = self.depth

        # Rule 1: warn on mixed output depths
        if self.outputs:
            output_depths = {}
            for code in self.outputs:
                feat = self._ref_features.get(code)
                cols = feat.columns if (feat is not None and hasattr(feat, "columns")) else []  # type: ignore[union-attr]
                d = (len(cols) - 1) if cols else 0
                output_depths[code] = d
            if len(set(output_depths.values())) > 1:
                self.logger.warning(
                    f"{name}: output features have mixed depths {output_depths}. "
                    f"The model will iterate at depth {op_depth}. Shallower outputs "
                    f"will be overwritten on each deeper iteration step."
                )

        # Rule 2: all output features must share the same named domain (None = scalar, allowed alongside any domain).
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

        # Rule 3: input_domain must match the domain of output features
        declared_domain = self.input_domain
        feature_domain = next(iter(named_domains)) if named_domains else None
        if feature_domain and declared_domain and declared_domain != feature_domain:
            raise ValueError(
                f"{name}: input_domain='{declared_domain}' does not match output feature domain '{feature_domain}'."
            )

        # Rule 4: input features must not exceed operational depth
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

