"""
Prediction Model Interface for AIXD architecture.

Defines abstract interface for prediction models that learn from experiment data
and predict features for new parameter combinations. Features can then be evaluated
to compute performance metrics.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional, Any, final, Tuple
import numpy as np

from .base_interface import BaseInterface
from ..utils.logger import PfabLogger
from ..core import DataObject, Dataset


class IPredictionModel(BaseInterface):
    """
    Abstract base class for prediction models.

    - Learn parameter→feature relationships from experiment data
    - Predict feature values for new parameter combinations (virtual experiments)
    - Enable feature-based evaluation and multi-objective optimization
    - Support export/import for production inference via InferenceBundle
    - Must be dataclasses with DataObject fields for parameters (schema generation)
    """

    def __init__(self, logger: PfabLogger):
        """Initialize evaluation system."""
        super().__init__(logger)

    @property
    def depth(self) -> int:
        """Operational depth: number of dimension levels this model iterates over.

        Inferred from output feature tensor shapes — equals the maximum number of
        iterator dimensions across all declared output features. Depth 0 means scalar
        outputs with no dimensional iteration.

        Requires set_ref_features() to have been called (done automatically by
        PredictionSystem during agent initialization).
        """
        max_depth = 0
        for code in self.outputs:
            feat = self._ref_features.get(code)
            # _ref_features stores DataArray instances (Feature.array() factory output);
            # Pyright sees Feature (factory class) which lacks .columns — type: ignore needed.
            if feat is not None and hasattr(feat, "columns") and feat.columns:  # type: ignore[union-attr]
                max_depth = max(max_depth, len(feat.columns) - 1)  # type: ignore[union-attr]
        return max_depth

    def validate_dimensional_coherence(self, schema: Any) -> None:
        """Validate that this model's input/output dimensions are coherent.

        Rules:
            1. (Warning) All output features should share the same depth. Mixed-depth
               outputs are allowed but produce a warning — the model iterates at the
               maximum depth, and shallower outputs overwrite on each deeper iteration.
            2. (Error) Any dimension codes declared in input_parameters must form a
               consecutive prefix of the schema hierarchy {level 1, 2, ..., k} — no
               gaps, must start at level 1.
            3. (Error) Input features must not have a depth exceeding the model's
               operational depth.

        Args:
            schema: DatasetSchema instance with fully initialized feature columns.

        Raises:
            ValueError: If Rule 2 or Rule 3 is violated.
        """
        name = self.__class__.__name__
        op_depth = self.depth

        # Rule 1: warn on mixed output depths
        if self.outputs:
            output_depths = {}
            for code in self.outputs:
                feat = self._ref_features.get(code)
                # _ref_features stores DataArray instances; Pyright sees Feature (factory).
                cols = feat.columns if (feat is not None and hasattr(feat, "columns")) else []  # type: ignore[union-attr]
                d = (len(cols) - 1) if cols else 0
                output_depths[code] = d
            if len(set(output_depths.values())) > 1:
                self.logger.warning(
                    f"{name}: output features have mixed depths {output_depths}. "
                    f"The model will iterate at depth {op_depth}. Shallower outputs "
                    f"will be overwritten on each deeper iteration step."
                )

        # Rule 2: declared dimension codes must form a consecutive prefix
        schema_dim_by_code = {dim.code: dim for dim in schema.parameters.get_sorted_dimensions()}
        declared_dim_levels = sorted(
            schema_dim_by_code[code].level
            for code in self.input_parameters
            if code in schema_dim_by_code
        )
        if declared_dim_levels:
            expected = list(range(1, max(declared_dim_levels) + 1))
            if declared_dim_levels != expected:
                raise ValueError(
                    f"{name}: input_parameters declare dimension levels "
                    f"{declared_dim_levels}, which are not a consecutive prefix from 1 "
                    f"(expected {expected}). Declared dimensions must start at level 1 "
                    f"with no gaps."
                )
            declared_max = max(declared_dim_levels)
            if op_depth > 0 and declared_max != op_depth:
                raise ValueError(
                    f"{name}: input_parameters declare dimensions up to level "
                    f"{declared_max}, but output features require operational depth "
                    f"{op_depth}. Declare dimensions matching the depth of your output "
                    f"features, or remove explicit dimension declarations."
                )

        # Rule 3: input features must not exceed operational depth
        for code in self.input_features:
            feat = self._ref_features.get(code)
            # _ref_features stores DataArray instances; Pyright sees Feature (factory).
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
        """
        Forward pass of given parameter values to retrieve features.
        
        Args:
            X: Numpy array with normalized parameter values (batch_size, n_params)
        
        Returns:
            Numpy array with normalized feature values (batch_size, n_features)
        """
        pass

    @abstractmethod
    def train(self, train_batches: List[Tuple[np.ndarray, np.ndarray]], val_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        """
        Train the prediction model on batched data.
        
        Args:
            train_batches: List of (X, y) tuples for training
            val_batches: List of (X, y) tuples for validation
            **kwargs: Additional training parameters
        """
        pass

    # === LATENT ENCODING ===

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Map normalized parameter vectors to latent representations.

        Override to provide a custom latent space (e.g. penultimate layer of a
        neural network). The default implementation is the identity map so that
        the normalized parameter space itself is used as the latent space.

        Args:
            X: Normalized parameter array (batch_size, n_params)

        Returns:
            Latent representation array (batch_size, n_latent)
        """
        return X

    # === ONLINE LEARNING ===

    def tuning(self, tune_batches: List[Tuple[np.ndarray, np.ndarray]], **kwargs) -> None:
        """
        Fine-tune model with new measurements during fabrication.
        
        Override this method to implement online learning/adaptation.
        Default behavior: Raises NotImplementedError.
        
        Args:
            tune_batches: List of (X, y) tuples for tuning
            **kwargs: Additional tuning parameters
        
        Raises:
            NotImplementedError: If tuning not supported (default behavior)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tuning. "
            f"Override tuning() method to enable online learning."
        )
    
    # === EXPORT/IMPORT SUPPORT ===
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        """
        Serialize trained model state for production export.
        
        Override this method to enable export to InferenceBundle. Return all
        artifacts needed to restore model: weights, configuration, trained objects
        (sklearn models, neural networks, etc.). All values must be picklable.
        Raise RuntimeError if model not trained.
        
        Returns:
            Dict containing complete model state for reconstruction
            (e.g., {'model': sklearn_model, 'scaler': fitted_scaler})
        
        Raises:
            NotImplementedError: If export not supported (default behavior)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support export. "
            f"Override _get_model_artifacts() and _set_model_artifacts() to enable export."
        )
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """
        Restore trained model state from exported artifacts.
        
        Override this method to enable import from InferenceBundle. Reconstruct
        model to its trained state from the dict returned by _get_model_artifacts().
        Must perfectly reverse _get_model_artifacts() so that round-trip
        export→import preserves model behavior.
        
        Args:
            artifacts: Dict containing model state (from _get_model_artifacts())
        
        Raises:
            NotImplementedError: If import not supported (default behavior)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support import. "
            f"Override _get_model_artifacts() and _set_model_artifacts() to enable import."
        )

