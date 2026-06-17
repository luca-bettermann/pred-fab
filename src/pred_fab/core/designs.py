"""Sampling designs over the parameter space — data-independent generators.

A *design* is the process that produces (and tags the provenance of) a set of
experiments. Designs cover the unit cube ``[0,1]^d``; the calibration system maps
those unit points to schema-valid proposals via each variable's norm→real decode
(``Variable.to_real``), so integers/categoricals/bounds are respected uniformly.

These are deliberately **not** the acquisition optimiser: a Sobol test set must be
independent of the model and the collected data to serve as a fair generalisation
yardstick. See the KB note *First-class dataset concept in pred-fab*.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class Design(Protocol):
    """Produces ``n`` points of coverage in the unit cube ``[0,1]^n_dims``."""
    name: str

    def unit_points(self, n: int, n_dims: int, seed: int | None = None) -> np.ndarray:
        """``(n, n_dims)`` array in ``[0,1]``. Empty dims → ``(n, 0)``."""
        ...


class SobolDesign:
    """Scrambled low-discrepancy (Sobol) space-filling design — data-independent."""
    name = "sobol"

    def unit_points(self, n: int, n_dims: int, seed: int | None = None) -> np.ndarray:
        if n <= 0:
            return np.empty((0, max(n_dims, 0)), dtype=float)
        if n_dims <= 0:
            return np.empty((n, 0), dtype=float)
        engine = torch.quasirandom.SobolEngine(dimension=n_dims, scramble=True, seed=seed)
        return engine.draw(n).cpu().numpy().astype(float)
