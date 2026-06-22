"""Guard: the MODEL import surface (rtde keystone) pulls neither torch nor pandas.

rtde imports the dim/experiment model + per-position traversal *without* the ML stack
(``pred-fab[ml]``). A clean subprocess proves that importing the model surface does not pull
torch/pandas into ``sys.modules`` (the test env has them installed, so we check what's
*imported*, not what's installed). If this fails, something on the model path regained a
module-level ``import torch``/``import pandas`` — make it lazy (see [[PFAB - Hierarchical Load Save]]).
"""
import subprocess
import sys

_CHECK = """
import sys
import pred_fab
from pred_fab.core import (
    Dimension, Domain, DatasetSchema, SchemaRegistry, Dataset, ExperimentData,
    Parameters, Features, PerformanceAttributes, ParameterProposal,
    ParameterTrajectory, ExperimentSpec, ExperimentSet, Provenance,
)
assert hasattr(ExperimentData, "get_effective_parameters_for_context")
leaked = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "pandas"))
assert not leaked, "model surface pulled heavy deps: " + repr(leaked)
print("OK")
"""


def test_model_surface_imports_without_torch_or_pandas():
    result = subprocess.run([sys.executable, "-c", _CHECK], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout
