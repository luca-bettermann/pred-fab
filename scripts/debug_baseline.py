"""Debug script: runs baseline flow in a single script for stepping through in a debugger.

Usage:
    python -m scripts.debug_baseline
    # or with debugger:
    python -m debugpy --wait-for-client --listen 5678 -m scripts.debug_baseline
"""
import sys
sys.path.insert(0, "src")

from pathlib import Path
import tempfile

# --- Setup (equivalent to init-schema + init-agent) ---
tmp = Path(tempfile.mkdtemp())

# Import lbp schema and agent setup
# Adjust this import path if running from a different location
from pred_fab.orchestration.agent import PfabAgent

# Use the test builders for a quick standalone setup
from tests.utils.builders import build_workflow_stack, evaluate_loaded_workflow_experiments, build_prepared_workflow_datamodule

agent, dataset, codes = build_workflow_stack(tmp)
evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
dm = build_prepared_workflow_datamodule(agent=agent, dataset=dataset, val_size=0.0, test_size=0.0, recompute=True)
agent.train(datamodule=dm, validate=False, test=False)

# --- Configure (equivalent to cli configure) ---
cs = agent.calibration_system
cs.engine.gradient_n_iters = 20
cs.engine.gradient_n_starts = 2
cs.engine.gradient_raw_samples = 0
agent.configure_trajectory("speed", "n_layers", delta=50.0)

# --- Baseline (equivalent to cli baseline) ---
print("\n=== Running baseline ===")
specs = agent.baseline_step(n=3)

print(f"\n=== Got {len(specs)} specs ===")
for i, spec in enumerate(specs):
    params = spec.initial_params.to_dict()
    n_traj = sum(len(t.entries) for t in spec.trajectories.values())
    print(f"  Exp {i}: {params}  ({n_traj} trajectory entries)")
