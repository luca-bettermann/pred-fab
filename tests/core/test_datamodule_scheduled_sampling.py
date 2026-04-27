"""Tests for DataModule scheduled-sampling perturbation of recursive features."""

import numpy as np

from pred_fab.core import DataModule, Dataset, DatasetSchema
from pred_fab.core.data_blocks import (
    Domains, Features, Parameters, PerformanceAttributes,
)
from pred_fab.core.data_objects import (
    Dimension, Domain, Feature, Parameter, PerformanceAttribute,
)
from pred_fab.utils import SplitType


def _build_recursive_schema(tmp_path) -> DatasetSchema:
    """Schema with one recursive feature: prev_grid_1 = grid shifted by layer."""
    spatial = Domain("spatial", [
        Dimension("n_layers", "layer_idx", 1, 10),
        Dimension("n_segments", "segment_idx", 1, 5),
    ])
    layer_dim, _ = spatial.axes
    grid_feat = Feature.array("grid", domain=spatial)
    return DatasetSchema(
        root_folder=str(tmp_path),
        name="ss_recursive_schema",
        parameters=Parameters.from_list([
            Parameter.real("p1", min_val=0.0, max_val=1.0),
        ]),
        features=Features.from_list([
            grid_feat,
            *Feature.recursive("prev_grid", source=grid_feat, dimensions=(layer_dim,), max_depth=1),
        ]),
        performance=PerformanceAttributes.from_list([
            PerformanceAttribute.score("perf_1"),
        ]),
        domains=Domains([spatial]),
    )


def _seed_experiment(dataset: Dataset, exp_code: str, n_layers: int, n_segments: int) -> None:
    dataset.create_experiment(
        exp_code,
        parameters={"p1": 0.5, "n_layers": n_layers, "n_segments": n_segments},
    )
    exp = dataset.get_experiment(exp_code)
    grid_rows = []
    for k in range(n_layers):
        for s in range(n_segments):
            grid_rows.append([k, s, k * 100.0 + s])
    grid = np.array(grid_rows, dtype=np.float64)
    grid_tensor = exp.features.table_to_tensor("grid", grid, exp.parameters)
    exp.features.set_value("grid", grid_tensor)
    # Manually fill the recursive prev_grid_1 tensor (FeatureSystem would do
    # this via tensor shifting in production; we bypass it here for test isolation).
    prev_tensor = np.full_like(grid_tensor, np.nan)
    if n_layers > 1:
        prev_tensor[1:, :] = grid_tensor[:-1, :]
    exp.features.set_value("prev_grid_1", prev_tensor)


def _build_dm(tmp_path, n_layers: int = 4, n_segments: int = 3) -> DataModule:
    schema = _build_recursive_schema(tmp_path)
    dataset = Dataset(schema=schema, debug_flag=True)
    _seed_experiment(dataset, "exp_001", n_layers, n_segments)
    dm = DataModule(dataset=dataset)
    dm.initialize(
        input_parameters=["p1", "n_layers", "n_segments"],
        input_features=["prev_grid_1"],
        output_columns=["grid"],
    )
    dm._split_codes[SplitType.TRAIN] = ["exp_001"]
    dm.fit_normalization(SplitType.TRAIN)
    return dm


def _stub_predictions(n_layers: int, n_segments: int, base: float = 999.0) -> dict:
    """Build a predictions tensor with distinguishable values per cell."""
    arr = np.full((n_layers, n_segments), np.nan)
    for k in range(n_layers):
        for s in range(n_segments):
            arr[k, s] = base + k * 10 + s
    return {"exp_001": {"grid": arr}}


def test_set_state_disabled_by_default(tmp_path):
    dm = _build_dm(tmp_path)
    assert dm._ss_p_student == 0.0
    assert dm._ss_predictions_by_exp is None
    assert dm._ss_rng is None


def test_p_zero_returns_unperturbed_batches(tmp_path):
    dm = _build_dm(tmp_path)
    rng = np.random.RandomState(0)
    dm.set_scheduled_sampling_state(_stub_predictions(4, 3), p_student=0.0, rng=rng)

    [(X, _)] = dm.get_batches(SplitType.TRAIN)
    prev_idx = dm.input_columns.index("prev_grid_1")
    # With p=0, prev_grid_1 stays at its measured-shift values (0 for boundary,
    # measured grid value for interior cells), normalised. Prediction values
    # would land at very different normalised magnitudes; verify none of them
    # leaked through by checking the column doesn't contain the stub base (999).
    raw_back = X[:, prev_idx] * dm._parameter_stats["prev_grid_1"]["std"] + dm._parameter_stats["prev_grid_1"]["mean"]
    assert np.all(raw_back < 500), "p=0 should never inject stub predictions"


def test_p_one_replaces_interior_cells_with_predictions(tmp_path):
    dm = _build_dm(tmp_path, n_layers=4, n_segments=3)
    rng = np.random.RandomState(42)
    preds = _stub_predictions(4, 3, base=1000.0)
    dm.set_scheduled_sampling_state(preds, p_student=1.0, rng=rng)

    # We can't read the perturbed X_df directly through get_batches because of
    # normalisation. Verify by calling the perturbation helper directly.
    X_df, y_df = dm.dataset.export_to_dataframe(["exp_001"])
    X_df = dm._inject_context_features(X_df, y_df)
    perturbed = dm._perturb_recursive_features(X_df, ["exp_001"])

    # 4 layers × 3 segments = 12 rows in C-order: (0,0),(0,1),(0,2),(1,0),...
    # For interior cells (k>0), prev_grid_1 should be the prediction at (k-1, s).
    # Cell (1, 0) at row 3: prev = pred[0, 0] = 1000.
    assert perturbed["prev_grid_1"].iloc[3] == 1000.0
    # Cell (1, 1) at row 4: prev = pred[0, 1] = 1001.
    assert perturbed["prev_grid_1"].iloc[4] == 1001.0
    # Cell (3, 2) at row 11: prev = pred[2, 2] = 1022.
    assert perturbed["prev_grid_1"].iloc[11] == 1022.0


def test_p_one_boundary_cells_get_nan(tmp_path):
    dm = _build_dm(tmp_path, n_layers=4, n_segments=3)
    rng = np.random.RandomState(42)
    dm.set_scheduled_sampling_state(_stub_predictions(4, 3), p_student=1.0, rng=rng)

    X_df, y_df = dm.dataset.export_to_dataframe(["exp_001"])
    X_df = dm._inject_context_features(X_df, y_df)
    perturbed = dm._perturb_recursive_features(X_df, ["exp_001"])

    # Cells (0, *) have no prior layer → prev should be NaN for rows 0..2.
    for j in range(3):
        assert np.isnan(perturbed["prev_grid_1"].iloc[j]), f"row {j} should be NaN"


def test_val_batches_never_perturbed(tmp_path):
    dm = _build_dm(tmp_path)
    dm._split_codes[SplitType.VAL] = ["exp_001"]
    rng = np.random.RandomState(0)
    dm.set_scheduled_sampling_state(_stub_predictions(4, 3), p_student=1.0, rng=rng)

    [(X, _)] = dm.get_batches(SplitType.VAL)
    prev_idx = dm.input_columns.index("prev_grid_1")
    raw_back = X[:, prev_idx] * dm._parameter_stats["prev_grid_1"]["std"] + dm._parameter_stats["prev_grid_1"]["mean"]
    assert np.all(raw_back < 500), "VAL batches must never carry SS perturbations"


def test_non_recursive_columns_untouched(tmp_path):
    dm = _build_dm(tmp_path)
    rng = np.random.RandomState(0)
    dm.set_scheduled_sampling_state(_stub_predictions(4, 3), p_student=1.0, rng=rng)

    X_df, y_df = dm.dataset.export_to_dataframe(["exp_001"])
    X_df = dm._inject_context_features(X_df, y_df)
    perturbed = dm._perturb_recursive_features(X_df, ["exp_001"])

    # Static parameter and dimension-size columns must remain unchanged
    assert (perturbed["p1"] == 0.5).all()
    assert (perturbed["n_layers"] == 4).all()
    assert (perturbed["n_segments"] == 3).all()


def test_p_intermediate_partially_replaces(tmp_path):
    """p_student=0.5 should replace approximately half of the interior cells."""
    dm = _build_dm(tmp_path, n_layers=8, n_segments=4)
    _seed_experiment(dm.dataset, "exp_002", n_layers=8, n_segments=4)  # ignored — single exp

    rng = np.random.RandomState(123)
    preds = _stub_predictions(8, 4, base=1000.0)
    dm.set_scheduled_sampling_state(preds, p_student=0.5, rng=rng)

    X_df, y_df = dm.dataset.export_to_dataframe(["exp_001"])
    X_df = dm._inject_context_features(X_df, y_df)
    perturbed = dm._perturb_recursive_features(X_df, ["exp_001"])

    # Interior cells (k > 0): rows 4..31 (8 layers × 4 segments → rows 4-31 are k>=1).
    interior_vals = perturbed["prev_grid_1"].iloc[4:].to_numpy()
    n_replaced = int(np.sum(interior_vals >= 1000.0))
    n_total = len(interior_vals)
    # With p=0.5 and 28 interior cells, ~14 ± stochastic noise should be replaced.
    assert 8 < n_replaced < 22, f"Expected ~50% replaced, got {n_replaced}/{n_total}"
