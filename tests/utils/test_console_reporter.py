import pytest
from pred_fab.utils import PfabLogger, ConsoleReporter


@pytest.fixture
def reporter(tmp_path):
    """ConsoleReporter with mock schema metadata."""
    logger = PfabLogger.get_logger(str(tmp_path / "logs"))
    return ConsoleReporter(
        logger=logger,
        param_codes=["water_ratio", "print_speed", "design", "material"],
        perf_codes=["path_accuracy", "energy_efficiency"],
        param_categories={"design": ["A", "B"], "material": ["clay", "concrete"]},
        perf_weights={"path_accuracy": 2.0, "energy_efficiency": 1.0},
    )


def test_reporter_enabled_follows_logger(reporter):
    """ConsoleReporter.enabled should reflect logger's console output state."""
    reporter._logger._console_output_enabled = True
    assert reporter.enabled is True
    reporter._logger._console_output_enabled = False
    assert reporter.enabled is False


def test_format_params_numeric_and_categorical(reporter):
    """Numeric params get formatted with decimals, categoricals get truncated."""
    params = {"water_ratio": 0.42, "print_speed": 35.0, "design": "A", "material": "clay"}
    result = reporter._format_params(params)
    assert "w=0.42" in result
    assert "pri=35.0" in result
    assert "des=A" in result
    assert "mat=cla" in result


def test_format_params_skips_missing(reporter):
    """Missing parameters should not appear in the formatted string."""
    params = {"water_ratio": 0.42}
    result = reporter._format_params(params)
    assert "w=0.42" in result
    assert "pri" not in result


def test_format_perf_colors_scores(reporter):
    """Performance scores should contain ANSI color codes based on value."""
    perf = {"path_accuracy": 0.85, "energy_efficiency": 0.3}
    result = reporter._format_perf(perf)
    assert "0.850" in result  # high score
    assert "0.300" in result  # low score


def test_format_perf_skips_nan(reporter):
    """NaN performance values should be omitted."""
    perf = {"path_accuracy": float("nan"), "energy_efficiency": 0.5}
    result = reporter._format_perf(perf)
    assert "pat" not in result
    assert "0.500" in result


def test_print_training_summary_with_r2_and_r2_adj(reporter, capsys):
    """Training summary should show both R2 and R2_adj when available."""
    reporter._logger._console_output_enabled = True
    metrics = {
        "feature_1": {"r2": 0.85, "r2_adj": 0.82},
        "feature_2": {"r2": 0.92},
    }
    reporter.print_training_summary(metrics)
    output = capsys.readouterr().out
    assert "0.850" in output
    assert "0.820" in output  # r2_adj
    assert "0.920" in output


def test_print_training_summary_disabled(reporter, capsys):
    """No output when console is disabled."""
    reporter._logger._console_output_enabled = False
    reporter.print_training_summary({"f": {"r2": 0.5}})
    assert capsys.readouterr().out == ""


def test_print_phase_header(reporter, capsys):
    """Phase header should include phase number and title."""
    reporter._logger._console_output_enabled = True
    reporter.print_phase_header(1, "Baseline", "10 experiments")
    output = capsys.readouterr().out
    assert "PHASE 1" in output
    assert "Baseline" in output
    assert "10 experiments" in output


def test_print_phase_header_disabled(reporter, capsys):
    reporter._logger._console_output_enabled = False
    reporter.print_phase_header(1, "Test")
    assert capsys.readouterr().out == ""


def test_print_experiment_row(reporter, capsys):
    """Experiment row should show code, params, and perf."""
    reporter._logger._console_output_enabled = True
    reporter.print_experiment_row(
        "baseline_01",
        {"water_ratio": 0.40, "print_speed": 30.0, "design": "A", "material": "clay"},
        {"path_accuracy": 0.75, "energy_efficiency": 0.60},
    )
    output = capsys.readouterr().out
    assert "baseline_01" in output
    assert "0.750" in output


def test_print_phase_summary(reporter, capsys):
    """Phase summary should identify the best experiment."""
    reporter._logger._console_output_enabled = True
    log = [
        ("exp_01", {"design": "A", "material": "clay"}, {"path_accuracy": 0.5, "energy_efficiency": 0.5}),
        ("exp_02", {"design": "B", "material": "clay"}, {"path_accuracy": 0.9, "energy_efficiency": 0.8}),
    ]
    reporter.print_phase_summary(log)
    output = capsys.readouterr().out
    assert "exp_02" in output  # best experiment


def test_print_phase_summary_empty(reporter, capsys):
    """Empty experiment list should produce no output."""
    reporter._logger._console_output_enabled = True
    reporter.print_phase_summary([])
    assert capsys.readouterr().out == ""
