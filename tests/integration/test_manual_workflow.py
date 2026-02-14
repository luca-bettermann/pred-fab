from tests.workflows.manual_workflow import run_workflow


def test_manual_workflow_runs_end_to_end(tmp_path):
    result = run_workflow(str(tmp_path))

    assert "param_1" in result
    assert "param_2" in result
    assert "dim_1" in result
    assert "dim_2" in result
    assert "param_3" in result
    assert result["param_3"] == "B"
