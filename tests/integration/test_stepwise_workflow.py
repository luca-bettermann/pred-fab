from tests.utils.builders import (
    build_prepared_workflow_datamodule,
    build_workflow_stack,
    collect_workflow_local_artifact_paths,
    configure_default_workflow_calibration,
    evaluate_loaded_workflow_experiments,
)


def test_workflow_evaluation_step_populates_features_and_performance(tmp_path):
    agent, dataset, _ = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")

    for exp in dataset.get_all_experiments():
        assert exp.parameters.get_value("param_3") == "B"
        assert exp.features.is_populated("feature_1")
        assert exp.features.is_populated("feature_2")
        assert exp.features.is_populated("feature_3")
        assert exp.performance.has_value("performance_1")
        assert exp.performance.has_value("performance_2")


def test_workflow_save_all_creates_local_artifacts(tmp_path):
    agent, dataset, codes = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    dataset.save_all(recompute_flag=True, verbose_flag=False)

    expected_paths = collect_workflow_local_artifact_paths(
        root_folder=str(tmp_path),
        schema_name=dataset.schema.name,
        exp_codes=codes,
    )
    missing_paths = [str(path) for path in expected_paths if not path.exists()]
    assert missing_paths == []


def test_workflow_exploration_step_runs_from_configured_calibration(tmp_path):
    agent, dataset, _ = build_workflow_stack(tmp_path)
    evaluate_loaded_workflow_experiments(agent=agent, dataset=dataset, category_value="B")
    configure_default_workflow_calibration(agent)

    datamodule = build_prepared_workflow_datamodule(
        agent=agent,
        dataset=dataset,
        val_size=0.0,
        test_size=0.0,
        recompute=True,
    )
    proposal = agent.exploration_step(datamodule=datamodule)

    assert proposal.source_step == "exploration_step"
    assert set(["param_1", "param_2", "dim_1", "dim_2", "param_3"]).issubset(set(proposal.keys()))
    assert proposal["param_3"] == "B"
    assert agent.calibration_system.fixed_params["param_3"] == "B"
    assert agent.calibration_system.param_bounds["param_2"] == (1, 4)
    assert agent.calibration_system.trust_regions["param_1"] == 0.1
