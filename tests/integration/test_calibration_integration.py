import pytest
from unittest.mock import MagicMock
from lbp_package.orchestration.agent import PfabAgent
from lbp_package.orchestration.calibration import CalibrationSystem

@pytest.fixture
def agent(tmp_path):
    root_folder = str(tmp_path)
    local_folder = str(tmp_path / "local")
    log_folder = str(tmp_path / "logs")
    
    agent = PfabAgent(
        root_folder=root_folder,
        local_folder=local_folder,
        log_folder=log_folder,
        debug_flag=True
    )
    return agent

def test_calibration_configuration(agent):
    # Manually initialize calibration system since we are not running full initialize()
    agent.calibration_system = CalibrationSystem(dataset=None, logger=agent.logger)
    
    weights = {'perf1': 0.5, 'perf2': 0.5}
    agent.configure_calibration(weights)
    
    assert agent.calibration_system.performance_weights == weights

def test_propose_experiments_flow(agent):
    # Mock calibration system
    agent.calibration_system = MagicMock(spec=CalibrationSystem)
    agent.calibration_system.propose_new_experiments.return_value = [{'p1': 0.5}]
    agent._initialized = True
    
    param_ranges = {'p1': (0.0, 1.0)}
    experiments = agent.propose_next_experiments(param_ranges)
    
    assert len(experiments) == 1
    assert experiments[0]['p1'] == 0.5
    agent.calibration_system.propose_new_experiments.assert_called_once()
