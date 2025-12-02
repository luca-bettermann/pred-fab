import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from lbp_package.orchestration.calibration import CalibrationSystem, BayesianCalibrationStrategy
from lbp_package.core.dataset import Dataset, ExperimentData
from lbp_package.utils.logger import LBPLogger

@pytest.fixture
def mock_logger():
    return MagicMock(spec=LBPLogger)

@pytest.fixture
def mock_dataset():
    dataset = MagicMock(spec=Dataset)
    dataset.get_experiment_codes.return_value = []
    return dataset

@pytest.fixture
def calibration_system(mock_dataset, mock_logger):
    return CalibrationSystem(mock_dataset, mock_logger, random_seed=42)

def test_initialization(calibration_system):
    assert isinstance(calibration_system.strategy, BayesianCalibrationStrategy)
    assert calibration_system.performance_weights == {}

def test_set_performance_weights(calibration_system):
    weights = {'perf1': 0.5, 'perf2': 0.5}
    calibration_system.set_performance_weights(weights)
    assert calibration_system.performance_weights == weights

def test_compute_system_performance(calibration_system):
    calibration_system.set_performance_weights({'perf1': 0.6, 'perf2': 0.4})
    
    mock_exp = MagicMock(spec=ExperimentData)
    mock_exp.performance = MagicMock()
    mock_exp.performance.has_value.side_effect = lambda k: True
    mock_exp.performance.get_value.side_effect = lambda k: {'perf1': 0.8, 'perf2': 0.5}[k]
    
    score = calibration_system.compute_system_performance(mock_exp)
    # 0.8 * 0.6 + 0.5 * 0.4 = 0.48 + 0.2 = 0.68
    assert score == pytest.approx(0.68)

def test_generate_baseline_experiments(calibration_system):
    param_ranges = {'p1': (0.0, 1.0), 'p2': (10.0, 20.0)}
    experiments = calibration_system.generate_baseline_experiments(n_samples=5, param_ranges=param_ranges)
    
    assert len(experiments) == 5
    for exp in experiments:
        assert 0.0 <= exp['p1'] <= 1.0
        assert 10.0 <= exp['p2'] <= 20.0

def test_propose_new_experiments(calibration_system, mock_dataset):
    # Setup mock history
    mock_dataset.get_experiment_codes.return_value = ['exp1', 'exp2']
    
    exp1 = MagicMock(spec=ExperimentData)
    exp1.parameters = MagicMock()
    exp1.parameters.get_value.side_effect = lambda k: {'p1': 0.1}.get(k)
    exp1.performance = MagicMock()
    exp1.performance.has_value.return_value = True
    exp1.performance.get_value.return_value = 0.5
    
    exp2 = MagicMock(spec=ExperimentData)
    exp2.parameters = MagicMock()
    exp2.parameters.get_value.side_effect = lambda k: {'p1': 0.9}.get(k)
    exp2.performance = MagicMock()
    exp2.performance.has_value.return_value = True
    exp2.performance.get_value.return_value = 0.8
    
    mock_dataset.get_experiment.side_effect = lambda code: {'exp1': exp1, 'exp2': exp2}[code]
    
    calibration_system.set_performance_weights({'perf1': 1.0})
    
    param_ranges = {'p1': (0.0, 1.0)}
    
    # Mock strategy to avoid running actual GP
    with patch.object(calibration_system.strategy, 'propose_next_points') as mock_propose:
        mock_propose.return_value = np.array([[0.5]])
        
        experiments = calibration_system.propose_new_experiments(param_ranges, n_points=1)
        
        assert len(experiments) == 1
        assert experiments[0]['p1'] == 0.5
        
        # Verify strategy was called with correct history
        args, _ = mock_propose.call_args
        X_hist, y_hist, bounds, n_points, mode = args
        assert len(X_hist) == 2
        assert len(y_hist) == 2
        assert np.array_equal(bounds, np.array([[0.0, 1.0]]))
