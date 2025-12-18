"""
Tests for _evaluate_from_params method - enabling calibration workflows.

Tests demonstrate that evaluation can be performed from raw parameters
without requiring a full ExperimentData instance.
"""

import pytest
import numpy as np
import tempfile
from dataclasses import dataclass

from lbp_package.core import Dataset, DatasetSchema, Parameter, Dimension
from lbp_package.orchestration import EvaluationSystem
from lbp_package.interfaces import IEvaluationModel, IFeatureModel
from lbp_package.utils import LBPLogger


@dataclass
class DummyFeatureModel(IFeatureModel):
    """Minimal feature model for testing."""
    
    def _load_data(self, **params):
        """No data loading needed for test."""
        return None
    
    def _compute_feature_logic(self, data, **params) -> dict:
        """Return simple function of parameters."""
        temp = params.get('temp', 200)
        speed = params.get('speed', 50)
        layer = params.get('layer', 0)
        return {'value': temp / 100 + speed / 10 + layer * 0.1}
    
    def _compute_feature_values(self, feature_name: str, visualize: bool = False, **params) -> float:
        """Return simple function of parameters."""
        data = self._load_data(**params)
        features = self._compute_feature_logic(data, **params)
        return features['value']


@dataclass
class DummyEvaluationModel(IEvaluationModel):
    """Minimal evaluation model for testing."""
    
    @property
    def feature_model_class(self):
        return DummyFeatureModel
    
    def _compute_target_value(self, **params) -> float:
        """Target is always 5.0."""
        return 5.0
    
    def _compute_scaling_factor(self, **params) -> float:
        """Scaling factor is 1.0."""
        return 1.0
    
    def compute_performance(self, exp_data) -> dict:
        """Return empty dict - aggregation happens automatically."""
        return {}


def test_evaluate_from_params_basic():
    """Test that _evaluate_from_params works without ExperimentData."""
    # Setup schema and dataset
    schema = DatasetSchema()
    schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
    schema.parameters.add('speed', Parameter.real(min_val=10, max_val=100))
    schema.parameters.add('n_layers', Dimension.integer(
        code='n_layers', dim_name='layers', iterator_code='layer',
        min_val=1, max_val=10
    ))
    schema.performance.add('deviation', Parameter.real())
    
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LBPLogger('test', tmpdir)
        eval_system = EvaluationSystem(dataset, logger)
        
        # Register evaluation model
        eval_model = DummyEvaluationModel(logger=logger)
        eval_system.add_evaluation_model(
            performance_code='deviation',
            evaluation_model=eval_model,
            feature_model_class=DummyFeatureModel
        )
        
        # Call _evaluate_from_params directly with raw parameters
        params = {
            'temp': 200.0,
            'speed': 50.0,
            'n_layers': 3
        }
        
        perf_results, metric_results = eval_system._compute_features_from_params(
            params=params,
            evaluate_from=0,
            evaluate_to=None,
            visualize=False,
            recompute=False
        )
        
        # Verify results
        assert 'deviation' in perf_results
        assert isinstance(perf_results['deviation'], (int, float))
        assert 'deviation_feature' in metric_results
        assert metric_results['deviation_feature'].shape == (3,)  # 3 layers


def test_evaluate_from_params_for_calibration():
    """Test that _evaluate_from_params can be used in calibration loop."""
    # Setup
    schema = DatasetSchema()
    schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
    schema.parameters.add('speed', Parameter.real(min_val=10, max_val=100))
    schema.parameters.add('n_layers', Dimension.integer(
        code='n_layers', dim_name='layers', iterator_code='layer',
        min_val=1, max_val=10
    ))
    schema.performance.add('deviation', Parameter.real())
    
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LBPLogger('test', tmpdir)
        eval_system = EvaluationSystem(dataset, logger)
        
        eval_model = DummyEvaluationModel(logger=logger)
        eval_system.add_evaluation_model(
            performance_code='deviation',
            evaluation_model=eval_model,
            feature_model_class=DummyFeatureModel
        )
        
        # Simulate calibration optimization loop
        candidate_params = [
            {'temp': 180.0, 'speed': 40.0, 'n_layers': 2},
            {'temp': 200.0, 'speed': 50.0, 'n_layers': 3},
            {'temp': 220.0, 'speed': 60.0, 'n_layers': 4},
        ]
        
        performances = []
        for params in candidate_params:
            perf_results, _ = eval_system._compute_features_from_params(
                params=params,
                evaluate_from=0,
                evaluate_to=None,
                visualize=False
            )
            performances.append(perf_results['deviation'])
        
        # Verify we got valid performance values for all candidates
        assert len(performances) == 3
        assert all(isinstance(p, (int, float)) for p in performances)
        assert all(not np.isnan(p) for p in performances)


def test_evaluate_experiment_wrapper_consistency():
    """Test that evaluate_experiment uses _evaluate_from_params internally."""
    # Setup
    schema = DatasetSchema()
    schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
    schema.parameters.add('n_layers', Dimension.integer(
        code='n_layers', dim_name='layers', iterator_code='layer',
        min_val=1, max_val=10
    ))
    schema.performance.add('deviation', Parameter.real())
    
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LBPLogger('test', tmpdir)
        eval_system = EvaluationSystem(dataset, logger)
        
        eval_model = DummyEvaluationModel(logger=logger)
        eval_system.add_evaluation_model(
            performance_code='deviation',
            evaluation_model=eval_model,
            feature_model_class=DummyFeatureModel
        )
        
        # Create exp_data
        exp_data = dataset.load_experiment('exp1', {'temp': 200.0, 'n_layers': 2})
        
        # Method 1: Call evaluate_experiment (wrapper)
        eval_system.compute_exp_features(exp_data)
        perf_from_wrapper = exp_data.performance.get_value('deviation')
        
        # Method 2: Call _evaluate_from_params directly
        params = {'temp': 200.0, 'n_layers': 2}
        perf_results, _ = eval_system._compute_features_from_params(params=params)
        perf_from_core = perf_results['deviation']
        
        # Both methods should produce the same result
        assert np.isclose(perf_from_wrapper, perf_from_core)
