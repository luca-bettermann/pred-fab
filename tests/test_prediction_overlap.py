import numpy as np
import pandas as pd
import pytest
from lbp_package.core import Dataset, DatasetSchema, ExperimentData, Parameter, Dimension, DataModule
from lbp_package.orchestration import PredictionSystem
import tempfile
import os
from lbp_package.utils import LBPLogger
from lbp_package.interfaces import IPredictionModel


class TrackingModel(IPredictionModel):
    """Model that tracks predictions and returns the position index directly."""
    def __init__(self, logger):
        super().__init__(logger)
        self.prediction_calls = []  # Track all forward_pass calls
    
    @property
    def predicted_features(self):
        return ['value']
    
    @property
    def features_as_input(self):
        return []
    
    def train(self, X, y, **kwargs):
        pass
    
    def forward_pass(self, X):
        # Track which positions were in this batch
        positions = X['n'].values.tolist()
        self.prediction_calls.append(positions)
        # Return the position index as the prediction value
        return pd.DataFrame({'value': X['n'].values.astype(float)})
    
    def _get_model_artifacts(self):
        return {}
    
    def _set_model_artifacts(self, artifacts):
        pass


def test_prediction_overlap_batching():
    """Test that overlap parameter correctly includes overlapping positions in batches."""
    schema = DatasetSchema()
    schema.parameters.add('n', Dimension.integer(
        param_name='n', dim_name='n', iterator_name='n', min_val=1, max_val=20
    ))
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    datamodule = DataModule(dataset, test_size=0.0, val_size=0.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LBPLogger('test', tmpdir)
        system = PredictionSystem(dataset, logger)
        model = TrackingModel(logger)
        system.add_prediction_model(model)
        
        # Fake training (datamodule needs to be set)
        system.datamodule = datamodule
        
        # Predict with overlap: 10 positions, batch_size=4, overlap=2
        # Expected batches:
        # Batch 0: [0, 1, 2, 3] (positions 0-3)
        # Batch 1: [2, 3, 4, 5] (positions 2-5, with 2 overlap from previous)
        # Batch 2: [4, 5, 6, 7] (positions 4-7, with 2 overlap from previous)
        # Batch 3: [6, 7, 8, 9] (positions 6-9, with 2 overlap from previous)
        result = system._predict_from_params({'n': 10}, batch_size=4, overlap=2)
        
        # Check result shape and values
        assert result['value'].shape == (10,)
        # Model returns position index, so values should be [0, 1, 2, ..., 9]
        assert np.allclose(result['value'], np.arange(10))
        
        # Verify batch calls with overlap
        assert len(model.prediction_calls) == 3  # 3 batches to cover 10 positions
        assert model.prediction_calls[0] == [0, 1, 2, 3]
        assert model.prediction_calls[1] == [2, 3, 4, 5, 6, 7]  # overlap + full batch
        assert model.prediction_calls[2] == [6, 7, 8, 9]  # overlap + remaining


def test_prediction_no_overlap():
    """Test that batching works correctly without overlap."""
    schema = DatasetSchema()
    schema.parameters.add('n', Dimension.integer(
        param_name='n', dim_name='n', iterator_name='n', min_val=1, max_val=20
    ))
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    datamodule = DataModule(dataset, test_size=0.0, val_size=0.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LBPLogger('test', tmpdir)
        system = PredictionSystem(dataset, logger)
        model = TrackingModel(logger)
        system.add_prediction_model(model)
        
        system.datamodule = datamodule
        
        # Predict without overlap: 10 positions, batch_size=4, overlap=0
        # Expected batches:
        # Batch 0: [0, 1, 2, 3]
        # Batch 1: [4, 5, 6, 7]
        # Batch 2: [8, 9]
        result = system._predict_from_params({'n': 10}, batch_size=4, overlap=0)
        
        assert result['value'].shape == (10,)
        assert np.allclose(result['value'], np.arange(10))
        
        # Verify batches without overlap
        assert len(model.prediction_calls) == 3
        assert model.prediction_calls[0] == [0, 1, 2, 3]
        assert model.prediction_calls[1] == [4, 5, 6, 7]
        assert model.prediction_calls[2] == [8, 9]


def test_prediction_overlap_validation():
    """Test that overlap parameter validation works correctly."""
    schema = DatasetSchema()
    schema.parameters.add('n', Dimension.integer(
        param_name='n', dim_name='n', iterator_name='n', min_val=1, max_val=20
    ))
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    datamodule = DataModule(dataset, test_size=0.0, val_size=0.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LBPLogger('test', tmpdir)
        system = PredictionSystem(dataset, logger)
        model = TrackingModel(logger)
        system.add_prediction_model(model)
        system.datamodule = datamodule
        
        # Negative overlap should raise error
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            system._predict_from_params({'n': 10}, batch_size=4, overlap=-1)
        
        # Overlap >= batch_size should raise error for multiple batches
        with pytest.raises(ValueError, match="overlap must be less than batch_size"):
            system._predict_from_params({'n': 10}, batch_size=4, overlap=4)
