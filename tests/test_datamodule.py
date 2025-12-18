"""
Tests for DataModule - ML preprocessing configuration.

Tests:
- Data extraction from Dataset
- Normalization fitting and application
- Denormalization
- Per-feature normalization overrides
- Batching
- Integration with PredictionSystem
"""

import pytest
import pandas as pd
import numpy as np

from lbp_package.core import Dataset, DatasetSchema, DataModule, ExperimentData
from lbp_package.core.data_objects import DataReal, DataInt, DataArray
from lbp_package.core.data_blocks import DataBlock, Features


@pytest.fixture
def sample_dataset():
    """Create a dataset with experiments containing features."""
    schema = DatasetSchema()
    schema.parameters.add("param_1", DataReal("param_1"))
    schema.parameters.add("param_2", DataInt("param_2"))
    schema.performance.add("accuracy", DataReal("accuracy"))
    
    dataset = Dataset(
        name="test_dataset",
        schema=schema,
        schema_id="test_001",
        local_data=None,
        external_data=None
    )
    
    # Add experiments with features in metric_arrays
    np.random.seed(42)
    
    for i in range(20):
        exp_code = f"exp_{i:03d}"
        
        # Parameters
        param_1 = float(i) * 0.1
        param_2 = i + 10
        
        # Create experiment with metric_arrays initialized
        exp_data = dataset.load_experiment(
            exp_code=exp_code,
            exp_params={"param_1": param_1, "param_2": param_2},
            performance={"accuracy": 0.8 + i * 0.01}
        )
        
        # Initialize metric_arrays (Dataset.add_experiment sets it to None)
        exp_data.features = Features()
        
        # Add scalar features to metric_arrays
        feature_1_arr = DataArray(code="feature_1", shape=())
        feature_2_arr = DataArray(code="feature_2", shape=())
        exp_data.features.add("feature_1", feature_1_arr)
        exp_data.features.add("feature_2", feature_2_arr)
        exp_data.features.set_value("feature_1", np.array(float(i) * 2.0))
        exp_data.features.set_value("feature_2", np.array(float(i) * 0.5 + 10.0))
    
    return dataset


class TestDataModuleBasics:
    """Test basic DataModule functionality."""
    
    def test_initialization(self, sample_dataset):
        """Test DataModule initialization."""
        dm = DataModule(sample_dataset, batch_size=8, normalize='standard')
        
        assert dm.dataset == sample_dataset
        assert dm.batch_size == 8
        assert dm._default_normalize == 'standard'
        assert dm._is_fitted is False
        assert len(dm._feature_overrides) == 0
        assert dm.X_data is not None
        assert dm.y_data is not None
        assert len(dm.X_data) == 20
    
    def test_data_loading(self, sample_dataset):
        """Test extracting all data from dataset."""
        dm = DataModule(sample_dataset, test_size=0.0, val_size=0.0)
        
        # Check shapes
        assert dm.X_data.shape == (20, 2)
        assert dm.y_data.shape == (20, 2)
        
        # Check columns
        assert set(dm.input_columns) == {'param_1', 'param_2'}
        assert set(dm.output_columns) == {'feature_1', 'feature_2'}
        
        # Check values (param_1 is first column, param_2 is second)
        # param_1 = i * 0.1, param_2 = i + 10
        # i=0: 0.0, 10.0
        assert dm.X_data[0, 0] == 0.0
        assert dm.X_data[0, 1] == 10.0
        
        # i=5: 0.5, 15.0
        assert dm.X_data[5, 0] == 0.5
        assert dm.X_data[5, 1] == 15.0
    
    def test_repr(self, sample_dataset):
        """Test string representation."""
        dm = DataModule(sample_dataset, batch_size=16, normalize='standard')
        repr_str = repr(dm)
        
        assert 'standard' in repr_str
        assert 'batch_size=16' in repr_str
        assert 'not fitted' in repr_str
        assert 'overrides=0' in repr_str


class TestNormalization:
    """Test normalization functionality."""
    
    def test_fit_normalize_standard(self, sample_dataset):
        """Test fitting standard normalization."""
        dm = DataModule(sample_dataset, normalize='standard', test_size=0.0, val_size=0.0)
        
        dm._fit_normalize('train')
        
        assert dm._is_fitted is True
        assert 'feature_1' in dm._feature_stats
        assert 'feature_2' in dm._feature_stats
        
        # Check stored statistics
        stats_1 = dm._feature_stats['feature_1']
        assert stats_1['method'] == 'standard'
        assert 'mean' in stats_1
        assert 'std' in stats_1
        
        # Verify mean/std values
        y_vals = dm.y_data[:, dm.output_columns.index('feature_1')]
        assert stats_1['mean'] == pytest.approx(np.mean(y_vals))
        assert stats_1['std'] == pytest.approx(np.std(y_vals))
    
    def test_normalize_application(self, sample_dataset):
        """Test applying standard normalization via get_batches."""
        dm = DataModule(sample_dataset, normalize='standard', test_size=0.0, val_size=0.0)
        
        dm._fit_normalize('train')
        batches = dm.get_batches('train')
        X_batch, y_batch = batches[0]
        
        # Check normalized values have mean~0, std~1
        idx_f1 = dm.output_columns.index('feature_1')
        assert np.mean(y_batch[:, idx_f1]) == pytest.approx(0.0, abs=1e-6)
        assert np.std(y_batch[:, idx_f1]) == pytest.approx(1.0, abs=1e-6)
    
    def test_denormalize_output(self, sample_dataset):
        """Test denormalization."""
        dm = DataModule(sample_dataset, normalize='standard', test_size=0.0, val_size=0.0)
        
        dm._fit_normalize('train')
        batches = dm.get_batches('train')
        _, y_batch = batches[0]
        
        y_denorm = dm.denormalize_output(y_batch)
        
        # Should recover original values
        np.testing.assert_allclose(dm.y_data, y_denorm, rtol=1e-6, atol=1e-6)
    
    def test_fit_normalize_minmax(self, sample_dataset):
        """Test fitting minmax normalization."""
        dm = DataModule(sample_dataset, normalize='minmax', test_size=0.0, val_size=0.0)
        
        dm._fit_normalize('train')
        batches = dm.get_batches('train')
        _, y_batch = batches[0]
        
        # Check values are in [0, 1]
        idx_f1 = dm.output_columns.index('feature_1')
        assert np.min(y_batch[:, idx_f1]) == pytest.approx(0.0, abs=1e-6)
        assert np.max(y_batch[:, idx_f1]) == pytest.approx(1.0, abs=1e-6)
    
    def test_normalize_none(self, sample_dataset):
        """Test no normalization."""
        dm = DataModule(sample_dataset, normalize='none', test_size=0.0, val_size=0.0)
        
        dm._fit_normalize('train')
        batches = dm.get_batches('train')
        _, y_batch = batches[0]
        
        # Should be unchanged
        np.testing.assert_allclose(dm.y_data, y_batch)


class TestPerFeatureOverrides:
    """Test per-feature normalization overrides."""
    
    def test_set_feature_normalize(self, sample_dataset):
        """Test setting per-feature normalization."""
        dm = DataModule(sample_dataset, normalize='standard')
        dm.set_feature_normalize('feature_1', 'minmax')
        
        assert dm._feature_overrides['feature_1'] == 'minmax'
        assert dm.get_normalize_method('feature_1') == 'minmax'
        assert dm.get_normalize_method('feature_2') == 'standard'  # Default
    
    def test_mixed_normalization(self, sample_dataset):
        """Test different normalization per feature."""
        dm = DataModule(sample_dataset, normalize='standard', test_size=0.0, val_size=0.0)
        dm.set_feature_normalize('feature_1', 'minmax')
        
        dm._fit_normalize('train')
        batches = dm.get_batches('train')
        _, y_batch = batches[0]
        
        idx_f1 = dm.output_columns.index('feature_1')
        idx_f2 = dm.output_columns.index('feature_2')
        
        # feature_1 should be minmax [0, 1]
        assert np.min(y_batch[:, idx_f1]) == pytest.approx(0.0, abs=1e-6)
        assert np.max(y_batch[:, idx_f1]) == pytest.approx(1.0, abs=1e-6)
        
        # feature_2 should be standard (mean~0, std~1)
        assert np.mean(y_batch[:, idx_f2]) == pytest.approx(0.0, abs=1e-6)
        assert np.std(y_batch[:, idx_f2]) == pytest.approx(1.0, abs=1e-6)


class TestDataModuleCopy:
    """Test DataModule copying."""
    
    def test_copy(self, sample_dataset):
        """Test deep copy of DataModule."""
        dm = DataModule(sample_dataset, batch_size=8, normalize='standard', test_size=0.0, val_size=0.0)
        dm.set_feature_normalize('feature_1', 'minmax')
        
        dm._fit_normalize('train')
        
        # Create copy
        dm_copy = dm.copy()
        
        # Verify independent copy
        assert dm_copy is not dm
        assert dm_copy.batch_size == dm.batch_size
        assert dm_copy._default_normalize == dm._default_normalize
        assert dm_copy._is_fitted == dm._is_fitted
        
        # Verify deep copy of stats
        assert dm_copy._feature_stats is not dm._feature_stats
        assert dm_copy._feature_stats == dm._feature_stats
        
        # Modify original shouldn't affect copy
        dm.set_feature_normalize('feature_2', 'robust')
        assert 'feature_2' not in dm_copy._feature_overrides


class TestSplits:
    """Test train/val/test splitting functionality."""
    
    def test_default_splits(self, sample_dataset):
        """Test default 80/10/10 split (test=0.2, val=0.1 of remaining)."""
        dm = DataModule(sample_dataset, test_size=0.2, val_size=0.1)
        
        sizes = dm.get_split_sizes()
        
        # Check sizes: 20 total, 4 test (20%), 2 val (10% of 16), 14 train (70%)
        assert sizes['test'] == 4
        assert sizes['val'] == 2  # Approximately 10% of remaining 16
        assert sizes['train'] == 14
        assert sum(sizes.values()) == 20
    
    def test_no_splits(self, sample_dataset):
        """Test with no splits - all data for training."""
        dm = DataModule(sample_dataset, test_size=0.0, val_size=0.0)
        
        sizes = dm.get_split_sizes()
        assert sizes['train'] == 20
        assert sizes['val'] == 0
        assert sizes['test'] == 0
        
        # Val and test batches should be empty
        assert len(dm.get_batches('val')) == 0
        assert len(dm.get_batches('test')) == 0
    
    def test_reproducible_splits(self, sample_dataset):
        """Test that splits are reproducible with same random_seed."""
        dm1 = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=42)
        dm2 = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=42)
        
        # Check indices directly
        assert dm1._split_indices['train'] == dm2._split_indices['train']
    
    def test_different_random_seeds(self, sample_dataset):
        """Test that different seeds produce different splits."""
        dm1 = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=42)
        dm2 = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=99)
        
        # Should NOT be equal
        assert dm1._split_indices['train'] != dm2._split_indices['train']


class TestBatching:
    """Test batching functionality."""
    
    def test_batch_size(self, sample_dataset):
        """Test batching with specific size."""
        dm = DataModule(sample_dataset, batch_size=5, test_size=0.0, val_size=0.0)
        
        batches = dm.get_batches('train')
        
        # 20 samples / 5 = 4 batches
        assert len(batches) == 4
        
        for X_batch, y_batch in batches:
            assert len(X_batch) == 5
            assert len(y_batch) == 5
            
    def test_no_batching(self, sample_dataset):
        """Test with batch_size=None (single batch)."""
        dm = DataModule(sample_dataset, batch_size=None, test_size=0.0, val_size=0.0)
        
        batches = dm.get_batches('train')
        
        assert len(batches) == 1
        X_batch, y_batch = batches[0]
        assert len(X_batch) == 20

