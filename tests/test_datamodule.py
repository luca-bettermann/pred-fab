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
from lbp_package.core.data_objects import DataReal, DataInt
from lbp_package.core.data_blocks import DataBlock


@pytest.fixture
def sample_dataset():
    """Create a dataset with experiments containing features."""
    schema = DatasetSchema()
    schema.parameters.add("param_1", DataReal("param_1"))
    schema.parameters.add("param_2", DataInt("param_2"))
    schema.performance_attrs.add("accuracy", DataReal("accuracy"))
    
    dataset = Dataset(
        name="test_dataset",
        schema=schema,
        schema_id="test_001",
        local_data=None,
        external_data=None
    )
    
    # Add experiments with features
    np.random.seed(42)
    for i in range(20):
        exp_code = f"exp_{i:03d}"
        
        # Parameters
        param_1 = float(i) * 0.1
        param_2 = i + 10
        
        # Create experiment
        exp_data = dataset.add_experiment(
            exp_code=exp_code,
            exp_params={"param_1": param_1, "param_2": param_2},
            performance={"accuracy": 0.8 + i * 0.01}
        )
        
        # Add features
        exp_data.features = DataBlock()
        exp_data.features.add("feature_1", DataReal("feature_1"))
        exp_data.features.add("feature_2", DataReal("feature_2"))
        
        exp_data.features.set_value("feature_1", float(i) * 2.0)
        exp_data.features.set_value("feature_2", float(i) * 0.5 + 10.0)
    
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
    
    def test_extract_all(self, sample_dataset):
        """Test extracting all data from dataset."""
        dm = DataModule(sample_dataset)
        X, y = dm.extract_all()
        
        # Check shapes
        assert len(X) == 20
        assert len(y) == 20
        
        # Check columns
        assert set(X.columns) == {'param_1', 'param_2'}
        assert set(y.columns) == {'feature_1', 'feature_2'}
        
        # Check values
        assert X.iloc[0]['param_1'] == 0.0
        assert X.iloc[5]['param_2'] == 15
        assert y.iloc[0]['feature_1'] == 0.0
        assert y.iloc[10]['feature_2'] == 15.0
    
    def test_get_feature_names(self, sample_dataset):
        """Test getting feature names from dataset."""
        dm = DataModule(sample_dataset)
        features = dm.get_feature_names()
        
        assert set(features) == {'feature_1', 'feature_2'}
    
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
        dm = DataModule(sample_dataset, normalize='standard')
        _, y = dm.extract_all()
        
        dm.fit_normalize(y)
        
        assert dm._is_fitted is True
        assert 'feature_1' in dm._feature_stats
        assert 'feature_2' in dm._feature_stats
        
        # Check stored statistics
        stats_1 = dm._feature_stats['feature_1']
        assert stats_1['method'] == 'standard'
        assert 'mean' in stats_1
        assert 'std' in stats_1
        
        # Verify mean/std values
        assert stats_1['mean'] == pytest.approx(y['feature_1'].mean())
        assert stats_1['std'] == pytest.approx(y['feature_1'].std())
    
    def test_normalize_standard(self, sample_dataset):
        """Test applying standard normalization."""
        dm = DataModule(sample_dataset, normalize='standard')
        _, y = dm.extract_all()
        
        dm.fit_normalize(y)
        y_norm = dm.normalize(y)
        
        # Check normalized values have mean~0, std~1
        assert y_norm['feature_1'].mean() == pytest.approx(0.0, abs=1e-10)
        assert y_norm['feature_1'].std() == pytest.approx(1.0, abs=1e-6)
        assert y_norm['feature_2'].mean() == pytest.approx(0.0, abs=1e-10)
        assert y_norm['feature_2'].std() == pytest.approx(1.0, abs=1e-6)
    
    def test_denormalize_standard(self, sample_dataset):
        """Test denormalization."""
        dm = DataModule(sample_dataset, normalize='standard')
        _, y = dm.extract_all()
        
        dm.fit_normalize(y)
        y_norm = dm.normalize(y)
        y_denorm = dm.denormalize(y_norm)
        
        # Should recover original values
        pd.testing.assert_frame_equal(y, y_denorm, rtol=1e-6, atol=1e-6)
    
    def test_fit_normalize_minmax(self, sample_dataset):
        """Test minmax normalization."""
        dm = DataModule(sample_dataset, normalize='minmax')
        _, y = dm.extract_all()
        
        dm.fit_normalize(y)
        y_norm = dm.normalize(y)
        
        # Check values are in [0, 1]
        assert y_norm['feature_1'].min() == pytest.approx(0.0, abs=1e-8)
        assert y_norm['feature_1'].max() == pytest.approx(1.0, abs=1e-8)
        assert y_norm['feature_2'].min() == pytest.approx(0.0, abs=1e-8)
        assert y_norm['feature_2'].max() == pytest.approx(1.0, abs=1e-8)
    
    def test_fit_normalize_robust(self, sample_dataset):
        """Test robust normalization."""
        dm = DataModule(sample_dataset, normalize='robust')
        _, y = dm.extract_all()
        
        dm.fit_normalize(y)
        
        stats = dm._feature_stats['feature_1']
        assert stats['method'] == 'robust'
        assert 'median' in stats
        assert 'q1' in stats
        assert 'q3' in stats
    
    def test_normalize_none(self, sample_dataset):
        """Test no normalization."""
        dm = DataModule(sample_dataset, normalize='none')
        _, y = dm.extract_all()
        
        dm.fit_normalize(y)
        y_norm = dm.normalize(y)
        
        # Should be unchanged
        pd.testing.assert_frame_equal(y, y_norm)
    
    def test_normalize_before_fit_raises(self, sample_dataset):
        """Test that normalize() before fit_normalize() raises error."""
        dm = DataModule(sample_dataset, normalize='standard')
        _, y = dm.extract_all()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            dm.normalize(y)


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
        dm = DataModule(sample_dataset, normalize='standard')
        dm.set_feature_normalize('feature_1', 'minmax')
        
        _, y = dm.extract_all()
        dm.fit_normalize(y)
        y_norm = dm.normalize(y)
        
        # feature_1 should be minmax [0, 1]
        assert y_norm['feature_1'].min() == pytest.approx(0.0, abs=1e-8)
        assert y_norm['feature_1'].max() == pytest.approx(1.0, abs=1e-8)
        
        # feature_2 should be standard (mean~0, std~1)
        assert y_norm['feature_2'].mean() == pytest.approx(0.0, abs=1e-10)
        assert y_norm['feature_2'].std() == pytest.approx(1.0, abs=1e-6)
    
    def test_override_to_none(self, sample_dataset):
        """Test disabling normalization for specific feature."""
        dm = DataModule(sample_dataset, normalize='standard')
        dm.set_feature_normalize('feature_1', 'none')
        
        _, y = dm.extract_all()
        dm.fit_normalize(y)
        y_norm = dm.normalize(y)
        
        # feature_1 should be unchanged
        assert (y_norm['feature_1'] == y['feature_1']).all()
        
        # feature_2 should be normalized
        assert y_norm['feature_2'].mean() == pytest.approx(0.0, abs=1e-10)


class TestBatching:
    """Test batching functionality."""
    
    def test_no_batching(self, sample_dataset):
        """Test single batch (default)."""
        dm = DataModule(sample_dataset)
        batches = dm.get_batches()
        
        assert len(batches) == 1
        X, y = batches[0]
        assert len(X) == 20
        assert len(y) == 20
    
    def test_batching(self, sample_dataset):
        """Test multiple batches."""
        dm = DataModule(sample_dataset, batch_size=8)
        batches = dm.get_batches()
        
        # 20 samples / 8 per batch = 3 batches
        assert len(batches) == 3
        
        # Check batch sizes
        assert len(batches[0][0]) == 8
        assert len(batches[1][0]) == 8
        assert len(batches[2][0]) == 4  # Remainder
    
    def test_batch_size_larger_than_data(self, sample_dataset):
        """Test batch size larger than dataset."""
        dm = DataModule(sample_dataset, batch_size=100)
        batches = dm.get_batches()
        
        assert len(batches) == 1
        assert len(batches[0][0]) == 20


class TestDataModuleCopy:
    """Test DataModule copying."""
    
    def test_copy(self, sample_dataset):
        """Test deep copy of DataModule."""
        dm = DataModule(sample_dataset, batch_size=8, normalize='standard')
        dm.set_feature_normalize('feature_1', 'minmax')
        
        _, y = dm.extract_all()
        dm.fit_normalize(y)
        
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


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self):
        """Test DataModule with empty dataset."""
        schema = DatasetSchema()
        schema.parameters.add("param_1", DataReal("param_1"))
        
        dataset = Dataset(
            name="empty",
            schema=schema,
            schema_id="empty_001",
            local_data=None,
            external_data=None
        )
        
        dm = DataModule(dataset)
        
        with pytest.raises(ValueError, match="No experiments with features"):
            dm.extract_all()
    
    def test_denormalize_before_fit(self, sample_dataset):
        """Test denormalize before fitting returns unchanged data."""
        dm = DataModule(sample_dataset, normalize='standard')
        _, y = dm.extract_all()
        
        # Should return copy without error
        y_denorm = dm.denormalize(y)
        pd.testing.assert_frame_equal(y, y_denorm)
    
    def test_partial_features(self, sample_dataset):
        """Test normalization with subset of features."""
        dm = DataModule(sample_dataset, normalize='standard')
        _, y = dm.extract_all()
        
        # Fit on all features
        dm.fit_normalize(y)
        
        # Normalize only one feature
        y_subset = y[['feature_1']]
        y_norm = dm.normalize(y_subset)
        
        assert y_norm.shape == y_subset.shape
        assert y_norm['feature_1'].mean() == pytest.approx(0.0, abs=1e-10)


class TestSplits:
    """Test train/val/test splitting functionality."""
    
    def test_default_splits(self, sample_dataset):
        """Test default 80/10/10 split (test=0.2, val=0.1 of remaining)."""
        dm = DataModule(sample_dataset, test_size=0.2, val_size=0.1)
        
        X_all, y_all = dm.extract_all()  # 20 total
        X_train, y_train = dm.extract_all(split='train')
        X_val, y_val = dm.extract_all(split='val')
        X_test, y_test = dm.extract_all(split='test')
        
        # Check sizes: 20 total, 4 test (20%), 2 val (10% of 16), 14 train (70%)
        assert len(X_test) == 4
        assert len(X_val) == 2  # Approximately 10% of remaining 16
        assert len(X_train) == 14
        assert len(X_all) == 20
    
    def test_no_splits(self, sample_dataset):
        """Test with no splits - all data for training."""
        dm = DataModule(sample_dataset, test_size=0.0, val_size=0.0)
        
        X_train, y_train = dm.extract_all(split='train')
        
        assert len(X_train) == 20
        
        # Val and test should be empty
        with pytest.raises(ValueError, match="Split 'val' is empty"):
            dm.extract_all(split='val')
        
        with pytest.raises(ValueError, match="Split 'test' is empty"):
            dm.extract_all(split='test')
    
    def test_only_test_split(self, sample_dataset):
        """Test with only test split, no validation."""
        dm = DataModule(sample_dataset, test_size=0.2, val_size=0.0)
        
        X_train, y_train = dm.extract_all(split='train')
        X_test, y_test = dm.extract_all(split='test')
        
        assert len(X_test) == 4
        assert len(X_train) == 16
        
        # Val should be empty
        with pytest.raises(ValueError, match="Split 'val' is empty"):
            dm.extract_all(split='val')
    
    def test_reproducible_splits(self, sample_dataset):
        """Test that splits are reproducible with same random_seed."""
        dm1 = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=42)
        dm2 = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=42)
        
        X_train1, _ = dm1.extract_all(split='train')
        X_train2, _ = dm2.extract_all(split='train')
        
        pd.testing.assert_frame_equal(X_train1, X_train2)
    
    def test_different_random_seeds(self, sample_dataset):
        """Test that different seeds produce different splits."""
        dm1 = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=42)
        dm2 = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=99)
        
        X_train1, _ = dm1.extract_all(split='train')
        X_train2, _ = dm2.extract_all(split='train')
        
        # Should NOT be equal (very unlikely to be same with different seeds)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(X_train1, X_train2)
    
    def test_get_split_sizes(self, sample_dataset):
        """Test get_split_sizes() method."""
        dm = DataModule(sample_dataset, test_size=0.2, val_size=0.1, random_seed=42)
        
        sizes = dm.get_split_sizes()
        
        assert sizes['train'] == 14
        assert sizes['val'] == 2
        assert sizes['test'] == 4
        assert sum(sizes.values()) == 20
    
    def test_invalid_split_name(self, sample_dataset):
        """Test error on invalid split name."""
        dm = DataModule(sample_dataset)
        
        with pytest.raises(ValueError, match="Invalid split 'invalid'"):
            dm.extract_all(split='invalid')
    
    def test_repr_includes_splits(self, sample_dataset):
        """Test __repr__ shows split configuration."""
        dm = DataModule(sample_dataset, test_size=0.2, val_size=0.1, normalize='standard')
        
        repr_str = repr(dm)
        
        assert 'test=0.2' in repr_str
        assert 'val=0.1' in repr_str
        assert "normalize='standard'" in repr_str
