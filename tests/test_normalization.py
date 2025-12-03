"""
Tests for DataModule normalization functionality including one-hot encoding.
"""
import pytest
import pandas as pd
import numpy as np

from lbp_package.core import (
    Dataset,
    DatasetSchema,
    Parameter,
    Dimension,
    DataArray,
    DataModule,
)


@pytest.fixture
def categorical_dataset():
    """Dataset with categorical parameters for testing one-hot encoding."""
    schema = DatasetSchema()
    schema.parameters.add('optimizer', Parameter.categorical(['adam', 'sgd', 'rmsprop']))
    schema.parameters.add('learning_rate', Parameter.real(min_val=0.001, max_val=0.1))
    schema.parameters.add('batch_size', Parameter.integer(min_val=16, max_val=256))
    schema.features.add('loss', DataArray(name='loss', shape=(1,)))
    
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    # Add experiments with different optimizers
    configs = [
        ('adam', 0.01, 32),
        ('sgd', 0.02, 64),
        ('rmsprop', 0.03, 128),
        ('adam', 0.015, 32),
        ('sgd', 0.025, 64),
    ]
    
    for i, (opt, lr, bs) in enumerate(configs):
        exp = dataset.load_experiment(f'exp_{i}', {
            'optimizer': opt,
            'learning_rate': lr,
            'batch_size': bs
        })
        exp.features.set_value('loss', np.array([0.5 - i * 0.05]))
    
    return dataset


@pytest.fixture
def dimensional_dataset():
    """Dataset with dimensional parameters for testing dimension normalization."""
    schema = DatasetSchema()
    schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
    schema.parameters.add('n_layers', Dimension.integer(
        param_name='n_layers', dim_name='layers', iterator_name='layer',
        min_val=1, max_val=10
    ))
    schema.features.add('deviation', DataArray(name='deviation', shape=(-1,)))
    
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    # Add experiments
    for i in range(3):
        n_layers = 5
        exp = dataset.load_experiment(f'exp_{i}', {
            'temp': 200.0 + i * 10,
            'n_layers': n_layers
        })
        exp.features.set_value('deviation', np.random.rand(n_layers) * 0.1)
    
    return dataset


@pytest.fixture
def mixed_dataset():
    """Dataset with mixed parameter types."""
    schema = DatasetSchema()
    schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
    schema.parameters.add('use_cooling', Parameter.boolean())
    schema.parameters.add('material', Parameter.categorical(['PLA', 'ABS', 'PETG']))
    schema.parameters.add('speed', Parameter.integer(min_val=10, max_val=100))
    schema.features.add('quality', DataArray(name='quality', shape=(1,)))
    
    dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
    
    configs = [
        (200.0, True, 'PLA', 50),
        (220.0, False, 'ABS', 60),
        (210.0, True, 'PETG', 55),
        (230.0, False, 'PLA', 70),
    ]
    
    for i, (temp, cooling, mat, speed) in enumerate(configs):
        exp = dataset.load_experiment(f'exp_{i}', {
            'temp': temp,
            'use_cooling': cooling,
            'material': mat,
            'speed': speed
        })
        exp.features.set_value('quality', np.array([0.8 + i * 0.02]))
    
    return dataset


class TestCategoricalOneHot:
    """Test one-hot encoding for categorical parameters."""
    
    def test_categorical_detection(self, categorical_dataset):
        """Test that categorical parameters are detected correctly."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0)
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        # Should detect 'optimizer' as categorical
        assert 'optimizer' in dm._categorical_mappings
        assert dm._categorical_mappings['optimizer'] == ['adam', 'rmsprop', 'sgd']
    
    def test_onehot_encoding(self, categorical_dataset):
        """Test that categorical values are one-hot encoded."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0)
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        
        # Original categorical column should be replaced with one-hot columns
        assert 'optimizer' not in X_norm.columns
        assert 'optimizer_adam' in X_norm.columns
        assert 'optimizer_sgd' in X_norm.columns
        assert 'optimizer_rmsprop' in X_norm.columns
        
        # Check one-hot values are binary
        for col in ['optimizer_adam', 'optimizer_sgd', 'optimizer_rmsprop']:
            assert X_norm[col].isin([0.0, 1.0]).all()
        
        # Each row should have exactly one 1.0
        onehot_cols = ['optimizer_adam', 'optimizer_sgd', 'optimizer_rmsprop']
        assert (X_norm[onehot_cols].sum(axis=1) == 1.0).all()
    
    def test_onehot_encoding_values(self, categorical_dataset):
        """Test that one-hot encoding produces correct values."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0)
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        
        # First row has 'adam'
        assert X_norm.iloc[0]['optimizer_adam'] == 1.0
        assert X_norm.iloc[0]['optimizer_sgd'] == 0.0
        assert X_norm.iloc[0]['optimizer_rmsprop'] == 0.0
        
        # Second row has 'sgd'
        assert X_norm.iloc[1]['optimizer_adam'] == 0.0
        assert X_norm.iloc[1]['optimizer_sgd'] == 1.0
        assert X_norm.iloc[1]['optimizer_rmsprop'] == 0.0
        
        # Third row has 'rmsprop'
        assert X_norm.iloc[2]['optimizer_adam'] == 0.0
        assert X_norm.iloc[2]['optimizer_sgd'] == 0.0
        assert X_norm.iloc[2]['optimizer_rmsprop'] == 1.0
    
    def test_onehot_decoding(self, categorical_dataset):
        """Test that one-hot encoded values can be decoded back."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0)
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        X_denorm = dm.denormalize_parameters(X_norm)
        
        # Should reconstruct original categorical column
        assert 'optimizer' in X_denorm.columns
        assert 'optimizer_adam' not in X_denorm.columns
        
        # Values should match original
        assert X['optimizer'].tolist() == X_denorm['optimizer'].tolist()
    
    def test_onehot_roundtrip(self, categorical_dataset):
        """Test complete round-trip: categorical → one-hot → categorical."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0)
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        X_denorm = dm.denormalize_parameters(X_norm)
        
        # All columns should match
        assert set(X.columns) == set(X_denorm.columns)
        
        # All values should match
        for col in X.columns:
            if X[col].dtype == 'object':
                assert X[col].tolist() == X_denorm[col].tolist()
            else:
                assert np.allclose(X[col], X_denorm[col])


class TestNormalizeStrategy:
    """Test normalize_strategy property on DataObjects."""
    
    def test_datareal_strategy(self):
        """DataReal should use 'default' strategy."""
        param = Parameter.real(min_val=0, max_val=100)
        param.name = 'temp'
        assert param.normalize_strategy == 'default'
    
    def test_dataint_strategy(self):
        """DataInt should use 'default' strategy."""
        param = Parameter.integer(min_val=0, max_val=10)
        param.name = 'count'
        assert param.normalize_strategy == 'default'
    
    def test_databool_strategy(self):
        """DataBool should use 'none' strategy."""
        param = Parameter.boolean()
        param.name = 'active'
        assert param.normalize_strategy == 'none'
    
    def test_datacategorical_strategy(self):
        """DataCategorical should use 'categorical' strategy."""
        param = Parameter.categorical(['A', 'B', 'C'])
        param.name = 'type'
        assert param.normalize_strategy == 'categorical'
    
    def test_datadimension_strategy(self):
        """DataDimension should use 'minmax' strategy."""
        dim = Dimension.integer('n_layers', 'layers', 'layer', min_val=1, max_val=10)
        assert dim.normalize_strategy == 'minmax'


class TestDimensionalNormalization:
    """Test normalization of dimensional parameters."""
    
    def test_dimension_minmax_normalization(self, dimensional_dataset):
        """Dimensional indices should be normalized with minmax."""
        dm = DataModule(dimensional_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        # 'layer' should use minmax (from DataDimension.normalize_strategy)
        assert dm.get_parameter_normalize_method('layer') == 'minmax'
        
        # 'temp' should use standard (DataModule default)
        assert dm.get_parameter_normalize_method('temp') == 'standard'
    
    def test_dimension_normalization_values(self, dimensional_dataset):
        """Test that dimensional indices are normalized to [0, 1]."""
        dm = DataModule(dimensional_dataset, test_size=0.0, val_size=0.0)
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        
        # Layer indices should be in [0, 1] range after minmax
        assert X_norm['layer'].min() >= 0.0
        assert X_norm['layer'].max() <= 1.0


class TestMixedParameterTypes:
    """Test normalization with mixed parameter types."""
    
    def test_mixed_normalization(self, mixed_dataset):
        """Test normalization with real, bool, categorical, and int parameters."""
        dm = DataModule(mixed_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        
        # Boolean should be unchanged (strategy = 'none')
        assert 'use_cooling' in X_norm.columns
        assert X_norm['use_cooling'].isin([True, False]).all()
        
        # Categorical should be one-hot encoded
        assert 'material' not in X_norm.columns
        assert 'material_PLA' in X_norm.columns
        assert 'material_ABS' in X_norm.columns
        assert 'material_PETG' in X_norm.columns
        
        # Real and int should be normalized (standard)
        assert 'temp' in X_norm.columns
        assert 'speed' in X_norm.columns
    
    def test_mixed_roundtrip(self, mixed_dataset):
        """Test round-trip with mixed parameter types."""
        dm = DataModule(mixed_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        X_denorm = dm.denormalize_parameters(X_norm)
        
        # Boolean values should be preserved
        assert X['use_cooling'].tolist() == X_denorm['use_cooling'].tolist()
        
        # Categorical values should be preserved
        assert X['material'].tolist() == X_denorm['material'].tolist()
        
        # Numeric values should be approximately preserved
        assert np.allclose(X['temp'], X_denorm['temp'])
        assert np.allclose(X['speed'], X_denorm['speed'])


class TestNormalizationOverrides:
    """Test user override functionality."""
    
    def test_parameter_override(self, mixed_dataset):
        """Test that user can override parameter normalization."""
        dm = DataModule(mixed_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        
        # Override temp to use minmax instead of standard
        dm.set_parameter_normalize('temp', 'minmax')
        
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        # Check that temp uses minmax
        assert dm.get_parameter_normalize_method('temp') == 'minmax'
        assert 'temp' in dm._parameter_stats
        assert dm._parameter_stats['temp']['method'] == 'minmax'
    
    def test_feature_override(self, mixed_dataset):
        """Test that user can override feature normalization."""
        dm = DataModule(mixed_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        
        # Override quality to use robust
        dm.set_feature_normalize('quality', 'robust')
        
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        # Check that quality uses robust
        assert dm.get_normalize_method('quality') == 'robust'
        assert dm._feature_stats['quality']['method'] == 'robust'
    
    def test_override_to_none(self, categorical_dataset):
        """Test that user can disable normalization for a parameter."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        
        # Override learning_rate to 'none'
        dm.set_parameter_normalize('learning_rate', 'none')
        
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        
        # learning_rate should be unchanged
        assert np.array_equal(X['learning_rate'].values, X_norm['learning_rate'].values)


class TestSeparateNormalizeMethods:
    """Test separate normalize_parameters and normalize_features methods."""
    
    def test_normalize_parameters_only(self, categorical_dataset):
        """Test normalizing only parameters."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        
        # X should be normalized
        assert 'optimizer_adam' in X_norm.columns
        assert X_norm['learning_rate'].mean() != X['learning_rate'].mean()
    
    def test_normalize_features_only(self, categorical_dataset):
        """Test normalizing only features."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        y_norm = dm.normalize_features(y)
        
        # y should be normalized
        assert abs(y_norm['loss'].mean()) < 1e-10  # Close to 0
    
    def test_denormalize_parameters_only(self, categorical_dataset):
        """Test denormalizing only parameters."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        X_norm = dm.normalize_parameters(X)
        X_denorm = dm.denormalize_parameters(X_norm)
        
        # Should reconstruct original X
        assert X['optimizer'].tolist() == X_denorm['optimizer'].tolist()
        assert np.allclose(X['learning_rate'], X_denorm['learning_rate'])
    
    def test_denormalize_features_only(self, categorical_dataset):
        """Test denormalizing only features."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0, normalize='standard')
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        y_norm = dm.normalize_features(y)
        y_denorm = dm.denormalize_features(y_norm)
        
        # Should reconstruct original y
        assert np.allclose(y['loss'], y_denorm['loss'])


class TestEdgeCases:
    """Test edge cases in normalization."""
    
    def test_empty_categorical_mappings(self, dimensional_dataset):
        """Test normalization when no categorical parameters exist."""
        dm = DataModule(dimensional_dataset, test_size=0.0, val_size=0.0)
        X, y = dm.get_split('train')
        dm.fit_normalize(X, y)
        
        # Should have no categorical mappings
        assert len(dm._categorical_mappings) == 0
        
        # Normalization should still work
        X_norm = dm.normalize_parameters(X)
        X_denorm = dm.denormalize_parameters(X_norm)
        
        assert np.allclose(X['temp'], X_denorm['temp'])
    
    def test_unfitted_normalization_error(self, categorical_dataset):
        """Test that normalizing before fitting raises error."""
        dm = DataModule(categorical_dataset, test_size=0.0, val_size=0.0)
        X, y = dm.get_split('train')
        
        # Should raise error if not fitted
        with pytest.raises(RuntimeError, match="not fitted"):
            dm.normalize_parameters(X)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            dm.normalize_features(y)
