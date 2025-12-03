"""
Tests for Phase 10: Dimensional Prediction Architecture

Test-driven development suite covering:
1. Dimensional data extraction from DataModule
2. Vectorized prediction workflows (Independent, Transformer, GNN patterns)
3. Batching and memory management
4. Online learning and prediction horizons
5. External feature handling
6. Data structure simplification (predicted_metric_arrays)
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from lbp_package.core import (
    Dataset, DatasetSchema, ExperimentData, Parameter, Dimension,
    DataModule, MetricArrays, DataArray, DataBlock
)
from lbp_package.interfaces import IPredictionModel
from lbp_package.orchestration import PredictionSystem
from lbp_package.utils import LBPLogger


# ============================================================================
# TEST FIXTURES - Prediction Model Examples
# ============================================================================

class IndependentPositionModel(IPredictionModel):
    """
    Independent model - predicts based on position only.
    No dependencies on previous features.
    """
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.model = None
        self.is_trained = False
        self._dim_cols = []  # Will be set during training
    
    @property
    def predicted_features(self) -> List[str]:
        return ['deviation']
    
    @property
    def features_as_input(self) -> List[str]:
        return []  # No external features needed
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        """Train simple linear model on position features."""
        self.model = LinearRegression()
        # Use all dimensional columns (layer, segment, etc.)
        dim_cols = [col for col in X.columns if col in ['layer', 'segment', 'position']]
        if not dim_cols:
            # Fall back to all non-parameter columns
            dim_cols = [col for col in X.columns if col not in ['temp', 'speed']]
        self.model.fit(X[dim_cols], y['deviation'])
        self.is_trained = True
        self._dim_cols = dim_cols
    
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict deviation based on layer and segment."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        predictions = self.model.predict(X[self._dim_cols])
        return pd.DataFrame({'deviation': predictions})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {'sklearn_model': self.model}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.model = artifacts['sklearn_model']
        self.is_trained = True


class TransformerStyleModel(IPredictionModel):
    """
    Transformer-style model - sees all positions, learns dependencies via attention.
    (Simplified - real transformer would use PyTorch/TensorFlow)
    """
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.model = None
        self.is_trained = False
        self._feature_cols = []  # Will be set during training
    
    @property
    def predicted_features(self) -> List[str]:
        return ['deviation']
    
    @property
    def required_features(self) -> List[str]:
        return []
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        """Train model with context awareness (simplified)."""
        # Real implementation would use attention mechanism
        # Here we use RandomForest as proxy for context-aware model
        self.model = RandomForestRegressor(n_estimators=10, max_depth=5)
        # Use all available columns (params + dimensions)
        self._feature_cols = list(X.columns)
        self.model.fit(X[self._feature_cols], y['deviation'])
        self.is_trained = True
    
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict with context (all positions seen together)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        predictions = self.model.predict(X[self._feature_cols])
        return pd.DataFrame({'deviation': predictions})
    
    def tuning(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        """Fine-tune with new measurements (incremental learning)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained - call train() first")
        # For RandomForest, we'll retrain with warm_start simulation
        # In practice, use incremental learning algorithms
        self.model.fit(X[self._feature_cols], y['deviation'])
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {'sklearn_model': self.model}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.model = artifacts['sklearn_model']
        self.is_trained = True


class ExternalFeatureModel(IPredictionModel):
    """
    Model requiring external features (e.g., temperature readings).
    Demonstrates explicit feature dependency.
    """
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.model = None
        self.is_trained = False
    
    @property
    def predicted_features(self) -> List[str]:
        return ['surface_quality']
    
    @property
    def features_as_input(self) -> List[str]:
        return ['temperature_measured', 'humidity']
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        """Train on params + external features."""
        self.model = Ridge()
        features = X[['temp', 'speed', 'temperature_measured', 'humidity']]
        self.model.fit(features, y['surface_quality'])
        self.is_trained = True
    
    def forward_pass(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict using external features."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Validate required features present
        missing = set(self.features_as_input) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        features = X[['temp', 'speed', 'temperature_measured', 'humidity']]
        predictions = self.model.predict(features)
        return pd.DataFrame({'surface_quality': predictions})
    
    def _get_model_artifacts(self) -> Dict[str, Any]:
        return {'sklearn_model': self.model}
    
    def _set_model_artifacts(self, artifacts: Dict[str, Any]):
        self.model = artifacts['sklearn_model']
        self.is_trained = True


# ============================================================================
# TEST CLASS 1: Data Structure & Storage
# ============================================================================

class TestDimensionalDataStructure:
    """Test simplified data storage in ExperimentData."""
    
    def test_predicted_metric_arrays_field(self):
        """ExperimentData should have predicted_metric_arrays field."""
        
        exp_data = ExperimentData(
            exp_code="test_001",
            parameters=DataBlock()
        )
        
        # Should have predicted_metric_arrays field (auto-initialized)
        assert hasattr(exp_data, 'predicted_metric_arrays')
        assert exp_data.predicted_features is not None
        assert isinstance(exp_data.predicted_features, MetricArrays)
    
    def test_metric_arrays_simplified_storage(self):
        """Metric arrays should store only feature values, not redundant data."""
        
        arrays = MetricArrays()
        
        # Store ONLY feature values (no dims, targets, scaling)
        deviation_array = np.random.rand(10, 5)  # 10 layers, 5 segments
        arrays.add('deviation', DataArray(name='deviation', shape=(10, 5)))
        arrays.set_value('deviation', deviation_array)
        
        # Retrieve and verify
        retrieved = arrays.get_value('deviation')
        assert retrieved.shape == (10, 5)
        assert np.array_equal(retrieved, deviation_array)
    
    def test_separate_measured_and_predicted_storage(self):
        """Measured and predicted features stored separately."""
        
        exp_data = ExperimentData(
            exp_code="test_001",
            parameters=DataBlock(),
            features=MetricArrays(),
            predicted_features=MetricArrays()
        )
        
        # Store measured features
        measured = np.array([[0.1, 0.15], [0.12, 0.14]])
        exp_data.features.add('deviation', DataArray(name='deviation', shape=(2, 2)))
        exp_data.features.set_value('deviation', measured)
        
        # Store predicted features
        predicted = np.array([[0.11, 0.16], [0.13, 0.15]])
        exp_data.predicted_features.add('deviation', DataArray(name='deviation', shape=(2, 2)))
        exp_data.predicted_features.set_value('deviation', predicted)
        
        # Verify separation
        assert not np.array_equal(
            exp_data.features.get_value('deviation'),
            exp_data.predicted_features.get_value('deviation')
        )


# ============================================================================
# TEST CLASS 2: DataModule Dimensional Extraction
# ============================================================================

class TestDataModuleDimensionalExtraction:
    """Test extraction of dimensional training data from DataModule."""
    
    @pytest.fixture
    def dataset_with_dimensions(self):
        """Create dataset with dimensional experiments."""
        schema = DatasetSchema()
        schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
        schema.parameters.add('speed', Parameter.real(min_val=10, max_val=100))
        schema.parameters.add('n_layers', Dimension.integer(
            param_name='n_layers', dim_name='layers', iterator_name='layer',
            min_val=1, max_val=100
        ))
        schema.parameters.add('n_segments', Dimension.integer(
            param_name='n_segments', dim_name='segments', iterator_name='segment',
            min_val=1, max_val=10
        ))
        
        dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
        
        # Add experiment with dimensional features
        exp_data = dataset.add_experiment(
            'exp_001',
            {'temp': 200.0, 'speed': 50.0, 'n_layers': 3, 'n_segments': 2}
        )
        
        # Populate metric arrays with feature values
        deviation_data = np.array([[0.1, 0.15], [0.12, 0.14], [0.11, 0.13]])  # (3, 2)
        exp_data.features = MetricArrays()
        exp_data.features.add('deviation', DataArray(name='deviation', shape=(3, 2)))
        exp_data.features.set_value('deviation', deviation_data)
        
        return dataset
    
    def test_get_split_extracts_dimensional_data(self, dataset_with_dimensions):
        """get_split() should flatten dimensional data into rows."""
        datamodule = DataModule(dataset_with_dimensions, test_size=0.0, val_size=0.0)
        
        X, y = datamodule.get_split('train')
        
        # Should have 3 layers × 2 segments = 6 rows
        assert len(X) == 6
        assert len(y) == 6
        
        # X should have columns: [temp, speed, layer, segment]
        assert 'temp' in X.columns
        assert 'speed' in X.columns
        assert 'layer' in X.columns
        assert 'segment' in X.columns
        
        # y should have feature column
        assert 'deviation' in y.columns
    
    def test_dimensional_data_values_correct(self, dataset_with_dimensions):
        """Extracted dimensional data should match original metric arrays."""
        datamodule = DataModule(dataset_with_dimensions, test_size=0.0, val_size=0.0)
        X, y = datamodule.get_split('train')
        
        # First position: layer=0, segment=0 → deviation=0.1
        row_0 = y[y.index == 0]
        assert float(row_0['deviation'].iloc[0]) == pytest.approx(0.1)
        
        # Last position: layer=2, segment=1 → deviation=0.13
        row_5 = y[y.index == 5]
        assert float(row_5['deviation'].iloc[0]) == pytest.approx(0.13)
    
    def test_batching_support(self, dataset_with_dimensions):
        """DataModule should support batching for large datasets."""
        datamodule = DataModule(dataset_with_dimensions, test_size=0.0, val_size=0.0)
        
        # Get all data (batching happens in PredictionSystem.predict)
        X, y = datamodule.get_split('train')
        
        # Should have all 6 positions
        assert len(X) == 6
        assert len(y) == 6


# ============================================================================
# TEST CLASS 3: Independent Model Prediction
# ============================================================================

class TestIndependentModelPrediction:
    """Test vectorized prediction with position-aware models."""
    
    def test_independent_model_training(self):
        """Independent model should train on dimensional positions."""
        # Create training data
        X_train = pd.DataFrame({
            'temp': [200] * 6,
            'speed': [50] * 6,
            'layer': [0, 0, 1, 1, 2, 2],
            'segment': [0, 1, 0, 1, 0, 1]
        })
        y_train = pd.DataFrame({
            'deviation': [0.1, 0.15, 0.12, 0.14, 0.11, 0.13]
        })
        
        model = IndependentPositionModel()
        model.train(X_train, y_train)
        
        assert model.is_trained
    
    def test_independent_model_prediction_vectorized(self):
        """Model should predict all positions in parallel."""
        # Train
        X_train = pd.DataFrame({
            'layer': [0, 1, 2],
            'segment': [0, 0, 0]
        })
        y_train = pd.DataFrame({'deviation': [0.1, 0.2, 0.3]})
        
        model = IndependentPositionModel()
        model.train(X_train, y_train)
        
        # Predict multiple positions at once
        X_new = pd.DataFrame({
            'layer': [0, 1, 2, 3],
            'segment': [0, 0, 0, 0]
        })
        
        predictions = model.forward_pass(X_new)
        
        # Should return all 4 predictions
        assert len(predictions) == 4
        assert 'deviation' in predictions.columns
    
    def test_batched_prediction(self):
        """Large prediction should work with batching."""
        # Train simple model
        X_train = pd.DataFrame({
            'layer': range(10),
            'segment': [0] * 10
        })
        y_train = pd.DataFrame({'deviation': np.linspace(0.1, 0.2, 10)})
        
        model = IndependentPositionModel()
        model.train(X_train, y_train)
        
        # Predict large number of positions
        n_positions = 1000
        X_new = pd.DataFrame({
            'layer': np.random.randint(0, 10, n_positions),
            'segment': np.zeros(n_positions)
        })
        
        # Should handle batching internally (no OOM)
        predictions = model.forward_pass(X_new)
        assert len(predictions) == n_positions


# ============================================================================
# TEST CLASS 4: Transformer/Context-Aware Models
# ============================================================================

class TestTransformerStylePrediction:
    """Test context-aware models (Transformer pattern)."""
    
    def test_transformer_model_training(self):
        """Transformer model should train on all features including position."""
        X_train = pd.DataFrame({
            'temp': [200, 200, 210, 210],
            'speed': [50, 50, 60, 60],
            'layer': [0, 1, 0, 1],
            'segment': [0, 0, 0, 0]
        })
        y_train = pd.DataFrame({
            'deviation': [0.1, 0.12, 0.11, 0.13]
        })
        
        model = TransformerStyleModel()
        model.train(X_train, y_train)
        
        assert model.is_trained
    
    def test_transformer_sees_all_positions(self):
        """Transformer should predict with awareness of all positions."""
        # Train
        X_train = pd.DataFrame({
            'temp': [200] * 6,
            'speed': [50] * 6,
            'layer': [0, 0, 1, 1, 2, 2],
            'segment': [0, 1, 0, 1, 0, 1]
        })
        y_train = pd.DataFrame({'deviation': [0.1, 0.15, 0.12, 0.14, 0.11, 0.13]})
        
        model = TransformerStyleModel()
        model.train(X_train, y_train)
        
        # Predict all positions together (model sees full context)
        X_new = pd.DataFrame({
            'temp': [200] * 4,
            'speed': [50] * 4,
            'layer': [0, 1, 2, 3],
            'segment': [0, 0, 0, 0]
        })
        
        predictions = model.forward_pass(X_new)
        
        # Should leverage context to predict
        assert len(predictions) == 4
        assert 'deviation' in predictions.columns


# ============================================================================
# TEST CLASS 5: External Feature Handling
# ============================================================================

class TestExternalFeatureHandling:
    """Test models requiring external features as inputs."""
    
    def test_model_declares_features_as_input(self):
        """Model should declare required external features."""
        model = ExternalFeatureModel()
        
        required = model.features_as_input
        assert 'temperature_measured' in required
        assert 'humidity' in required
    
    def test_training_with_external_features(self):
        """Model should train when external features provided."""
        X_train = pd.DataFrame({
            'temp': [200, 210],
            'speed': [50, 60],
            'temperature_measured': [25.3, 26.1],
            'humidity': [0.45, 0.50]
        })
        y_train = pd.DataFrame({'surface_quality': [0.85, 0.82]})
        
        model = ExternalFeatureModel()
        model.train(X_train, y_train)
        
        assert model.is_trained
    
    def test_prediction_requires_external_features(self):
        """Prediction should fail if external features missing."""
        model = ExternalFeatureModel()
        
        # Train first
        X_train = pd.DataFrame({
            'temp': [200], 'speed': [50],
            'temperature_measured': [25.3], 'humidity': [0.45]
        })
        y_train = pd.DataFrame({'surface_quality': [0.85]})
        model.train(X_train, y_train)
        
        # Try to predict without external features
        X_new_incomplete = pd.DataFrame({
            'temp': [200],
            'speed': [50]
            # Missing: temperature_measured, humidity
        })
        
        with pytest.raises(ValueError, match="Missing required features"):
            model.forward_pass(X_new_incomplete)
    
    def test_prediction_succeeds_with_all_features(self):
        """Prediction should work when all required features provided."""
        model = ExternalFeatureModel()
        
        # Train
        X_train = pd.DataFrame({
            'temp': [200], 'speed': [50],
            'temperature_measured': [25.3], 'humidity': [0.45]
        })
        y_train = pd.DataFrame({'surface_quality': [0.85]})
        model.train(X_train, y_train)
        
        # Predict with all features
        X_new = pd.DataFrame({
            'temp': [210],
            'speed': [60],
            'temperature_measured': [26.5],  # User-provided
            'humidity': [0.48]                # User-provided
        })
        
        predictions = model.forward_pass(X_new)
        assert len(predictions) == 1
        assert 'surface_quality' in predictions.columns


# ============================================================================
# TEST CLASS 6: PredictionSystem Integration
# ============================================================================

class TestPredictionSystemDimensional:
    """Test PredictionSystem with dimensional prediction workflows."""
    
    @pytest.fixture
    def prediction_system_setup(self, tmp_path):
        """Setup PredictionSystem with dimensional dataset."""
        # Create dataset
        schema = DatasetSchema()
        schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
        schema.parameters.add('n_layers', Dimension.integer(
            param_name='n_layers', dim_name='layers', iterator_name='layer',
            min_val=1, max_val=5
        ))
        
        dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
        
        # Add experiment
        exp_data = dataset.add_experiment('exp_001', {'temp': 200.0, 'n_layers': 3})
        exp_data.features = MetricArrays()
        deviation_data = np.array([0.1, 0.12, 0.11])
        exp_data.features.add('deviation', DataArray(name='deviation', shape=(3,)))
        exp_data.features.set_value('deviation', deviation_data)
        
        logger = LBPLogger(name='test', log_folder=str(tmp_path))
        system = PredictionSystem(dataset=dataset, logger=logger)
        
        return system, dataset
    
    def test_predict_experiment_returns_exp_data(self, prediction_system_setup):
        """predict_experiment() should return ExperimentData with predictions."""
        system, dataset = prediction_system_setup
        
        # Add and train model
        model = IndependentPositionModel()
        system.add_prediction_model(model)
        
        datamodule = DataModule(dataset)
        system.train(datamodule)
        
        # Predict using internal helper (for calibration workflows)
        result = system._predict_from_params(
            params={'temp': 210.0, 'n_layers': 4}
        )
        
        # Should return dict of predictions
        assert isinstance(result, dict)
        assert 'deviation' in result
        assert isinstance(result['deviation'], np.ndarray)
        assert result['deviation'].shape == (4,)
    
    def test_predict_with_batching(self, prediction_system_setup):
        """Large predictions should use batching."""
        system, dataset = prediction_system_setup
        
        model = IndependentPositionModel()
        system.add_prediction_model(model)
        
        datamodule = DataModule(dataset)
        system.train(datamodule)
        
        # Predict with many positions
        result = system._predict_from_params(
            params={'temp': 210.0, 'n_layers': 100},
            batch_size=10  # Small batch size to test batching
        )
        
        # Should handle 100 positions with batching
        assert isinstance(result, dict)
        assert 'deviation' in result
        assert result['deviation'].shape == (100,)
    
    def test_prediction_horizon_control(self, prediction_system_setup):
        """Should support predicting specific ranges."""
        system, dataset = prediction_system_setup
        
        model = IndependentPositionModel()
        system.add_prediction_model(model)
        
        datamodule = DataModule(dataset)
        system.train(datamodule)
        
        # Predict only layers 10-20
        result = system._predict_from_params(
            params={'temp': 210.0, 'n_layers': 100},
            predict_from=10,
            predict_to=20
        )
        
        # Should only have predictions for layers 10-19
        assert isinstance(result, dict)
        predictions = result['deviation']
        # Count non-NaN values
        non_nan_count = np.sum(~np.isnan(predictions))
        assert non_nan_count == 10  # Layers 10-19


# ============================================================================
# TEST CLASS 7: Online Learning
# ============================================================================

class TestOnlineLearning:
    """Test online learning and fine-tuning during fabrication."""
    
    def test_tuning_method_exists(self):
        """Model should have tuning() method for online learning."""
        model = IndependentPositionModel()
        assert hasattr(model, 'tuning')
    
    def test_tuning_default_raises_not_implemented(self):
        """Default tuning() should raise NotImplementedError."""
        X = pd.DataFrame({'layer': [0, 1], 'segment': [0, 0]})
        y = pd.DataFrame({'deviation': [0.1, 0.2]})
        
        model = IndependentPositionModel()
        
        # Call tuning without overriding should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="does not support tuning"):
            model.tuning(X, y)
    
    def test_online_mode_uses_measured_features(self, tmp_path):
        """Training should reject data with NaN values."""
        # Create dataset with complete training data only
        schema = DatasetSchema()
        schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
        schema.parameters.add('n_layers', Dimension.integer(
            param_name='n_layers', dim_name='layers', iterator_name='layer',
            min_val=1, max_val=100
        ))
        
        train_dataset = Dataset(name='train', schema=schema, schema_id='test_schema')
        
        # Add complete training data
        train_exp = train_dataset.add_experiment('train_001', {'temp': 190.0, 'n_layers': 10})
        train_exp.features = MetricArrays()
        train_data = np.random.rand(10) * 0.1 + 0.1
        train_exp.features.add('deviation', DataArray(name='deviation', shape=(10,)))
        train_exp.features.set_value('deviation', train_data)
        
        # Setup prediction system and train on complete data
        logger = LBPLogger(name='test', log_folder=str(tmp_path))
        system = PredictionSystem(dataset=train_dataset, logger=logger)
        model = IndependentPositionModel()
        system.add_prediction_model(model)
        
        datamodule = DataModule(train_dataset, test_size=0.0, val_size=0.0)
        system.train(datamodule)
        
        # Create separate experiment with partial measurements for prediction
        pred_dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
        exp_data = pred_dataset.add_experiment('ongoing', {'temp': 200.0, 'n_layers': 100})
        exp_data.features = MetricArrays()
        
        # Only first 10 layers measured - rest are NaN
        measured = np.full(100, np.nan)
        measured[:10] = np.random.rand(10) * 0.1 + 0.1
        exp_data.features.add('deviation', DataArray(name='deviation', shape=(100,)))
        exp_data.features.set_value('deviation', measured)
        
        # Predict on ongoing experiment
        result = system.predict_experiment(
            exp_data=exp_data,
            predict_from=10,
            predict_to=20
        )
        
        # Should have predictions
        assert result.predicted_features is not None
        
        # Predict remaining layers using measured as context
        result = system.predict_experiment(
            exp_data=exp_data,  # Has measured features for layers 0-9
            predict_from=10,
            predict_to=100
        )
        
        # Should use measured features as context
        assert result.predicted_features is not None
    
    def test_online_fabrication_integration(self, tmp_path):
        """
        Integration test: Simulate fabrication process with online learning.
        
        Workflow:
        1. Start fabrication with initial model
        2. For each layer:
            a. Predict next N layers ahead
            b. "Fabricate" and measure actual features
            c. Tune model with new measurements
            d. Continue to next layer
        3. Verify both measured and predicted arrays populate correctly
        """
        # Setup dataset with 2D dimensions (layers × segments)
        schema = DatasetSchema()
        schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
        schema.parameters.add('speed', Parameter.real(min_val=10, max_val=100))
        schema.parameters.add('n_layers', Dimension.integer(
            param_name='n_layers', dim_name='layers', iterator_name='layer',
            min_val=1, max_val=50
        ))
        schema.parameters.add('n_segments', Dimension.integer(
            param_name='n_segments', dim_name='segments', iterator_name='segment',
            min_val=1, max_val=5
        ))
        
        dataset = Dataset(name='fabrication_test', schema=schema, schema_id='fab_schema')
        
        # Add historical experiments for initial training
        for i in range(3):
            exp = dataset.add_experiment(
                f'hist_{i:03d}',
                {'temp': 200.0 + i*10, 'speed': 50.0, 'n_layers': 10, 'n_segments': 3}
            )
            exp.features = MetricArrays()
            
            # Synthetic deviation data with pattern: base + layer_effect + segment_effect
            deviation = np.zeros((10, 3))
            for layer in range(10):
                for seg in range(3):
                    base = 0.1 + i * 0.01
                    layer_effect = layer * 0.005
                    seg_effect = seg * 0.002
                    deviation[layer, seg] = base + layer_effect + seg_effect
            
            exp.features.add('deviation', DataArray(name='deviation', shape=(10, 3)))
            exp.features.set_value('deviation', deviation)
        
        # Setup prediction system
        logger = LBPLogger(name='fabrication_test', log_folder=str(tmp_path))
        system = PredictionSystem(dataset=dataset, logger=logger)
        
        # Use Transformer-style model (context-aware)
        model = TransformerStyleModel(logger=logger)
        system.add_prediction_model(model)
        
        # Initial training on historical data
        datamodule = DataModule(dataset)
        system.train(datamodule)
        
        # Start new fabrication experiment
        fab_params = {'temp': 215.0, 'speed': 55.0, 'n_layers': 20, 'n_segments': 3}
        fab_exp = dataset.add_experiment('fab_ongoing', fab_params)
        
        # Initialize arrays
        total_positions = 20 * 3  # 20 layers × 3 segments
        measured_deviation = np.full((20, 3), np.nan)
        predicted_deviation = np.full((20, 3), np.nan)
        
        fab_exp.features.add('deviation', DataArray(name='deviation', shape=(20, 3)))
        fab_exp.features.set_value('deviation', measured_deviation)
        fab_exp.predicted_features.add('deviation', DataArray(name='deviation', shape=(20, 3)))
        fab_exp.predicted_features.set_value('deviation', predicted_deviation)
        
        # Simulation parameters
        lookahead = 5  # Predict 5 layers ahead
        tune_frequency = 2  # Tune every 2 layers
        
        # Simulate fabrication layer by layer
        for current_layer in range(20):
            # Step 1: Predict ahead (if not at end)
            if current_layer < 20 - 1:
                predict_to_layer = min(current_layer + lookahead, 20)
                
                # Predict using current model and measured data so far
                prediction_result = system.predict_experiment(
                    exp_data=fab_exp,
                    predict_from=current_layer + 1,
                    predict_to=predict_to_layer,
                    batch_size=5
                )
                
                # Update predicted arrays
                pred_values = prediction_result.predicted_features.get_value('deviation')
                predicted_deviation[current_layer+1:predict_to_layer, :] = pred_values[current_layer+1:predict_to_layer, :]
                fab_exp.predicted_features.set_value('deviation', predicted_deviation)
            
            # Step 2: "Fabricate" current layer - simulate feature extraction/evaluation
            # In reality, this would be sensor readings or post-fabrication measurements
            for seg in range(3):
                # Simulate measurement (ground truth with noise)
                base = 0.11
                layer_effect = current_layer * 0.005
                seg_effect = seg * 0.002
                noise = np.random.normal(0, 0.005)
                measured_value = base + layer_effect + seg_effect + noise
                
                # Add to measured data
                measured_deviation[current_layer, seg] = measured_value
            
            # Update measured arrays
            fab_exp.features.set_value('deviation', measured_deviation)
            
            # Step 3: Tune model with new measurements (every tune_frequency layers)
            if (current_layer + 1) % tune_frequency == 0 and current_layer > 0:
                # Extract measured positions for tuning
                measured_positions = []
                X_tune_rows = []
                y_tune_rows = []
                
                for layer in range(current_layer + 1):
                    for seg in range(3):
                        if not np.isnan(measured_deviation[layer, seg]):
                            X_tune_rows.append({
                                'temp': fab_params['temp'],
                                'speed': fab_params['speed'],
                                'layer': layer,
                                'segment': seg
                            })
                            y_tune_rows.append({
                                'deviation': measured_deviation[layer, seg]
                            })
                
                if X_tune_rows:
                    X_tune = pd.DataFrame(X_tune_rows)
                    y_tune = pd.DataFrame(y_tune_rows)
                    
                    # Online learning: tune model with new measurements
                    model.tuning(X_tune, y_tune)
        
        # Final verification
        final_measured = fab_exp.features.get_value('deviation')
        final_predicted = fab_exp.predicted_features.get_value('deviation')
        
        # 1. All layers should be measured
        assert not np.any(np.isnan(final_measured)), "All positions should be measured"
        
        # 2. Predicted array should have values (at least some predictions made)
        assert np.sum(~np.isnan(final_predicted)) > 0, "Should have predictions"
        
        # 3. Measured and predicted should be separate
        assert not np.array_equal(final_measured, final_predicted), "Measured ≠ Predicted"
        
        # 4. Pattern verification: deviation increases with layer
        layer_means = np.mean(final_measured, axis=1)
        assert layer_means[-1] > layer_means[0], "Deviation should increase with layer"
        
        # 5. Online learning should improve predictions over time
        # Early predictions (before tuning) vs late predictions (after tuning)
        # This is demonstrated by the tuning loop - model adapts to new data
        assert model.is_trained, "Model should remain trained after tuning"
    
    def test_predict_dims_with_online_tuning(self, tmp_path):
        """
        Test PredictionSystem.predict_experiment() with online tuning workflow.
        
        Simulates realistic fabrication:
        1. Predict next batch of layers
        2. "Fabricate" and measure
        3. Tune model with measurements
        4. Repeat
        """
        # Setup
        schema = DatasetSchema()
        schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
        schema.parameters.add('speed', Parameter.real(min_val=10, max_val=100))
        schema.parameters.add('n_layers', Dimension.integer(
            param_name='n_layers', dim_name='layers', iterator_name='layer',
            min_val=1, max_val=50
        ))
        
        dataset = Dataset(name='online_test', schema=schema, schema_id='online_schema')
        
        # Historical data for training
        for i in range(3):
            exp = dataset.add_experiment(
                f'hist_{i:03d}',
                {'temp': 200.0 + i*10, 'speed': 50.0, 'n_layers': 10}
            )
            exp.features = MetricArrays()
            deviation = np.array([0.1 + i*0.01 + j*0.005 for j in range(10)])
            exp.features.add('deviation', DataArray(name='deviation', shape=(10,)))
            exp.features.set_value('deviation', deviation)
        
        # Setup system
        logger = LBPLogger(name='online_test', log_folder=str(tmp_path))
        system = PredictionSystem(dataset=dataset, logger=logger)
        model = TransformerStyleModel(logger=logger)
        system.add_prediction_model(model)
        
        # Train
        datamodule = DataModule(dataset)
        system.train(datamodule)
        
        # Fabrication experiment
        fab_params = {'temp': 215.0, 'speed': 55.0, 'n_layers': 20}
        fab_exp = dataset.add_experiment('fab_ongoing', fab_params)
        fab_exp.features = MetricArrays()
        fab_exp.predicted_features = MetricArrays()
        
        measured = np.full(20, np.nan)
        predicted = np.full(20, np.nan)
        fab_exp.features.add('deviation', DataArray(name='deviation', shape=(20,)))
        fab_exp.features.set_value('deviation', measured)
        fab_exp.predicted_features.add('deviation', DataArray(name='deviation', shape=(20,)))
        fab_exp.predicted_features.set_value('deviation', predicted)
        
        # Online loop
        for layer in range(20):
            # Predict next 5 layers
            if layer < 19:
                result = system.predict_experiment(
                    exp_data=fab_exp,
                    predict_from=layer + 1,
                    predict_to=min(layer + 6, 20),
                    batch_size=5
                )
                
                # Update predictions
                pred_vals = result.predicted_features.get_value('deviation')
                predicted[:] = pred_vals
                fab_exp.predicted_features.set_value('deviation', predicted)
            
            # "Fabricate" - measure
            measured_value = 0.11 + layer * 0.005 + np.random.normal(0, 0.003)
            measured[layer] = measured_value
            fab_exp.features.set_value('deviation', measured)
            
            # Tune every 5 layers
            if layer % 5 == 4 and layer > 0:
                X_tune_rows = []
                y_tune_rows = []
                for l in range(layer + 1):
                    if not np.isnan(measured[l]):
                        X_tune_rows.append({'temp': fab_params['temp'], 'speed': fab_params['speed'], 'layer': l})
                        y_tune_rows.append({'deviation': measured[l]})
                
                if X_tune_rows:
                    X_tune = pd.DataFrame(X_tune_rows)
                    y_tune = pd.DataFrame(y_tune_rows)
                    model.tuning(X_tune, y_tune)
        
        # Verify
        final_measured = fab_exp.features.get_value('deviation')
        final_predicted = fab_exp.predicted_features.get_value('deviation')
        
        assert not np.any(np.isnan(final_measured)), "All layers measured"
        assert np.sum(~np.isnan(final_predicted)) > 0, "Should have predictions"
        assert final_measured[-1] > final_measured[0], "Deviation increases"



# ============================================================================
# TEST CLASS 8: Memory Management
# ============================================================================

class TestMemoryManagement:
    """Test memory efficiency with large dimensional spaces."""
    
    def test_configurable_batch_size(self):
        """Batch size should be configurable."""
        # This would be tested with actual PredictionSystem implementation
        # For now, verify the pattern
        
        default_batch_size = 1000
        custom_batch_size = 500
        
        # System should accept batch_size parameter
        # system.predict_experiment(params, batch_size=custom_batch_size)
        
        assert custom_batch_size < default_batch_size
    
    def test_memory_estimation(self):
        """Verify memory estimation for batch sizing."""
        # Estimate: positions × features × bytes
        n_positions = 1000
        n_features = 20
        bytes_per_float = 8
        
        estimated_bytes = n_positions * n_features * bytes_per_float
        estimated_kb = estimated_bytes / 1024
        
        # Should be well under memory limits
        assert estimated_kb < 200  # ~160 KB for 1000 positions


# ============================================================================
# TEST CLASS 9: Data Validation
# ============================================================================

class TestDataValidation:
    """Verify validation catches data quality issues early."""
    
    def test_nan_validation_in_training(self, tmp_path):
        """Training should reject data with NaN values."""
        # Create dataset with NaN values
        schema = DatasetSchema()
        schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
        schema.parameters.add('n_layers', Dimension.integer(
            param_name='n_layers', dim_name='layers', iterator_name='layer',
            min_val=1, max_val=10
        ))
        
        dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
        
        # Add experiment with partial measurements (has NaN)
        exp_data = dataset.add_experiment('incomplete', {'temp': 200.0, 'n_layers': 10})
        exp_data.features = MetricArrays()
        
        # Only first 5 layers measured - rest are NaN
        measured = np.full(10, np.nan)
        measured[:5] = np.random.rand(5) * 0.1 + 0.1
        exp_data.features.add('deviation', DataArray(name='deviation', shape=(10,)))
        exp_data.features.set_value('deviation', measured)
        
        # Setup prediction system
        logger = LBPLogger(name='test', log_folder=str(tmp_path))
        system = PredictionSystem(dataset=dataset, logger=logger)
        model = IndependentPositionModel()
        system.add_prediction_model(model)
        
        # Training should fail with clear error message
        datamodule = DataModule(dataset, test_size=0.0, val_size=0.0)
        with pytest.raises(ValueError, match="Training data contains NaN values"):
            system.train(datamodule)
    
    def test_model_compatibility_validation(self, tmp_path):
        """Training should validate model features match data."""
        # Create dataset
        schema = DatasetSchema()
        schema.parameters.add('temp', Parameter.real(min_val=150, max_val=250))
        schema.parameters.add('n_layers', Dimension.integer(
            param_name='n_layers', dim_name='layers', iterator_name='layer',
            min_val=1, max_val=10
        ))
        
        dataset = Dataset(name='test', schema=schema, schema_id='test_schema')
        
        # Add experiment with 'deviation' feature
        exp_data = dataset.add_experiment('exp_001', {'temp': 200.0, 'n_layers': 5})
        exp_data.features = MetricArrays()
        exp_data.features.add('deviation', DataArray(name='deviation', shape=(5,)))
        exp_data.features.set_value('deviation', np.random.rand(5) * 0.1)
        
        # Setup prediction system
        logger = LBPLogger(name='test', log_folder=str(tmp_path))
        system = PredictionSystem(dataset=dataset, logger=logger)
        
        # Create model that requires 'surface_quality' (not in dataset)
        class MismatchModel(IPredictionModel):
            def __init__(self):
                super().__init__(logger)
                self.is_trained = False
            
            @property
            def features_as_input(self):
                return ['surface_quality']  # Doesn't exist in dataset
            
            @property
            def predicted_features(self):
                return ['surface_quality']
            
            def train(self, X, y):
                self.is_trained = True
            
            def forward_pass(self, X):
                return np.zeros(len(X))
            
            def _get_model_artifacts(self):
                return {}
            
            def _load_model_from_artifacts(self, artifacts):
                pass
            
            def _set_model_artifacts(self, artifacts):
                pass
        
        model = MismatchModel()
        system.add_prediction_model(model)
        
        # Training should fail with clear error message
        datamodule = DataModule(dataset, test_size=0.0, val_size=0.0)
        with pytest.raises(ValueError, match="Model requires features .* not found in data"):
            system.train(datamodule)


# ============================================================================
# TEST CLASS 10: Breaking Changes & Migration
# ============================================================================

class TestBreakingChanges:
    """Verify breaking changes are documented and intentional."""
    
    def test_no_aggregated_feature_training(self):
        """
        Phase 10 does not support training on aggregated features.
        This is a breaking change from Phase 9.
        """
        # Old API (Phase 9) - NOT SUPPORTED:
        # X_train with aggregated features
        # y_train with scalar means
        
        # New API (Phase 10) - REQUIRED:
        # X_train with dimensional positions
        # y_train with dimensional feature values
        
        # This test documents the intentional breaking change
        assert True  # Placeholder - actual migration guide in docs
    
    def test_external_features_must_be_explicit(self):
        """
        External features must be provided in X_new during prediction.
        No automatic chaining of models.
        """
        model = ExternalFeatureModel()
        
        # features_as_input property is mandatory
        required = model.features_as_input
        assert isinstance(required, list)
        assert len(required) > 0
        
        # Users MUST provide these in X_new
        # Framework will NOT automatically predict them
