import pytest
import os
import numpy as np
from src.lbp_package.utils.folder_navigator import FolderNavigator
from examples.example_evaluation_models import PathDeviationFeature, EnergyFeature


class TestPathDeviationFeature:
    """Test path deviation feature model."""
    
    def test_initialization(self, temp_dir, test_logger, mock_study_params):
        """Test path deviation feature initialization."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        feature_model = PathDeviationFeature(
            performance_code="path_deviation",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        assert feature_model.tolerance_xyz == 0.1
        assert feature_model.n_layers == 2
        assert feature_model.n_segments == 2
        assert "path_deviation" in feature_model.performance_codes
    
    def test_feature_computation(self, temp_dir, test_logger, mock_study_params, setup_test_data):
        """Test path deviation feature computation."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        feature_model = PathDeviationFeature(
            performance_code="path_deviation",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        # Set runtime parameters
        feature_model.set_runtime_parameters(layer_id=0, segment_id=0)
        
        # Load test data
        data = feature_model._load_data(1)
        
        # Compute features
        features = feature_model._compute_features(data, visualize_flag=False)
        
        assert "path_deviation" in features
        assert isinstance(features["path_deviation"], float)
        assert features["path_deviation"] > 0  # Should have some deviation
        assert features["path_deviation"] < 1.0  # Should be reasonable
    
    def test_data_loading(self, temp_dir, test_logger, mock_study_params, setup_test_data):
        """Test data loading functionality."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        feature_model = PathDeviationFeature(
            performance_code="path_deviation",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        # Load test data
        data = feature_model._load_data(1)
        
        assert "designed" in data
        assert "measured" in data
        assert "layers" in data["designed"]
        assert "layers" in data["measured"]
        assert len(data["designed"]["layers"]) == 2
        assert len(data["measured"]["layers"]) == 2


class TestEnergyFeature:
    """Test energy feature model."""
    
    def test_initialization(self, temp_dir, test_logger, mock_study_params):
        """Test energy feature initialization."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        feature_model = EnergyFeature(
            performance_code="energy_consumption",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        assert feature_model.power_rating == 50.0
        assert feature_model.layerTime == 30.0
        assert "energy_consumption" in feature_model.performance_codes
    
    def test_feature_computation(self, temp_dir, test_logger, mock_study_params):
        """Test energy feature computation."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        feature_model = EnergyFeature(
            performance_code="energy_consumption",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        # Set experiment parameters
        feature_model.set_experiment_parameters(layerTime=30.0)
        
        # Load data (None for energy calculation)
        data = feature_model._load_data(1)
        assert data is None
        
        # Compute features
        features = feature_model._compute_features(data, visualize_flag=False)
        
        assert "energy_consumption" in features
        assert features["energy_consumption"] == 50.0 * 30.0  # power * time
        assert features["energy_consumption"] == 1500.0
    
    def test_parameter_updates(self, temp_dir, test_logger, mock_study_params):
        """Test parameter updates affect feature computation."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        feature_model = EnergyFeature(
            performance_code="energy_consumption",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        # Test with different layer times
        feature_model.set_experiment_parameters(layerTime=60.0)
        features = feature_model._compute_features(None, visualize_flag=False)
        assert features["energy_consumption"] == 50.0 * 60.0  # 3000.0
        
        # Test with different power rating
        feature_model.set_model_parameters(power_rating=100.0)
        features = feature_model._compute_features(None, visualize_flag=False)
        assert features["energy_consumption"] == 100.0 * 60.0  # 6000.0
