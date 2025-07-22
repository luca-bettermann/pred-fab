import pytest
import os
import yaml
from src.lbp_package.orchestration import EvaluationSystem
from src.lbp_package.utils.folder_navigator import FolderNavigator
from examples.example_evaluation_models import PathDeviationEvaluation, EnergyConsumption


class TestEvaluationSystem:
    """Test evaluation system functionality."""
    
    def test_initialization(self, temp_dir, test_logger, mock_data_interface):
        """Test evaluation system initialization."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        assert eval_system.nav == nav
        assert eval_system.interface == mock_data_interface
        assert eval_system.logger == test_logger
        assert len(eval_system.evaluation_models) == 0
    
    def test_add_evaluation_model(self, temp_dir, test_logger, mock_data_interface, mock_study_params):
        """Test adding evaluation models."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        # Add path deviation model
        eval_system.add_evaluation_model(
            PathDeviationEvaluation, 
            "path_deviation", 
            mock_study_params
        )
        
        # Add energy consumption model
        eval_system.add_evaluation_model(
            EnergyConsumption,
            "energy_consumption", 
            mock_study_params
        )
        
        assert len(eval_system.evaluation_models) == 2
        assert "path_deviation" in eval_system.evaluation_models
        assert "energy_consumption" in eval_system.evaluation_models
        
        # Verify model types
        assert isinstance(eval_system.evaluation_models["path_deviation"], PathDeviationEvaluation)
        assert isinstance(eval_system.evaluation_models["energy_consumption"], EnergyConsumption)
    
    def test_add_feature_model_instances(self, temp_dir, test_logger, mock_data_interface, mock_study_params):
        """Test adding feature model instances."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        # Add evaluation models
        eval_system.add_evaluation_model(PathDeviationEvaluation, "path_deviation", mock_study_params)
        eval_system.add_evaluation_model(EnergyConsumption, "energy_consumption", mock_study_params)
        
        # Add feature model instances
        eval_system.add_feature_model_instances(mock_study_params)
        
        # Verify feature models are set
        path_eval = eval_system.evaluation_models["path_deviation"]
        energy_eval = eval_system.evaluation_models["energy_consumption"]
        
        assert path_eval.feature_model is not None
        assert energy_eval.feature_model is not None
        
        # Verify feature models have correct performance codes
        assert "path_deviation" in path_eval.feature_model.performance_codes
        assert "energy_consumption" in energy_eval.feature_model.performance_codes
    
    def test_feature_model_sharing(self, temp_dir, test_logger, mock_data_interface, mock_study_params):
        """Test feature model sharing between evaluation models."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        # Add two evaluation models with same feature model type
        eval_system.add_evaluation_model(EnergyConsumption, "energy_consumption_1", mock_study_params)
        eval_system.add_evaluation_model(EnergyConsumption, "energy_consumption_2", mock_study_params)
        
        # Add feature model instances
        eval_system.add_feature_model_instances(mock_study_params)
        
        # Verify feature models are shared (same instance)
        energy_eval_1 = eval_system.evaluation_models["energy_consumption_1"]
        energy_eval_2 = eval_system.evaluation_models["energy_consumption_2"]
        
        assert energy_eval_1.feature_model is energy_eval_2.feature_model
        
        # Verify both performance codes are registered
        assert "energy_consumption_1" in energy_eval_1.feature_model.performance_codes
        assert "energy_consumption_2" in energy_eval_2.feature_model.performance_codes
    
    def test_evaluation_workflow(self, temp_dir, test_logger, mock_data_interface, mock_study_params, setup_test_data):
        """Test complete evaluation workflow."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        # Add evaluation models
        eval_system.add_evaluation_model(PathDeviationEvaluation, "path_deviation", mock_study_params)
        eval_system.add_evaluation_model(EnergyConsumption, "energy_consumption", mock_study_params)
        
        # Add feature model instances
        eval_system.add_feature_model_instances(mock_study_params)
        
        # Create mock experiment record
        exp_record = {"id": "test_exp", "fields": {"Code": "TEST_STUDY_001"}}
        
        # Run evaluation in debug mode (no database operations)
        eval_system.run(
            exp_nr=1,
            exp_record=exp_record,
            visualize_flag=False,
            debug_flag=True,
            **mock_study_params
        )
        
        # Verify evaluation completed
        path_eval = eval_system.evaluation_models["path_deviation"]
        energy_eval = eval_system.evaluation_models["energy_consumption"]
        
        assert path_eval.performance_metrics["Value"] is not None
        assert energy_eval.performance_metrics["Value"] is not None
        
        # Verify performance arrays are populated
        assert path_eval.performance_array is not None
        assert energy_eval.performance_array is not None
        
        # Path deviation should have 2x2 dimensions
        assert path_eval.performance_array.shape == (2, 2, 3)
        
        # Energy consumption should be scalar
        assert energy_eval.performance_array.shape == (3,)
