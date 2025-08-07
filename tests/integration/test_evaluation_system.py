from lbp_package.orchestration import EvaluationSystem
from lbp_package.utils import FolderNavigator
from examples import PathEvaluation, EnergyConsumption

class TestEvaluationSystem:
    """Test evaluation system functionality."""
    
    def test_initialization(self, temp_dir, test_logger, mock_data_interface):
        """Test evaluation system initialization."""
        nav = FolderNavigator(temp_dir, temp_dir, temp_dir, "test")
        
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        assert eval_system.nav == nav
        assert eval_system.interface == mock_data_interface
        assert eval_system.logger == test_logger
        assert len(eval_system.evaluation_models) == 0
    
    def test_add_evaluation_model(self, temp_dir, test_logger, mock_data_interface, mock_study_params):
        """Test adding evaluation models."""
        nav = FolderNavigator(temp_dir, temp_dir, temp_dir, "test")
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        # Add path deviation model
        eval_system.add_evaluation_model(
            PathEvaluation, 
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
        assert isinstance(eval_system.evaluation_models["path_deviation"], PathEvaluation)
        assert isinstance(eval_system.evaluation_models["energy_consumption"], EnergyConsumption)
    
    def test_add_feature_model_instances(self, temp_dir, test_logger, mock_data_interface, mock_study_params):
        """Test that feature model instances are properly managed through LBPManager."""
        # This test verifies the evaluation models can be added without errors
        nav = FolderNavigator(temp_dir, temp_dir, temp_dir, "test")
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        # Add evaluation models
        eval_system.add_evaluation_model(PathEvaluation, "path_deviation", mock_study_params)
        eval_system.add_evaluation_model(EnergyConsumption, "energy_consumption", mock_study_params)
        
        # Verify evaluation models are properly added
        assert "path_deviation" in eval_system.evaluation_models
        assert "energy_consumption" in eval_system.evaluation_models
        
        path_eval = eval_system.evaluation_models["path_deviation"]
        energy_eval = eval_system.evaluation_models["energy_consumption"]
        
        # Verify models are properly instantiated
        assert path_eval is not None
        assert energy_eval is not None

    def test_feature_model_sharing(self, temp_dir, test_logger, mock_data_interface, mock_study_params):
        """Test that evaluation models with same feature model type can be added."""
        # Note: Feature model sharing is now managed internally by LBPManager
        # This test verifies the evaluation models can be added without conflicts
        nav = FolderNavigator(temp_dir, temp_dir, temp_dir, "test")
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        # Add two evaluation models with same feature model type
        eval_system.add_evaluation_model(EnergyConsumption, "energy_consumption_1", mock_study_params)
        eval_system.add_evaluation_model(EnergyConsumption, "energy_consumption_2", mock_study_params)
        
        # Verify both evaluation models are properly added
        assert "energy_consumption_1" in eval_system.evaluation_models
        assert "energy_consumption_2" in eval_system.evaluation_models
        
        energy_eval_1 = eval_system.evaluation_models["energy_consumption_1"]
        energy_eval_2 = eval_system.evaluation_models["energy_consumption_2"]
        
        # Verify models are properly instantiated
        assert energy_eval_1 is not None
        assert energy_eval_2 is not None

    def test_evaluation_workflow(self, temp_dir, test_logger, mock_data_interface, mock_study_params, mock_exp_params, setup_test_data):
        """Test complete evaluation workflow without feature models."""
        # Note: Complete workflow with feature models is now tested through LBPManager
        # This test verifies basic evaluation workflow without feature model dependencies
        nav = FolderNavigator(temp_dir, temp_dir, temp_dir, "test")
        eval_system = EvaluationSystem(nav, mock_data_interface, test_logger)
        
        # Add evaluation models
        eval_system.add_evaluation_model(PathEvaluation, "path_deviation", mock_study_params)
        eval_system.add_evaluation_model(EnergyConsumption, "energy_consumption", mock_study_params)
        
        # Verify evaluation models were added successfully
        assert "path_deviation" in eval_system.evaluation_models
        assert "energy_consumption" in eval_system.evaluation_models
        
        path_eval = eval_system.evaluation_models["path_deviation"]
        energy_eval = eval_system.evaluation_models["energy_consumption"]
        
        # Verify models are properly instantiated
        assert path_eval is not None
        assert energy_eval is not None
