import pytest
import os
import yaml
from src.lbp_package.orchestration import LBPManager


class TestLBPManager:
    """Test LBP Manager end-to-end functionality."""
    
    def test_initialization(self, temp_dir, mock_data_interface):
        """Test LBP Manager initialization."""
        log_dir = os.path.join(temp_dir, "logs")
        
        manager = LBPManager(
            root_folder=temp_dir,
            local_folder=temp_dir,
            server_folder=temp_dir,
            log_folder=log_dir,
            data_interface=mock_data_interface
        )
        
        assert manager.nav is not None
        assert manager.interface == mock_data_interface
        assert manager.logger is not None
        assert manager.eval_system is None
    
    def test_study_initialization(self, temp_dir, mock_data_interface, mock_config):
        """Test study initialization workflow."""
        log_dir = os.path.join(temp_dir, "logs")
        
        # Create mock config file
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(mock_config, f)
        
        # Change to temp directory for config loading
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            manager = LBPManager(
                root_folder=temp_dir,
                local_folder=temp_dir,
                server_folder=temp_dir,
                log_folder=log_dir,
                data_interface=mock_data_interface
            )
            
            # Initialize study
            manager.initialize_study("test")
            
            assert manager.study_code == "test"
            assert manager.nav.study_code == "test"
            assert manager.eval_system is not None
            assert len(manager.eval_system.evaluation_models) == 2
            assert "path_deviation" in manager.eval_system.evaluation_models
            assert "energy_consumption" in manager.eval_system.evaluation_models
            
        finally:
            os.chdir(original_cwd)
    
    def test_complete_workflow(self, temp_dir, mock_data_interface, mock_config, setup_test_data):
        """Test complete LBP Manager workflow."""
        log_dir = os.path.join(temp_dir, "logs")
        
        # Create mock config file
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(mock_config, f)
        
        # Change to temp directory for config loading
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            manager = LBPManager(
                root_folder=temp_dir,
                local_folder=temp_dir,
                server_folder=temp_dir,
                log_folder=log_dir,
                data_interface=mock_data_interface
            )
            
            # Initialize study
            manager.initialize_study("test")
            
            # Run evaluation
            manager.run_evaluation(
                exp_nr=1,
                visualize_flag=False,
                debug_flag=True  # No database operations
            )
            
            # Verify evaluation completed
            assert manager.eval_system is not None
            assert len(manager.eval_system.evaluation_models) == 2
            
            # Verify performance metrics were computed
            path_eval = manager.eval_system.evaluation_models["path_deviation"]
            energy_eval = manager.eval_system.evaluation_models["energy_consumption"]
            
            assert path_eval.performance_metrics["Value"] is not None
            assert energy_eval.performance_metrics["Value"] is not None
            
            # Verify performance values are reasonable
            assert 0 <= path_eval.performance_metrics["Value"] <= 1
            assert 0 <= energy_eval.performance_metrics["Value"] <= 1
            
        finally:
            os.chdir(original_cwd)
    
    def test_error_handling(self, temp_dir, mock_data_interface):
        """Test error handling in LBP Manager."""
        log_dir = os.path.join(temp_dir, "logs")
        
        manager = LBPManager(
            root_folder=temp_dir,
            local_folder=temp_dir,
            server_folder=temp_dir,
            log_folder=log_dir,
            data_interface=mock_data_interface
        )
        
        # Test running without initialization
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.run_evaluation(exp_nr=1)
        
        # Test initialization with missing config
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            manager.initialize_study("test")
    
