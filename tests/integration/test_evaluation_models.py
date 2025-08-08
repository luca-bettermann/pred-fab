from lbp_package.utils import FolderNavigator
from examples import PathEvaluation
from examples import EnergyConsumption


class TestPathDeviationEvaluation:
    """Test path deviation evaluation model."""
    
    def test_initialization(self, temp_dir, test_logger, mock_study_params):
        """Test path deviation evaluation initialization."""
        nav = FolderNavigator(temp_dir, temp_dir, temp_dir, "TEST_STUDY")
        
        eval_model = PathEvaluation(
            performance_code="path_deviation",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        assert eval_model.performance_code == "path_deviation"
        assert eval_model.target_deviation == 0.0
        assert eval_model.max_deviation == 0.5
        assert eval_model.n_layers == None # experiment not set yet
        assert eval_model.n_segments == None # experiment not set yet
        assert len(eval_model.dim_names) == 2
        assert "layers" in eval_model.dim_names
        assert "segments" in eval_model.dim_names
    
    def test_target_and_scaling_computation(self, temp_dir, test_logger, mock_study_params):
        """Test target value and scaling factor computation."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        eval_model = PathEvaluation(
            performance_code="path_deviation",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        assert eval_model._compute_target_value() == 0.0
        assert eval_model._declare_scaling_factor() == 0.5
    
    def test_dimensional_setup(self, temp_dir, test_logger, mock_study_params, mock_exp_params):
        """Test dimensional configuration."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        eval_model = PathEvaluation(
            performance_code="path_deviation",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        # Test dimension configuration
        assert eval_model.dim_names == ['layers', 'segments']
        assert eval_model.dim_iterator_names == ['layer_id', 'segment_id']
        assert eval_model.dim_param_names == ['n_layers', 'n_segments']
        
        # Test dimension sizes after setting experiment parameters
        eval_model.set_experiment_parameters(**mock_exp_params)
        dim_sizes = eval_model._compute_dim_sizes()
        assert dim_sizes == [2, 2]
        
        # Test dimension ranges
        dim_ranges = eval_model._compute_dim_ranges()
        assert len(dim_ranges) == 2
        assert list(dim_ranges[0]) == [0, 1]
        assert list(dim_ranges[1]) == [0, 1]


class TestEnergyConsumption:
    """Test energy consumption evaluation model."""
    
    def test_initialization(self, temp_dir, test_logger, mock_study_params, mock_exp_params):
        """Test energy consumption evaluation initialization."""
        nav = FolderNavigator(temp_dir, temp_dir, temp_dir, "TEST_STUDY")
        
        eval_model = EnergyConsumption(
            performance_code="energy_consumption",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )

        # Test initialization   
        assert eval_model.performance_code == "energy_consumption"
        assert eval_model.target_energy == 0.0
        assert eval_model.max_energy == None # Experiment parameter not set yet
        assert len(eval_model.dim_names) == 0  # No dimensions

        # Test setting experiment parameters
        mock_exp_params.update(mock_study_params)
        eval_model.set_experiment_parameters(**mock_exp_params)
        assert eval_model.max_energy == 10000.0 # Experiment parameter set
    
    def test_target_and_scaling_computation(self, temp_dir, test_logger, mock_study_params, mock_exp_params):
        """Test target value and scaling factor computation."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        eval_model = EnergyConsumption(
            performance_code="energy_consumption",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        assert eval_model._compute_target_value() == 0.0

        mock_exp_params.update(mock_study_params)
        eval_model.set_experiment_parameters(**mock_exp_params)
        assert eval_model._declare_scaling_factor() == 10000.0

    def test_scalar_evaluation(self, temp_dir, test_logger, mock_study_params):
        """Test scalar evaluation (no dimensions)."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        eval_model = EnergyConsumption(
            performance_code="energy_consumption",
            folder_navigator=nav,
            logger=test_logger,
            **mock_study_params
        )
        
        # Test dimension configuration for scalar
        assert eval_model.dim_names == []
        assert eval_model.dim_iterator_names == []
        assert eval_model.dim_param_names == []
        
        # Test dimension sizes for scalar
        dim_sizes = eval_model._compute_dim_sizes()
        assert dim_sizes == []
        
        # Test dimension ranges for scalar
        dim_ranges = eval_model._compute_dim_ranges()
        assert len(dim_ranges) == 0
        assert len(dim_ranges) == 0
