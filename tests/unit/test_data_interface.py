import pytest
from examples.file_data_interface import FileDataInterface


class TestDataInterface:
    """Test data interface functionality using ExampleDataInterface."""
    
    def test_example_data_interface_initialization(self, temp_dir):
        """Test ExampleDataInterface initialization."""
        interface = FileDataInterface(temp_dir)
        
        assert interface.local_folder == temp_dir
        assert interface.client is None
    
    def test_study_record_retrieval(self, mock_data_interface):
        """Test study record retrieval."""
        study_record = mock_data_interface.get_study_record("test")
        
        assert study_record["id"] == "study_test"
        assert study_record["fields"]["Code"] == "test"
        assert "test" in study_record["fields"]["Name"]
    
    def test_experiment_record_retrieval(self, mock_data_interface):
        """Test experiment record retrieval."""
        exp_record = mock_data_interface.get_exp_record("test_001")
        
        assert exp_record["id"] == "exp_test_001"
        assert exp_record["fields"]["Code"] == "test_001"
        assert "test_001" in exp_record["fields"]["Name"]
    
    def test_study_parameters_retrieval(self, mock_data_interface, mock_study_params):
        """Test study parameters retrieval."""
        study_record = mock_data_interface.get_study_record("test")
        
        study_params = mock_data_interface.get_study_parameters(study_record)
        
        assert study_params == mock_study_params
        assert study_params["target_deviation"] == 0.0
        assert study_params["max_deviation"] == 0.5
    
    def test_performance_records_retrieval(self, mock_data_interface):
        """Test performance records retrieval."""
        study_record = mock_data_interface.get_study_record("test")
        
        perf_records = mock_data_interface.get_performance_records(study_record)
        
        assert len(perf_records) == 2
        assert perf_records[0]["Code"] == "path_deviation"
        assert perf_records[1]["Code"] == "energy_consumption"
    
    def test_experiment_variables_retrieval(self, mock_data_interface, mock_exp_params):
        """Test experiment variables retrieval."""
        exp_record = mock_data_interface.get_exp_record("test_001")
        
        exp_vars = mock_data_interface.get_exp_variables(exp_record)
        
        assert exp_vars == mock_exp_params
        assert exp_vars["layerTime"] == 30.0
        assert exp_vars["n_layers"] == 2
        assert exp_vars["n_segments"] == 2
    
    def test_client_check_with_none_client(self, mock_data_interface):
        """Test client validation with None client."""
        # Should raise error with None client in base implementation
        with pytest.raises(ValueError, match="Client not initialized"):
            mock_data_interface._client_check()        
    
    def test_client_check_with_valid_client(self, mock_data_interface):
        """Test client validation with valid client."""
        mock_data_interface.client = "mock_client"
        mock_data_interface._client_check()  # Should not raise error

