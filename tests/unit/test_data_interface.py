import pytest
from examples.mock_data_interface import ExampleDataInterface


class TestDataInterface:
    """Test data interface functionality using ExampleDataInterface."""
    
    def test_example_data_interface_initialization(self, temp_dir, mock_study_params, mock_exp_params):
        """Test ExampleDataInterface initialization."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        
        assert interface.test_data_dir == temp_dir
        assert interface.study_params == mock_study_params
        assert interface.exp_params == mock_exp_params
        assert interface.client is None
    
    def test_study_record_retrieval(self, temp_dir, mock_study_params, mock_exp_params):
        """Test study record retrieval."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        
        study_record = interface.get_study_record("TEST_STUDY")
        
        assert study_record["id"] == "mock_study_001"
        assert study_record["fields"]["Code"] == "TEST_STUDY"
        assert study_record["fields"]["Name"] == "Test Study TEST_STUDY"
    
    def test_experiment_record_retrieval(self, temp_dir, mock_study_params, mock_exp_params):
        """Test experiment record retrieval."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        
        exp_record = interface.get_exp_record("TEST_STUDY_001")
        
        assert exp_record["id"] == "mock_exp_001"
        assert exp_record["fields"]["Code"] == "TEST_STUDY_001"
        assert exp_record["fields"]["Name"] == "Test Experiment TEST_STUDY_001"
    
    def test_study_parameters_retrieval(self, temp_dir, mock_study_params, mock_exp_params):
        """Test study parameters retrieval."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        study_record = interface.get_study_record("TEST_STUDY")
        
        study_params = interface.get_study_parameters(study_record)
        
        assert study_params == mock_study_params
        assert study_params["target_deviation"] == 0.0
        assert study_params["max_deviation"] == 0.5
    
    def test_performance_records_retrieval(self, temp_dir, mock_study_params, mock_exp_params):
        """Test performance records retrieval."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        study_record = interface.get_study_record("TEST_STUDY")
        
        perf_records = interface.get_performance_records(study_record)
        
        assert len(perf_records) == 2
        assert perf_records[0]["Code"] == "path_deviation"
        assert perf_records[1]["Code"] == "energy_consumption"
        assert all(record["Active"] for record in perf_records)
    
    def test_experiment_variables_retrieval(self, temp_dir, mock_study_params, mock_exp_params):
        """Test experiment variables retrieval."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        exp_record = interface.get_exp_record("TEST_STUDY_001")
        
        exp_vars = interface.get_exp_variables(exp_record)
        
        assert exp_vars == mock_exp_params
        assert exp_vars["layerTime"] == 30.0
        assert exp_vars["n_layers"] == 2
    
    def test_database_push_mock(self, temp_dir, mock_study_params, mock_exp_params):
        """Test database push functionality (mock)."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        exp_record = interface.get_exp_record("TEST_STUDY_001")
        
        test_values = {"Value": 0.85, "Performance": 0.90}
        interface.push_to_database(exp_record, "path_deviation", test_values)
        
        assert hasattr(interface, 'pushed_data')
        assert "path_deviation" in interface.pushed_data
        assert interface.pushed_data["path_deviation"] == test_values
    
    def test_system_performance_update_mock(self, temp_dir, mock_study_params, mock_exp_params):
        """Test system performance update functionality (mock)."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        study_record = interface.get_study_record("TEST_STUDY")
        
        interface.update_system_performance(study_record)
        
        assert hasattr(interface, 'system_performance_updated')
        assert interface.system_performance_updated is True
    
    def test_client_check_with_none_client(self, temp_dir, mock_study_params, mock_exp_params):
        """Test client validation with None client."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        
        # Should raise error with None client in base implementation
        with pytest.raises(ValueError, match="Client not initialized"):
            interface._client_check()        
    
    def test_client_check_with_valid_client(self, temp_dir, mock_study_params, mock_exp_params):
        """Test client validation with valid client."""
        interface = ExampleDataInterface(temp_dir, mock_study_params, mock_exp_params)
        
        interface.client = "mock_client"
        interface._client_check()  # Should not raise error
