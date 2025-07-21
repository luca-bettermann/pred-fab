import pytest
from lbp_package.lbp_package.data_interface import DataInterface


class ConcreteDataInterface(DataInterface):
    """Concrete implementation for testing."""
    
    def get_study_record(self, study_code: str):
        return {"id": "test_study", "code": study_code}
    
    def get_exp_record(self, exp_code: str):
        return {"id": "test_exp", "code": exp_code}
    
    def get_study_parameters(self, study_record):
        return {"param1": "value1", "param2": "value2"}
    
    def get_performance_records(self, study_record):
        return [{"Code": "test_performance"}]
    
    def get_exp_variables(self, exp_record):
        return {"var1": "value1", "var2": "value2"}


class TestDataInterface:
    """Test data interface functionality."""
    
    def test_concrete_implementation(self):
        """Test concrete data interface implementation."""
        interface = ConcreteDataInterface()
        
        # Test study record
        study_record = interface.get_study_record("TEST_STUDY")
        assert study_record["id"] == "test_study"
        assert study_record["code"] == "TEST_STUDY"
        
        # Test experiment record
        exp_record = interface.get_exp_record("TEST_EXP")
        assert exp_record["id"] == "test_exp"
        assert exp_record["code"] == "TEST_EXP"
        
        # Test study parameters
        study_params = interface.get_study_parameters(study_record)
        assert study_params["param1"] == "value1"
        assert study_params["param2"] == "value2"
        
        # Test performance records
        perf_records = interface.get_performance_records(study_record)
        assert len(perf_records) == 1
        assert perf_records[0]["Code"] == "test_performance"
        
        # Test experiment variables
        exp_vars = interface.get_exp_variables(exp_record)
        assert exp_vars["var1"] == "value1"
        assert exp_vars["var2"] == "value2"
    
    def test_client_check(self):
        """Test client validation."""
        interface = ConcreteDataInterface()
        
        # Should not raise error with None client (base implementation)
        interface._client_check()
        
        # Test with client set
        interface.client = "mock_client"
        interface._client_check()  # Should not raise error
        
        # Test with None client
        interface.client = None
        with pytest.raises(ValueError, match="Client not initialized"):
            interface._client_check()
    
    def test_optional_methods(self):
        """Test optional methods have default implementations."""
        interface = ConcreteDataInterface()
        
        # These should not raise errors (default implementations)
        interface.push_to_database({}, "test_code", {})
        interface.update_system_performance({})
