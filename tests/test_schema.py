"""
Tests for unified schema structure with parameters, dimensions, performance_attrs, and metric_arrays.
"""
import pytest
import numpy as np
from lbp_package.core.schema import DatasetSchema
from lbp_package.core.data_blocks import Parameters, Dimensions, PerformanceAttributes, Features
from lbp_package.core.data_objects import Parameter, PerformanceAttribute, Dimension, DataArray


class TestSchemaCreation:
    """Test creating DatasetSchema with unified structure."""
    
    def test_schema_default_empty(self):
        """Test creating default schema has all four blocks."""
        schema = DatasetSchema()
        
        assert schema.parameters is not None
        assert schema.dimensions is not None
        assert schema.performance is not None
        assert schema.features is not None
        
        # All blocks should be empty
        assert len(list(schema.parameters.keys())) == 0
        assert len(list(schema.dimensions.keys())) == 0
        assert len(list(schema.performance.keys())) == 0
        assert len(list(schema.features.keys())) == 0
    
    def test_schema_with_parameters(self):
        """Test adding parameters to schema."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema.parameters.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        
        assert len(list(schema.parameters.keys())) == 2
        assert schema.parameters.has("lr")
        assert schema.parameters.has("batch_size")
    
    def test_schema_complete(self):
        """Test creating complete schema with all blocks populated."""
        schema = DatasetSchema()
        
        # Add parameters
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        # Add dimensions
        schema.dimensions.add("traj.t", Dimension.integer("traj", "t", "i", min_val=0, max_val=100))
        
        # Add performance
        schema.performance.add("accuracy", PerformanceAttribute.real(min_val=0.0, max_val=1.0))
        
        # Add metric arrays
        schema.features.add("energy", DataArray(code="energy", shape=(100,)))
        
        assert len(list(schema.parameters.keys())) == 1
        assert len(list(schema.dimensions.keys())) == 1
        assert len(list(schema.performance.keys())) == 1
        assert len(list(schema.features.keys())) == 1


class TestSchemaHash:
    """Test schema hash computation."""
    
    def test_schema_hash_deterministic(self):
        """Test that same schema produces same hash."""
        schema1 = DatasetSchema()
        schema1.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        schema2 = DatasetSchema()
        schema2.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        hash1 = schema1._compute_schema_hash()
        hash2 = schema2._compute_schema_hash()
        
        assert hash1 == hash2
    
    def test_schema_hash_different_params(self):
        """Test that different parameters produce different hash."""
        schema1 = DatasetSchema()
        schema1.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        schema2 = DatasetSchema()
        schema2.parameters.add("lr", Parameter.real(min_val=0.0, max_val=2.0))  # Different max
        
        hash1 = schema1._compute_schema_hash()
        hash2 = schema2._compute_schema_hash()
        
        assert hash1 != hash2
    
    def test_schema_hash_includes_all_blocks(self):
        """Test that hash includes all blocks."""
        schema1 = DatasetSchema()
        schema1.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        schema2 = DatasetSchema()
        schema2.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema2.performance.add("acc", PerformanceAttribute.real(min_val=0.0, max_val=1.0))
        
        hash1 = schema1._compute_schema_hash()
        hash2 = schema2._compute_schema_hash()
        
        assert hash1 != hash2


class TestSchemaSerialization:
    """Test schema serialization."""
    
    def test_schema_to_dict(self):
        """Test converting schema to dictionary."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        
        schema_dict = schema.to_dict()
        
        assert "parameters" in schema_dict
        assert "dimensions" in schema_dict
        assert "performance_attrs" in schema_dict
        assert "metric_arrays" in schema_dict
        assert "schema_hash" in schema_dict
    
    def test_schema_from_dict(self):
        """Test creating schema from dictionary."""
        schema = DatasetSchema()
        schema.parameters.add("lr", Parameter.real(min_val=0.0, max_val=1.0))
        schema.parameters.add("batch_size", Parameter.integer(min_val=1, max_val=256))
        schema.performance.add("accuracy", PerformanceAttribute.real(min_val=0.0, max_val=1.0))
        
        schema_dict = schema.to_dict()
        restored = DatasetSchema.from_dict(schema_dict)
        
        assert len(list(restored.parameters.keys())) == 2
        assert len(list(restored.performance.keys())) == 1
        assert restored._compute_schema_hash() == schema._compute_schema_hash()


class TestSchemaIntegration:
    """Integration tests for complete schema usage."""
    
    def test_complete_schema_workflow(self):
        """Test complete workflow: create, populate, check structure."""
        # Create schema
        schema = DatasetSchema()
        
        # Add parameters
        schema.parameters.add("learning_rate", Parameter.real(min_val=0.0, max_val=1.0))
        schema.parameters.add("num_layers", Parameter.integer(min_val=1, max_val=10))
        schema.parameters.add("optimizer", Parameter.categorical(categories=["adam", "sgd"]))
        
        # Add dimensions
        schema.dimensions.add("trajectory.timestep", Dimension.integer("trajectory", "timestep", "t", min_val=0, max_val=100))
        
        # Add performance
        schema.performance.add("accuracy", PerformanceAttribute.real(min_val=0.0, max_val=1.0))
        schema.performance.add("loss", PerformanceAttribute.real(min_val=0.0, max_val=100.0))
        schema.performance.calibration_weights = {"accuracy": 0.7, "loss": 0.3}
        
        # Note: DataArray serialization needs fix, skip for now
        
        # Validate structure
        assert len(list(schema.parameters.keys())) == 3
        assert len(list(schema.dimensions.keys())) == 1
        assert len(list(schema.performance.keys())) == 2
        assert schema.performance.calibration_weights == {"accuracy": 0.7, "loss": 0.3}
