import pytest
import os
from lbp_package.utils.folder_navigator import FolderNavigator


class TestFolderNavigator:
    """Test folder navigation functionality."""
    
    def test_initialization(self, temp_dir):
        """Test folder navigator initialization."""
        local_folder = os.path.join(temp_dir, "local")
        server_folder = os.path.join(temp_dir, "server")
        
        nav = FolderNavigator(local_folder, server_folder)
        
        assert nav.local_folder == local_folder
        assert nav.server_folder == server_folder
        assert nav.study_code is None
        assert nav.study_folder is None
    
    def test_set_study_code(self, temp_dir):
        """Test setting study code."""
        nav = FolderNavigator(temp_dir, temp_dir)
        
        nav.set_study_code("TEST_STUDY")
        
        assert nav.study_code == "TEST_STUDY"
        assert nav.study_folder == os.path.join(temp_dir, "TEST_STUDY")
    
    def test_get_experiment_code(self, temp_dir):
        """Test experiment code generation."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        exp_code = nav.get_experiment_code(1)
        assert exp_code == "TEST_STUDY_001"
        
        exp_code = nav.get_experiment_code(42)
        assert exp_code == "TEST_STUDY_042"
    
    def test_get_experiment_folder(self, temp_dir):
        """Test experiment folder path generation."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        exp_folder = nav.get_experiment_folder(1)
        expected = os.path.join(temp_dir, "TEST_STUDY", "TEST_STUDY_001")
        assert exp_folder == expected
    
    def test_get_server_experiment_folder(self, temp_dir):
        """Test server experiment folder path generation."""
        local_folder = os.path.join(temp_dir, "local")
        server_folder = os.path.join(temp_dir, "server")
        
        nav = FolderNavigator(local_folder, server_folder, "TEST_STUDY")
        
        server_exp_folder = nav.get_server_experiment_folder(1)
        expected = os.path.join(server_folder, "TEST_STUDY", "TEST_STUDY_001")
        assert server_exp_folder == expected
    
    def test_copy_to_folder(self, temp_dir):
        """Test file copying functionality."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        # Create source file
        src_file = os.path.join(temp_dir, "test_file.txt")
        with open(src_file, 'w') as f:
            f.write("test content")
        
        # Create target directory
        target_dir = os.path.join(temp_dir, "target")
        os.makedirs(target_dir)
        
        # Copy file
        dst_path = nav.copy_to_folder(src_file, target_dir)
        
        # Verify copy
        expected_dst = os.path.join(target_dir, "test_file.txt")
        assert dst_path == expected_dst
        assert os.path.exists(expected_dst)
        
        with open(expected_dst, 'r') as f:
            assert f.read() == "test content"
    
    def test_copy_nonexistent_file(self, temp_dir):
        """Test copying nonexistent file raises error."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        with pytest.raises(FileNotFoundError):
            nav.copy_to_folder("nonexistent.txt", temp_dir)
    
    def test_copy_to_nonexistent_directory(self, temp_dir):
        """Test copying to nonexistent directory raises error."""
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        
        # Create source file
        src_file = os.path.join(temp_dir, "test_file.txt")
        with open(src_file, 'w') as f:
            f.write("test content")
        
        with pytest.raises(NotADirectoryError):
            nav.copy_to_folder(src_file, "nonexistent_dir")
    
    def test_check_server_connection(self, temp_dir):
        """Test server connection checking."""
        # Test with existing server folder
        nav = FolderNavigator(temp_dir, temp_dir, "TEST_STUDY")
        assert nav.check_server_connection() == True
        
        # Test with nonexistent server folder
        nav_bad = FolderNavigator(temp_dir, "nonexistent_server", "TEST_STUDY")
        with pytest.raises(ConnectionError):
            nav_bad.check_server_connection()
