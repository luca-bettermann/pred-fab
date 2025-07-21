import pytest
import os
import logging
from lbp_package.utils.log_manager import LBPLogger


class TestLBPLogger:
    """Test logging functionality."""
    
    def test_logger_initialization(self, temp_dir):
        """Test logger initialization."""
        log_folder = os.path.join(temp_dir, "logs")
        logger = LBPLogger("TestLogger", log_folder)
        
        assert logger.name == "TestLogger"
        assert logger.log_folder == log_folder
        assert logger.debug_mode == False
        assert os.path.exists(log_folder)
        assert len(logger.logger.handlers) == 1
    
    def test_debug_mode_initialization(self, temp_dir):
        """Test debug mode initialization."""
        log_folder = os.path.join(temp_dir, "logs")
        logger = LBPLogger("TestLogger", log_folder, debug_mode=True)
        
        assert logger.debug_mode == True
        
        # Check debug log file exists
        debug_log = os.path.join(log_folder, "TestLogger_debug.log")
        assert os.path.exists(debug_log)
    
    def test_switch_to_debug_mode(self, temp_dir):
        """Test switching to debug mode."""
        log_folder = os.path.join(temp_dir, "logs")
        logger = LBPLogger("TestLogger", log_folder, debug_mode=False)
        
        assert logger.debug_mode == False
        
        logger.switch_to_debug_mode()
        
        assert logger.debug_mode == True
        debug_log = os.path.join(log_folder, "TestLogger_debug.log")
        assert os.path.exists(debug_log)
    
    def test_logging_methods(self, temp_dir):
        """Test different logging methods."""
        log_folder = os.path.join(temp_dir, "logs")
        logger = LBPLogger("TestLogger", log_folder)
        
        # Test logging methods don't raise errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Test console methods
        logger.console_info("Console info")
        logger.console_success("Console success")
        logger.console_warning("Console warning")
        logger.console_summary("Console summary")
    
    def test_log_and_raise(self, temp_dir):
        """Test log and raise functionality."""
        log_folder = os.path.join(temp_dir, "logs")
        logger = LBPLogger("TestLogger", log_folder)
        
        with pytest.raises(ValueError, match="Test error message"):
            logger.log_and_raise("Test error message")
        
        with pytest.raises(RuntimeError, match="Runtime error"):
            logger.log_and_raise("Runtime error", RuntimeError)
    
    def test_ansi_code_stripping(self, temp_dir):
        """Test ANSI code stripping."""
        log_folder = os.path.join(temp_dir, "logs")
        logger = LBPLogger("TestLogger", log_folder)
        
        # Test with ANSI codes
        ansi_text = "\033[1mBold text\033[0m and \033[31mRed text\033[0m"
        clean_text = logger._strip_ansi_codes(ansi_text)
        
        assert clean_text == "Bold text and Red text"
        assert "\033[" not in clean_text
