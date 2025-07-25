import logging
import os
import re
from datetime import datetime


class LBPLogger:
    """
    Enhanced logger with integrated console output and error handling.
    
    Provides dual logging to file and console with different formatting
    for each output channel. Supports debug mode switching and ANSI
    code stripping for clean log files.
    """
    
    def __init__(self, name: str, log_folder: str):
        """
        Initialize logger with file and console handlers.
        
        Args:
            name: Logger name identifier
            log_folder: Directory for log file storage
        """
        self.name = name
        self.log_folder = log_folder
        self.debug_mode = False
        self.current_log_file = None
        
        # Create logger instance
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers to avoid duplication
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Setup initial session-based file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_file = f"{name}_session_{timestamp}.log"
        self._setup_file_handler(session_log_file)
    
    def _setup_file_handler(self, log_file) -> None:
        """Configure file handler with appropriate naming and formatting."""
        os.makedirs(self.log_folder, exist_ok=True)
        
        log_file_path = os.path.join(self.log_folder, log_file)
        
        # Track current log file
        self.current_log_file = log_file_path
        
        # Configure file handler
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        # Remove %(name)s from formatter to exclude logger name from log rows
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str) -> None:
        """Log debug message to file only."""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message to file only."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message to file only."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message to file only."""
        self.logger.error(message)
    
    def console_info(self, message: str) -> None:
        """Print to console and log as info."""
        print(message)
        self.logger.info(f"CONSOLE: \n\n{message}\n")
    
    def console_success(self, message: str) -> None:
        """Print success message to console and log."""
        print(f"✅ {message}")
        self.logger.info(f"CONSOLE SUCCESS: \n\n{message}\n")
    
    def console_warning(self, message: str) -> None:
        """Print warning to console and log."""
        print(f"⚠️ {message}")
        self.logger.warning(f"CONSOLE WARNING: \n\n{message}\n")

    def console_summary(self, message: str) -> None:
        """Print formatted summary to console and clean version to log."""
        # Display formatted version to console
        print(message)

        # Log clean version without ANSI codes
        clean_message = self._strip_ansi_codes(message)  
        self.logger.info(f"CONSOLE SUMMARY:\n\n{clean_message}")

    def switch_to_debug_mode(self) -> None:
        """Switch logger to debug mode by renaming current file."""
        if not self.debug_mode:
            self.debug_mode = True
            
            # Close current file handler to release file lock
            if self.logger.handlers:
                self.logger.handlers[0].close()
                self.logger.handlers.clear()
            
            # Define debug file path
            debug_file = os.path.join(self.log_folder, "debug.log")
            
            # Simply rename current log file to debug file (overwrites if exists)
            if self.current_log_file and os.path.exists(self.current_log_file):
                os.rename(self.current_log_file, debug_file)
                self.current_log_file = debug_file
            
            # Recreate handler for the renamed file
            self._setup_file_handler("debug.log")

    def _strip_ansi_codes(self, text: str) -> str:
        """
        Remove ANSI escape codes from text for clean log files.
        
        Args:
            text: Text potentially containing ANSI codes
            
        Returns:
            Clean text without ANSI escape sequences
        """
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
