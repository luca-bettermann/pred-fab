import logging
import os
import re
from datetime import datetime

class PfabLogger:
    """Singleton logger with dual file/console output, ANSI stripping, and structured console methods."""
    
    _instance: 'PfabLogger | None' = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PfabLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, name: str = "pred_fab", log_folder: str = "./logs") -> None:
        """Initialize logger with file and console handlers."""
        # Prevent re-initialization
        if getattr(self, "_initialized", False):
            return
            
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
        
        self._initialized = True
        self._console_new_line = False
        self._console_output_enabled = True
    
    @classmethod
    def get_logger(cls, log_folder: str) -> 'PfabLogger':
        """Get the singleton logger instance, creating default if undefined."""
        if cls._instance is None:
            # Create default instance if none exists
            return cls(log_folder=log_folder)
        return cls._instance

    def set_console_output(self, enabled: bool) -> None:
        """Enable or disable console printing for info/success methods (warnings always shown)."""
        self._console_output_enabled = enabled

    # === PUBLIC API METHODS ===
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
        if self._console_output_enabled:
            print(message)
            self._console_new_line = False
        self.logger.info(f"CONSOLE: \n\n{message}\n")

    def console_success(self, message: str) -> None:
        """Print success message to console and log."""
        if self._console_output_enabled:
            print(f"\033[32m\u2713\033[0m {message}")
            self._console_new_line = False
        self.logger.info(f"CONSOLE SUCCESS: \n\n{message}\n")

    def console_warning(self, message: str) -> None:
        """Print warning to console and log."""
        print(f"\033[33m!\033[0m {message}")
        self._console_new_line = False
        self.logger.warning(f"CONSOLE WARNING: \n\n{message}\n")

    def console_execute(self, message: str) -> None:
        """Print execute message to console and log."""
        if self._console_output_enabled:
            print(f"\033[36m>\033[0m {message}")
            self._console_new_line = False
        self.logger.warning(f"CONSOLE EXECUTE: \n\n{message}\n")

    def console_loaded(self, message: str) -> None:
        """Print loaded message to console and log."""
        if self._console_output_enabled:
            print(f"\033[36m>\033[0m {message}")
            self._console_new_line = False
        self.logger.info(f"CONSOLE LOADED: \n\n{message}\n")

    def console_saved(self, message: str) -> None:
        """Print saved message to console and log."""
        if self._console_output_enabled:
            print(f"\033[32m\u2713\033[0m {message}")
            self._console_new_line = False
        self.logger.info(f"CONSOLE SAVED: \n\n{message}\n")

    def console_pushed(self, message: str) -> None:
        """Print pushed message to console and log."""
        if self._console_output_enabled:
            print(f"\033[32m\u2713\033[0m {message}")
            self._console_new_line = False
        self.logger.info(f"CONSOLE PUSHED: \n\n{message}\n")

    def console_pulled(self, message: str) -> None:
        """Print pulled message to console and log."""
        if self._console_output_enabled:
            print(f"\033[32m\u2713\033[0m {message}")
            self._console_new_line = False
        self.logger.info(f"CONSOLE PULLED: \n\n{message}\n")

    def console_summary(self, message: str) -> None:
        """Print formatted summary to console and clean version to log."""
        if self._console_output_enabled:
            print(message)
        clean_message = self._strip_ansi_codes(message)
        self.logger.info(f"CONSOLE SUMMARY:\n\n{clean_message}")

    def console_new_line(self) -> None:
        """Print a new line to console."""
        if self._console_output_enabled and not self._console_new_line:
            print("")
            self._console_new_line = True

    # === PRIVATE METHODS ===
    def _setup_file_handler(self, log_file: str) -> None:
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

    def _strip_ansi_codes(self, text: str) -> str:
        """Remove ANSI escape codes from text for clean log files."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
