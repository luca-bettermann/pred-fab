# folder_navigator.py
import os
from typing import List, Optional
import shutil


class FolderNavigator:
    """
    File system navigation utility for managing study and experiment folders.

    Provides centralized access to local and server file systems with
    consistent naming conventions and error handling.
    """

    def __init__(
            self, 
            root_folder: str,
            local_folder: str, 
            server_folder: Optional[str] = None, 
            study_code: Optional[str] = None
            ) -> None:
        """
        Initialize folder navigator with base paths.

        Args:
            local_folder: Path to local data storage
            server_folder: Path to server data storage
            study_code: Optional study code for immediate initialization
        """
        self.root_folder: str = root_folder
        self.local_folder: str = local_folder
        self.server_folder: Optional[str] = server_folder
        self.study_code: Optional[str] = study_code
        self.study_folder: Optional[str] = None

        if study_code is not None:
            self.set_study_code(study_code)

    def set_study_code(self, study_code: str) -> None:
        """
        Set or change the study code.

        Args:
            study_code: Unique identifier for the study
        """
        self.study_code = study_code
        self.study_folder = os.path.join(self.local_folder, self.study_code)

    def get_experiment_code(self, exp_nr: int) -> str:
        """
        Generate experiment code from experiment number.

        Args:
            exp_nr: Experiment number

        Returns:
            Formatted experiment code (e.g., "STUDY_001")
        """
        if self.study_code is None:
            raise ValueError("Study code must be set before generating experiment code")
        return f"{self.study_code}_{str(exp_nr).zfill(3)}"

    def get_experiment_folder(self, exp_code: str) -> str:
        """
        Get full path to local experiment folder.

        Args:
            exp_nr: Experiment number

        Returns:
            Full path to experiment folder
        """
        if self.study_folder is None:
            raise ValueError("Study code must be set before getting experiment folder")
        return os.path.join(self.study_folder, exp_code)

    def get_server_experiment_folder(self, exp_code: str) -> str:
        """
        Get full path to server experiment folder.

        Args:
            exp_nr: Experiment number

        Returns:
            Full path to server experiment folder
        """
        if self.server_folder is None:
            raise ValueError("Server folder must be set before getting server experiment folder")
        if self.study_code is None:
            raise ValueError("Study code must be set before getting experiment folder")
        return os.path.join(self.server_folder, self.study_code, exp_code)

    def list_experiments(self) -> List[str]:
        """
        List all experiment folder names within the study folder.

        Returns:
            List of experiment folder names
        """
        if self.study_folder is None:
            raise ValueError("Study code must be set before listing experiments")
        
        return [
            d
            for d in os.listdir(self.study_folder)
            if os.path.isdir(os.path.join(self.study_folder, d))
        ]

    def copy_to_folder(self, src_path: str, target_folder: str) -> str:
        """
        Copy file or folder to target directory.

        Args:
            src_path: Source file or folder path
            target_folder: Target directory path

        Returns:
            Path to copied file or folder

        Raises:
            FileNotFoundError: If source path doesn't exist
            NotADirectoryError: If target is not a directory
            RuntimeError: If copy operation fails
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source path {src_path} does not exist.")

        if not os.path.isdir(target_folder):
            raise NotADirectoryError(f"Target folder {target_folder} does not exist or is not a directory.")

        base_name = os.path.basename(src_path.rstrip(os.sep))
        dst_path = os.path.join(target_folder, base_name)

        try:
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)

            return dst_path

        except Exception as e:
            raise RuntimeError(f"Failed to copy {src_path}: {str(e)}")

    def check_folder_access(self, folder_path: str) -> bool:
        """
        Verify folder accessibility.

        Returns:
            True if folder is accessible

        Raises:
            ConnectionError: If folder is not accessible
        """
        if folder_path is None:
            raise ValueError("Folder path must be set to check access.")

        is_connected = os.path.exists(folder_path)
        if not is_connected:
            raise ConnectionError(f"Folder access {folder_path}: FAILED")

        return is_connected


