import os
from pathlib import Path
from src.lbp_package.orchestration import LBPManager
from examples.file_data_interface import FileDataInterface
from tests.test_data import create_test_data_files

def main():
    # Get paths relative to this file
    root_dir = Path(__file__).parent
    local_dir = root_dir / "local"
    logs_dir = root_dir / "logs"

    # Define study code and experiment number
    study_code = "test"
    exp_nr = 1

    # Ensure directories exist
    local_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # Create data interface that reads from local files
    interface = FileDataInterface(str(local_dir))

    lbp_manager = LBPManager(
        root_folder=str(root_dir),
        local_folder=str(local_dir),
        log_folder=str(logs_dir),
        data_interface=interface
    )

    # Generate example data files
    generate_example_files(study_code, exp_nr)
    
    lbp_manager.initialize_study(study_code=study_code)
    lbp_manager.run_evaluation(
        exp_nr=exp_nr,
        visualize_flag=False,
        debug_flag=True
    )


def generate_example_files(study_code: str, exp_nr: int):
    """Generate example data files."""
    # Get the examples directory
    examples_dir = Path(__file__).parent
    local_dir = examples_dir / "local"
    
    # Create the local directory if it doesn't exist
    local_dir.mkdir(exist_ok=True)
    
    # Generate test data for the "test" study
    create_test_data_files(str(local_dir), study_code=study_code, exp_nr=exp_nr)

    print("✓ Example data generated successfully!")
    print(f"  Study: {study_code}")
    print(f"  Experiment: {study_code}_{exp_nr:03d}")
    print(f"  Location: {local_dir}/{study_code}/")


if __name__ == "__main__":
    main()
