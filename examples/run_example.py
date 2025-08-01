import os
from pathlib import Path
from lbp_package.orchestration import LBPManager
from examples.file_data_interface import FileDataInterface
from tests.conftest import create_study_json_files, create_exp_json_files

def main():
    # Get paths relative to this file
    root_dir = Path(__file__).parent
    local_dir = root_dir / "local"
    logs_dir = root_dir / "logs"

    # Define study code and experiment number
    study_code = "test"

    # Ensure directories exist
    local_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    
    # Create data interface that reads from local files
    interface = FileDataInterface(str(local_dir))

    # Initialize LBPManager with the local folder and data interface
    lbp_manager = LBPManager(
        root_folder=str(root_dir),
        local_folder=str(local_dir),
        log_folder=str(logs_dir),
        data_interface=interface
    )

    # Get the examples directory
    examples_dir = Path(__file__).parent
    local_dir = examples_dir / "local"
    
    # Create the local directory if it doesn't exist
    local_dir.mkdir(exist_ok=True)
    
    # Generate data for 3 experiments of "test" study
    create_study_json_files(str(local_dir), study_code=study_code)
    create_exp_json_files(str(local_dir), study_code=study_code, exp_nr=1, n_layers=2, n_segments=2)
    create_exp_json_files(str(local_dir), study_code=study_code, exp_nr=2, n_layers=2, n_segments=3)
    create_exp_json_files(str(local_dir), study_code=study_code, exp_nr=3, n_layers=3, n_segments=2)

    # Initialize the study and run evaluation
    lbp_manager.initialize_study(study_code)

    # Run evaluations for each experiment
    for exp_nr in range(1, 4):
        lbp_manager.run_evaluation(
            exp_nr=exp_nr,
            visualize_flag=False,
            debug_flag=True
        )



if __name__ == "__main__":
    main()
