from pathlib import Path
from lbp_package import LBPManager

from examples.energy_consumption import EnergyConsumption
from examples.path_deviation import PathEvaluation
from examples.predict_all import PredictExample
from examples.file_data_interface import FileDataInterface
from tests.conftest import create_study_json_files, create_exp_json_files

# TODO NOW
# - Incorporate "Value Sign" field from airtable in code
# - Remove self.nav from interfaces. Only orchestration should handle navigation.

# TODO FUTURE
# - Features are immutable, whereas performance can change based on targets.
# - Hence, prediction models should predict features, not performance.
# - Therefore, features should be stored in the database, 
#   if we want to train prediction models based on that database.
# - Evaluation only becomes relevant once we want to optimize.
#   In the most elegant structure, evaluation should happen in the optimizer stage.
# - However, that part of the structure is not crucial. Important is,
#   how we plan to store features in the database. Making the airtable more complex
#   does not sound appealing. Switching to a more traditional, query-based
#   database solution might be worth considering.

def main():
    # Get paths relative to this file
    root_dir = Path(__file__).parent
    local_dir = root_dir / "local"
    logs_dir = root_dir / "logs"

    # Define study code
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
        external_data_interface=interface
    )

    # Add the example evaluation models to the LBPManager
    lbp_manager.add_evaluation_model("energy_consumption", EnergyConsumption)
    lbp_manager.add_evaluation_model("path_deviation", PathEvaluation)

    # Add the example prediction model to the LBPManager
    lbp_manager.add_prediction_model(["energy_consumption", "path_deviation"], PredictExample)

    # Get the parent directory and create local directory if it doesn't exist
    parent_dir = Path(__file__).parent
    local_dir = parent_dir / "local"
    local_dir.mkdir(exist_ok=True)
    
    # Generate data for 3 experiments of "test" study
    create_study_json_files(str(local_dir), study_code=study_code)
    create_exp_json_files(str(local_dir), study_code=study_code, exp_nr=1, layer_time=30.0, layer_height=0.25, n_layers=2, n_segments=2)
    create_exp_json_files(str(local_dir), study_code=study_code, exp_nr=2, layer_time=40.0, layer_height=0.20, n_layers=2, n_segments=3)
    create_exp_json_files(str(local_dir), study_code=study_code, exp_nr=3, layer_time=50.0, layer_height=0.15, n_layers=3, n_segments=2)

    # Initialize the study and run evaluation
    lbp_manager.initialize_for_study(study_code)

    # Run evaluations for each experiment
    lbp_manager.run_evaluation(exp_nr=1)
    lbp_manager.run_evaluation(exp_nr=2)
    lbp_manager.run_evaluation(exp_nr=3)
    # ERROR:
    # FIX RESULTS FILE MANAGEMENT

    # Run predictions for all experiments
    # lbp_manager.run_training()  # TODO: Fix training implementation

    # Calibrate the upcoming experiment
    # lbp_manager.run_calibration(exp_nr=4)  # TODO: Fix calibration implementation



if __name__ == "__main__":
    main()
