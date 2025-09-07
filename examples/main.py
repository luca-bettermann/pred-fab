from pathlib import Path
from lbp_package import LBPManager

from examples.evaluation_energy import EnergyConsumption
from examples.evaluation_geometry import PathEvaluation
from examples.prediction import PredictExample
from examples.external_data import MockDataInterface

# TODO NOW
# - Implement prediction Example

# TODO FUTURE
# - Evaluation only becomes relevant once we want to optimize.
#   In the most elegant structure, evaluation should happen in the optimizer stage.

def main():
    # Get paths relative to this file and make sure directories exist
    root_dir = Path(__file__).parent
    local_dir = root_dir / "local"
    logs_dir = root_dir / "logs"
    local_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Define study code
    study_code = "test"
    
    # Create data interface that reads from local files
    interface = MockDataInterface(str(local_dir))

    # Initialize LBPManager with the local folder and data interface
    lbp_manager = LBPManager(
        root_folder=str(root_dir),
        local_folder=str(local_dir),
        log_folder=str(logs_dir),
        external_data_interface=interface
    )

    # Add the example evaluation models to the LBPManager
    # Add any additional parameters that should be passed to the EvaluationModel or its FeatureModel (optional)
    lbp_manager.add_evaluation_model("energy_consumption", EnergyConsumption, additional_param=None)
    lbp_manager.add_evaluation_model("path_deviation", PathEvaluation, round_digits=3)

    # Add the example prediction model to the LBPManager
    # Add any additional parameters that should be passed to the PredictExample or its FeatureModel (optional)
    lbp_manager.add_prediction_model(["energy_consumption", "path_deviation"], PredictExample, round_digits=4, additional_param=None)
    
    # Initialize the study and run evaluation
    lbp_manager.initialize_for_study(study_code)

    # Run evaluations for each experiment
    lbp_manager.run_evaluation(study_code, exp_nr=1)
    lbp_manager.run_evaluation(study_code, exp_nrs=[2, 3])

    # Run predictions for all experiments
    lbp_manager.run_training(study_code, exp_nrs=[1, 2, 3])

    # Calibrate the upcoming experiment
    # lbp_manager.run_calibration(exp_nr=4)  # TODO: Fix calibration implementation    


if __name__ == "__main__":
    main()
