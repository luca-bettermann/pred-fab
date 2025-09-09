from pathlib import Path
from lbp_package import LBPManager
import time

from examples.evaluation_energy import EnergyConsumption
from examples.evaluation_geometry import PathEvaluation
from examples.prediction import PredictExample
from examples.external_data import MockDataInterface
from examples.calibration import RandomSearchCalibration, DifferentialEvolutionCalibration

# TODO FUTURE
# - Evaluation only becomes relevant once we want to optimize.
#   In the most elegant structure, evaluation should happen in the optimizer stage.
# - Right now, the feature model does not contain dimensionalities. This means, that
#   that when we connect PredictionModel with FeatureModel, it is without
#   dimensionality information. This needs to be adjusted.

def main():
    # Get paths relative to this file and make sure directories exist
    root_dir = Path(__file__).parent
    local_dir = root_dir / "local"
    logs_dir = root_dir / "logs"
    local_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Define study code
    study_code = "test"

    # Initialize LBPManager with the local folder and data interface
    lbp_manager = LBPManager(
        root_folder=str(root_dir),
        local_folder=str(local_dir),
        log_folder=str(logs_dir),
        external_data_interface=MockDataInterface(str(local_dir))
    )

    # Add the example evaluation models to the LBPManager
    # Add any additional parameters that should be passed to the EvaluationModel or its FeatureModel (optional)
    lbp_manager.add_evaluation_model("energy_consumption", EnergyConsumption, weight=0.3, additional_param=None)
    lbp_manager.add_evaluation_model("path_deviation", PathEvaluation, weight=0.7, round_digits=3)

    # Add the example prediction model to the LBPManager
    # Add any additional parameters that should be passed to the PredictExample or its FeatureModel (optional)
    lbp_manager.add_prediction_model(["energy_consumption", "path_deviation"], PredictExample, round_digits=4, additional_param=None)
    
    # Add calibration model
    
    # Initialize the study and run evaluation
    time.sleep(1)
    lbp_manager.initialize_for_study(study_code)

    # Run evaluations for each experiment
    time.sleep(1)
    lbp_manager.run_evaluation(study_code, exp_nr=1)
    time.sleep(1)
    lbp_manager.run_evaluation(study_code, exp_nrs=[2, 3])

    # Run predictions for all experiments
    time.sleep(1)
    lbp_manager.run_training(study_code, exp_nrs=[1, 2, 3])

    # Calibrate the upcoming experiment
    param_ranges = {
        "layerTime": (0.0, 1.0),
        "layerHeight": (10, 100),
    }

    # Calibrate using Random Search
    time.sleep(1)
    lbp_manager.set_calibration_model(RandomSearchCalibration, n_evaluations=100)
    lbp_manager.run_calibration(exp_nr=4, param_ranges=param_ranges)
    
    # Calibrate using Differential Evolution
    time.sleep(1)
    lbp_manager.set_calibration_model(DifferentialEvolutionCalibration, maxiter=10, seed=42)
    lbp_manager.run_calibration(exp_nr=4, param_ranges=param_ranges)


if __name__ == "__main__":
    main()
