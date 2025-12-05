from typing import Any, Dict, Tuple, Type, Optional, List
import numpy as np

from ..utils import LBPLogger
from ..interfaces.evaluation import IEvaluationModel
from ..interfaces.features import IFeatureModel
from ..core import Dataset, ExperimentData, DataArray, DataReal, Dimensions, Parameters
from .base import BaseOrchestrationSystem


class EvaluationSystem(BaseOrchestrationSystem):
    """
    Orchestrates multiple evaluation models using Dataset.
    
    Manages evaluation model execution and stores results in ExperimentData.
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize evaluation system."""
        super().__init__(dataset, logger)
        self.evaluation_models: List[IEvaluationModel] = []

    # === PUBLIC API ===
    
    # def add_evaluation_model(
    #     self, 
    #     performance_code: str, 
    #     evaluation_model: IEvaluationModel,
    #     feature_model_class: Type[IFeatureModel],
    #     **feature_model_params
    # ) -> None:
    #     """Add an evaluation model with its feature model."""
    #     if not isinstance(evaluation_model, IEvaluationModel):
    #         raise TypeError(f"Expected IEvaluationModel instance, got {type(evaluation_model).__name__}")
        
    #     self.logger.info(f"Adding '{type(evaluation_model).__name__}' model for '{performance_code}'")
        
    #     # Create feature model with dataset reference
    #     feature_model = feature_model_class(
    #         dataset=self.dataset,
    #         logger=self.logger,
    #         **feature_model_params
    #     )
        
    #     # Connect feature model to evaluation model
    #     evaluation_model.add_feature_model(feature_model)
        
    #     self.evaluation_models[performance_code] = evaluation_model
    
    def get_models(self) -> List[IEvaluationModel]:
        """Return registered evaluation models."""
        return self.evaluation_models
    
    def get_model_specs(self) -> Dict[str, Dict[str, Any]]:
        """Extract input/output DataObject specifications from registered evaluation models."""        
        # Get base specs (inputs)
        specs = super().get_model_specs()
        
        # Add outputs (performance attributes)
        specs["outputs"] = {}
        for eval_model in self.evaluation_models:
            specs["outputs"][eval_model.performance_code] = DataReal(eval_model.performance_code)
        
        return specs

    # def run(self, exp_code: str, feature_name: str, performance_attr_name: str,
    #         evaluate_from: int = 0, evaluate_to: Optional[int] = None,
    #         visualize: bool = False, recompute: bool = False) -> ExperimentData:
    #     """Execute evaluation for an experiment."""
    #     # Get experiment from dataset
    #     exp_data = self.dataset.get_experiment(exp_code)
        
    #     # Find evaluation model for this performance attribute
    #     if performance_attr_name not in self.evaluation_models:
    #         raise ValueError(f"No evaluation model for '{performance_attr_name}'")
        
    #     eval_model = self.evaluation_models[performance_attr_name]
        
    #     self.logger.info(f"Running evaluation for '{performance_attr_name}' on experiment {exp_code}")
        
    #     # Handle recompute logic
    #     if recompute:
    #         self.logger.info("Recompute flag set - clearing cached features")
    #         self.dataset.clear_feature_cache()
        
    #     # Run evaluation - stores results directly in exp_data
    #     eval_model.run(
    #         feature_name, 
    #         performance_attr_name, 
    #         exp_data, 
    #         evaluate_from=evaluate_from,
    #         evaluate_to=evaluate_to,
    #         visualize=visualize
    #     )
        
    #     self.logger.info("Evaluation completed successfully")
    #     return exp_data

    def evaluate_experiment(
        self,
        exp_data: ExperimentData,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> None:
        """Execute all evaluations for an experiment and mutate exp_data with results."""

        # Initialize arrays in exp_data if not present
        for eval_model in self.evaluation_models:
            # TODO: initialie arrays on a per-eval-model basis
            
            required_dims = eval_model._get_required_dimensions()
            shape = exp_data.get_array_shape(dims=required_dims)


        shape = exp_data.get_array_shape()
        if not exp_data.features.values:
            exp_data.initialize_array(block_type="feature", feature_code="feature", shape=shape)
        else:
            if exp_data.features.values() != shape:
                raise ValueError("Existing feature arrays in exp_data have incompatible shape")

        # Get evaluation results from core logic
        perf_results, metric_results = self._evaluate_from_params(
            parameters=exp_data.parameters,
            dimensions=exp_data.dimensions,
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize,
            recompute=recompute
        )

        # Update exp_data with results
        exp_data.features.set_values(metric_results)
        exp_data.performance.set_values(perf_results)
        
        # Store results in exp_data
        self._store_results_in_exp_data(exp_data, perf_results, metric_results)

    def _evaluate_from_params(
        self,
        parameters: Parameters,
        dimensions: Dimensions,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Core evaluation logic from raw parameters."""

        # Create temporary exp_data for evaluation models
        # temp_exp_data = ExperimentData(
        #     exp_code="temp_eval",
        #     parameters=DataBlock()
        # )
        
        # # Populate parameters
        # for name, value in params.items():
        #     if name in self.dataset.schema.parameters.keys():
        #         data_obj = self.dataset.schema.parameters.data_objects[name]
        #         temp_exp_data.parameters.add(name, data_obj)
        #         temp_exp_data.parameters.set_value(name, value)
        
        # # Initialize performance and metric_arrays
        # for name, data_obj in self.dataset.schema.performance_attrs.items():
        #     temp_exp_data.performance.add(name, data_obj)
        
        # for name, data_obj in self.dataset.schema.features.items():
        #     temp_exp_data.features.add(name, data_obj)
        
        # Run evaluation for each performance code

        # TODO: do not pass exp_data to interfaces
        # TODO: encode feature code and performance code in the respective models.

        # Handle recompute logic
        if recompute:
            self.logger.info(f"Recompute flag set - clearing cache")
            self.dataset.clear_feature_cache()

        performance_dict: Dict[str, float] = {}
        feature_dict: Dict[str, np.ndarray] = {}

        for eval_model in self.evaluation_models:
            
            self.logger.info(f"Evaluating '{eval_model.performance_code}' with provided parameters")

            # Run evaluation
            features, performance = eval_model.run(
                parameters=parameters,
                dimensions=dimensions,
                evaluate_from=evaluate_from,
                evaluate_to=evaluate_to,
                visualize=visualize
            )

            # Collect results
            performance_dict[eval_model.performance_code] = performance
            feature_dict[eval_model.feature_input_code] = features

        # Clear feature cache after evaluation to free memory
        self.dataset.clear_feature_cache()

        return feature_dict, performance_dict
    
    def _store_results_in_exp_data(
        self,
        exp_data: ExperimentData,
        perf_results: Dict[str, float],
        metric_results: Dict[str, np.ndarray]
    ) -> None:
        """Store evaluation results in exp_data."""
        # Store performance values
        for perf_code, perf_value in perf_results.items():
            if not exp_data.performance.has(perf_code):
                data_obj = self.dataset.schema.performance_attrs.data_objects[perf_code]
                exp_data.performance.add(perf_code, data_obj)
            exp_data.performance.set_value(perf_code, perf_value)
        
        # Store metric arrays
        for array_name, array_value in metric_results.items():
            if not exp_data.features.has(array_name):
                data_obj = DataArray(code=array_name, shape=array_value.shape)
                exp_data.features.add(array_name, data_obj)
            exp_data.features.set_value(array_name, array_value)

    # def get_evaluation_model_dict(self) -> Dict[str, IEvaluationModel]:
    #     """Get a dict of performance code to evaluation models."""
    #     eval_models = {}
    #     for model in self.evaluation_models:
    #         eval_models[model.performance_code] = model
    #     return eval_models