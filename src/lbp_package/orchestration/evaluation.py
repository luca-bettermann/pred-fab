from typing import Any, Dict, Type, Optional, List
import numpy as np

from ..utils import LBPLogger
from ..interfaces.evaluation import IEvaluationModel
from ..interfaces.features import IFeatureModel
from ..core.dataset import Dataset, ExperimentData
from ..core.data_objects import DataArray, DataReal
from ..core.data_blocks import DataBlock
from .base import BaseOrchestrationSystem


class EvaluationSystem(BaseOrchestrationSystem):
    """
    Orchestrates multiple evaluation models using Dataset.
    
    Manages evaluation model execution and stores results in ExperimentData.
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize evaluation system."""
        super().__init__(dataset, logger)
        self.evaluation_models: Dict[str, IEvaluationModel] = {}

    # === PUBLIC API ===
    
    def add_evaluation_model(
        self, 
        performance_code: str, 
        evaluation_model: IEvaluationModel,
        feature_model_class: Type[IFeatureModel],
        **feature_model_params
    ) -> None:
        """Add an evaluation model with its feature model."""
        if not isinstance(evaluation_model, IEvaluationModel):
            raise TypeError(f"Expected IEvaluationModel instance, got {type(evaluation_model).__name__}")
        
        self.logger.info(f"Adding '{type(evaluation_model).__name__}' model for '{performance_code}'")
        
        # Create feature model with dataset reference
        feature_model = feature_model_class(
            dataset=self.dataset,
            logger=self.logger,
            **feature_model_params
        )
        
        # Connect feature model to evaluation model
        evaluation_model.add_feature_model(feature_model)
        
        self.evaluation_models[performance_code] = evaluation_model
    
    def get_models(self) -> Dict[str, IEvaluationModel]:
        """Return registered evaluation models."""
        return self.evaluation_models
    
    def get_model_specs(self) -> Dict[str, Dict[str, Any]]:
        """Extract input/output DataObject specifications from registered evaluation models."""        
        # Get base specs (inputs)
        specs = super().get_model_specs()
        
        # Add outputs (performance attributes)
        specs["outputs"] = {}
        for perf_code in self.evaluation_models.keys():
            specs["outputs"][perf_code] = DataReal(perf_code)
        
        return specs

    def run(self, exp_code: str, feature_name: str, performance_attr_name: str,
            evaluate_from: int = 0, evaluate_to: Optional[int] = None,
            visualize: bool = False, recompute: bool = False) -> ExperimentData:
        """Execute evaluation for an experiment."""
        # Get experiment from dataset
        exp_data = self.dataset.get_experiment(exp_code)
        
        # Find evaluation model for this performance attribute
        if performance_attr_name not in self.evaluation_models:
            raise ValueError(f"No evaluation model for '{performance_attr_name}'")
        
        eval_model = self.evaluation_models[performance_attr_name]
        
        self.logger.info(f"Running evaluation for '{performance_attr_name}' on experiment {exp_code}")
        
        # Handle recompute logic
        if recompute:
            self.logger.info("Recompute flag set - clearing cached features")
            self.dataset.clear_feature_cache()
        
        # Run evaluation - stores results directly in exp_data
        eval_model.run(
            feature_name, 
            performance_attr_name, 
            exp_data, 
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize
        )
        
        self.logger.info("Evaluation completed successfully")
        return exp_data

    def evaluate_experiment(
        self,
        exp_data: ExperimentData,
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> None:
        """Execute all evaluations for an experiment and mutate exp_data with results."""
        # Extract parameters from exp_data
        params = self._extract_params_from_exp_data(exp_data)
        
        # Get evaluation results from core logic
        perf_results, metric_results = self._evaluate_from_params(
            params=params,
            evaluate_from=evaluate_from,
            evaluate_to=evaluate_to,
            visualize=visualize,
            recompute=recompute
        )
        
        # Store results in exp_data
        self._store_results_in_exp_data(exp_data, perf_results, metric_results)

    def _evaluate_from_params(
        self,
        params: Dict[str, Any],
        evaluate_from: int = 0,
        evaluate_to: Optional[int] = None,
        visualize: bool = False,
        recompute: bool = False
    ) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Core evaluation logic from raw parameters."""

        # Create temporary exp_data for evaluation models
        temp_exp_data = ExperimentData(
            exp_code="temp_eval",
            parameters=DataBlock()
        )
        
        # Populate parameters
        for name, value in params.items():
            if name in self.dataset.schema.parameters.keys():
                data_obj = self.dataset.schema.parameters.data_objects[name]
                temp_exp_data.parameters.add(name, data_obj)
                temp_exp_data.parameters.set_value(name, value)
        
        # Initialize performance and metric_arrays
        for name, data_obj in self.dataset.schema.performance_attrs.items():
            temp_exp_data.performance.add(name, data_obj)
        
        for name, data_obj in self.dataset.schema.features.items():
            temp_exp_data.features.add(name, data_obj)
        
        # Run evaluation for each performance code
        # TODO: encode feature code and performance code in the respective models.
        for perf_code in self.evaluation_models.keys():
            feature_name = f"{perf_code}_feature"
            eval_model = self.evaluation_models[perf_code]
            
            self.logger.info(f"Evaluating '{perf_code}' with provided parameters")
            
            # Run evaluation - mutates temp_exp_data
            eval_model.run(
                feature_name=feature_name,
                performance_attr_name=perf_code,
                exp_data=temp_exp_data,
                evaluate_from=evaluate_from,
                evaluate_to=evaluate_to,
                visualize=visualize
            )
        
        # Extract results
        perf_results = temp_exp_data.performance.get_values_dict()
        metric_results = {
            name: temp_exp_data.features.get_value(name)
            for name in temp_exp_data.features.keys()
            if temp_exp_data.features.has_value(name)
        }
        
        return perf_results, metric_results
    
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
                data_obj = DataArray(name=array_name, shape=array_value.shape)
                exp_data.features.add(array_name, data_obj)
            exp_data.features.set_value(array_name, array_value)

    def get_evaluation_models(self) -> Dict[str, IEvaluationModel]:
        """Get all evaluation models."""
        return self.evaluation_models