from typing import Any, Dict, Type, Optional, List
import numpy as np

from ..utils import LBPLogger
from ..interfaces.evaluation import IEvaluationModel
from ..interfaces.features import IFeatureModel
from ..core.dataset import Dataset, ExperimentData


class EvaluationSystem:
    """
    Orchestrates multiple evaluation models using Dataset.
    
    Manages evaluation model execution and stores results in ExperimentData.
    """
    
    def __init__(self, dataset: Dataset, logger: LBPLogger):
        """Initialize evaluation system."""
        self.dataset = dataset
        self.logger = logger
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

    def run(self, exp_code: str, feature_name: str, performance_attr_name: str,
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
        eval_model.run(feature_name, performance_attr_name, exp_data, visualize=visualize)
        
        self.logger.info("Evaluation completed successfully")
        return exp_data

    def evaluate_experiment(
        self,
        exp_data: ExperimentData,
        visualize: bool = False,
        recompute: bool = False
    ) -> None:
        """Execute all evaluations for an experiment and mutate exp_data with results."""
        from ..core.data_blocks import PerformanceAttributes, MetricArrays
        
        exp_code = exp_data.exp_code
        
        # Check if already computed (unless recompute=True)
        if not recompute and self.dataset.has_experiment(exp_code):
            existing_exp = self.dataset.get_experiment(exp_code)
            if existing_exp.performance is not None:
                self.logger.info(f"Experiment '{exp_code}' already evaluated (use recompute=True to override)")
                # Copy existing results to exp_data
                exp_data.performance = existing_exp.performance
                exp_data.metric_arrays = existing_exp.metric_arrays
                return
        
        # Initialize performance and metric_arrays blocks if needed
        if exp_data.performance is None:
            exp_data.performance = PerformanceAttributes()
            for name, data_obj in self.dataset.schema.performance_attrs.items():
                exp_data.performance.add(name, data_obj)
        
        if exp_data.metric_arrays is None:
            exp_data.metric_arrays = MetricArrays()
            for name, data_obj in self.dataset.schema.metric_arrays.items():
                exp_data.metric_arrays.add(name, data_obj)
        
        # Run evaluation for each performance code
        for perf_code in self.evaluation_models.keys():
            feature_name = f"{perf_code}_feature"
            
            self.logger.info(f"Evaluating '{perf_code}' for experiment '{exp_code}'")
            
            # Run single performance evaluation
            result_exp_data = self.run(
                exp_code=exp_code,
                feature_name=feature_name,
                performance_attr_name=perf_code,
                visualize=visualize,
                recompute=recompute
            )
            
            # Merge results into exp_data
            if result_exp_data.performance is not None:
                perf_value = result_exp_data.performance.get_value(perf_code)
                exp_data.performance.set_value(perf_code, perf_value)
            
            if result_exp_data.metric_arrays is not None:
                for array_name in result_exp_data.metric_arrays.keys():
                    if result_exp_data.metric_arrays.has_value(array_name):
                        array_value = result_exp_data.metric_arrays.get_value(array_name)
                        exp_data.metric_arrays.set_value(array_name, array_value)
        
        self.logger.info(f"Completed evaluation for experiment '{exp_code}'")

    def get_evaluation_models(self) -> Dict[str, IEvaluationModel]:
        """Get all evaluation models."""
        return self.evaluation_models