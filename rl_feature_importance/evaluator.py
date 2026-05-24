import numpy as np
from typing import Dict, Any

from .adapters import SB3Adapter, TorchAdapter
from .methods import PermutationFeatureImportance

class FeatureImportanceEvaluator:
    def __init__(self, model: Any, model_type: str = "sb3", **kwargs):
        """
        Initializes the Evaluator with the given model.
        
        :param model: The RL model to evaluate.
        :param model_type: "sb3" for Stable Baselines 3 or "torch" for raw PyTorch models.
        :param kwargs: Additional arguments passed to the adapter.
        """
        self.model_type = model_type.lower()
        
        if self.model_type == "sb3":
            self.adapter = SB3Adapter(model, **kwargs)
        elif self.model_type == "torch":
            self.adapter = TorchAdapter(model, **kwargs)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'sb3' or 'torch'.")

    def evaluate(self, observations: np.ndarray, method: str = "permutation", **kwargs) -> Dict[int, float]:
        """
        Evaluates the feature importance on a given set of observations.
        
        :param observations: A batch of observations (shape: [batch_size, n_features]).
        :param method: The method to use (currently only "permutation" is supported).
        :param kwargs: Additional arguments passed to the method (e.g., n_iterations).
        :return: A dictionary mapping feature index to importance score.
        """
        method = method.lower()
        
        if method == "permutation":
            evaluator_method = PermutationFeatureImportance(self.adapter, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}. Currently supported: 'permutation'.")
            
        return evaluator_method.evaluate(observations)
