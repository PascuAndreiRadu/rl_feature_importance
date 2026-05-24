import numpy as np
import pytest
from rl_feature_importance.adapters import BaseModelAdapter
from rl_feature_importance.evaluator import FeatureImportanceEvaluator

class DeterministicAdapter(BaseModelAdapter):
    """
    A dummy adapter where the action is completely determined by feature 0.
    Action = Obs[:, 0] * 2 + 1
    """
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return obs[:, 0] * 2 + 1

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(obs.shape[0])

def test_permutation_importance():
    batch_size = 100
    n_features = 3
    
    # Generate random observations
    observations = np.random.randn(batch_size, n_features).astype(np.float32)
    
    adapter = DeterministicAdapter()
    # We can inject our custom adapter into the evaluator if we bypass model_type,
    # or test the method directly. Let's test the method directly.
    from rl_feature_importance.methods import PermutationFeatureImportance
    
    method = PermutationFeatureImportance(adapter, n_iterations=10)
    scores = method.evaluate(observations)
    
    # Feature 0 should have a very high importance score (divergence > 0)
    # Feature 1 and 2 should have 0 importance because they don't affect the action
    assert scores[0] > 0.1
    assert np.isclose(scores[1], 0.0)
    assert np.isclose(scores[2], 0.0)

def test_evaluator_integration():
    import torch
    import torch.nn as nn
    
    class SimpleNet(nn.Module):
        def forward(self, obs):
            # Action depends only on feature 1
            return obs[:, 1:2] * 5.0
            
    model = SimpleNet()
    evaluator = FeatureImportanceEvaluator(model, model_type="torch")
    
    observations = np.random.randn(50, 4).astype(np.float32)
    scores = evaluator.evaluate(observations, method="permutation", n_iterations=5)
    
    # Feature 1 should be important
    assert scores[1] > 0.1
    # Others should be 0
    assert np.isclose(scores[0], 0.0)
    assert np.isclose(scores[2], 0.0)
    assert np.isclose(scores[3], 0.0)
