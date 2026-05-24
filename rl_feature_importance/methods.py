import numpy as np
from typing import Dict
from .adapters import BaseModelAdapter

class PermutationFeatureImportance:
    def __init__(self, adapter: BaseModelAdapter, n_iterations: int = 5):
        """
        :param adapter: The model adapter.
        :param n_iterations: Number of times to shuffle and evaluate to reduce variance.
        """
        self.adapter = adapter
        self.n_iterations = n_iterations

    def evaluate(self, observations: np.ndarray) -> Dict[int, float]:
        """
        Calculates feature importance by measuring the change in the model's
        action output when a feature is randomly shuffled across the batch.
        
        :param observations: A batch of observations (shape: [batch_size, n_features]).
        :return: A dictionary mapping feature index to importance score (higher = more important).
        """
        if observations.ndim != 2:
            raise ValueError("Observations must be a 2D array of shape [batch_size, n_features]")

        n_samples, n_features = observations.shape
        if n_samples < 2:
            raise ValueError("Batch size must be at least 2 for permutation to have an effect.")

        # Baseline predictions
        baseline_actions = self.adapter.get_action(observations)
        
        importance_scores = {}

        for feature_idx in range(n_features):
            feature_divergences = []
            
            for _ in range(self.n_iterations):
                # Create a perturbed batch
                perturbed_obs = observations.copy()
                
                # Shuffle the specific feature column
                np.random.shuffle(perturbed_obs[:, feature_idx])
                
                # Get predictions on perturbed batch
                perturbed_actions = self.adapter.get_action(perturbed_obs)
                
                # Compute divergence (Mean Squared Error)
                mse = np.mean((baseline_actions - perturbed_actions) ** 2)
                feature_divergences.append(mse)
                
            # Average the divergences over iterations
            importance_scores[feature_idx] = float(np.mean(feature_divergences))
            
        return importance_scores
