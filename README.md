# RL Feature Importance

A unified Python package for calculating feature importance in Reinforcement Learning models. It currently supports models trained with PyTorch directly or via Stable Baselines 3 (SB3).

## Installation

You can install this package locally:
```bash
pip install -e .
```

If you want to use it with Stable Baselines 3:
```bash
pip install -e .[sb3]
```

## Usage

Here is a quick example of how to use the Permutation Feature Importance evaluator:

```python
import numpy as np
import torch
from stable_baselines3 import PPO
from rl_feature_importance.evaluator import FeatureImportanceEvaluator

# Load your model (SB3 example)
model = PPO.load("my_ppo_model")

# Collect some observations (e.g., from an environment)
observations = np.random.randn(100, 10)  # 100 samples, 10 features

# Initialize evaluator
evaluator = FeatureImportanceEvaluator(model, model_type="sb3")

# Compute importance using Permutation Feature Importance
importance_scores = evaluator.evaluate(observations, method="permutation")

for i, score in importance_scores.items():
    print(f"Feature {i} importance: {score:.4f}")
```
