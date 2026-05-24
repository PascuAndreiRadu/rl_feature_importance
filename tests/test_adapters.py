import numpy as np
import pytest
import torch
import torch.nn as nn
from rl_feature_importance.adapters import TorchAdapter, BaseModelAdapter

class DummyTorchModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, act_dim)

    def forward(self, obs):
        return self.fc(obs)

class DummyModelTupleReturn(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc_act = nn.Linear(obs_dim, act_dim)
        self.fc_val = nn.Linear(obs_dim, 1)

    def forward(self, obs):
        return self.fc_act(obs), self.fc_val(obs)

def test_torch_adapter_simple():
    obs_dim = 4
    act_dim = 2
    batch_size = 5
    model = DummyTorchModel(obs_dim, act_dim)
    adapter = TorchAdapter(model)
    
    obs = np.random.randn(batch_size, obs_dim).astype(np.float32)
    actions = adapter.get_action(obs)
    
    assert actions.shape == (batch_size, act_dim)
    assert isinstance(actions, np.ndarray)
    
    values = adapter.get_value(obs)
    assert values.shape == (batch_size,)
    assert (values == 0).all() # Default fallback

def test_torch_adapter_tuple():
    obs_dim = 4
    act_dim = 2
    batch_size = 3
    model = DummyModelTupleReturn(obs_dim, act_dim)
    adapter = TorchAdapter(model)
    
    obs = np.random.randn(batch_size, obs_dim).astype(np.float32)
    actions = adapter.get_action(obs)
    
    assert actions.shape == (batch_size, act_dim)
    assert isinstance(actions, np.ndarray)

# Testing SB3 adapter requires stable-baselines3. We can optionally skip if not installed,
# but we specified it in pyproject.toml [dev].
def test_sb3_adapter():
    try:
        from stable_baselines3 import PPO
        import gymnasium as gym
    except ImportError:
        pytest.skip("stable-baselines3 not installed")
        
    from rl_feature_importance.adapters import SB3Adapter
    
    # Create a dummy env and model
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, n_steps=16, verbose=0)
    
    adapter = SB3Adapter(model)
    
    batch_size = 2
    obs = np.random.randn(batch_size, 4).astype(np.float32) # Cartpole has 4 dims
    
    actions = adapter.get_action(obs)
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (batch_size,) # Discrete actions
    
    values = adapter.get_value(obs)
    assert isinstance(values, np.ndarray)
    assert values.shape == (batch_size, 1) or values.shape == (batch_size,)
