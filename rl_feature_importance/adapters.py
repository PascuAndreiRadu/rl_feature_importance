import abc
import numpy as np
import torch

class BaseModelAdapter(abc.ABC):
    @abc.abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Given a batch of observations, return the model's deterministic actions.
        """
        pass

    @abc.abstractmethod
    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """
        Given a batch of observations, return the model's value estimates.
        If the model does not estimate values (e.g., pure policy gradient), return zeros.
        """
        pass

class SB3Adapter(BaseModelAdapter):
    def __init__(self, model):
        """
        Adapter for Stable Baselines 3 models.
        :param model: An SB3 model (e.g., PPO, SAC).
        """
        self.model = model

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # predict returns a tuple (actions, states)
        actions, _ = self.model.predict(obs, deterministic=True)
        return actions

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        # Not all SB3 models have a value function accessible the same way.
        # We try to use the policy's predict_values if it exists.
        if hasattr(self.model, "policy") and hasattr(self.model.policy, "predict_values"):
            # Convert obs to tensor for the policy
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                values = self.model.policy.predict_values(obs_tensor)
            return values.cpu().numpy()
        else:
            return np.zeros(obs.shape[0])

class TorchAdapter(BaseModelAdapter):
    def __init__(self, model, device="cpu"):
        """
        Adapter for raw PyTorch models.
        Assumes the model's forward pass returns the action.
        If it returns a tuple, the user should provide a wrapper.
        :param model: A torch.nn.Module.
        :param device: The device to run the model on.
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self.model(obs_tensor)
            
        # Try to handle common output formats
        if isinstance(output, tuple):
            actions = output[0]  # Assume first element is action
        else:
            actions = output
            
        if isinstance(actions, torch.Tensor):
            return actions.cpu().numpy()
        return actions

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        # Default fallback if the Torch model doesn't explicitly have a value method.
        # Can be subclassed by users for specific architectures.
        return np.zeros(obs.shape[0])
