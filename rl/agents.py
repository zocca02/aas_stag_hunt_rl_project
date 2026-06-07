from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F


class RLAgent(ABC):
    
    @abstractmethod
    def policy(self, state):
        raise NotImplemented
    
    @abstractmethod
    def on_action_performed(self, state, action, reward, new_state, terminated, truncated) -> None:
        raise NotImplemented

class RandomAgent(RLAgent):
    
    def __init__(self, action_space):
        self.action_space = action_space

    def policy(self, state) -> int:
        return self.action_space.sample()

    def on_action_performed(self, state, action, reward, new_state, terminated, truncated) -> None:
        return

    def on_end(self):
        return
