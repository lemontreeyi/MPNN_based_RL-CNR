import torch
import numpy as np
from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.size = 0  # buffer size

        self.buffer = deque(maxlen=self.max_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, action_dist, next_state, reward, done):
        """传入sasr数据组"""
        experience = {
            "features_critic": state,
            "action": action,
            "action_dist": action_dist,
            "next_features_critic": next_state,
            "reward": reward,
            "not_done": 1 - done,
        }
        self.buffer.append(experience)
        self.size = len(self.buffer)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
