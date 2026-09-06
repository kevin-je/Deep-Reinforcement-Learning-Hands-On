from typing import Tuple

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn

from dataclasses import dataclass

# 模型超参数
HIDDEN_SIZE = 256

# 环境超参数
NUM_EPISODES = 100
RENDER = True

class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

def play(net: nn.Module, env: gym.Env = gym.make('CartPole-v0'), num_episodes: int = NUM_EPISODES, render: bool = RENDER):
    net.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            observation, _ = env.reset()
            total_reward = 0
            
            while True:
                action = net(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    env.render()