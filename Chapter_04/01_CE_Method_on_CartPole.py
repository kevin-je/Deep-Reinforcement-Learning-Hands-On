from typing import List

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn

from dataclasses import dataclass

# 模型超参数
HIDDEN_SIZE = 128

# 环境超参数
NUM_EPISODES = 100
RENDER = True

# 设备参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

@dataclass
class Episode:
    step: List[EpisodeStep]
    reward: float

def play(net: nn.Module,
         env: gym.Env,
         num_episodes: int = NUM_EPISODES
) -> Episode:

    sm = nn.Softmax(dim=0)

    episode = Episode(step=[], reward=0.0)

    net.eval()
    with torch.no_grad():

        for _ in range(num_episodes):

            observation, _ = env.reset()

            while True:
                # 神经网络预测动作
                action_prob = sm(net(torch.Tensor(observation).unsqueeze(0).to(DEVICE))[0]).to("cpu").numpy()
                action = int(np.random.choice(np.arange(len(action_prob)), p=action_prob))

                next_observation, reward, terminated, truncated, _ = env.step(action)
                env.render()

                episode.step.append(EpisodeStep(observation=observation, action=action))
                episode.reward += reward
                observation = next_observation

                if terminated or truncated:
                    break

    env.close()
    return episode

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode = "human" if RENDER else None)

    net = Net(input_size=4, output_size=2).to(DEVICE)
    play(net, env)