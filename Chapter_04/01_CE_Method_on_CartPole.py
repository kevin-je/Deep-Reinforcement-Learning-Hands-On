from typing import List

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass

from tqdm import tqdm

# 模型超参数
HIDDEN_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
PERCENTAGE = 30

# 环境超参数
NUM_EPISODES = 32
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


class StepDataset(Dataset):
    def __init__(self, elite_episodes: List[Episode]):
        # 获取所有的step
        self.episodes_steps: List[EpisodeStep] = []
        for episode in elite_episodes:
            self.episodes_steps.extend(episode.step)

    def __len__(self) -> int:
        return len(self.episodes_steps)

    def __getitem__(self, idx: int) -> EpisodeStep:
        return self.episodes_steps[idx]


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
    # 创建环境
    env = gym.make("CartPole-v1", render_mode = "human" if RENDER else None)
    # 创建网络
    net = Net(input_size=4, output_size=2).to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        print(f"------ Episode {epoch:03d} ------")

        # 获取观测数据
        print("游戏中......")
        episodes: List[Episode] = []

        for _ in tqdm(range(NUM_EPISODES), colour="green"):
            episode = play(net=net, env=env)
            episodes.append(episode)

        # 构建 elite_episodes
        percentile = np.percentile([episode.reward for episode in episodes], PERCENTAGE)
        elite_episodes: List[Episode] = []
        for episode in episodes:
            if episode.reward >= percentile:
                elite_episodes.append(episode)

        # 创建数据集
        step_dataset: Dataset = StepDataset(elite_episodes)
        dataloader: DataLoader = DataLoader(step_dataset, batch_size=BATCH_SIZE, shuffle=True)