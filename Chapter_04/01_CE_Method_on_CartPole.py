from typing import List, Tuple

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
NUM_ITERATIONS = 36
PERCENTAGE = 70
LEARNING_RATE = 3e-3

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

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.episodes_steps[idx].observation, self.episodes_steps[idx].action


def play(net: nn.Module,
         env: gym.Env,
) -> Episode:

    episode = Episode(step=[], reward=0.0)

    net.eval()
    with torch.no_grad():

        observation, _ = env.reset()
        while True:
            # 神经网络预测动作
            action_prob_1 = torch.sigmoid(net(torch.tensor(observation, device=DEVICE))).item()
            action = int(np.random.choice([0.0, 1.0], p=[1-action_prob_1, action_prob_1]))

            next_observation, reward, terminated, truncated, _ = env.step(action)
            env.render()

            episode.step.append(EpisodeStep(observation=observation, action=action))
            episode.reward += reward
            observation = next_observation

            if terminated or truncated:
                break

    return episode

if __name__ == "__main__":
    # 创建环境
    env = gym.make("CartPole-v1", render_mode = "human" if RENDER else None)

    # 创建网络
    net = Net(input_size=4, output_size=1).to(DEVICE)

    # 创建标准与优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"------ ITERATION {iteration:03d} ------")

        # 获取观测数据
        print("游戏中......")
        episodes: List[Episode] = []

        loops = tqdm(range(NUM_EPISODES), colour="green")
        for _ in loops:
            episode = play(net=net, env=env)
            loops.set_postfix({"reward": episode.reward})
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

        loops = tqdm(dataloader, colour="green")
        for batch_idx, (obs, action) in enumerate(loops, start=1):
            loops.set_description(f"Batch: {batch_idx:03d}")

            obs, action = obs.to(device=DEVICE), action.to(dtype=torch.float32, device=DEVICE)
            action_prob = net(obs).squeeze(-1)
            loss = criterion(action_prob, action)
            loops.set_postfix({"loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存模型权重及优化器
        torch.save(net.state_dict(), "./01_Weights.pth")
        torch.save(optimizer.state_dict(), "01_Optimizer.pth")

    env.close()