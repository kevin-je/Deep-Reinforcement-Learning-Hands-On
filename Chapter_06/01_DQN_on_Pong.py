import ale_py
import gymnasium as gym

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch import optim

from PIL import Image

import matplotlib.pyplot as plt

from dataclasses import dataclass

from collections import deque

from typing import List, Tuple


# 超参数
LEARNING_RATE = 3e-3
BATCH_SIZE = 32
EPOCHS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON_START = 0.9
EPSILON_END = 0.05
DECAY_STEPS = 100_000

TGT_UPDATE_FREQ = 1_000

NUM_EPISODES = 1
NUM_ITERATIONS = 36
NUM_FRAMES = 4

GAMMA = 0.99

IMG_SIZE = (84, 84)

# 构建 DQN 网络
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        relu = nn.ReLU(inplace=True)
        flatten = nn.Flatten()

        self.sequential_conv: nn.Sequential = nn.Sequential(conv1, relu, conv2, relu, conv3, flatten)

        in_features = self.sequential_conv(torch.zeros(1, *input_shape)).size(-1)

        linear1 = nn.Linear(in_features, 512)
        linear2 = nn.Linear(512, num_actions)

        self.sequential_ff = nn.Sequential(linear1, relu, linear2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.sequential_ff(self.sequential_conv(obs))


@dataclass
class Transition:
    obs_stack: torch.Tensor
    action: int
    reward: float
    next_obs_stack: torch.Tensor
    done: bool

@dataclass
class ReplayBuffer:
    buffer: List[Transition]


def play_single_step(env: gym.Env,
        obs_stack: torch.Tensor,
        dqn: DQN,
        dqn_tgt: DQN,
        num_steps: int
) -> Tuple[int, float, np.ndarray, bool]:

    # 如果条件满足，将 dqn 的权重复制到 dqn_tgt
    if num_steps % TGT_UPDATE_FREQ == 0:
        dqn_tgt.load_state_dict(dqn.state_dict())


    if obs_stack.size(0) < NUM_FRAMES:
        action: int = env.action_space.sample()

    else:
        # 神经网络预测 Q 值
        actions_val: torch.Tensor = dqn(obs_stack.unsqueeze(0))
        # 计算 epsilon 的值
        epsilon = cal_epsilon(num_steps, EPSILON_START, EPSILON_END)
        # 采用 epsilon 贪心策略
        if np.random.random() < epsilon:
            action: int = env.action_space.sample()
        else:
            action: int = int(actions_val.argmax(dim=1).item())

    # 将动作值输入环境
    next_obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    done: bool = True if terminated or truncated else False

    return action, reward, next_obs, done


def play_single_episode(
        env: gym.Env,
        dqn: DQN,
        dqn_tgt: DQN,
        num_steps: int,
        replay_buffer: ReplayBuffer,
        num_frames: int = NUM_FRAMES
) -> Tuple[ReplayBuffer, int]:

    obs_stack: deque = deque(maxlen=num_frames)       # [tensor(84, 84)]
    next_obs_stack: deque = deque(maxlen=num_frames)

    # 初始化环境
    obs, _ = env.reset()
    assert type(obs) is np.ndarray

    while True:
        obs_stack.append(obs)
        obs_stack_t = torch.tensor(
            np.asarray(obs_stack),
            dtype=torch.float32
        )

        action, reward, next_obs, done = play_single_step(
            env,
            obs_stack_t,
            dqn,
            dqn_tgt,
            num_steps
        )
        num_steps += 1

        next_obs_stack.append(next_obs)

        # 生成一条 transition
        if len(obs_stack) == num_frames:
            trans = Transition(
                obs_stack_t,
                action,
                reward,
                torch.tensor(np.asarray(next_obs_stack), dtype=torch.float32),
                done
            )

            # 将生成的转移放入回放缓冲区
            replay_buffer.buffer.append(trans)

        obs = next_obs

        if done: break

    return replay_buffer, num_steps


class ResizeImg(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, img_size: Tuple[int, int] = IMG_SIZE) -> None:
        super().__init__(env)

        self.img_size = img_size

    def observation(self, obs: np.ndarray) -> np.ndarray:
        img = Image.fromarray(obs)
        # 转换为灰度图
        img = img.convert("L")
        # 裁剪
        img = img.crop((0, 30, 160, 190))
        # 缩放
        img = img.resize(self.img_size)
        # 转 numpy 数组
        obs = np.asarray(img, dtype=np.float32)
        # 归一化
        return obs / 255.0


def cal_q_val_tgt(dqn_tgt: DQN, trans: Transition, gamma: float = GAMMA) -> float:
    if trans.done:
        q_val_tgt = trans.reward

    else:
        next_obs_stack = trans.next_obs_stack
        q_val_tgt = dqn_tgt(next_obs_stack.unsqueeze(0)).argmax(dim=1).item() * gamma + trans.reward

    return q_val_tgt


def cal_epsilon(
        step: int,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        decay_steps: int = DECAY_STEPS) -> float:
    return max(epsilon_start - (step / decay_steps), epsilon_end)

class PongDataset(Dataset):
    def __init__(self, replay_buffer: ReplayBuffer, dqn_tgt: DQN) -> None:
        super().__init__()

        # 获取观测值
        self.observations = list(map(lambda x: x.obs_stack, replay_buffer.buffer))

        # 获取目标 Q 值
        fn = lambda x: cal_q_val_tgt(dqn_tgt, x)
        dqn_tgt.eval()
        with torch.no_grad():
            self.q_vals_tgt = list(map(fn, replay_buffer.buffer))

        # 获取每个转移的动作值
        self.actions = list(map(lambda x: x.action, replay_buffer.buffer))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, item):
        return self.observations[item], self.q_vals_tgt[item], self.actions[item]


def train(env: gym.Env, dqn: DQN, dqn_tgt: DQN) -> None:

    # 创建 criterion 和 optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)

    num_steps: int = 0

    for _ in range(NUM_ITERATIONS):

        # 开始游玩
        dqn.eval()
        dqn = dqn.to("cpu")
        with torch.no_grad():
            replay_buffer = ReplayBuffer([])
            for i in range(1, NUM_EPISODES + 1):

                buffer, steps = play_single_episode(
                    env,
                    dqn,
                    dqn_tgt,
                    num_steps,
                    replay_buffer
                )
                num_steps += steps

        # 创建数据集
        pong_dataset = PongDataset(replay_buffer, dqn_tgt)
        dataloader = DataLoader(pong_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 模型训练
        dqn.train()
        dqn = dqn.to(DEVICE)

        for obs_stacks_t, q_vals_tgt, actions in dataloader:

            q_vals = dqn(obs_stacks_t.to(DEVICE))
            q_vals_tgt = q_vals_tgt.to(DEVICE)
            actions = actions.to(DEVICE)

            q_vals = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

            # 计算损失
            loss = criterion(q_vals, q_vals_tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    gym.register_envs(ale_py)
    env = gym.make('ALE/Pong-v5', render_mode='human', obs_type="grayscale")
    env = ResizeImg(env, IMG_SIZE)

    dqn = DQN((NUM_FRAMES, *IMG_SIZE), env.action_space.n)
    dqn_tgt = DQN((NUM_FRAMES, *IMG_SIZE), env.action_space.n)

    train(env, dqn, dqn_tgt)