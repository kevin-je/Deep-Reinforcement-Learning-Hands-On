import ale_py
import gymnasium as gym

import numpy as np

import random

import torch
import torch.nn as nn
from torch import optim

import cv2 as cv

import os
from copy import copy

from dataclasses import dataclass

from collections import deque

from typing import Tuple, List

import json


# 超参数
LEARNING_RATE = 2e-4
BATCH_SIZE = 32

BUFFER_SIZE = 1_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON_START = 0.99
EPSILON_END = 0.05
DECAY_STEPS = 150_000

SCORE_BOUNDARY= 21

TGT_UPDATE_FREQ = 1_000

NUM_EPISODES = 32
NUM_FRAMES = 4

RENDER_MODE = None

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

        assert obs.dtype is torch.uint8
        return self.sequential_ff(self.sequential_conv(obs/255.0))


@dataclass
class Transition:
    obs_stack: deque[np.ndarray]
    action: int
    reward: int | float
    next_obs_stack: deque[np.ndarray]
    done: bool


@torch.no_grad()
def play_single_step(
        env: gym.Env,
        obs_stack: deque[np.ndarray],
        dqn: DQN,
        num_steps: int
) -> Tuple[int, float, np.ndarray, bool]:

    if len(obs_stack) < NUM_FRAMES:
        action: int = env.action_space.sample()

    else:
        # 神经网络预测 Q 值
        actions_val: torch.Tensor = dqn(
            torch.as_tensor(
                np.asarray(
                    obs_stack,
                    dtype=np.uint8
                ),
                device=DEVICE,
                dtype=torch.uint8
            ).unsqueeze(0)
        )
        # 计算 epsilon 的值
        epsilon = cal_epsilon(num_steps)
        # 采用 epsilon 贪心策略
        if np.random.random() < epsilon:
            action: int = env.action_space.sample()
        else:
            action: int = int(actions_val.argmax(dim=1).item())

    # 将动作值输入环境
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done: bool = True if terminated or truncated else False

    # 渲染画面
    if RENDER_MODE == "human":
        env.render()

    return action, reward, next_obs, done


def init_reply_buffer(
        env: gym.Env,
        replay_buffer: deque[Transition]
) -> Tuple[deque[Transition], int]:
    assert len(replay_buffer) == 0

    obs_stack = deque(maxlen=NUM_FRAMES)
    obs, _ = env.reset()
    next_obs_stack = deque(maxlen=NUM_FRAMES)

    action = 0
    reward: int = 0
    done = False
    score = 0

    for _ in range(NUM_FRAMES):
        obs_stack.append(obs)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs_stack.append(next_obs)
        obs = next_obs
        score += reward

    trans = Transition(
        obs_stack=obs_stack,
        action=action,
        reward=reward,
        next_obs_stack=next_obs_stack,
        done=done
    )
    replay_buffer.append(trans)
    return replay_buffer, score


class ResizeImg(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, img_size: Tuple[int, int] = IMG_SIZE) -> None:
        super().__init__(env)

        self.img_size = img_size

    def observation(self, observation: np.ndarray) -> np.ndarray:
        observation = observation[30:191, :161]
        # 缩放
        observation = cv.resize(observation, self.img_size)
        return observation


def cal_epsilon(
        step: int,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        decay_steps: int = DECAY_STEPS) -> float:
    return max(epsilon_start - (step / decay_steps), epsilon_end)


def get_replay_buffer(
        env: gym.Env,
        dqn: DQN,
        replay_buffer: deque[Transition],
        num_steps: int,
        score: int
) -> Tuple[deque[Transition], int, int]:

    obs_stack = replay_buffer[-1].next_obs_stack
    done = replay_buffer[-1].done

    if done:
        obs, _ = env.reset()
        obs_stack.append(obs)
        score = 0

    action, reward, next_obs, done = play_single_step(env, obs_stack, dqn, num_steps)
    num_steps += 1
    score += reward

    # 获取 next_obs_stack
    next_obs_stack = copy(obs_stack)
    next_obs_stack.append(next_obs)

    # 生成一条 transition 并放入回放缓冲区
    trans = Transition(
        obs_stack = obs_stack,
        action = action,
        reward = reward,
        next_obs_stack = next_obs_stack,
        done = done
    )
    replay_buffer.append(trans)
    return replay_buffer, num_steps, score


@torch.no_grad()
def get_samples(
        batch: List[Transition],
        dqn_tgt: DQN,
        gamma: float = GAMMA
) -> Tuple[torch.ByteTensor, torch.Tensor, torch.FloatTensor]:

    obs_stacks = []
    actions = []
    rewards =[]
    next_obs_stacks = []
    dones = []

    for trans in batch:
        obs_stacks.append(trans.obs_stack)
        actions.append(trans.action)
        rewards.append(trans.reward)
        next_obs_stacks.append(trans.next_obs_stack)
        dones.append(trans.done)

    # 转换成tensor
    obs_stacks_t = torch.as_tensor(
        np.asarray(obs_stacks),
        device=DEVICE,
        dtype=torch.uint8
    )
    actions_t = torch.as_tensor(
        np.asarray(actions),
        device=DEVICE,
    )
    rewards_t = torch.as_tensor(
        np.asarray(rewards),
        device=DEVICE
    )
    next_obs_stacks_t = torch.as_tensor(
        np.asarray(next_obs_stacks, dtype=np.uint8),
        device=DEVICE,
        dtype=torch.uint8
    )
    dones_t = torch.as_tensor(
        np.asarray(
            dones,
            dtype=np.bool
        ),
        device=DEVICE,
        dtype=torch.bool
    )
    assert dones_t.dtype is torch.bool

    # 计算 tgt_q_vals
    tgt_q_vals_t = dqn_tgt(next_obs_stacks_t).max(dim=1).values * gamma + rewards_t
    tgt_q_vals_t[dones_t] = rewards_t[dones_t]

    return obs_stacks_t, actions_t, tgt_q_vals_t


def cal_loss(
        replay_buffer: deque[Transition],
        dqn: DQN,
        dqn_tgt: DQN,
        criterion: nn.MSELoss
) -> torch.Tensor:

    # 获取样本
    batch = random.sample(replay_buffer, BATCH_SIZE)
    obs_stacks_t, actions_t, tgt_q_vals_t = get_samples(batch, dqn_tgt)

    # 损失计算
    q_vals_t: torch.Tensor = dqn(obs_stacks_t).gather(1, actions_t.unsqueeze(1))
    loss = criterion(q_vals_t, tgt_q_vals_t)
    return loss

def train(env: gym.Env, dqn: DQN, dqn_tgt: DQN) -> None:

    # 创建 criterion 和 optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)

    # 加载模型权重、优化器和步数
    if os.path.exists("./01_Optimizer.pth"):
        optimizer.load_state_dict(torch.load("./01_Optimizer.pth"))

    if os.path.exists("./01_Weights.pth"):
        dqn.load_state_dict(torch.load("./01_Weights.pth"))

    if os.path.exists("./01_checkpoint.json"):
        with open("./01_checkpoint.json", "r") as f:
            num_steps = json.load(f)["num_steps"]
    else:
        num_steps: int = 0

    # 创建回放缓冲区
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    # 先玩几步，初始化缓冲区
    replay_buffer, score = init_reply_buffer(env, replay_buffer)

    dqn_tgt.eval()

    episode_idx: int = 0
    while True:
        # 开始正式游玩
        dqn.eval()

        replay_buffer, num_steps, score = get_replay_buffer(env, dqn, replay_buffer, num_steps, score)
        if num_steps % TGT_UPDATE_FREQ == 0:
            dqn_tgt.load_state_dict(dqn.state_dict())

        # 记录 episode 数
        if replay_buffer[-1].done:
            episode_idx += 1
            print(f"Episode: {episode_idx:06d}; Score: {int(score):3d}")

            if score >= SCORE_BOUNDARY:
                print("Solved!")
                break

        if len(replay_buffer) == BUFFER_SIZE:

            # 模型训练
            dqn.train()

            # 计算损失
            loss = cal_loss(replay_buffer, dqn, dqn_tgt, criterion)
            print(f"loss: {loss.item():2.3f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 保存模型权重、优化器和步数
            torch.save(dqn.state_dict(), "./01_Weights.pth")
            torch.save(optimizer.state_dict(), "./01_Optimizer.pth")
            with open("./01_checkpoint.json", "w") as f:
                json.dump({"num_steps": num_steps}, f)

            print()


if __name__ == '__main__':

    gym.register_envs(ale_py)
    env = gym.make('ALE/Pong-v5', render_mode=RENDER_MODE, obs_type="grayscale")
    env = ResizeImg(env, IMG_SIZE)

    dqn = DQN((NUM_FRAMES, *IMG_SIZE), env.action_space.n).to(DEVICE)
    dqn_tgt = DQN((NUM_FRAMES, *IMG_SIZE), env.action_space.n).to(DEVICE)

    train(env, dqn, dqn_tgt)
    env.close()