import ale_py
import gymnasium as gym

import numpy as np

import random

import torch
import torch.nn as nn
from torch import optim

import cv2 as cv

import os

from dataclasses import dataclass

from collections import deque

from typing import Tuple, List

from torchvision.datasets import flickr
from tqdm import tqdm

import json


# 超参数
LEARNING_RATE = 2e-4
BATCH_SIZE = 32

BUFFER_SIZE = 10_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON_START = 0.2
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
        return self.sequential_ff(self.sequential_conv(obs/255.0))


@dataclass
class Transition:
    obs_stack: torch.Tensor
    action: int
    reward: int | float
    next_obs_stack: torch.Tensor
    done: bool


def play_single_step(
        env: gym.Env,
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
        actions_val: torch.Tensor = dqn(
            torch.as_tensor(
                obs_stack,
                dtype=torch.uint8,
                device = DEVICE
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
    assert type(next_obs) is np.ndarray
    assert type(reward) is int
    done: bool = True if terminated or truncated else False

    # 渲染画面
    if RENDER_MODE == "human":
        env.render()

    return action, reward, next_obs, done


def play_single_episode(
        env: gym.Env,
        dqn: DQN,
        dqn_tgt: DQN,
        num_steps: int,
        replay_buffer: deque,
        num_frames: int = NUM_FRAMES
) -> Tuple[deque, int, float]:

    obs_stack: deque = deque(maxlen=num_frames)       # [tensor(84, 84)]
    next_obs_stack: deque = deque(maxlen=num_frames)

    score = 0

    # 初始化环境
    obs, _ = env.reset()

    while True:
        obs_stack.append(obs)
        obs_stack_t = torch.as_tensor(
            np.asarray(obs_stack),
            dtype=torch.uint8,
            device=DEVICE
        )

        action, reward, next_obs, done = play_single_step(
            env,
            obs_stack_t,
            dqn,
            dqn_tgt,
            num_steps
        )
        num_steps += 1
        score += reward

        next_obs_stack.append(next_obs)

        # 生成一条 transition
        if len(obs_stack) == num_frames:
            trans = Transition(
                obs_stack_t,
                action,
                reward,
                torch.as_tensor(
                    np.asarray(next_obs_stack),
                    dtype=torch.uint8
                ),
                done
            )

            # 将生成的转移放入回放缓冲区
            replay_buffer.append(trans)

        obs = next_obs

        if done: break

    return replay_buffer, num_steps, score


class ResizeImg(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, img_size: Tuple[int, int] = IMG_SIZE) -> None:
        super().__init__(env)

        self.img_size = img_size

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # img = Image.fromarray(obs)
        # 转换为灰度图
        # img = img.convert("L")
        # 裁剪
        # img = img.crop((0, 30, 160, 190))
        observation = observation[30:191, :161]
        # 缩放
        observation = cv.resize(observation, self.img_size)
        return observation


def cal_q_val_tgt(dqn_tgt: DQN, next_obs_stack_t: torch.ByteTensor, gamma: float = GAMMA) -> float:
    if trans.done:
        q_val_tgt = trans.reward

    else:
        next_obs_stack = trans.next_obs_stack
        q_val_tgt = dqn_tgt(next_obs_stack.unsqueeze(0)).max(dim=1).values.item() * gamma + trans.reward

    return q_val_tgt


def cal_epsilon(
        step: int,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        decay_steps: int = DECAY_STEPS) -> float:
    return max(epsilon_start - (step / decay_steps), epsilon_end)


def get_samples(batch: List[Transition], dqn_tgt: DQN) -> Tuple[torch.ByteTensor, torch.FloatTensor]:
    obs_stacks_t = []
    actions_t = []
    rewards_t =[]

    for trans in batch:
        obs_stacks_t.append(trans.obs_stack)
        actions_t.append(trans.action)
        rewards_t.append(trans.reward)




def train(env: gym.Env, dqn: DQN, dqn_tgt: DQN) -> None:
    # 创建回放缓冲区
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    # 创建 criterion 和 optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dqn.parameters(), lr=LEARNING_RATE)

    # 加载模型权重、优化器和步数
    if os.path.exists("./01_Optimizer.pth"):
        optimizer.load_state_dict(torch.load("./01_Optimizer.pth"))

    if os.path.exists("./01_Weights.pth"):
        dqn.load_state_dict(torch.load("./01_Weights.pth"))

    if os.path.exists("./01_check_point.json"):
        with open("./01_check_point.json", "r") as f:
            num_steps = json.load(f)["num_steps"]
    else: num_steps: int = 0

    episode_idx: int = 0
    while True:
        # 开始游玩
        print("Playing...")
        dqn.eval()
        with torch.no_grad():

            replay_buffer, num_steps, score = play_single_episode(
                env,
                dqn,
                dqn_tgt,
                num_steps,
                replay_buffer
            )

            episode_idx += 1
            print(f"Episode: {episode_idx:06d}; Score: {score:3d}")
            if score == SCORE_BOUNDARY:
                print("Solved!")
                break


        # 模型训练
        print("Training...")
        # 从回放缓冲区采样
        batch = random.sample(replay_buffer, BATCH_SIZE)

        dqn.train()
        dqn = dqn.to(DEVICE)
        loops = tqdm(dataloader, colour="green")
        for batch_idx, (obs_stacks_t, q_vals_tgt, actions) in enumerate(loops, 1):
            loops.set_description(f"Batch {batch_idx}")

            q_vals = dqn(obs_stacks_t.to(DEVICE))
            q_vals_tgt = q_vals_tgt.to(DEVICE)
            actions = actions.to(DEVICE)

            q_vals = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

            # 计算损失
            loss = criterion(q_vals, q_vals_tgt)
            loops.set_postfix({"loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存模型权重、优化器和步数
        torch.save(dqn.state_dict(), "./01_Weights.pth")
        torch.save(optimizer.state_dict(), "./01_Optimizer.pth")
        with open("./01_check_point.json", "w") as f:
            json.dump({"num_steps": num_steps}, f)

        print()


if __name__ == '__main__':

    gym.register_envs(ale_py)
    env = gym.make('ALE/Pong-v5', render_mode=RENDER_MODE, obs_type="grayscale")
    env = ResizeImg(env, IMG_SIZE)

    dqn = DQN((NUM_FRAMES, *IMG_SIZE), env.action_space.n).to(DEVICE)
    dqn_tgt = DQN((NUM_FRAMES, *IMG_SIZE), env.action_space.n)


    train(env, dqn, dqn_tgt)
    env.close()