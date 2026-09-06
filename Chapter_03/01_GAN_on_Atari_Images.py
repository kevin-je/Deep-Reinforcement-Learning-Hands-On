import os

import gymnasium as gym
import ale_py

import numpy as np
from numpy.lib.format import open_memmap
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from typing import List, Iterator, Tuple

from tqdm import tqdm

from matplotlib import pyplot as plt

ENV_NAMES = ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5"]

NUM_IMGS = 10_000
SAMPLE_STEP = 100
IMGS_FILE_PATH = "./01_Images_Array.npy"

RESIZED_IMG_SIZE = (64, 64)
NOISE_CHANNELS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
OPTIM_BETAS = (0.5, 0.999)
EPOCHS = 20
BATCH_SIZE = 64

def get_observations(env: gym.Env) -> Iterator[gym.core.ObsType]:
    observation, _ = env.reset()
    yield observation

    while True:
        action = env.action_space.sample()

        observation, _, terminated, truncated, _ = env.step(action)

        # 先把这一步得到的 observation 交出去
        yield observation

        # 然后再判断是否需要开始下一局
        if terminated or truncated:
            observation, _ = env.reset()
            yield observation


def array_to_image(array: np.ndarray, file_name: str) -> Image.Image:
    img = Image.fromarray(array)
    img.save(file_name)
    return img


class InputWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32
        )

    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        img = Image.fromarray(observation)
        # 图像缩放
        img: np.ndarray = np.array(img.resize(RESIZED_IMG_SIZE), dtype=np.float32) * 2 / 255.0 - 1.0
        # 将通道维度移至最前面
        img = np.moveaxis(img, -1, 0)
        return img


def save_imgs_array(
        envs: List[gym.core.Env],
        num_imgs: int = NUM_IMGS,
        sample_step: int = SAMPLE_STEP,
        img_size: Tuple[int, int] = RESIZED_IMG_SIZE,
        file_path: str = IMGS_FILE_PATH
) -> None:

    if os.path.exists(file_path):
        return

    # 创建一个npy文件存储所有图像数组，并开启内存映射
    data = open_memmap(file_path, mode="w+", dtype=np.float32, shape=(len(envs) * num_imgs, 3, *img_size))

    for i, env in enumerate(envs):
        env = InputWrapper(env)
        imgs_gen = get_observations(env)

        for j, img in enumerate(imgs_gen, 1):
            if j % sample_step == 0:
                data[i*num_imgs + j//sample_step - 1] = img

            if j // sample_step == num_imgs:
                break

        data.flush()
        env.close()

    # 断言 data 的长度与预期一致
    assert len(data) == num_imgs * len(envs)
    # 释放 data
    del data

class Generator(nn.Module):
    def __init__(self, inputs_channel: int):
        super().__init__()

        conv_t1 = nn.ConvTranspose2d(
            inputs_channel, 1024,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )

        conv_t2 = nn.ConvTranspose2d(
            1024, 512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        conv_t3 = nn.ConvTranspose2d(
            512, 256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        conv_t4 = nn.ConvTranspose2d(
            256, 128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        conv_t5 = nn.ConvTranspose2d(
            128, 3,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

        self.net = nn.Sequential(
            conv_t1,
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            conv_t2,
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            conv_t3,
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            conv_t4,
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            conv_t5,
            nn.Tanh()
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        conv1 = nn.Conv2d(
            3, 64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        conv2 = nn.Conv2d(
            64, 128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        conv3 = nn.Conv2d(
            128, 256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        conv4 = nn.Conv2d(
            256, 512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        conv5 = nn.Conv2d(
            512, 1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )

        self.net = nn.Sequential(
            conv1,
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            conv2,
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            conv3,
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            conv4,
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            conv5,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).view(-1)


class RealImgsDataset(Dataset):
    def __init__(self, file_path: str = IMGS_FILE_PATH):
        self.imgs_array = np.load(file_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.imgs_array)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.imgs_array[idx].copy()


class GenLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, dis_results: torch.Tensor) -> torch.Tensor:
        return self.loss(dis_results, torch.ones(dis_results.size(0), device=dis_results.device))


class DisLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = nn.BCEWithLogitsLoss()
        self.loss2 = nn.BCEWithLogitsLoss()

    def forward(self, real_outputs: torch.Tensor, fake_outputs: torch.Tensor) -> torch.Tensor:
        return (self.loss1(real_outputs, torch.ones_like(real_outputs, device=real_outputs.device)) +
                self.loss2(fake_outputs, torch.zeros_like(fake_outputs, device=fake_outputs.device)))

def iterate_batch(
        batch: torch.Tensor,
        gen_net: nn.Module,
        dis_net: nn.Module,
        gen_optimizer: optim.Optimizer,
        dis_optimizer: optim.Optimizer,
        criterion_gen: nn.Module = GenLoss(),
        criterion_dis: nn.Module = DisLoss(),
) -> Tuple[float, float]:

    # 生成假图像
    noise = torch.randn(batch.size(0), NOISE_CHANNELS, 1, 1, device=batch.device)
    fake_imgs = gen_net(noise)

    # 训练判别器
    real_outputs = dis_net(batch)
    fake_outputs = dis_net(fake_imgs.detach())
    dis_loss: torch.Tensor = criterion_dis(real_outputs, fake_outputs)
    dis_optimizer.zero_grad()
    dis_loss.backward()
    dis_optimizer.step()

    # 训练生成器
    dis_outputs = dis_net(fake_imgs)
    gen_loss: torch.Tensor = criterion_gen(dis_outputs)
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    return dis_loss.item(), gen_loss.item()


def inference(num_h: int, num_w: int) -> None:
    gen_net: nn.Module = Generator(inputs_channel=NOISE_CHANNELS)
    gen_net.load_state_dict(torch.load("01_Weights/gen_net.pth"))

    gen_net.eval()
    with torch.no_grad():

        noise = torch.randn(num_h*num_w, NOISE_CHANNELS, 1, 1)
        imgs = (gen_net(noise).permute(0, 2, 3, 1) + 1) / 2

        fig, axes = plt.subplots(num_h, num_w, figsize=(10, 10))

        for i, ax in enumerate(axes.flat):
            ax.imshow(imgs[i])
            ax.axis("off")

        plt.tight_layout()
        plt.show()
        plt.close()

def train(require_infer: bool = True) -> None:
    # 将 ale_py 与 gymnasium 绑定
    gym.register_envs(ale_py)
    envs = [gym.make(env_name) for env_name in ENV_NAMES]

    # 获取并保存真实图像
    save_imgs_array(envs=envs)

    # 创建数据集
    real_imgs_dataset = RealImgsDataset()
    real_imgs_dataloader = DataLoader(
        real_imgs_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 创建网络
    gen_net = Generator(inputs_channel=NOISE_CHANNELS).to(DEVICE)
    dis_net = Discriminator().to(DEVICE)

    # 创建优化器
    gen_optimizer = optim.Adam(
        gen_net.parameters(),
        lr=LEARNING_RATE_G,
        betas=OPTIM_BETAS
    )
    dis_optimizer = optim.Adam(
        dis_net.parameters(),
        lr=LEARNING_RATE_D,
        betas=OPTIM_BETAS
    )

    # 加载模型权重与优化器
    if os.path.exists("01_Weights/gen_net.pth"):
        gen_net.load_state_dict(torch.load("01_Weights/gen_net.pth"))
    if os.path.exists("01_Weights/dis_net.pth"):
        dis_net.load_state_dict(torch.load("01_Weights/dis_net.pth"))

    if os.path.exists("01_Optimizers/gen_optimizer.pth"):
        gen_optimizer.load_state_dict(torch.load("01_Optimizers/gen_optimizer.pth"))
    if os.path.exists("01_Optimizers/dis_optimizer.pth"):
        dis_optimizer.load_state_dict(torch.load("01_Optimizers/dis_optimizer.pth"))

    # 批次迭代
    for epoch in range(1, EPOCHS + 1):

        print(f"------ Epoch {epoch:03d} ------")

        loops = tqdm(real_imgs_dataloader, colour="green")

        for batch_idx, real_imgs in enumerate(loops):
            real_imgs = real_imgs.to(DEVICE)

            loops.set_description(f"Batch {epoch:04d}")
            dis_loss, gen_loss = iterate_batch(
                batch=real_imgs,
                gen_net=gen_net,
                dis_net=dis_net,
                gen_optimizer=gen_optimizer,
                dis_optimizer=dis_optimizer
            )
            loops.set_postfix({"dis_loss": dis_loss, "gen_loss": gen_loss})

        print()

        # 保存权重
        os.makedirs("01_Weights", exist_ok=True)
        torch.save(gen_net.state_dict(), "01_Weights/gen_net.pth")
        torch.save(dis_net.state_dict(), "01_Weights/dis_net.pth")

        # 保存优化器
        os.makedirs("01_Optimizers", exist_ok=True)
        torch.save(gen_optimizer.state_dict(), "01_Optimizers/gen_optimizer.pth")
        torch.save(dis_optimizer.state_dict(), "01_Optimizers/dis_optimizer.pth")

        # 生成器训练效果可视化
        if require_infer:
            inference(num_h=4, num_w=4)


if __name__ == "__main__":

    train(require_infer=True)