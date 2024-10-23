import argparse
import os
import random
import pandas as pd
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler

from dataset import DatasetSG
from models.simple_net.model import SGR
from utils import normize, signal_show, signal_postprocess

"""
    Train for signal recovery model.
"""


def train(
    train_data: str,
    epochs: int,
    lr: float,
    batch_size: int,
    stop_loss: float,
    device: torch.device,
):
    """
        执行模型的训练过程。

        参数：
        train_data (str): 训练数据的路径，指向用于训练的文件或目录。
        epochs (int): 模型训练的轮数，决定模型将遍历数据集的次数。
        lr (float): 学习率，控制每次参数更新时的步长大小。
        batch_size (int): 批处理大小，决定每次梯度更新时所用的样本数量。
        stop_loss (float): 提前停止训练的损失值门限，当损失低于此值时停止训练。
        device (torch.device): 模型训练的设备，可以是 CPU 或 GPU（"cuda"）。

        返回：
        None
    """

    data_normalize = normize(1000, 0.5, 0.5)

    train_dataset = DatasetSG(root_dir=train_data, transformer=data_normalize)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    model = SGR.from_pretrained("ckpt/sgr").to(device)
    scheduler = DDIMScheduler.from_pretrained("models/scheduler")
    scheduler.set_timesteps(1000, device)

    Loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs // 2, eta_min=1e-6)

    train_log = {
        "epoch": [],
        "loss": [],
    }
    for epoch in range(epochs):
        model.train()
        _loss = 0.
        for idx, sigs in enumerate(train_loader):
            b, _, _ = sigs.size()
            t_idx = random.randint(0, len(scheduler.timesteps) - 1)
            t = scheduler.timesteps[t_idx].unsqueeze(0).repeat(b)

            # TODO: Unsure whether need signal sample?

            noise = torch.rand_like(sigs)
            sigs, noise = sigs.to(device), noise.to(device)
            sigs_noised = scheduler.add_noise(sigs, noise, t-1)

            pred = model(sigs_noised, t-1)
            loss = Loss(pred, sigs)
            _loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        opt_scheduler.step()
        print(f"[{epoch}/{epochs}] loss={format(_loss/len(train_loader), '.4f')}")
        train_log["epoch"].append(epoch)
        train_log["loss"].append(_loss/len(train_loader))
        torch.save(model.state_dict(), "ckpt/model.pth")

        df = pd.DataFrame(train_log)
        df.to_csv("logs/train_log_0.3t.csv", index=False)
        if _loss/len(train_loader) <= stop_loss:
            break


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--train_data", type=str, default="data/2eval_data/", help="train data path")
    arg.add_argument("--epochs", type=int, default=512, help="number of epochs")
    arg.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    arg.add_argument("--batch_size", type=int, default=8, help="batch size")
    arg.add_argument("--device", type=str, default="cuda:1", help="device")
    arg.add_argument("--stop_loss", type=float, default=0.0003, help="stop train loss")
    args = arg.parse_args()
    train(**vars(args))
