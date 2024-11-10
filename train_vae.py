import argparse
import os
import random
import pandas as pd
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch.utils.data import DataLoader

from dataset import DatasetSG
from models.simple_net.vae import VAE, VAE_MIN
from utils import normize

"""
    Train for VAE model.
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

    data_normalize = normize(10, 0, 1)
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     # torchvision.transforms.Normalize((0.5), (0.5)),
    # ])

    # train_dataset = torchvision.datasets.MNIST(download=True, root="min", train=True, transform=transform)
    train_dataset = DatasetSG(root_dir=train_data, transformer=data_normalize)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    model = VAE.from_pretrained("ckpt/vae").to(device)
    # model = VAE_MIN(in_channels=1, hidden_channels=16).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs // 2, eta_min=1e-4)

    train_log = {
        "epoch": [],
        "loss": [],
    }
    for epoch in range(epochs):
        model.train()
        _loss = 0.
        _kl_loss = 0.
        _rec_loss = 0.
        for idx, sigs in enumerate(train_loader):
            sigs = sigs.to(device)
            z, mu, logvar = model.encoder(sigs)
            pred = model.decoder(z)

            loss, kl_loss, rec_loss = model.loss(pred, sigs, mu, logvar)
            _loss += loss.item()
            _kl_loss += kl_loss.item()
            _rec_loss += rec_loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()

        opt_scheduler.step()
        print(f"[{epoch}/{epochs}] loss={format(_loss/len(train_loader), '.4f')} kl_loss={format(_kl_loss/len(train_loader), '.4f')} rec_loss={format(_rec_loss/len(train_loader), '.4f')}")

        train_log["epoch"].append(epoch)
        train_log["loss"].append(_loss/len(train_loader))
        # torch.save(model.state_dict(), "ckpt/vae_model.pth")

        df = pd.DataFrame(train_log)
        # df.to_csv("logs/train_log_vae.csv", index=False)
        if _loss/len(train_loader) <= stop_loss:
            break


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--train_data", type=str, default="data/same_datas", help="train data path")
    arg.add_argument("--epochs", type=int, default=512, help="number of epochs")
    arg.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    arg.add_argument("--batch_size", type=int, default=32, help="batch size")
    arg.add_argument("--device", type=str, default="cuda:1", help="device")
    arg.add_argument("--stop_loss", type=float, default=0.0001, help="stop train loss")
    args = arg.parse_args()
    train(**vars(args))
