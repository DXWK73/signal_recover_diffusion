import os
import argparse

import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch.utils.data import DataLoader
from diffusers import *

from dataset import DatasetSG
from models.simple_net.model import DiT
from utils import normize, cosine_similarity

"""
    Train for signal recovery model.
"""


def eval(
    data: str,
    batch_size: int,
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

    eval_dataset = DatasetSG(root_dir=data, transformer=data_normalize, train=False)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        model = DiT.from_pretrained("ckpt/sgr").to(device)
        # vae = VAE.from_pretrained("ckpt/vae").to(device)
        scheduler = PNDMScheduler.from_pretrained("models/scheduler")
        scheduler.set_timesteps(50, device) # 推理步数，需根据噪声比例手动调整

        model.eval()
        thr = 0.985
        noise_weight = 0.1
        right_num = all_num = 0

        idx = 0
        for sigs in tqdm.tqdm(eval_loader, desc="Eval precessing"):
            if idx == len(eval_loader) - 1:
                break
            noise = torch.randn_like(sigs)
            xt = (1-noise_weight)*sigs + noise_weight*noise
            noised_sigs = xt
            sigs, xt = sigs.to(device), xt.to(device)
            xt *= scheduler.init_noise_sigma

            start_timestep = 46  # 起始时间步，需根据噪声幅度手动调整
            for t in scheduler.timesteps[start_timestep:]:
                t = t.unsqueeze(0).repeat(xt.size()[0])
                model_input = scheduler.scale_model_input(xt, t[0])
                pred = model(model_input, t)
                xt = scheduler.step(model_output=pred, timestep=t[0], sample=xt, return_dict=False)[0]

            # xt = noised_sigs.to(device)
            for ori_sig, denoised_sig in zip(sigs, xt):
                cos_similarity_score = cosine_similarity(ori_sig, denoised_sig)
                if cos_similarity_score > thr:
                    right_num += 1
                all_num += 1
            idx += 1

        print(f"accuracy for {noise_weight} noised signals: {format(right_num/all_num*100, '.2f')}%")

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--data", type=str, default="data/same_datas", help="train data path")
    arg.add_argument("--batch_size", type=int, default=32, help="batch size")
    arg.add_argument("--device", type=str, default="cuda:1", help="device")
    args = arg.parse_args()
    eval(**vars(args))
