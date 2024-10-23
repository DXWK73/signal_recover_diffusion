import os
import random
import tqdm
import argparse

import torch
from diffusers import DDIMScheduler

from models.simple_net.model import SGR
from utils import signal_postprocess, signal_save, signal_show

"""
    Inference for signal recovery model.
"""

def infer(
    model_weights: str,
    diffusion_steps: int,
    device: torch.device,
):
    """
    执行模型的推理过程。

    参数：
    model_weights (str): 模型权重的路径，指向预训练的模型文件，用于加载模型参数。
    diffusion_steps (int): 扩散模型推理过程中使用的步数，影响生成质量与速度。
    device (torch.device): 模型推理运行的设备，可以是 CPU 或 GPU（"cuda"）。

    返回：
    None
    """

    with torch.no_grad():
        model = SGR.from_pretrained(model_weights).to(device)
        scheduler = DDIMScheduler.from_pretrained("models/scheduler")
        scheduler.set_timesteps(diffusion_steps, device)

        generator = torch.Generator()
        rand = random.randint(1, 1000)
        generator.manual_seed(rand)
        xt = torch.randn(1, 5000, 1, generator=generator).to(device)
        xt *= scheduler.init_noise_sigma

        for t in tqdm.tqdm(scheduler.timesteps, desc="diffusion process"):
            t = t.unsqueeze(0)
            model_input = scheduler.scale_model_input(xt, t[0]-1)
            pred = model(model_input, t-1)
            xt = scheduler.step(model_output=pred, timestep=t[0]-1, sample=xt, return_dict=False)[0]

        signal = signal_postprocess(xt)[0]
        signal_show(signal)
        signal_save(signal, "assets/generate_signal.txt")

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--model_weights", type=str, default="ckpt/sgr", help="model weights")
    arg.add_argument("--diffusion_steps", type=int, default=50, help="diffusion steps")
    arg.add_argument("--device", type=str, default="cuda:1", help="device")
    args = arg.parse_args()
    infer(**vars(args))
