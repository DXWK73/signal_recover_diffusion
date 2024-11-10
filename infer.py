import random

import tqdm
import argparse

import torch
from diffusers import *

from models.simple_net.model import DiT
from utils import signal_postprocess, signal_save_to_img, signal_save_to_txt, signal_load, cosine_similarity

"""
    Evaluating for signal recovery model.
"""

def infer(
    ori_signal: str,
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
        model = DiT.from_pretrained(model_weights).to(device)
        scheduler = PNDMScheduler.from_pretrained("models/scheduler")
        scheduler.set_timesteps(50, device=device) # 推理步数，需根据噪声比例手动调整

        generator = torch.Generator()
        rand = random.randint(1, 1000)
        generator.manual_seed(rand)
        # xt = torch.randn(1, 500, 1, generator=generator).to(device)
        xt = signal_load(ori_signal).to(device)[:, 1:, :]
        ori_signal_tensor = xt
        ori_signal = signal_postprocess(xt)[0]
        signal_save_to_img(ori_signal, "examples/ori_signal.png")

        b = 0.1
        xt = (1-b)*xt + b*torch.randn_like(xt, device=device)
        noised_signal = signal_postprocess(xt)[0]
        signal_save_to_img(noised_signal, "examples/noised_signal.png")

        xt *= scheduler.init_noise_sigma

        start_timestep = 46 # 起始时间步，需根据噪声幅度手动调整
        for t in tqdm.tqdm(scheduler.timesteps[start_timestep:], desc="diffusion process"):
            t = t.unsqueeze(0)
            model_input = scheduler.scale_model_input(xt, t[0])
            pred = model(model_input, t)
            xt = scheduler.step(model_output=pred, timestep=t[0], sample=xt, return_dict=False)[0]

        print(cosine_similarity(ori_signal_tensor, xt))
        signal = signal_postprocess(xt)[0]
        signal_save_to_img(signal, "examples/generate_signal.png")
        signal_save_to_txt(signal, "examples/generate_signal.txt")

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("--model_weights", type=str, default="ckpt/sgr", help="model weights")
    arg.add_argument("--diffusion_steps", type=int, default=50, help="diffusion steps")
    arg.add_argument("--device", type=str, default="cuda:1", help="device")
    arg.add_argument("--ori_signal", type=str, default="examples/original_signal1.txt", help="ori signal input")
    args = arg.parse_args()
    infer(**vars(args))
