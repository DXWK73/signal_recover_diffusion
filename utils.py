import os

import torch
import PIL.Image
import matplotlib.pyplot as plt

def normize(mx, mean, std):
    def get_normize(x):
        x = torch.clamp(x, 0, 1000)
        return (x/mx - mean) / std
    return get_normize

def cosine_similarity(tensor1, tensor2):
    # 计算分子：点积
    dot_product = torch.sum(tensor1 * tensor2)

    # 计算分母：张量的范数乘积
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)

    # 计算余弦相似度
    similarity = dot_product / (norm1 * norm2)
    return similarity.item()

def signal_postprocess(signal):
    signal = signal*10
    return signal

def signal_save_to_txt(signal, save_file):
    s, _ = signal.size()
    freq = 0
    with open(save_file, 'w') as f:
        f.write("Frequency(Hz)	Amplitude \n")
        while freq < s:
            amp = signal[freq, 0].item()
            f.write(f"{freq}, {amp:.1f}\n")
            freq += 1

def signal_save_to_img(signal, save_path):
    freqs = [i for i in range(signal.size()[0])]
    amps = [signal[freq, 0].item() for freq in freqs]

    y_min, y_max = -10, 20
    # 固定 y 轴范围
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.plot(freqs, amps)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    # 覆盖保存
    plt.savefig(save_path, bbox_inches='tight')  # 去掉多余边框
    plt.close()  # 关闭图表以释放内存

def get_int_num(num):
    num1, num2 = map(float, num.split('e'))
    return num1*10**num2

def signal_load(signal_path):
    data = []
    transformer = normize(10, 0, 1)
    with open(signal_path, mode='rb') as f:
        while True:
            freq_and_amp = f.readline().strip()
            if not freq_and_amp:
                break
            freq, amp = freq_and_amp.split()
            freq, amp = str(freq)[2:-1], str(amp)[2:-1]
            if not freq[0].isdigit():
                continue
            freq, amp = float(freq), get_int_num(amp)
            data.append(amp)
    return transformer(torch.tensor([data])).unsqueeze(2)
