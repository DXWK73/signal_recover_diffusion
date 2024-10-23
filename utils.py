import os

import torch
import PIL.Image
import matplotlib.pyplot as plt

def normize(mx, mean, std):
    def get_normize(x):
        x = torch.clamp(x, 0, 1000)
        return (x/mx - mean) / std
    return get_normize

def signal_postprocess(signal):
    signal = (signal*0.5+0.5)*1000
    return torch.clamp(signal, min=0)

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
    plt.plot(freqs, amps)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.savefig(save_path)
