import os
from signal import signal

import matplotlib.pyplot as plt

def get_int_num(num):
    num1, num2 = map(float, num.split('e'))
    return num1*10**num2

def signal_load(signal_path):
    data = []
    with open(signal_path, mode='rb') as f:
        while True:
            freq_and_amp = f.readline().strip()
            if not freq_and_amp:
                break
            freq, amp = freq_and_amp.split()
            freq, amp = str(freq)[2:-1], str(amp)[2:-1]
            if not freq[0].isdigit():
                continue
            # freq, amp = get_int_num(freq), get_int_num(amp)
            data.append(float(amp))
    return data

def signal_save_to_img(signal, save_path):
    freqs = [i for i in range(len(signal))]
    amps = signal
    plt.plot(freqs, amps)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.savefig(save_path)

if __name__ == '__main__':
    signal = signal_load("1train_data_clean/generated_spectrum_data_0.txt")
    signal_save_to_img(signal, "generated_spectrum_data_0.png")

