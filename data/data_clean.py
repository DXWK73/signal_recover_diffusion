import os

from mpmath.tests.extratest_gamma import maxdps

"""
    Clean the datas for removing the datas with single peak signal.
"""

def signal_save_to_txt(signal, save_file):
    s = len(signal)
    freq = 0
    with open(save_file, 'w') as f:
        f.write("Frequency(Hz)	Amplitude \n")
        while freq < s:
            amp = signal[freq]
            f.write(f"{freq}, {amp:.1f}\n")
            freq += 1

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
            freq, amp = get_int_num(freq), get_int_num(amp)
            data.append(amp)
    return data

def clean_datas(ori_path, save_path):
    ori_data_names = sorted(os.listdir(ori_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    save_datas = []

    # find need to save files
    for name in ori_data_names:
        amps = signal_load(os.path.join(ori_path, name))
        if max(amps) <= 1000:
            save_datas.append(amps)

    # save files
    id = 0
    for data in save_datas:
        signal_save_to_txt(data, os.path.join(save_path, f"generated_spectrum_data_{id}.txt"))
        id += 1

if __name__ == '__main__':
    ori_path = "1train_data"
    save_path = "1train_data_clean"
    clean_datas(ori_path, save_path)
