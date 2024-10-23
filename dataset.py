import os

import torch
from torch.utils.data import Dataset

class DatasetSG(Dataset):
    def __init__(self, root_dir, transformer=None):
        self.root_dir = root_dir
        self.transformer = transformer
        data_names = [
            name for name in os.listdir(self.root_dir)
        ]
        data_names = sorted(data_names, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        self.datas = []
        for name in data_names[:1000]:
            data = []
            with open(os.path.join(self.root_dir, name), mode='rb') as f:
                while True:
                    freq_and_amp = f.readline().strip()
                    if not freq_and_amp:
                        break
                    freq, amp = freq_and_amp.split()
                    freq, amp = str(freq)[2:-1], str(amp)[2:-1]
                    if not freq[0].isdigit():
                        continue
                    freq, amp = self._get_int_num(freq), self._get_int_num(amp)
                    data.append(amp)
            self.datas.append(data)

    def __getitem__(self, idx):
        return self.transformer(torch.tensor(self.datas[idx]).unsqueeze(1))

    def __len__(self):
        return len(self.datas)

    def _get_int_num(self, num):
        num1, num2 = map(float, num.split('e'))
        return num1*10**num2
