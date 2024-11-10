import os

import torch
from torch.utils.data import Dataset

class DatasetSG(Dataset):
    def __init__(self, root_dir, transformer=None, train=True):
        self.root_dir = root_dir
        self.transformer = transformer
        data_names = [
            name for name in os.listdir(self.root_dir)
        ]
        data_names = sorted(data_names, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        train_num = 5000
        eval_num = len(data_names) - train_num
        start, end = 0, train_num
        if not train:
            start, end = train_num, train_num + eval_num

        self.datas = []
        for name in data_names[start:end]:
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
                    freq, amp = freq, self._get_int_num(amp)
                    data.append(amp)
            self.datas.append(data)

        # data0 = self.datas[1]
        # self.datas = []
        # for i in range(2000):
        #     self.datas.append(data0.copy())

    def __getitem__(self, idx):
        return self.transformer(torch.tensor(self.datas[idx][1:]).unsqueeze(1))

    def __len__(self):
        return len(self.datas)

    def _get_int_num(self, num):
        num1, num2 = map(float, num.split('e'))
        return num1*10**num2
