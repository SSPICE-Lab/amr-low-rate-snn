import os
import pickle

import numpy as np
import torch.utils.data


class RML16Loader(torch.utils.data.Dataset):
    _modulations = [
        'BPSK',
        'QPSK',
        '8PSK',
        'GFSK',
        'CPFSK',
        'QAM16',
        'QAM64',
        'PAM4',
        'AM-SSB',
        'AM-DSB',
        'WBFM'
    ]
    _snrs = np.arange(-20, 19, 2)

    def __init__(self,
                 path,
                 cache=True,
                 cache_file="",
                 indices=None,
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform

        if cache:
            if os.path.exists(f"{cache_file}_data.npy"):
                self.data = np.load(f"{cache_file}_data.npy")
                self.labels = np.load(f"{cache_file}_labels.npy")
                return

        with open(path, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')

        data = []
        labels = []
        for i, modulation in enumerate(self._modulations):
            modulation_data = np.concatenate([pickle_data[(modulation, snr)] for snr in self._snrs], axis=0)
            data.append(modulation_data)
            labels.append(np.zeros(modulation_data.shape[0]) + i)
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        if indices is not None:
            data = data[indices]
            labels = labels[indices]

        self.data = data
        self.labels = labels.astype(np.int64)

        if cache:
            if not os.path.exists(cache_file):
                os.makedirs("cache", exist_ok=True)
                np.save(f"{cache_file}_data.npy", self.data)
                np.save(f"{cache_file}_labels.npy", self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.target_transform:
            label = self.target_transform(label)

        return sample, label
