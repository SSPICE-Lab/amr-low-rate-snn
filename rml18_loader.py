import os

import h5py
import numpy as np
import torch.utils.data


class RML18Loader(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 indices=None,
                 snr=None,
                 modulation=None,
                 gains=None,
                 cache=False,
                 cache_str="",
                 transform=None,
                 target_transform=None):
        self.path = path
        self.transform = transform
        self.target_transform = target_transform

        if cache:
            if snr is not None:
                if isinstance(snr, (list, np.ndarray)):
                    snr_str = f"{snr[0]}_{snr[-1]}"
                else:
                    snr_str = f"{snr}"
            else:
                snr_str = "snr_all"
            if modulation is not None:
                if not isinstance(modulation, list):
                    modulation = [modulation]
                mod_str = "_".join(str(x) for x in modulation)
            else:
                mod_str = "mod_all"
            cache_file = f"cache/rml18_{cache_str}_{snr_str}_{mod_str}"
            if os.path.exists(f"{cache_file}_data.npy"):
                self.data = np.load(f"{cache_file}_data.npy")
                self.labels = np.load(f"{cache_file}_labels.npy")
                return

        with h5py.File(self.path, "r") as f:
            self.data = f["X"][:]
            self.labels = f["Y"][:]
            self.labels = np.argmax(self.labels, axis=1)
            self.snrs = f["Z"][:]
            if indices is not None:
                self.data = self.data[indices]
                self.labels = self.labels[indices]
                self.snrs = self.snrs[indices]

        if snr is not None:
            if isinstance(snr, (list, np.ndarray)):
                snr_indices = np.zeros(0, dtype=int)
                for s in snr:
                    snr_indices = np.concatenate([snr_indices, np.where(self.snrs == s)[0]])
                indices = snr_indices
            else:
                indices = np.where(self.snrs == snr)[0]
            self.data = self.data[indices]
            self.labels = self.labels[indices]

        if modulation is not None:
            if not isinstance(modulation, list):
                modulation = [modulation]

            modulation_indices = np.zeros(0, dtype=int)
            for m in modulation:
                modulation_indices = np.concatenate([modulation_indices, np.where(self.labels == m)[0]])
            indices = modulation_indices
            self.data = self.data[indices]
            self.labels = self.labels[indices]
            # modulation is a list of classes, so we need to update the labels
            for i, m in enumerate(modulation):
                self.labels[self.labels == m] = i

        if gains is not None:
            for i, gain in enumerate(gains):
                self.data[self.labels == i] *= gain

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
