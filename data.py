import numpy as np
import torch.utils.data as torchdata

class Dataset(torchdata.Dataset):
    def __init__(self, data:np.ndarray, num_negative:int=0):
        self.data = data
        self.num_negative = num_negative
        self.idx = np.arange(np.max(data[:,1]), dtype=int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item:int) -> list:
        if self.num_negative == 0:
            return self.data[item].tolist()
        else:
            return self.data[item], self._sample_negative(self.data[item][1])

    def _sample_negative(self, pos:int):
        neg = np.random.choice(self.idx, self.num_negative + 1, replace=False)
        idx = np.nonzero(pos == neg)[0]
        if len(idx) > 0:
            idx = idx.item()
            return np.concatenate([neg[:idx], neg[idx+1:]])
        else:
            return neg[:self.num_negative]


class DataLoader(torchdata.DataLoader):
    def __init__(self, dataset:Dataset, batch_size:int, shuffle:bool, num_workers:int):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, num_workers=num_workers)
