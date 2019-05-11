import numpy as np
import torch.utils.data as torchdata

class Dataset(torchdata.Dataset):
    def __init__(self, data:np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item:int) -> list:
        return self.data[item].tolist()


class DataLoader(torchdata.DataLoader):
    def __init__(self, dataset:Dataset, batch_size:int, shuffle:bool, num_workers:int):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, num_workers=num_workers)
