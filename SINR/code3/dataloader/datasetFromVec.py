import torch
from torch.utils.data import Dataset


class DatasetFromVec(Dataset):
    def __init__(self, series, memory_size, future, transform=None, target_transform=None):
        super(Dataset, self).__init__()

        self.series = series
        self.transform = transform
        self.target_transform = target_transform
        self.memory_size = memory_size
        self.future = future

        self.series_num = series.shape[0]
        self.T = series.shape[1]

        self.x = torch.empty(self.series_num, self.T - self.memory_size - self.future, self.memory_size)
        self.y = torch.empty(self.series_num, self.T - self.memory_size - self.future, self.future)

        self.CreateXandY()

    def __len__(self):
        return self.y.shape[1]

    def __getitem__(self, index):
        x = self.x[:, index, :]
        y = self.y[:, index, :]
        return x, y

    def CreateXandY(self):
        future = self.future
        memory_size = self.memory_size
        T = self.series.shape[1]

        for idx in range(T - memory_size - future):
            self.x[:, idx, :] = self.series[:, idx: idx + memory_size]
        self.x = torch.unsqueeze(self.x, dim=3)
        for idx in range(T - memory_size - future):
            self.y[:, idx, :] = self.series[:, idx + memory_size: idx + memory_size + future]

