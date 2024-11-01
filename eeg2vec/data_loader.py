import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # Shape: [num_samples, num_channels (5), sequence_length]
        self.labels = labels  # Shape: [num_samples, 5]
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[idx]

def get_dataloader(data, labels, batch_size, shuffle=True):
    dataset = EEGDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)