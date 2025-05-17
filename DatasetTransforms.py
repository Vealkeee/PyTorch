import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset

class WineDataset(Dataset):

    # Data loading
    def __init__(self, transform=None):
        xy = np.loadtxt('./TextData/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # fuck the first collumn, but not in self.y
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1 
        self.n_samples = xy.shape[0]

        self.transform = transform

    # Allows for indexing, as if we do: Dataset[0]
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    # Allows to call lenght of the Dataset
    def __len__(self):
        return self.n_samples
    
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))