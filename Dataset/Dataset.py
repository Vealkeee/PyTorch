import torch
import torchvision
import numpy as np
import math

from torch.utils.data import DataLoader, Dataset

class WineDataset(Dataset):

    # Data loading
    def __init__(self):
        xy = np.loadtxt('./TextData/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # fuck the first collum, but not in self.y
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1 
        self.n_samples = xy.shape[0]

    # Allows for indexing, as if we do: Dataset[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    # Allows to call lenght of the Dataset
    def __len__(self):
        return self.n_samples
    
dataset = WineDataset()

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
n_epochs = 200
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

# print(total_samples, n_iterations)

for epoch in range(n_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if (i+1) % 10 == 0:
            print(f'epoch={epoch+1}/{n_epochs}, step={i+1}/{n_iterations}, inputs={inputs.shape}')


# Only getting the next iter.
# dataiter = iter(dataloader)
# data = next(dataiter)
# features, labels = data
# print(features, labels)

# Thing below used to take a look at the dataset
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)