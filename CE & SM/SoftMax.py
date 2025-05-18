import torch
import torch.nn as nn
import numpy as np

def SoftMax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

np_arrays = np.array([2.0, 0.4, 1.2])
np_output = SoftMax(np_arrays)
print(f"numpy={np_output}")

t_arrays = torch.tensor([2.0, 0.4, 1.2])
t_output = torch.softmax(t_arrays, dim=0)
print(f"torch={t_output}")