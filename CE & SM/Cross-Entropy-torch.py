import torch
import torch.nn as nn
import numpy as np

# Hello myself, today we're gonna learn 
# about Cross-Entropy-Loss!

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# nsamples x nclasses 
Y_predicted = torch.tensor([[2.0, 1.0, 0.1]])
X_predicted = torch.tensor([[0.1, 1.0, 2.0]])

l1 = loss(Y_predicted, Y)
l2 = loss(X_predicted, Y)

print(f"Good Result: {l1.item()}")
print(f"Bad Result: {l2.item()}")