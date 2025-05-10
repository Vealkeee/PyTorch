import torch
import numpy as np

# Short explanation of the things below, we're creating a tensor with require of Gradient
# Gradient calculation is happens as we getting some sort of number manipulation which we want to get
# then we can get the S(scala) value with .mean() and call the .backward() fn with the respect to X value, 
# after which we're gonna call X's gradient value, otherwise we can creator the V(vector probably) value
# ang give it a tensor, with a dtype of torch.float32 for example, and use this V in .backward(v) fn.

# Requires_grad, mean that's the X value requires the gradient
# that we can calculate with backpropagation 

x = torch.randn(3, requires_grad=True)
y = x + 3
z = y * x * 3
print(z)

z = z.mean()
# no need in Z value if we're creating the V value to put in the backward
v = torch.tensor([0.1, 0.2, 0.03], dtype=torch.float32)
z.backward(v)
print(x.grad)

# if we want to prevent our tensor to have requires_grad, we can use three type of functions below
x = torch.randn(3, requires_grad=True)
x.requires_grad_(False)
print(x)
y = x.detach()
print(y)
with torch.no_grad():
    print(x)

# training example below:
weight = torch.ones(4, requires_grad=True)

for epochs in range(5):
    model_output = (weight * 3).sum()
    model_output.backward()
    print(weight.grad)
    # to delete changes with gradient we've done
    weight.grad.zero_()