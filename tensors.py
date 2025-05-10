import torch 
import numpy as np

# empty tensor, and demensions
# (1, ...) = 1D
# (1, 2, ...) = 2D
# (1, 2, 3...) = 3D
x = torch.empty(1, 3, 5)
print(x)


# random tensor
x = torch.rand(2, 2, 2)
print(x)


# .size used to define size of the tensor
# calls the type of a tensor, and .ones replacing all with 1
x = torch.ones(2, 2, 3, dtype=int)
print(x.dtype)
print(x.size())


# basic tensor
x = torch.tensor([1, 2, 3])
print(x)


# completing an addition, etc. with 2 tensors
# input = tensor([[[0.9262]]])
# or we can just do torch in addition

x = torch.rand(1, 1, 1)
y = torch.rand(1, 1, 1)

# 1. z = x + y 
# 2. z = torch.add(x, y) || z = torch.sub(x, y) || z = torch.mul(x, y) || z = torch.div(x, y)
# 3. y.add_(x) || y.sub_(x) || y.mul_(x) || y.div_(x)
# print(z)


# [:, 1] by column, [1, :] by rows

# .item() can be used to show the number

# if there's only one number etc.
x = torch.rand(2, 5)
print(x)
print(x[1, :])


# .view is changing the tensor element
# by adding the -1, we're getting correct tensor
x = torch.rand(4, 3)
print(x)
y = x.view(-1, 3)
print(y.size())


# converting tensor to a numpy array
x = torch.ones(5)
print(x)
y = x.numpy()
print(y)

# converting np arrays to torch tensors
x = np.ones(5)
print(x)
b = torch.from_numpy(x)
print(b)


# applying cuda(gpu) to the tensor
if torch.cuda.is_available:
    device = torch.device("cuda")
    x = torch.rand(5, device=device)
    y = torch.rand(5, device=device)
    z = x + y
    print(z)
