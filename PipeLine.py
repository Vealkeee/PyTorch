# 1) Design the model ( input size, output size, forward pass )
# 2) Construct loss and optimizer
# 3) Training loop:

#   - forward pass: compute the prediction
#   - backward pass: gets gradients
#   - update weights

# Iterating until it's done after all

import torch
import torch.nn as nn

# f = w * x
# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

AV = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

# If we want to create a model by yourself

# class LinearRegression(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegression, self).__init__()

#         define layers
#         self.lin = nn.Linear(input_dim, output_dim)
    
#     def forward(self, x):
#         return self.lin(x)

model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(5) = {model(AV).item():.3f}')

learning_rate = 0.01
n_iters = 2000

# initialize loss and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
for epoch in range(n_iters):
    # prediction
    y_predicted = model(X)
    # loss
    l = loss(Y, y_predicted)
    # gradient
    l.backward()
    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

# Prediction after the training
print(f'Prediction after training: f(5) = {model(AV).item():.3f}')