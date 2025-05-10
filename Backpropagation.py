import torch

x = torch.tensor(5.0)
y = torch.tensor(3.0)

w = torch.tensor(2.0, requires_grad=True)

# computing the loss
y_hat = x * w
loss = (y_hat - y) ** 2
print(loss)
# getting gradient
loss.backward()
print(w.grad)