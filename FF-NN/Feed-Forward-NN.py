import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# Device configure
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-Parameters
input_size = 784 # 28*28
hidden_size = 100
num_classes = 10 # since we've 9 numbers
n_iterations = 10
learning_rate = 0.01
batch_size = 100

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                            transform=transforms.ToTensor(),
                            )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                           shuffle=False)

# 100 = batch size; 1 = single channel; 
# 28, 28 = image size, because input_size = 784
# labels = 100, for each class label, 1 value in tensor
examples = iter(train_loader)
samples, labels = next(examples)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')

plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size=input_size, 
                  hidden_size=hidden_size, 
                  num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for step in range(n_iterations):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        prediction = model(images)
        loss = criterion(prediction, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"step={step+1}/{n_iterations} loss={loss.item():.4f}")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # .max returns value & index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    accuracy = 100.0 * n_correct / n_samples
    print(f'accuracy of the number prediction = {accuracy}')