import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from cifar10.model import CIFARCNN
from utils import get_device, accuracy

print(torch.cuda.is_available())

device = get_device()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

train_data = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

model = CIFARCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_acc = accuracy(preds, labels)
        total_correct += batch_acc * labels.size(0)
        total_samples += labels.size(0)

    epoch_acc = total_correct / total_samples

    print(
        "Epoch",
        epoch,
        "Loss",
        total_loss,
        "Accuracy",
        epoch_acc
    )