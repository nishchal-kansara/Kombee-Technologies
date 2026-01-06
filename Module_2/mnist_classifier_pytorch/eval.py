import torch
from torchvision import datasets, transforms
from model import MNISTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=1000,
    shuffle=False
)

model = MNISTCNN().to(device)
model.load_state_dict(torch.load("saved_model/mnist_cnn.pth", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        _, predicted = torch.max(preds, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print("Test Accuracy", correct / total)
