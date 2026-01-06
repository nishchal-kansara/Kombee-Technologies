import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)
