import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMClassifier

torch.manual_seed(0)

vocab_size = 50
embed_dim = 32
hidden_dim = 64
num_classes = 2

model = LSTMClassifier(
    vocab_size,
    embed_dim,
    hidden_dim,
    num_classes
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 16
seq_len = 10

for epoch in range(5):
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, num_classes, (batch_size,))

    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs, 1)
    acc = (preds == labels).float().mean().item()

    print("Epoch", epoch, "Loss", loss.item(), "Accuracy", acc)
