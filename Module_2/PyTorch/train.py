import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import NumberDataset
from models.simple_nn import SimpleNN

dataset = NumberDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = SimpleNN()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):
    total_loss = 0

    for x, y in loader:
        x = x.float().unsqueeze(1)
        y = y.float().unsqueeze(1)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch, "Loss", total_loss)
