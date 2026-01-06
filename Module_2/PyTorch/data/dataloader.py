from torch.utils.data import DataLoader
from dataset import NumberDataset

dataset = NumberDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for x, y in loader:
    print("Input", x)
    print("Target", y)
