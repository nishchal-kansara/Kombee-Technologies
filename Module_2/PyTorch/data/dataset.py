import torch
from torch.utils.data import Dataset

class NumberDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = x * 2
        return torch.tensor(x), torch.tensor(y)

if __name__ == "__main__":
    dataset = NumberDataset()
    print(len(dataset))
    print(dataset[0])
