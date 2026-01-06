import torch

# scalar tensor
a = torch.tensor(5)
print(a)

# vector tensor
b = torch.tensor([1, 2, 3])
print(b)

# matrix tensor
c = torch.tensor([[1, 2], [3, 4]])
print(c)

# tensor operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

z = x + y
print(z)

# tensor on GPU if available
if torch.cuda.is_available():
    x = x.cuda()
    print("Tensor moved to GPU")