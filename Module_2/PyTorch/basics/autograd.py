import torch

# enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)

# simple operation
y = x * x

# calculate gradient
y.backward()

print("Value of y", y.item())
print("Gradient of x", x.grad)