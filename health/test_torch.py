import torch
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
c = torch.matmul(a, b)
print(c.shape)