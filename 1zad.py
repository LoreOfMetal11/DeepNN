import torch
x = torch.randint(1, 10, (1,)) 
x = x.to(torch.float32)
x.requires_grad = True
n = 2
x_e = x**n
x_e = x_e * torch.randint(1, 11, (1,)).to(torch.float32)
x_e = torch.exp(x_e) 
x_e.backward()
print(f"Производная для n={n}: {x.grad}")