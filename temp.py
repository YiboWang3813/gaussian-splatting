import torch 

mask = [True, False, True] 
x = torch.randn((3, 10)) 
y = x[mask] 

print(y.shape)  

z = y.repeat(2, 1)

print(z.shape)