#cuda 
import torch
print(torch.cuda.is_available)

device=torch.device("cuda")
x=torch.randn(2,2).to(device)

#we cannot perform operators with cuda and cpu tensors at once
#both the tensors either have to be CPU or Cuda tensors