#perceptron with pytroch

import torch
import torch.nn as nn
import torch.functional as f
from torch.autograd import Variable

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.fc1=nn.Linear(1,1)
	
	def forward(self,x):
		x=self.fc1(x)
		return x

net=Net()
print(net)
'''Net(
  (fc1): Linear(in_features=1, out_features=1, bias=True)
)'''

#to print the parameters of the neural network
print(list(net.parameters()))
'''[Parameter containing:
tensor([[0.4780]], requires_grad=True), Parameter containing:
tensor([0.1686], requires_grad=True)]'''

input = Variable(torch.randn(1,1,1), requires_grad=True)
print(input) #tensor([[[0.7907]]], requires_grad=True)

output=net(input)
print(output) #tensor([[[0.5466]]], grad_fn=<AddBackward0>)

import torch.optim as optim
def criterion(out, label):
    return (label - out)**2
