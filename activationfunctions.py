#activation functions

import matplotlib.pyplot as plt
import torch
import numpy as np

fig, ax = plt.subplots(2,2)
fig.suptitle('Activation Functions')

def sigmoid():
	x=torch.range(-5,5,0.1)
	y=torch.sigmoid(x)
	ax[0,0].grid()
	ax[0,0].plot(x.numpy(), y.numpy())
	ax[0,0].set_title('Sigmoid')

def tanh():
	x=torch.range(-5,5,0.1)
	y=torch.tanh(x)
	ax[0,1].grid()
	ax[0,1].plot(x.numpy(), y.numpy(),color='orange')
	ax[0,1].set_title('Tanh')

def relu():
	x=torch.range(-5,5,0.1)
	y=torch.relu(x)
	ax[1,0].grid()
	ax[1,0].plot(x.numpy(), y.numpy(),color='g')
	ax[1,0].set_title('RelU')

def prelu():
	prelu = torch.nn.PReLU(num_parameters=1)
	x=torch.range(-5,5,0.1)
	y=prelu(x)
	ax[1,1].grid()
	ax[1,1].plot(x.numpy(), y.detach().numpy(),color='r')
	ax[1,1].set_title('PRelU')

sigmoid()
tanh()
relu()
prelu()