import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def mse():
	mse_loss = nn.MSELoss()
	outputs = torch.randn(3, 5, requires_grad=True)
	targets = torch.randn(3, 5)
	loss = mse_loss(outputs, targets)
	print(loss)

def crossentropy():
	ce_loss = nn.CrossEntropyLoss()
	outputs = torch.randn(3, 5, requires_grad=True)
	targets = torch.tensor([1, 0, 3], dtype=torch.int64)
	loss = ce_loss(outputs, targets)
	print(loss)
	
def bce(): #binary cross entropy
	bce_loss = nn.BCELoss()
	sigmoid = nn.Sigmoid()
	probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
	targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4, 1)
	oss = bce_loss(probabilities, targets)
	print(probabilities)
	print(loss)	
	