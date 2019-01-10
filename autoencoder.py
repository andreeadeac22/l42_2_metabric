import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from constants import *


class autoencoder(nn.Module):
	def __init__(self):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(genes_number, latent_dim),
			nn.ReLU(),
			)

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, genes_number),
			nn.Tanh()
			)

	def forward(self, x):
		print("input is ", x)
		x = self.encoder(x)
		print("encoded is ", x)
		x = self.decoder(x)
		return x

