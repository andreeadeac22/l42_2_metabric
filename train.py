
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from process_data import *
from autoencoder import *
from constants import *


def shuffle(data):
	#TODO: implement
	return data

def main():

	data_train, data_test = read_data()

	total_train = torch.tensor(data_train).float()

	print(total_train.shape)


	#dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

	#model = autoencoder().cuda()
	model = autoencoder()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(
		model.parameters(), lr=learning_rate, weight_decay=1e-5)

	for epoch in range(num_epochs):
		for j in range(batch_size, data_train.shape[0], batch_size):
			interval = [x for x in range(j-batch_size, j)]
			indices = torch.tensor(interval)
			batch = torch.index_select(total_train, 0, indices)
			input = Variable(batch, requires_grad=True)
			# ===================forward=====================
			output = model(input)
			loss = criterion(output, input)
			# ===================backward====================
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		# ===================log========================
		print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, tensor.item(loss.data[0])))


	torch.save(model.state_dict(), './sim_autoencoder.pth')

	"""
	model.eval()

	output_test = model(data_test)
	loss = criterion(data_test, output_test)

	print('test loss ', loss.data[0])
	"""

if __name__ == '__main__':
	main()
