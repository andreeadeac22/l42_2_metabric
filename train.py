
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

from process_data import *
from autoencoder import *
from constants import *

from sklearn.decomposition import PCA



def shuffle(data):
	#TODO: implement
	return data

def vis_pca(data):
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(data)
	print(pca.components_)
	print(pca.explained_variance_)
	# c = digit.targets in scatter
	# cmap=plt.cm.get_cmap('spectral', 10)
	plt.scatter(principalComponents[:, 0], principalComponents[:, 1], edgecolor='none', alpha=0.5)
	plt.xlabel('component 1')
	plt.ylabel('component 2')
	#plt.colorbar();
	plt.show()

	"""""
	principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
	finalDf = pd.concat([principalDf, df[['target']]], axis=1)
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_xlabel('Principal Component 1', fontsize=15)
	ax.set_ylabel('Principal Component 2', fontsize=15)
	ax.set_title('2 component PCA', fontsize=20)
	targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	colors = ['r', 'g', 'b']
	for target, color in zip(targets, colors):
		indicesToKeep = finalDf['target'] == target
		ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
		           , finalDf.loc[indicesToKeep, 'principal component 2']
		           , c=color
		           , s=50)
	ax.legend(targets)
	ax.grid()
	"""

def main():

	data_train, data_test = read_cna_data()

	total_train = torch.tensor(data_train).float()
	data_test = torch.tensor(data_test).float()

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
			output, encoded = model(input)
			loss = criterion(output, input)
			# ===================backward====================
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		# ===================log========================
		print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))


	torch.save(model.state_dict(), './sim_autoencoder.pth')

	model.eval()
	output_test, encoded_test = model(data_test)
	loss = criterion(data_test, output_test)
	print('test loss ', loss.data)

	vis_pca(encoded_test.detach().numpy())


if __name__ == '__main__':
	main()
