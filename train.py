
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_rand_score


from process_data import *
from autoencoder import *
from constants import *

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def shuffle(data):
	#TODO: implement
	return data

def vis_pca(data, labels):
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(data)
	plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=labels, edgecolor='none', alpha=0.5)
	plt.xlabel('component 1')
	plt.ylabel('component 2')
	plt.savefig('pca.png')


def vis_tsne(data, labels):
	data_embedded = TSNE(n_components=2).fit_transform(data)
	plt.scatter(data_embedded[:, 0], data_embedded[:, 1], c=labels, edgecolor='none', alpha=0.5)
	plt.savefig('tsne.png')


def cluster_kmeans(data):
	kmeans = KMeans(n_clusters=10, random_state=0).fit(data)

	y_kmeans = kmeans.predict(data)
	_, counts = np.unique(kmeans.labels_, return_counts=True)
	print(counts)

	vis_pca(data, y_kmeans)
	vis_tsne(data, y_kmeans)
	return kmeans


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

	#vis_pca(encoded_test.detach().numpy())
	kmeans = cluster_kmeans(encoded_test.detach().numpy())

	print(adjusted_rand_score(kmeans.labels_, patient_data['INTCLUST'].values))



if __name__ == '__main__':
	main()
