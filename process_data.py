import pandas as pd


def read_cna_data(input_file = "data_CNA.txt"):
	data = pd.read_csv(input_file, sep='\t')

	cna_data = data.select_dtypes(['number']).dropna(axis=1)
	cna_data = cna_data / 2
	cna_data = cna_data.iloc[:, 1:].values.transpose()
	cna_data_size = cna_data.shape[0]
	mid = int(cna_data_size/2)
	print(cna_data_size)
	cna_data_train = cna_data[:mid, :]
	cna_data_test = cna_data[mid:, :]

	print(cna_data_train.shape)
	print(cna_data_test.shape)
	return cna_data_train, cna_data_test


def read_patient_data(input_file = "data_clinical_patient.txt"):
	data_frame = pd.read_csv(input_file, sep='\t')

	df = data_frame[['#Patient Identifier','Overall Survival Status']]
	df = df.select_dtypes(['number']).dropna(axis=1)

	print(df)
	return df

read_patient_data()