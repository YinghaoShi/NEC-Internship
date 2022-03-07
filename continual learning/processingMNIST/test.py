import numpy as np 
from urllib import request
import pickle
import os

Cifar10_filename = [
					["training_images",["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]],
					["test_images",["test_batch"]],
					["training_labels",["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]],
					["test_labels",["test_batch"]]
]
Cifar10_Size = 32


def process_cifar(filename,save_path,target_name,size):
	cifar = {}
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	for name in filename[:2]:
		for i in name[1]:
			with open(i, 'rb') as f:
				data = pickle.load(f,encoding='bytes')[b'data']
				for j in range(len(data[0])):
					img = data[j]
					img = img.reshape(-1,size*size)
					img = img.astype(float) / 255.
					
				if name[0] not in cifar.keys():
					cifar[name[0]] = data
				else:
					cifar[name[0]] += data

	for name in filename[-2:]:
		for i in name[1]:
			with open(name[1], 'rb') as f:
				label = pickle.load(f, encoding='bytes')[b'labels']
				if name[0] not in cifar.keys():
					cifar[name[0]] = label
				else:
					cifar[name[0]] += label

	path_to_save = os.path.join(save_path,target_name+".pkl")
	with open(path_to_save, 'wb') as f:
		pickle.dump(cifar,f)
	print("Save complete.")

if __name__ == '__main__':
	cifar10_save_path = "Data/Cifar10"
	process_cifar(Cifar10_filename,cifar10_save_path,"cifar10",Cifar10_Size)
