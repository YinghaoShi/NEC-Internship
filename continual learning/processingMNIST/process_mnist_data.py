import numpy as np 
from urllib import request
import gzip
import pickle
import os

Emnist_filename = [
	["training_images","Data/gzip-emnist/gzip/emnist-mnist-train-images-idx3-ubyte.gz"],
	["test_images","Data/gzip-emnist/gzip/emnist-mnist-test-images-idx3-ubyte.gz"],
	["training_labels","Data/gzip-emnist/gzip/emnist-mnist-train-labels-idx1-ubyte.gz"],
	["test_labels","Data/gzip-emnist/gzip/emnist-mnist-test-labels-idx1-ubyte.gz"]
	]

Mnist_filename = [
	["training_images","train-images-idx3-ubyte.gz"],
	["test_images","t10k-images-idx3-ubyte.gz"],
	["training_labels","train-labels-idx1-ubyte.gz"],
	["test_labels","t10k-labels-idx1-ubyte.gz"]
	]
Fashion_filename = [
	["training_images","Data/fashion/train-images-idx3-ubyte.gz"],
	["test_images","Data/fashion/t10k-images-idx3-ubyte.gz"],
	["training_labels","Data/fashion/train-labels-idx1-ubyte.gz"],
	["test_labels","Data/fashion/t10k-labels-idx1-ubyte.gz"]
	]
Mnist_Size = 28

def process_mnist(filename,save_path,target_name,size):
	mnist = {}
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	for name in filename[:2]:
		with gzip.open(name[1], 'rb') as f:
			img = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,size*size)
			img = img.astype(float) / 255.
			print(np.max(img))
			mnist[name[0]] = img
	for name in filename[-2:]:
		with gzip.open(name[1], 'rb') as f:
			label = np.frombuffer(f.read(), np.uint8, offset=8)
			mnist[name[0]] = label

	path_to_save = os.path.join(save_path,target_name+".pkl")
	with open(path_to_save, 'wb') as f:
		pickle.dump(mnist,f)
	print("Save complete.")
	
def mnist(save_path, filename):

	base_url = "http://yann.lecun.com/exdb/mnist/"
	for name in filename:
		print("Downloading "+name[1]+"...")
		request.urlretrieve(base_url+name[1], name[1])
	print("Download complete.")
	process_mnist(filename,save_path,"mnist",28)


if __name__ == '__main__':
	mnist_save_path = "Data/MNIST"
	mnist(mnist_save_path, Mnist_filename)
	emnist_save_path = "Data/EMNIST"
	process_mnist(Emnist_filename,emnist_save_path,"emnist",Mnist_Size)
	fashion_save_path = "Data/fashion"
	process_mnist(Fashion_filename,fashion_save_path,"fashionMnist",Mnist_Size)




