import numpy as np 
import os 
import random
from scipy import ndimage
from tqdm import tqdm
import torch

def permute(data, num_task,save_type):
	'''
	data: dict
			key: (training_images, test_images, training_labels, test_labels)
	training images: either has shape 
			[N, (H*W)] mnist data or 
			[N,H,W,C] normal 3 channel image
	
	num_task: type: int

	data_name: type: str
				either be mnist, fashion mnist .....

	'''

	train_data = data['training_images']
	test_data = data['test_images']
	train_label = data['training_labels']
	test_label = data['test_labels']

	is_RGB = False

	if len(train_data.shape) == 2:
		N, HWC = train_data.shape
	else:
		assert len(train_data.shape) == 4
		N, H, W, C = train_data.shape
		train_data = train_data.reshape(N, -1)
		HWC = H * W *C
		is_RGB = True

	task_train_data = []
	task_test_data = []
	if save_type == "torch":
		train_label, test_label = torch.from_numpy(train_label), torch.from_numpy(test_label)

	for i in tqdm(range(num_task)):

		perm = np.random.permutation(HWC)

		train_data_perm = train_data[:,perm]
		test_data_perm = test_data[:, perm]

		if is_RGB:
			train_data_perm = train_data_perm.reshape(N,H,W,C)
			test_data_perm = test_data_perm.reshape(N,H,W,C)

		if save_type == "torch":
			train_data_perm, test_data_perm = torch.from_numpy(train_data_perm), torch.from_numpy(test_data_perm)
			

		task_train_data.append(["random permutation", train_data_perm, train_label])
		task_test_data.append(["random permutation", test_data_perm, test_label])

	return task_train_data, task_test_data
	#save_data_pickle(task_train_data, task_test_data, target_name)

def rotate(data, num_task, save_type,reshape_img = False):
	'''
	data: dict
			key: (training_images, test_images, training_labels, test_labels)
	training images: either has shape 
			[N, (H*W)] mnist data or 
			[N,H,W,C] normal 3 channel image
	
	num_task: type: int

	data_name: type: str
				either be mnist, fashion mnist .....

	save_type: type: str
				either be pkl or torch 

	reshape_img:  type: bool
				if true, the input array is contained completely in the roatated output
				see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
				for examples
	'''

	train_data = data['training_images']
	test_data = data['test_images']
	train_label = data['training_labels']
	test_label = data['test_labels']

	is_RGB = False

	if len(train_data.shape) == 2:
		N, HWC = train_data.shape
		H = np.sqrt(HWC).astype(int)
		W = H
		train_data = train_data.reshape(-1,H,W)
		test_data = test_data.reshape(-1,H,W)
	else:
		assert len(train_data.shape) == 4
		N, H, W, C = train_data.shape
		HWC = H * W *C
		is_RGB = True

	task_train_data = []
	task_test_data = []

	angles_list = []

	if save_type == "torch":
		train_label, test_label = torch.from_numpy(train_label), torch.from_numpy(test_label)

	for i in tqdm(range(num_task)):
		angle = np.random.randint(-60,60)
		angles_list.append(angle)
		train_rotate = []
		test_rotate = []

		for i in range(train_data.shape[0]):
			img = train_data[i]	
			train_rotate.append(ndimage.rotate(img,angle,reshape=reshape_img))

		for i in range(test_data.shape[0]):
			img = test_data[i]	
			test_data[i] = ndimage.rotate(img,angle,reshape=reshape_img)
			test_rotate.append(ndimage.rotate(img,angle,reshape=reshape_img))

		train_rotate = np.array(train_rotate)
		test_rotate = np.array(test_rotate)

		if not is_RGB:
			train_rotate = train_rotate.reshape(N,-1)
			test_rotate = test_rotate.reshape(test_data.shape[0],-1)

		if save_type == "torch":
			train_rotate, test_rotate = torch.from_numpy(train_rotate), torch.from_numpy(test_rotate)

		task_train_data.append(["random rotation", train_rotate, train_label])
		task_test_data.append(["random rotation", test_rotate, test_label])

	with open('angle_record_each_time.txt','a+') as f:
		f.write(str(angles_list) + '\n')

	return task_train_data, task_test_data
	#save_data_pickle(task_train_data, task_test_data, target_name)


