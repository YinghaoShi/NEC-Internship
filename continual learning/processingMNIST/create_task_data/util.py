import numpy as np 
import os 
import random
import pickle
import torch


def save_data_pickle(task_train_data, task_test_data, target_name, save_path = "./"):
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	data = [task_train_data, task_test_data]

	path_to_save_data = os.path.join(save_path,target_name+".pkl")
	with open(path_to_save_data, 'wb') as f:
		pickle.dump(data,f)

	print("Data is already saved!")

def save_data_torch(task_train_data, task_test_data, target_name, save_path = "./"):
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	data = [task_train_data, task_test_data]
	path_to_save_data = os.path.join(save_path,target_name+".pt")
	torch.save(data, path_to_save_data)
	print("Data is already saved!")

def load_ori_data(path):

	with open(path, 'rb') as f:
		data = pickle.load(f)
	return data