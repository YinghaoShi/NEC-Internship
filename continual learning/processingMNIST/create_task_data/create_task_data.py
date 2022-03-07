import numpy as np 
import os 
import random
import pickle
import torch
from util import save_data_pickle, save_data_torch, load_ori_data
from operations import permute, rotate
import argparse


def do_operation(data_path, num_task, data_name, operation_type, save_path,save_type='pkl'):
	'''
	data_path:
		str  ----> path of the data
	num_task:
		int  ----> number tasks to be created
	data_name:
		str  ----> name of the dataset
	operation_type:
		str  ----> type of operation, permutation, rotation
	save_type:
		str  ----> type for save, choose pkl or torch
				   recommend torch when the final size is too large
	'''

	data = load_ori_data(data_path)

	if operation_type == "rotation":
		target_name = data_name + "_rotation"
		task_train_data, task_test_data = rotate(data, num_task, save_type, reshape_img = False)

	elif operation_type == "permutation":
		target_name = data_name + "_permutation"
		task_train_data, task_test_data = permute(data, num_task, save_type)


	if save_type == 'pkl':
		save_data_pickle(task_train_data, task_test_data, target_name, save_path)
	else:
		save_data_torch(task_train_data, task_test_data, target_name, save_path)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='create tasks')

	# data parameters
	parser.add_argument('--data_path', default='Y:/Data/fashion/fashionMnist.pkl',help='path where data is located')   
	parser.add_argument("--data_name", default='fashion', help = "name of dataset under processing")
	parser.add_argument("--num_task", default=10, type=int , help="number of tasks")
	parser.add_argument("--operation_type", default="permutation", help = "type of operations including permutation and rotation")
	parser.add_argument("--save_type", default="torch", help = "type for saving data")
	parser.add_argument("--save_path", default="Y:/Data/Data_Processed_STL/fashion", help = "path for saving data")
	parser.add_argument("--seed", default=1234, type=int, help = "seed for random")
	args = parser.parse_args()

	np.random.seed(args.seed)
	data_path = do_operation(args.data_path, args.num_task, args.data_name,args.operation_type, args.save_path,args.save_type)

# python create_task_data.py --data_path ../../Data/EMNIST/emnist.pkl --save_path ../../Data/EMNIST --data_name emnist 