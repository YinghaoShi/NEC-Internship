### This is a modified version of main.py from https://github.com/facebookresearch/GradientEpisodicMemory 
### The most significant changes are to the arguments: 1) allowing for new models and 2) changing the default settings in some cases

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
""" example call """
""" python mainREWC_omni.py --n_tasks 4 --data_path data --save_path results  --samples_per_task 20 --cuda no --data_file omniglot.pt --num_samples_per_class 1 --seed 4 --meta_lambda 100 --lr 0.0001 """

import importlib
import datetime
import argparse
import random
import uuid
import time
import os

import numpy as np

from random import randint
import torch
from scipy.linalg import logm
import scipy.io
from numpy.linalg import inv
from numpy.linalg import eig
from numpy import transpose as trans
from copy import deepcopy
import pdb
import tensorflow as tf
import numpy as np
from copy import deepcopy


from copy import deepcopy
from utils.MLP_utils import MLP
from utils.ResNet18 import ResNet18
#from utils import mnist_utils
from utils import plot_utils
import torch
import os
import pdb
# continuum iterator #########################################################

def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def train_task(sess, model, num_epochs, trainset, testsets, x, y_, lr, batch_size=100):

	num_batch = int(np.floor(trainset.train.images.shape[0]/batch_size))
	
	# Reassign optimal weights from previous training session
	model.restore(sess)
	# Initialize test accuracy array for each task
	test_accs = []
	for task in range(len(testsets)):
		test_accs.append(np.zeros(num_epochs))
	# Train on current task
	for epoch in range(num_epochs):
		img = np.random.permutation(trainset.train.images.shape[0]) 
		for b in range(num_batch):
			# randomly sample a batch of images, and corresponding labels
			batch_x = trainset.train.images[img[b*batch_size:(b+1)*batch_size]]
			batch_y = trainset.train.labels[img[b*batch_size:(b+1)*batch_size]]
			one_hot_targets = np.eye(20)[batch_y]
			# train batch
			model.train_step.run(feed_dict={x: batch_x, y_: one_hot_targets})
		# Plotting
		#plt.subplot(1, 1, 1)
		#plots = []
		#colors = ['r', 'b']
		for task in range(len(testsets)):
			feed_dict={x: testsets[task].test.images, y_: np.eye(20)[testsets[task].test.labels]}
			test_accs[task][epoch] = model.accuracy.eval(feed_dict=feed_dict)
			#c = chr(ord('A') + task)
			#plot_h, = plt.plot(range(1,epoch+2), test_accs[task][:epoch+1], colors[task], label="task " + c + " (%1.2f)" % test_accs[task][epoch])
			#plots.append(plot_h)
		#plot_utils.plot_test_acc(plots)
		#plt.gcf().set_size_inches(10, 7)

	return test_accs

def train_task_online(sess, model, trainset, testsets, x, y_, lr, buffer_size=50, class_number=10):
	
	# Reassign optimal weights from previous training session
	model.restore(sess)
	# Initialize test accuracy array for each task
	test_accs = [0 for task in range(len(testsets))]

	# Train on current task
	img = np.random.permutation(trainset.train.images.shape[0]) 
	buffer_ind = img[:buffer_size]
	for b in range(buffer_size,len(img)):
		# randomly sample a batch of images, and corresponding labels
		batch_x = trainset.train.images[buffer_ind]
		batch_y = trainset.train.labels[buffer_ind]
		one_hot_targets = np.eye(class_number)[batch_y]
		# train batch
		model.train_step.run(feed_dict={x: batch_x, y_: one_hot_targets})
		
		buffer_ind[randint(0, buffer_size-1)] = buffer_ind[-1]
		buffer_ind[-1] =img[b]
		
	batch_x = trainset.train.images[buffer_ind]
	batch_y = trainset.train.labels[buffer_ind]
	one_hot_targets = np.eye(class_number)[batch_y]
	# train batch
	model.train_step.run(feed_dict={x: batch_x, y_: one_hot_targets})

	for task in range(len(testsets)):
		feed_dict={x: testsets[task].test.images, y_: np.eye(class_number)[testsets[task].test.labels]}
		test_accs[task] = model.accuracy.eval(feed_dict=feed_dict)

	return test_accs

def load_datasets(args):
	#pdb.set_trace()
	d_tr, d_te = torch.load(args.data_path + os.sep + args.data_file)
	#d_tr, d_te = torch.load(args.data_path + '/' + args.data_file)
	n_inputs = d_tr[0][1].size(1)
	n_outputs = 0
	for i in range(len(d_tr)):
		n_outputs = max(n_outputs, d_tr[i][2].max())
		n_outputs = max(n_outputs, d_te[i][2].max())
	n_outputs = n_outputs.numpy().astype(int)
	
	d_val =[[None,None,None] for  i in range(len(d_tr))]
	if args.samples_per_task > 0:
		for i in range(len(d_tr)):
			n_sample_tr = len(d_tr[i][1])
			perm = np.random.permutation(n_sample_tr)    
			perm = perm.tolist()
			
			d_tr[i][1] = d_tr[i][1][perm[:args.samples_per_task]]
			d_tr[i][2] = d_tr[i][2][perm[:args.samples_per_task]]
			
			all_labels = np.unique(d_tr[i][2])
			for lab in all_labels:
				ind = np.where(d_tr[i][2] == lab)[0]
				np.random.shuffle(ind)		

				d_val[i][1] = d_tr[i][1][ind[:args.num_samples_per_class]]
				d_val[i][2] = d_tr[i][2][ind[:args.num_samples_per_class]]          
			
			
				#d_val[i][1] = d_tr[i][1][:int(args.validation_perc*args.samples_per_task)]
				#d_val[i][2] = d_tr[i][2][:int(args.validation_perc*args.samples_per_task)]
	
	Data = []
	for i in range(len(d_tr)):
		mnist = lambda:0
		mnist.train = lambda:0
		mnist.validation = lambda:0
		mnist.test = lambda:0
		# and we remake the structure as the original one

		mnist.validation.images = d_val[i][1]
		mnist.validation.labels = d_val[i][2]
		
		mnist.train.images = d_tr[i][1]
		mnist.train.labels = d_tr[i][2]

		mnist.test.images = d_te[i][1]
		mnist.test.labels = d_te[i][2]
		
		mnist.train.images = mnist.train.images.reshape((-1,28,28,1))
		mnist.validation.images = mnist.validation.images.reshape((-1,28,28,1))
		mnist.test.images = mnist.test.images.reshape((-1,28,28,1))
		
		Data +=[mnist]
	Data = Data[0:args.n_tasks]
	return Data


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Continuum learning')

	parser.add_argument('--n_tasks', type=int, default=20,
						help='Number of tasks')
	
	# model details
	parser.add_argument('--n_hiddens', type=int, default=100,
						help='number of hidden neurons at each layer')
	parser.add_argument('--n_layers', type=int, default=2,
						help='number of hidden layers')
	parser.add_argument('--finetune', default='yes', type=str,help='whether to initialize nets in indep. nets')
	
	# optimizer parameters influencing all models
	parser.add_argument('--n_epochs', type=int, default=1,
						help='Number of epochs per task')
	parser.add_argument('--batch_size', type=int, default=1,
						help='the amount of items received by the algorithm at one time (set to 1 across all experiments). Variable name is from GEM project.')
	parser.add_argument('--lr', type=float, default=1e-3,
						help='learning rate')

	# memory parameters for R-EWC baselines
	parser.add_argument('--meta_lambda', type=int, default=100,
						help='meta_lambda')
	parser.add_argument('--num_samples_per_class', default=0, type=int,
						help='num_samples per class')

	# memory parameters for GEM baselines
	parser.add_argument('--n_memories', type=int, default=0,
						help='number of memories per task')
	parser.add_argument('--memory_strength', default=0, type=float,
						help='memory strength (meaning depends on memory)')

	# parameters specific to models in https://openreview.net/pdf?id=B1gTShAct7 
	
	parser.add_argument('--memories', type=int, default=5120, help='number of total memories stored in a reservoir sampling based buffer')
	
	# experiment parameters
	parser.add_argument('--cuda', type=str, default='no',
						help='Use GPU?')
	parser.add_argument('--seed', type=int, default=0,
						help='random seed of model')
	parser.add_argument('--log_every', type=int, default=100,
						help='frequency of logs, in minibatches')
	parser.add_argument('--save_path', type=str, default='results_fashion/',
						help='save models at the end of training')

	# data parameters
	parser.add_argument('--data_path', default='data',
						help='path where data is located')
	parser.add_argument('--data_file', default='fashion_mnist_permutations_reduced.pt',
						help='data file')
	parser.add_argument('--samples_per_task', type=int, default=-1,
						help='training samples per task (all if negative)')
	parser.add_argument('--shuffle_tasks', type=str, default='no',
						help='present tasks in order')

	args = parser.parse_args()

	args.cuda = True if args.cuda == 'yes' else False
	args.finetune = True if args.finetune == 'yes' else False

	# taskinput model has one extra layer

	rotate_layer =  [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]
	# unique identifier
	uid = uuid.uuid4().hex

	# initialize seeds    
	torch.backends.cudnn.enabled = False
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed_all(args.seed)


	###############################
	data = load_datasets(args)
	class_number = []
	for i in range(len(data)):
		class_current_task = max(np.unique(data[i].train.labels).max(),np.unique(data[i].test.labels).max())+1
		class_number.append(class_current_task)
	class_num = sum(class_number)
	print("class_number:{}".format(class_num))
	###############################
	tf.compat.v1.reset_default_graph()
	# define input and target placeholders
	x = tf.compat.v1.placeholder(tf.float32, shape=[None,28,28,1])
	y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, class_num])
	# instantiate new model
	model = ResNet18(x, y_, doDecom = rotate_layer,class_number=class_num,class_number_list=class_number,nf=20)
	#model = LeNet(x, y_, doDecom = rotate_layer)
	model.set_vanilla_loss(args.lr)
	# initialize variables
	sess = tf.compat.v1.InteractiveSession()
	sess.run(tf.compat.v1.global_variables_initializer())
	results =[]    
	for i in range(len(data)):
		mnist = data[i]
		print("MNIST- Image Shape: {}".format(mnist.train.images[0].shape))
		print("MNIST- Labels: {}".format(np.unique(mnist.train.labels)))
		
	#    mnist2 = data[1]
	#    print("MNIST-2 Labels: {}".format(np.unique(mnist2.train.labels)))
	#    test_accs1 = train_task(sess, model, num_epochs1, mnist2, [mnist1, mnist2], x, y_, lr)
		###############################
		# training ist task
		#test_accs = train_task(sess, model, 5, mnist, [data[j] for j in range(i+1)], x, y_, 0.001)    
		test_accs = train_task_online(sess, model, mnist, [data[j] for j in range(0,i+1)], x, y_, args.lr, buffer_size=50, class_number=class_num)
		results+=[test_accs]
		print(test_accs)
		model.star()
		###############################    
		# Compute the M and L matrices as described in the article, and use them to calculate the rotations
		
		#if i==0:
		#model.reset()
		model.compute_M_L(mnist.validation, sess, args.num_samples_per_class , eq_distrib=True)
		model.compute_svd(sess,i)
		###############################    
		tf.compat.v1.reset_default_graph()
		# Define input and target placeholders
		#x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
		x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28,28,1])
		y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, class_num])
		sess = tf.compat.v1.InteractiveSession()
		# Construct a new model
		model.rebuild_decom(x, y_,i)
		# Initialize variables
		sess.run(tf.compat.v1.global_variables_initializer())
		model.star()
		# Compute the Fisher Information necessary for the EWC loss term
		model.compute_fisher(mnist.validation, sess, args.num_samples_per_class, eq_distrib=True)    
		###############################    
		tf.compat.v1.reset_default_graph()
		# Define input and target placeholders
		#x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
		x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28,28,1])
		y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, class_num])
		# Construct a new model
		model.rebuild_decom(x, y_,i)
		model.update_ewc_loss(args.lr, args.meta_lambda, class_num)
		## Initialize variables
		sess = tf.compat.v1.InteractiveSession()
		sess.run(tf.compat.v1.global_variables_initializer())
		###############################        
	





	# prepare saving path and file name
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	fname = "R-EWC" + '_' + str(args.data_file) + '_' 
	fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	fname += '_' + uid
	fname = os.path.join(args.save_path, fname)

	# save confusion matrix and print one line of stats
	one_liner = str(vars(args))

	f = open(fname + '.txt', 'a')
	
	print(one_liner, file=f)
	print('Reuslts', file=f)
	print('============', file=f)
	print(results,file = f)

	f.close()








""" for i in range(len(data)):
	mnist = data[i]
	print("MNIST- Image Shape: {}".format(mnist.train.images[0].shape))
	print("MNIST- Labels: {}".format(np.unique(mnist.train.labels)))
	

	if i ==0:
		test_accs = train_task_online(sess, model, mnist, mnist, x, y_, args.lr, buffer_size=50)
		results+=[test_accs]
		print(test_accs)
		model.star()
	#    mnist2 = data[1]
	#    print("MNIST-2 Labels: {}".format(np.unique(mnist2.train.labels)))
	#    test_accs1 = train_task(sess, model, num_epochs1, mnist2, [mnist1, mnist2], x, y_, lr)
	###############################
	# training ist task
	#test_accs = train_task(sess, model, num_epochs, mnist, [data[j] for j in range(i+1)], x, y_, lr)    
	test_accs = train_task_online(sess, model, mnist, [data[j] for j in range(i+1)], x, y_, args.lr, buffer_size=50)
	results+=[test_accs]
	print(test_accs)
	model.star()
	###############################    
	# Compute the M and L matrices as described in the article, and use them to calculate the rotations
	
	#if i==0:
	#model.reset()
	model.compute_M_L(mnist.validation, sess, args.num_samples_per_class , eq_distrib=True)
	model.compute_svd(sess,i)
	###############################    
	tf.compat.v1.reset_default_graph()
	# Define input and target placeholders
	#x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
	x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28* 1])
	y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
	sess = tf.compat.v1.InteractiveSession()
	# Construct a new model
	model.rebuild_decom(x, y_,i)
	# Initialize variables
	sess.run(tf.compat.v1.global_variables_initializer())
	model.star()
	# Compute the Fisher Information necessary for the EWC loss term
	model.compute_fisher(mnist.validation, sess, args.num_samples_per_class, eq_distrib=True)    
	###############################    
	tf.compat.v1.reset_default_graph()
	# Define input and target placeholders
	#x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
	x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28*28* 1])
	y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
	# Construct a new model
	model.rebuild_decom(x, y_,i)
	model.update_ewc_loss(args.lr, args.meta_lambda, 5)
	## Initialize variables
	sess = tf.compat.v1.InteractiveSession()
	sess.run(tf.compat.v1.global_variables_initializer())
	###############################     """