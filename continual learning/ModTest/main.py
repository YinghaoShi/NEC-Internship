### This is a modified version of main.py from https://github.com/facebookresearch/GradientEpisodicMemory 
### The most significant changes are to the arguments: 1) allowing for new models and 2) changing the default settings in some cases

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import datetime
import argparse
import random
import uuid
import time
import os

import numpy as np

import torch
from torch.autograd import Variable
from metrics.metrics import confusion_matrix, confusion_matrix_forgetting
from scipy.linalg import logm
import scipy.io
from numpy.linalg import inv
from numpy.linalg import eig
from numpy import transpose as trans
from copy import deepcopy
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
	return d_tr, d_te, n_inputs, n_outputs + 1, len(d_tr)


class Continuum:

	def __init__(self, data, args):
		self.data = data
		self.batch_size = args.batch_size
		n_tasks = len(data) #20
		if n_tasks!=args.n_tasks:
			n_tasks=args.n_tasks

		task_permutation = range(n_tasks)	

		if args.shuffle_tasks == 'yes':
			task_permutation = torch.randperm(n_tasks).tolist()

		sample_permutations = []

		for t in range(n_tasks):
			N = data[t][1].size(0)
			if args.samples_per_task <= 0:
				n = N
			else:
				n = min(args.samples_per_task, N)

			p = torch.randperm(N)[0:n]
			sample_permutations.append(p) # inside each task, do shuffle

		self.sample_permutations = sample_permutations
		self.permutation = [] 
		# each element is a list [task_id, task_samples]
		# saved with the order of task

		for t in range(n_tasks):
			task_t = task_permutation[t]
			for _ in range(args.n_epochs):
				task_p = [[task_t, i] for i in sample_permutations[task_t]]
				random.shuffle(task_p)
				self.permutation += task_p

		self.length = len(self.permutation)
		print("Report: total number of iteration is {}".format(self.length))
		self.current = 0

	def __iter__(self):
		return self

	def next(self):
		return self.__next__()

	def __next__(self):
		if self.current >= self.length:
			raise StopIteration
		else:
			ti = self.permutation[self.current][0] # task_id
			j = []
			i = 0
			while (((self.current + i) < self.length) and
				   (self.permutation[self.current + i][0] == ti) and
				   (i < self.batch_size)):
				j.append(self.permutation[self.current + i][1])
				i += 1
			self.current += i
			j = torch.LongTensor(j) # j indicates a batch of sample id of specific task_id 
			return self.data[ti][1][j], ti, self.data[ti][2][j]  
			# should retrun a batch sample of a task per iteration

# train handle ###############################################################


def eval_tasks(model, tasks, args, iter):
	model.eval()
	result = []
	for i, task in enumerate(tasks):
		t = i
		x = task[1]
		y = task[2]
		rt = 0
		
		eval_bs = x.size(0)

		for b_from in range(0, x.size(0), eval_bs):
			b_to = min(b_from + eval_bs, x.size(0) - 1)
			if b_from == b_to:
				xb = x[b_from].view(1, -1)
				yb = torch.LongTensor([y[b_to]]).view(1, -1)
			else:
				xb = x[b_from:b_to]
				yb = y[b_from:b_to]
			if args.cuda:
				xb = xb.cuda()
			xb = Variable(xb, volatile=True).float()
			_, pb = torch.max(model(xb, t).data.cpu(), 1, keepdim=False)
			rt += (pb == yb.long()).float().sum()

		accuracy = rt / x.size(0)
		result.append(accuracy)

	return result

def life_experience(model, continuum, x_te, args):
	result_a = []
	result_t = []

	current_task = 0
	time_start = time.time()

	t1 = time.time()
	for (i, (x, t, y)) in enumerate(continuum):
		if(((i % args.log_every) == 0) or (t != current_task)):
			eval_result = eval_tasks(model, x_te, args, current_task)
			result_a.append(eval_result)
			result_t.append(current_task)
			current_task = t

		if(((i % (args.log_every*10)) == 0) or (t != current_task)):
			t2 = time.time()
			print("Task {} Every {} steps, time {}".format(t, args.log_every*10, t2-t1))  
			print(["%.5f"%tensor.item() for tensor in result_a[-1]])   
			t1 = time.time()

		v_x = x.view(x.size(0), -1).float()
		v_y = y.long().view(x.size(0), -1) # bug before. v_y should also has the shape of [batch, -1]

		if args.cuda:
			v_x = v_x.cuda()
			v_y = v_y.cuda()

		model.train()
		model.observe(Variable(v_x), t, Variable(v_y))

	result_a.append(eval_tasks(model, x_te, args, current_task))
	result_t.append(current_task)

	time_end = time.time()
	time_spent = time_end - time_start

	return torch.Tensor(result_t), torch.Tensor(result_a), time_spent

def life_experience_forget_task(model, continuum, x_te, forgetting_task_ids, args):
	#pdb.set_trace()
	result_a = []	
	task_history = {}
	
	current_task = 0
	time_start = time.time()

	t1 = time.time()
	for (i, (x, t, y)) in enumerate(continuum):
		if(((i % args.log_every) == 0) or (t != current_task)):
			current_task = t
		task_history[t] = task_history.get(t,0)+1
		if task_history[t]>args.forgetting_resee_size:
			continue

		if(((i % (args.log_every*10)) == 0) or (t != current_task)):
			t2 = time.time()
			print("Task {} Every {} steps, time {}".format(t, args.log_every*10, t2-t1))  
			t1 = time.time()
		if t in forgetting_task_ids:
			continue
		v_x = x.view(x.size(0), -1).float()
		v_y = y.long().view(x.size(0), -1) # bug before. v_y should also has the shape of [batch, -1]

		if args.cuda:
			v_x = v_x.cuda()
			v_y = v_y.cuda()

		######## here must correct
		model.train()
		model.observe(Variable(v_x), t, Variable(v_y),startForgetting=True,forgetting_task_ids=forgetting_task_ids)		

	#pdb.set_trace()
	result_a.append(eval_tasks(model, x_te, args, current_task))

	time_end = time.time()
	time_spent = time_end - time_start

	return torch.Tensor(result_a), time_spent	



def von_Neumann_divergence( A,B):

	t1 = time.time()
	#Divergence = np.trace(np.dot(A, logm(A)) - np.dot(A, logm(B)) - A + B)
	Aeig_val, Aeig_vec = eig(A)
	Beig_val, Beig_vec = eig(B) 
	Aeig_val, Aeig_vec = abs(Aeig_val), np.real(Aeig_vec)
	Beig_val, Beig_vec = abs(Beig_val), np.real(Beig_vec)
	Aeig_val[Aeig_val<1e-10] = 0
	Beig_val[Beig_val<1e-10] = 0

	A_val_temp, B_val_temp = deepcopy(Aeig_val), deepcopy(Beig_val)
	A_val_temp[Aeig_val <= 0] = 1
	B_val_temp[Beig_val <= 0] = 1

	part1 = np.sum(Aeig_val * np.log(A_val_temp) - Aeig_val + Beig_val) 

	lambda_log_theta = np.dot(Aeig_val.reshape(len(Aeig_val),1), np.log(B_val_temp.reshape(1, len(B_val_temp))))
	part2 = (np.dot(Aeig_vec.T, Beig_vec) **2) * lambda_log_theta
	part2 = -np.sum(part2)
	#print((np.dot(Aeig_vec, Beig_vec) **2).shape)

	Divergence = part1 + part2

	t2 = time.time()
	#print("time used for computing vm divergence {}".format(t2-t1))
	return Divergence

def compute_relatedness(C_x1_y1, C_x2_y2):
	C_x1 = C_x1_y1[0:-1,0:-1]
	C_x2 = C_x2_y2[0:-1,0:-1]


	v1 = von_Neumann_divergence(C_x1,C_x2)
	v2 = von_Neumann_divergence(C_x2,C_x1)
	v3 = von_Neumann_divergence(C_x1_y1,C_x2_y2)
	v4 = von_Neumann_divergence(C_x2_y2,C_x1_y1)

	r= 0.5*(v3+v4-v1-v2)

	return r

def compute_relatedness2(C_x1_y1, C_x2_y2):
	C_x1 = C_x1_y1[0:-1,0:-1]
	C_x2 = C_x2_y2[0:-1,0:-1]


	v1 = von_Neumann_divergence_Eff_norm(C_x1,C_x2)
	v2 = von_Neumann_divergence_Eff_norm(C_x2,C_x1)
	v3 = von_Neumann_divergence_Eff_norm(C_x1_y1,C_x2_y2)
	v4 = von_Neumann_divergence_Eff_norm(C_x2_y2,C_x1_y1)

	r= 0.5*(v3+v4-v1-v2)

	return r

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Continuum learning')

	parser.add_argument('--n_tasks', type=int, default=20,
						help='Number of tasks')
	
	# model details
	parser.add_argument('--model', type=str, default='single',
						help='model to train')
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

	# memory parameters for GEM baselines
	parser.add_argument('--n_memories', type=int, default=0,
						help='number of memories per task')
	parser.add_argument('--memory_strength', default=0, type=float,
						help='memory strength (meaning depends on memory)')

	# parameters specific to models in https://openreview.net/pdf?id=B1gTShAct7 
	
	parser.add_argument('--memories', type=int, default=5120, help='number of total memories stored in a reservoir sampling based buffer')

	parser.add_argument('--gamma', type=float, default=1.0,
						help='gamma learning rate parameter') #gating net lr in roe 

	parser.add_argument('--batches_per_example', type=float, default=1,
						help='the number of batch per incoming example')

	parser.add_argument('--s', type=float, default=1,
						help='current example learning rate multiplier (s)')

	parser.add_argument('--replay_batch_size', type=float, default=20,
						help='The batch size for experience replay. Denoted as k-1 in the paper.')

	parser.add_argument('--beta', type=float, default=1.0,
						help='beta learning rate parameter') # exploration factor in roe
	
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
	parser.add_argument('--data_path', default='data/',
						help='path where data is located')
	parser.add_argument('--data_file', default='fashion_mnist_permutations_reduced.pt',
						help='data file')
	parser.add_argument('--samples_per_task', type=int, default=-1,
						help='training samples per task (all if negative)')
	parser.add_argument('--shuffle_tasks', type=str, default='no',
						help='present tasks in order')

	parser.add_argument('--divergence', type = str, default = 'von_Neumann')
	parser.add_argument('--num_groups', type = int, default = 5)
	parser.add_argument('--if_output_cov', type = str2bool, default = True)
	parser.add_argument('--cov_recompute_every', type = int, default = 20)
	parser.add_argument('--cov_first_task_buffer', type = int, default = 200)
	parser.add_argument('--create_random_groups', type = str2bool, default = False)
	parser.add_argument('--create_group_per_unit', type = str2bool, default = False)
	
	parser.add_argument('--forgetting_mode', type = str2bool, default = False)
	parser.add_argument('--forgetting_task_ids', type = str, default = "")
	parser.add_argument('--forgetting_resee_size', type = int, default =100)
	parser.add_argument('--sign_attacked', type=float, default=-1.0,
						help='the sign for attacked tasks')

	parser.add_argument('--gem_reverse', type = str2bool, default = True)
	parser.add_argument('--gem_sub_mean', type = str2bool, default = True)
	parser.add_argument('--gem_ignore_relatedness', type = str2bool, default = False)
	
	parser.add_argument('--ewc_reverse', type = str2bool, default = True)

	parser.add_argument('--recompute_groups_every_task', type = str2bool, default = False)
	parser.add_argument('--create_mod_independent_groups', type = str2bool, default = False)
	parser.add_argument('--create_mod_independent_groups_first_task', type = str2bool, default = False)
	parser.add_argument('--create_mod_independent_groups_last_tasks', type = str2bool, default = False)
	parser.add_argument('--create_mod_independent_groups_all_tasks', type = str2bool, default = False)

	#parser.add_argument('--pre_compute_cov', type = str2bool, default = True)
	#parser.add_argument('--batch_size_group', type = int, default = 200)
	#parser.add_argument('--if_normalize_re', type = str2bool, default = False)

	args = parser.parse_args()

	args.cuda = True if args.cuda == 'yes' else False
	args.finetune = True if args.finetune == 'yes' else False

	# taskinput model has one extra layer
	if args.model == 'taskinput':
		args.n_layers -= 1

	# unique identifier
	uid = uuid.uuid4().hex

	if "omni" in args.data_file:	
		args.is_omniglot= True
	else:
		args.is_omniglot = False

	# initialize seeds    
	torch.backends.cudnn.enabled = False
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed_all(args.seed)

	# load data
	x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)
	# trim the remaining tasks
	if n_tasks!=args.n_tasks:
		n_tasks=args.n_tasks
		x_tr = x_tr[:n_tasks]
		x_te = x_te[:n_tasks]


	# set up continuum
	continuum = Continuum(x_tr, args)

	# load model
	Model = importlib.import_module('model.' + args.model)
	model = Model.Net(n_inputs, n_outputs, n_tasks, args)
	if args.cuda:
		try:
			model.cuda()
		except:
			pass 

	# run model on continuum
	result_t, result_a, spent_time = life_experience(model, continuum, x_te, args)


		# prepare saving path and file name
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	fname = args.model + '_' + str(args.memory_strength) + '_' 
	fname += str(args.if_output_cov) + '_' + str(args.num_groups) + '_' + args.data_file + '_'
	fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	fname += '_' + uid
	fname = os.path.join(args.save_path, fname)

	# save confusion matrix and print one line of stats
	one_liner = str(vars(args))

	f = open(fname + '.txt', 'a')
	print(one_liner, file=f)
	f.close()
	stats = confusion_matrix(result_t, result_a, fname + '.txt')

	
	one_liner += ' '.join(["%.3f" % stat for stat in stats])
	print(fname + ': ' + one_liner + ' # ' + str(spent_time))

	#pdb.set_trace()
	if args.forgetting_mode:
		# set up continuum
		continuum = Continuum(x_tr, args)
		forgetting_task_ids = [int(i) for i in args.forgetting_task_ids.split(",")]
		# run model on continuum
		result_a, spent_time = life_experience_forget_task (model, continuum, x_te, forgetting_task_ids ,args)
		# save confusion matrix and print one line of stats
		#pdb.set_trace()		
		stats = confusion_matrix_forgetting(result_a,forgetting_task_ids, fname + '.txt')
		one_liner += '######## After the attack ########'
		one_liner += ' '.join(["%.3f" % stat for stat in stats])

	if "ewc_GMod2Cov_IndepDel" in args.model:
		f = open(fname + '.txt', 'a')
		print('', file=f)
		print('Stat Reuslts', file=f)
		print('============', file=f)
		print(model.likelihoods,file = f)
		print(model.Independent_first,file = f)
		print(model.Independent_last,file = f)
		print(model.Independent_all,file = f)
		f.close()
	# save all results in binary file
	torch.save((result_t, result_a, model.state_dict(),
				stats, one_liner, args), fname + '.pt')
