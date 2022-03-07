### We use the same version of EwC https://www.pnas.org/content/114/13/3521 originally used in https://github.com/facebookresearch/GradientEpisodicMemory
### We directly copied the ewc.py model file from the GEM project https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common import MLP, ResNet18, von_Neumann_divergence, MLP_layerWise, find_groups, Covariance
#from mdp.utils import CovarianceMatrix
from numpy.linalg import matrix_power
from scipy.linalg import fractional_matrix_power
from scipy.linalg import logm
import scipy.io
from numpy.linalg import inv
from numpy.linalg import eig
from numpy import transpose as trans
import numpy as np
from scipy.linalg import logm
from types import SimpleNamespace
from copy import deepcopy
import pdb, pickle
import collections
import random

class Net(torch.nn.Module):

	def __init__(self,
				 n_inputs,
				 n_outputs,
				 n_tasks,
				 args):
		super(Net, self).__init__()
		self.nl, self.nh = args.n_layers, args.n_hiddens
		self.args = args
		self.reg = args.memory_strength

		# setup network
		self.is_cifar = (args.data_file == 'cifar100.pt')
		if self.is_cifar:
			self.net = ResNet18(n_outputs)
		else:
			self.net = MLP_layerWise([n_inputs] + [self.nh] * self.nl + [n_outputs])

		# setup optimizer
		self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr)

		# setup losses
		self.bce = torch.nn.CrossEntropyLoss()

		# setup memories
		self.current_task = 0
		self.fisher = {}
		self.optpar = {}
		self.task_history = {}

		# groups, relatedness and covariance		
		self.groups = []
		self.cov_list = []
		self.first_task_x_list = []
		self.first_task_y_list = []

		self.memx = None
		self.memy = None		
		self.num_groups = args.num_groups
		self.relatedness_list = [None]
		self.relatedness_list_normalized = [None]
		self.relatedness_list_normalized_reverse = [None]


		if self.is_cifar:
			self.nc_per_task = n_outputs / n_tasks
		else:
			self.nc_per_task = n_outputs
		self.n_outputs = n_outputs
		self.n_memories = args.n_memories
		
		# handle gpus if specified
		self.cuda = args.cuda
		if self.cuda:
			self.net = self.net.cuda()

		self.iter = 0
		self.n_tasks = 0

		#fixed parameters
		self.xi =0.069
		self.group_finding_iterations =1000
		self.group_finding_likelihood_threshold=0.1
		self.group_finding_trials=10






	def compute_offsets(self, task):
		if self.is_cifar:
			offset1 = task * self.nc_per_task
			offset2 = (task + 1) * self.nc_per_task
		else:
			offset1 = 0
			offset2 = self.n_outputs
		return int(offset1), int(offset2)

	def forward(self, x, t):
		outputs = self.net(x)
		output = outputs[-1]

		if self.is_cifar:
			# make sure we predict classes within the current task
			offset1, offset2 = self.compute_offsets(t)
			if offset1 > 0:
				output[:, :offset1].data.fill_(-10e10)
			if offset2 < self.n_outputs:
				output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
		return output


	def von_Neumann_divergence(self,A,B):
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
		#print("time used for computing vm divergence {}".format(t2-t1))
		return Divergence

	def compute_relatedness(self, C_x1_y1, C_x2_y2):		
		C_x1 = C_x1_y1[0:-1,0:-1]
		C_x2 = C_x2_y2[0:-1,0:-1]

		#try:
		v1 = self.von_Neumann_divergence(C_x1,C_x2)
		v2 = self.von_Neumann_divergence(C_x2,C_x1)
		v3 = self.von_Neumann_divergence(C_x1_y1,C_x2_y2)
		v4 = self.von_Neumann_divergence(C_x2_y2,C_x1_y1)

		#except :
			##pdb.set_trace()

		r= max(0,0.5*(v3+v4-v1-v2))

		return r 

	def creategroups(self):		
		W_ij_ab = None			
		for i, p in enumerate(self.net.parameters()): # per layer weight matrix
			if len(p.shape)==1:
				continue
			
			arguments = SimpleNamespace(c=self.num_groups,xi =self.xi,iterations=self.group_finding_iterations,threshold=self.group_finding_likelihood_threshold)				

			args = SimpleNamespace(**vars(arguments))

			#index_layer = i // 2 # as w, b are seperate in net.parameters()
			groups_layer = []
			#W_ij_bc = p.data.numpy().transpose()
			W_ij_bc = p.cpu().data.numpy().transpose()

			if i%2==0 and i!=0:
			#if index_layer >0 :
				#print(index_layer)				
				trials, prev_memberships,prev_likelihood = 0, [], -np.Inf
				while True:
					converge,memberships,diff,t, likelihood = find_groups(arguments,W_ij_ab,W_ij_bc)
					if converge:
						trials+=1						
						if likelihood>prev_likelihood:
							prev_memberships,prev_likelihood = memberships,likelihood						
						if trials>self.group_finding_trials:
							for c in range(max(prev_memberships)+1):							
								groups_layer.append(np.where(prev_memberships==c)[0])
							#print(memberships)
							break
					args.c = min(max(3,args.c+np.random.randint(-1,2)),int(W_ij_bc.shape[0]/4))
				self.groups.append(groups_layer)
			if i%2==0 and i !=self.nl*2:
				W_ij_ab = W_ij_bc


	def creategroup_per_unit(self):
		W_ij_ab = None
		for i, p in enumerate(self.net.parameters()): # per layer weight matrix
			if len(p.shape)==1:
				continue            

			index_layer = i // 2 # as w, b are seperate in net.parameters()						
			W_ij_bc = p.cpu().data.numpy().transpose()            
			if index_layer >0 :
				self.groups.append([np.array([i]) for i in range(W_ij_ab.shape[1])])				
			W_ij_ab = W_ij_bc

	def creategroups_Random(self):
		W_ij_ab = None          
		for i, p in enumerate(self.net.parameters()): # per layer weight matrix
			if len(p.shape)==1:
				continue            

			index_layer = i // 2 # as w, b are seperate in net.parameters()
			groups_layer = []
			num_groups_temp = self.num_groups
			#W_ij_bc = p.data.numpy().transpose()            
			W_ij_bc = p.cpu().data.numpy().transpose()            
			if index_layer >0 :
				while True:                 					
					memberships = np.random.randint(0,num_groups_temp,size=W_ij_ab.shape[1])
					if min(collections.Counter(memberships).values())< ((W_ij_ab.shape[1])/(self.num_groups*3)):
						num_groups_temp = min(max(3,num_groups_temp+np.random.randint(-1,2)),int(W_ij_bc.shape[0]/4))
						continue
					for c in range(max(memberships)+1):                         
						groups_layer.append(np.where(memberships==c)[0])
					break

				self.groups.append(groups_layer)
			W_ij_ab = W_ij_bc			
			

	def get_hidden_cov(self, x, y,cov_list=None):
		#pdb.set_trace()
		x_iterator = (x,) if not isinstance(x, (list)) else x
		y_iterator = (y,) if not isinstance(y, (list)) else y

		for x,y in zip(x_iterator,y_iterator):			
			outputs = self.net(x)#.type(torch.FloatTensor))
			num_layers = len(outputs) - 1 
			
			if cov_list is None:
				cov_list = [[Covariance() for k in self.groups[i]] for  i in range(num_layers)]
			
			for i in range(num_layers):
				layer = outputs[i]			
				for k_id,k in enumerate(self.groups[i]):
					#pdb.set_trace()
					group = layer[:,k]
					if self.args.if_output_cov:
						group_xy = torch.cat([group, outputs[-1]], 1)
					else:						
						#group_xy = torch.cat([group, y.type(torch.FloatTensor).unsqueeze(-1)], 1)						
						#group_xy = torch.cat([group, y], 1)
						group_xy = torch.cat([group, y.float()], 1)

					#group_xy = group_xy.data.numpy()
					group_xy = group_xy.data.cpu().numpy()
					cov_list[i][k_id].update(group_xy)
				
		return cov_list

	def get_group_relatedness(self, cov_list1, cov_list2):

		#if self.current_task==3:
		#	pdb.set_trace()
		relatedness = [[] for i in range(len(cov_list1))] 

		for layer in range(len(cov_list1)):
			for k in range(len(self.groups[layer])):
				cov1, cov2 = cov_list1[layer][k], cov_list2[layer][k]
				score = self.compute_relatedness(cov1.get_cov(), cov2.get_cov())
				relatedness[layer].append(score)
		return relatedness

	def observe(self, x, t, y,startForgetting=False,forgetting_task_ids=None):
		if self.cuda:
			x=x.cuda()
			y=y.cuda()


		self.observe_learn(x, t, y)

	def observe_learn(self, x, t, y):
		#pdb.set_trace()
		y = y.squeeze(0)
		self.net.train()

		self.task_history[t] = self.task_history.get(t,0)+1
		# next task?
		if t != self.current_task:
			print("Task:\t",t)
			self.n_tasks +=1
			if self.current_task==0:
				# Creating the groups and computing the covariance matrix for the fist task
				# and then emptying the buffers
				

				if self.args.create_group_per_unit:
					self.creategroup_per_unit()
				elif self.args.create_random_groups:
					self.creategroups_Random()
				else:					
					self.creategroups()					

				self.cov_list.append(self.get_hidden_cov(self.first_task_x_list,self.first_task_y_list))
				self.first_task_x_list, self.first_task_y_list = [] ,[]

			self.net.zero_grad()
			if self.is_cifar:
				offset1, offset2 = self.compute_offsets(self.current_task)
				self.bce((self.net(Variable(self.memx))[:, offset1: offset2]),
						 Variable(self.memy) - offset1).backward()
			else:
				self.bce(self(Variable(self.memx),
							  self.current_task),
						 Variable(self.memy)).backward()
			self.fisher[self.current_task] = []
			self.optpar[self.current_task] = []
			for p in self.net.parameters():
				pd = p.data.clone()
				pg = p.grad.data.clone().pow(2)
				self.optpar[self.current_task].append(pd)
				self.fisher[self.current_task].append(pg)
			self.current_task = t
			self.iter = 0
			self.memx = None
			self.memy = None
			###pdb.set_trace()
		
		# for the first task: accumulate the input data
		if (t==0):			
			if len(self.first_task_x_list) < self.args.cov_first_task_buffer:                
				self.first_task_x_list.append(x)
				self.first_task_y_list.append(y.unsqueeze(0))
			else:
				p = random.randint(0,self.args.cov_first_task_buffer)
				if p < self.args.cov_first_task_buffer:
					self.first_task_x_list[p] = x
					self.first_task_y_list[p] = y.unsqueeze(0)

		# for the other tasks: accumulate the input data
		else:			
			if len(self.cov_list)<= t:				
				#initializing the covariance matrix of the new task
				self.cov_list.append(self.get_hidden_cov(x,y.unsqueeze(0)))
				#initializing the reltedness to 1 to all tasks
				self.relatedness_list.append([[[1] *(len(l))  for l in self.groups] for i in range(t)])
				self.relatedness_list_normalized.append([np.array([np.exp(-1)*t]*(len(l)))  for l in self.groups])
				self.relatedness_list_normalized_reverse.append([np.array([t/np.exp(-1)]*(len(l)))  for l in self.groups])
			else:
				self.get_hidden_cov(x,y.unsqueeze(0),self.cov_list[t])
			if self.task_history[t] % self.args.cov_recompute_every==0:

				current_cov = self.cov_list[self.current_task]
				self.relatedness_list[t] = [self.get_group_relatedness(current_cov, self.cov_list[i]) for i in range(self.current_task)]
			
				self.relatedness_list_normalized[t] = [np.zeros((len(l))) for l in self.relatedness_list[t][0]]
				self.relatedness_list_normalized_reverse[t] = [np.zeros((len(l))) for l in self.relatedness_list[t][0]]

				for i in range(self.current_task):
					for li,layer in enumerate(self.relatedness_list[t][i]):
				 		self.relatedness_list_normalized[t][li] += np.exp(-np.array(layer))
				 		self.relatedness_list_normalized_reverse[t][li] += 1/np.exp(-np.array(layer))



				####################################################				 		

				for tt in range(self.current_task):
					for l_id in range(len(self.relatedness_list[t][tt])):
						for gi,k in enumerate(self.groups[l_id]):
							if self.args.ewc_reverse:
								self.relatedness_list[t][tt][l_id][gi] = (1/np.exp(-self.relatedness_list[t][tt][l_id][gi])) /(self.relatedness_list_normalized_reverse[t][l_id][gi])
							else:
								self.relatedness_list[t][tt][l_id][gi] = np.exp(-self.relatedness_list[t][tt][l_id][gi]) /(self.relatedness_list_normalized[t][l_id][gi])

				####################################################				 		
		

		if self.memx is None:
			self.memx = x.data.clone()
			self.memy = y.data.clone()
		else:
			if self.memx.size(0) < self.n_memories:
				self.memx = torch.cat((self.memx, x.data.clone()))
				self.memy = torch.cat((self.memy, y.data.clone()))
				if self.memx.size(0) > self.n_memories:
					self.memx = self.memx[:self.n_memories]
					self.memy = self.memy[:self.n_memories]
		 
		self.net.zero_grad()
		if self.is_cifar:
			offset1, offset2 = self.compute_offsets(t)
			loss = self.bce((self.net(x)[:, offset1: offset2]),
							y - offset1)
		else:
			result = self(x,t)
			loss = self.bce(result, y)#squeeze(0)	

		for tt in range(t):
			for i, p in enumerate(self.net.parameters()): # per layer weight matrix
				
				index_layer = i // 2 # as w, b are seperate in net.parameters()

				if index_layer > 0:
					###pdb.set_trace()
					if len(self.fisher[tt][i].shape) == 1:
						
						if self.args.ewc_reverse:
							s_weights = sum([self.relatedness_list[t][tt][index_layer - 1][gi]*self.relatedness_list_normalized_reverse[t][index_layer - 1][gi] for gi in range(len(self.groups[index_layer-1]))])
						else:
							s_weights = sum([self.relatedness_list[t][tt][index_layer - 1][gi]*self.relatedness_list_normalized[t][index_layer - 1][gi] for gi in range(len(self.groups[index_layer-1]))])

						#s_weights = sum([self.relatedness_list[t][tt][index_layer - 1][gi] for gi in range(len(self.groups[index_layer-1]))])
						#s_weights = sum([np.exp(-self.relatedness_list[t][tt][index_layer - 1][gi]) for gi in range(len(self.groups[index_layer-1]))])
						l = self.reg * s_weights/(len(self.groups[index_layer-1])*t)
						l = l* (p - Variable(self.optpar[tt][i])).pow(2)
						loss += l.sum()
						
					else:
						#for k in range(self.num_groups):
						###pdb.set_trace()
						for gi,k in enumerate(self.groups[index_layer-1]):
							l =self.reg * self.relatedness_list[t][tt][index_layer - 1][gi]													
							#if self.args.ewc_reverse:
							#	l =self.reg * (1/np.exp(-self.relatedness_list[t][tt][index_layer - 1][gi])) /(self.relatedness_list_normalized_reverse[t][index_layer - 1][gi])
							#else:
							#	l =self.reg * np.exp(-self.relatedness_list[t][tt][index_layer - 1][gi]) /(self.relatedness_list_normalized[t][index_layer - 1][gi])


							l = l * (p[:, k] - \
								Variable(self.optpar[tt][i][:, k])).pow(2)
							loss += l.sum()

				else:
					l = self.reg * Variable(self.fisher[tt][i])
					l = l * (p - Variable(self.optpar[tt][i])).pow(2)
					loss += l.sum()					

		loss.backward()
		self.opt.step()

