### This is a copy of GEM from https://github.com/facebookresearch/GradientEpisodicMemory. 
### In order to ensure complete reproducability, we do not change the file and treat it as a baseline.

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from .common import MLP, ResNet18, von_Neumann_divergence, MLP_layerWise, find_groups, Covariance

import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import fractional_matrix_power
from scipy.linalg import logm
import scipy.io
from scipy.linalg import block_diag
from numpy.linalg import inv
from numpy.linalg import eig
import random

import quadprog
from .common import MLP, ResNet18
from types import SimpleNamespace
from copy import deepcopy
import pdb, pickle
import collections
import sys

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid,grads_groups=None,groups=None):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layer
        tid: task id
    """
    
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    index_layer= 0
    for param in pp():        
        #print("para shape:{}".format(param.shape))
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        
        index_layer= cnt //2
        
        #revision for conv layers
        if (len(param.grad.data.shape)==1):       
            for k in range(grads_groups[tid][index_layer].shape[0]):
                grads_groups[tid][index_layer][k,beg:en] = param.grad.data.cpu().clone()
        elif (index_layer == 0 or len(param.grad.data.shape)==4):
            grads_groups[tid][index_layer] = np.zeros((1,sum(grad_dims)))
            grads_groups[tid][index_layer][0,beg: en] =  param.grad.data.view(-1).cpu().clone()
        else:
            #pdb.set_trace()
            #print("groups:{}".format(groups))
            grads_groups[tid][index_layer] = []
            for k_id,k in enumerate(groups[index_layer-1]):
                #print("kid:{}".format(k_id))
                #print("k content:{}".format(k))
                #print("index layer:{}".format(index_layer))
                #print("parameter grad data shape:{}".format(param.grad.data.shape))
                ind = np.hstack([k+beg+param.grad.data.shape[1]*i for i in range(param.grad.data.shape[0])])                
                #print("shape 0:{}".format(param.grad.data.shape[0]))    #10
                #print("shape 1:{}".format(param.grad.data.shape[1]))    #100
                #print("ind content:{}".format(ind.shape))  #1800, 1800, 2100, 2600, 1700, 300, 190, 150, 170, 190 sum=11000
                group = param.grad.data[:,k]                
                v = np.zeros(sum(grad_dims))
                v[ind] = param.grad.data[:,k].view(-1).cpu().clone()
                #print("k part gard:{}".format(param.grad.data[:,k].view(-1).shape))
                #print("v1 number:{}".format(v.shape[0]-np.sum(v==0)))
                #print("v shape:{}".format(v.shape))
                grads_groups[tid][index_layer].append(v)
                #print("shape:{}".format(grads_groups[tid][index_layer].shape))
            grads_groups[tid][index_layer] = np.vstack(grads_groups[tid][index_layer])

        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def transform_relatedness(relatedness,args=None):    
    #pdb.set_trace()
    h = []    
    for i,l1 in enumerate(relatedness): 
        #h += [float('inf')]
        h += []
        #print("h:{}".format(h))
        for j,l2 in enumerate(relatedness[i]): 
            
            #if args.gem_reverse:
            #    relatedness_copy = [1/np.exp(-e) for e in l2]
            #    relatedness_copy = [e/divergence_list_normalized_reverse[j][m] for m,e in enumerate(relatedness_copy)]
            #else:
            #    relatedness_copy = [np.exp(-e) for e in l2]
            #    relatedness_copy = [e/divergence_list_normalized[j][m] for m,e in enumerate(relatedness_copy)]

            relatedness_copy = l2.copy()
            #relatedness_copy = np.ones_like(relatedness_copy).tolist()
            #print("relatedness_copy:{}".format(relatedness_copy))
            if args.gem_sub_mean:
                relatedness_copy = [e-np.mean(relatedness_copy) for e in relatedness_copy]
            
            h+=relatedness_copy
    return h


def project2cone2(gradient, memories, current_Tid, margin=0.5,grads_groups=None,relatedness=None,args=None):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """    
    #gradient for current task
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    
    #divergence
    h = transform_relatedness(relatedness[current_Tid],args)
    
    #concenate the individual group gradients
    #e.g. number of groups=5 & number of layers=2 -> shape of pp is (11,p) 
    # -> first one is remaining gradients except group gradients, the last two is (5,p) and (5,p) associated with groups
    for tid in range(current_Tid):        
        for pi, p in enumerate(grads_groups[tid]):
            pp = p if ((tid==0) & (pi==0)) else np.concatenate((pp, p), axis=0)        
    
    print("pp:{}".format(pp))
    #np h
    h = np.array(h) + margin
    if args.gem_ignore_relatedness:
        h = np.zeros_like(h)

    #concenate h from vector (t-1)*(number of groups)*(number of layers) to matrix ((t-1), (number of groups)*(numvber of layers))
    h=np.expand_dims(h,axis=0)
    for i in range(current_Tid):
        if i==0:
            portion = h[:,i*(args.n_layers*args.num_groups):(i+1)*(args.n_layers*args.num_groups)]
        else:
            portion_ = h[:,i*(args.n_layers*args.num_groups):(i+1)*(args.n_layers*args.num_groups)]
            portion = np.vstack((portion, portion_))
    h = portion
    
    #normalization of inverted relatedness columnwisely
    #h_col_sum = np.sum(h,axis=0)
    #for i in range(len(h)):
    #    for j in range(args.n_layers*args.num_groups):
    #        h[i][j] = h[i][j]/h_col_sum[j]
    
    #move the mean of inverted relatedness in each column to 1
    h_col_mean= np.mean(h,axis=0)
    for i in range(len(h)):
        for j in range(args.n_layers*args.num_groups):
            h[i][j] = h[i][j]-h_col_mean[j]+1
    
    #set threshold for inverted relatedness
    #h = np.where(h<0.4,0.4,h)
    #h = np.where(h>1.6,1.6,h)
    
    #weight value for remaining gradients except group gradients
    value = 1

    #reconstruction
    for i in range(current_Tid):
        if i == 0:
            lastfive = h[i,:].tolist()
            lastfive.insert(0,value)
            lastfive = np.expand_dims(lastfive,axis=0)
            temp = np.dot(lastfive,\
            pp[i*(1+args.n_layers*args.num_groups):(i+1)*(1+args.n_layers*args.num_groups),:])
        else:
            lastfive = h[i,:].tolist()
            lastfive.insert(0,value)
            lastfive = np.expand_dims(lastfive,axis=0)
            temp_ = np.dot(lastfive,\
            pp[i*(1+args.n_layers*args.num_groups):(i+1)*(1+args.n_layers*args.num_groups),:])
            temp = np.vstack((temp,temp_))
    
    #sum up task level gradients like AGEM
    #temp = np.sum(temp,axis=0)
    #temp = np.expand_dims(temp, axis=0)
    
    #D=GGT
    P = np.dot(temp, temp.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(P.shape[0])*(1e-3)

    #d=-Gg
    q =  np.dot(temp, gradient.cpu().contiguous().view(-1).double().numpy()) * -1

    #A and m
    t = temp.shape[0]
    G = np.eye(t)
    G = np.eye(t) + np.eye(t)*0.00001
    h_ = np.zeros(t)
    try:
        v = quadprog.solve_qp(P, q, G, h_)[0]
    except:
        pdb.set_trace()
    x = np.dot(v, temp) + gradient_np  
    gradient.copy_(torch.Tensor(x).view(-1, 1))

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.is_cifar = (args.data_file == 'cifar100.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        else:
            #self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])
            self.net = MLP_layerWise([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.args = args

        self.cuda = args.cuda
        #if self.cuda:
        #    self.net = self.net.cuda()


        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)        
        self.grads_groups = [[ None for i in range(len(self.grad_dims)//2)] for j in range(n_tasks)]

        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
        
        if args.cuda:
            #self.cuda()
            self.net = self.net.cuda()

        self.age = 0
        self.n_tasks = 0
        self.task_history = {}
        # groups, relatedness and covariance        
        self.groups = []
        self.cov_list = []
        self.first_task_x_list = []
        self.first_task_y_list = []

        self.num_groups = args.num_groups
        self.divergence_list = [None]
        self.divergence_list_normalized = [None]
        self.divergence_list_normalized_reverse = [None]

        #fixed parameters
        self.xi =0.1
        self.group_finding_iterations =1000
        self.group_finding_likelihood_threshold=0.1
        self.group_finding_trials=10

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

    def compute_divergence(self, C_x1_y1, C_x2_y2):        
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
        #pdb.set_trace()
        W_ij_ab = None          
        for i, p in enumerate(self.net.parameters()): # per layer weight matrix
            if len(p.shape)==1 or len(p.shape)==4:
                continue
            
            arguments = SimpleNamespace(c=self.num_groups,xi =self.xi,iterations=self.group_finding_iterations,threshold=self.group_finding_likelihood_threshold)               

            args = SimpleNamespace(**vars(arguments))

            index_layer = i // 2 # as w, b are seperate in net.parameters()
            groups_layer = []
            #W_ij_bc = p.data.numpy().transpose()
            W_ij_bc = p.cpu().data.numpy().transpose()

            if index_layer >0 :
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
                    ###pdb.set_trace()
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
                    group = layer[:,k]
                    if self.args.if_output_cov:
                        group_xy = torch.cat([group, outputs[-1]], 1)
                    else:
                        #pdb.set_trace()         
                        group_xy = torch.cat([group, y.squeeze(-1).float()], 1)                        

                    #group_xy = group_xy.data.numpy()
                    group_xy = group_xy.data.cpu().numpy()
                    cov_list[i][k_id].update(group_xy)
                
        return cov_list

    def get_group_divergence(self, cov_list1, cov_list2):

        #if self.current_task==3:
        #   pdb.set_trace()
        relatedness = [[] for i in range(len(cov_list1))] 

        for layer in range(len(cov_list1)):
            for k in range(len(self.groups[layer])):
                cov1, cov2 = cov_list1[layer][k], cov_list2[layer][k]
                score = self.compute_divergence(cov1.get_cov(), cov2.get_cov())
                relatedness[layer].append(score)
        return relatedness

    def forward(self, x, t):
        outputs = self.net(x)
        output = outputs[-1]
        #print("output:{}".format(output.size()))
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            #print("offset1:{}".format(offset1))
            #print("offset2:{}".format(offset2))
            #print("n_outputs:{}".format(self.n_outputs))
            if offset1 > 0:
                #output[:, :offset1].data.fill_(-10e10)
                output[:offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                #output[:, offset2:self.n_outputs].data.fill_(-10e10)
                output[offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y,startForgetting=False,forgetting_task_ids=None):
        #pdb.set_trace()
        if self.cuda:
            x=x.cuda()
            y=y.cuda()

        if not startForgetting:
            self.observe_learn(x, t, y)
        #else:
        #    self.observe_attack(x, t, y, forgetting_task_ids)


    def observe_learn(self, x, t, y):
        # update memory
        self.age += 1

        self.task_history[t] = self.task_history.get(t,0)+1
        # next task?        
        if t != self.old_task:
            print("Task:\t",t)
            self.observed_tasks.append(t)

            self.n_tasks +=1
            if self.old_task==0:
                # Creating the groups and computing the covariance matrix for the fist task
                # and then emptying the buffers
                #pdb.set_trace()
                if not self.args.create_random_groups:
                    self.creategroups()
                else:
                    self.creategroups_Random()
                
                self.cov_list.append(self.get_hidden_cov(self.first_task_x_list,self.first_task_y_list))
                self.first_task_x_list, self.first_task_y_list = [] ,[]

            self.old_task = t
        
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
                self.divergence_list.append([[[1] *(len(l))  for l in self.groups] for i in range(t)])
                #print("relatedness list:{}".format(self.divergence_list))
                self.divergence_list_normalized.append([np.array([np.exp(-1)*t]*(len(l)))  for l in self.groups])
                self.divergence_list_normalized_reverse.append([np.array([t/np.exp(-1)]*(len(l)))  for l in self.groups])
            else:
                self.get_hidden_cov(x,y.unsqueeze(0),self.cov_list[t])
            if self.task_history[t] % self.args.cov_recompute_every==0:
                #if t>2:
                #    pdb.set_trace()
                current_cov = self.cov_list[self.old_task]
                self.divergence_list[t] = [self.get_group_divergence(current_cov, self.cov_list[i]) for i in range(self.old_task)]
                #pdb.set_trace()
                self.divergence_list_normalized[t] = [np.zeros((len(l))) for l in self.divergence_list[t][0]]
                self.divergence_list_normalized_reverse[t] = [np.zeros((len(l))) for l in self.divergence_list[t][0]]
                #for i in range(self.old_task):
                for i in range(t):                    
                    for li,layer in enumerate(self.divergence_list[t][i]):
                        self.divergence_list_normalized[t][li] += np.exp(-np.array(layer))    #transform divergence from a postive value to a value (0,1) ->  relatedness      
                        self.divergence_list_normalized_reverse[t][li] += 1/np.exp(-np.array(layer))  #transform divergence from a postive value to a value (1,inf)   ->divergence
                
                for tt in range(t):
                    for l_id in range(len(self.divergence_list[t][tt])):
                        for gi,k in enumerate(self.groups[l_id]):
                            if self.args.ewc_reverse:    #true: divergence  false: relatedness
                                self.divergence_list[t][tt][l_id][gi] = (1/np.exp(-self.divergence_list[t][tt][l_id][gi])) /(self.divergence_list_normalized_reverse[t][l_id][gi])
                            else:
                                self.divergence_list[t][tt][l_id][gi] = np.exp(-self.divergence_list[t][tt][l_id][gi]) /(self.divergence_list_normalized[t][l_id][gi])


        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                   self.is_cifar)
                ptloss = self.ce(
                    self.forward(
                        Variable(self.memory_data[past_task]),
                        past_task)[:, offset1: offset2],
                    Variable(self.memory_labs[past_task] - offset1))
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task,self.grads_groups,self.groups)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        #pdb.set_trace()        
        #loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y.squeeze(0) - offset1)        
        
        loss.backward()

        #print("relatedness:{}".format(self.divergence_list))
        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t,self.grads_groups,self.groups)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.cuda \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), t, self.margin,self.grads_groups,self.divergence_list,self.args)

                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)

        self.opt.step()
