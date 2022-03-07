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

import numpy as np
import quadprog
import pdb
from .common import MLP, ResNet18

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


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
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


def project2cone2(gradient, memories, margin=0.5):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """    
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
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
        #if self.is_cifar:
        #    self.net = ResNet18(n_outputs)
        #else:
        #    self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.is_cifar = (args.data_file == 'cifar100.pt')
        self.is_omniglot = args.is_omniglot

        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        elif self.is_omniglot:
            self.outputs_per_task = args.outputs_per_task
            if args.RESNET:
                #self.net = ResNet18(sum(self.outputs_per_task))                        
                self.net = ResNet18(sum(self.outputs_per_task))
            else:
                self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])            
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])


        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda

        # allocate episodic memory
        #self.memory_data = torch.FloatTensor(
        #    n_tasks, self.n_memories, n_inputs)        
        #self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        self.memory_data = torch.FloatTensor(size=[n_tasks, self.n_memories, np.prod(n_inputs)])        
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        #self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        self.grads = torch.Tensor(sum(self.grad_dims), 2)

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
            self.cuda()

        self.age = 0

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        # update memory
        self.age += 1
        
        dev = self.grads.device 

        #print(t,self.age)
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        #pdb.set_trace()
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            #self.memory_labs[t, self.mem_cnt: endcnt].copy_(
            #    y.data[: effbsz])
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz][:,0])
            
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                #offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                #                                   self.is_cifar)
                #ptloss = self.ce(
                #    self.forward(
                #        Variable(self.memory_data[past_task]),
                #        past_task)[:, offset1: offset2],
                #    Variable(self.memory_labs[past_task] - offset1))
                #ptloss.backward()
                #store_grad(self.parameters, self.grads, self.grad_dims,
                #           past_task)

                

                offset1, offset2 = compute_offsets(0, self.nc_per_task,
                                                   self.is_cifar)
                ptloss = self.ce(self.forward(Variable(self.memory_data[0]),
                        0)[:, offset1: offset2],Variable(self.memory_labs[0] - offset1))

                for task in range(1,past_task+1):
                    offset1, offset2 = compute_offsets(task, self.nc_per_task,
                                                       self.is_cifar)
                    ptloss += self.ce(self.forward(Variable(self.memory_data[task]),
                            task)[:, offset1: offset2],Variable(self.memory_labs[task] - offset1))
                ptloss = ptloss/(len(self.observed_tasks)-1)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,0)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        #loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        #loss = self.ce(self.forward(x, t)[:, offset1: offset2], y.squeeze(0) - offset1)        
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y.squeeze(1) - offset1)        
        
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, 1)
            #indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
            #    else torch.LongTensor(self.observed_tasks[:-1])
            #dotp = torch.mm(self.grads[:, t].unsqueeze(0),
            #                self.grads.index_select(1, indx))
            dotp = torch.mm(self.grads[:, 1].unsqueeze(0),
                            self.grads.index_select(1, torch.tensor([0],device=dev)))
            if (dotp < 0).sum() != 0:
                #project2cone2(self.grads[:, t].unsqueeze(1),
                #              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                #overwrite_grad(self.parameters, self.grads[:, t],
                #               self.grad_dims)
                project2cone2(self.grads[:, 1].unsqueeze(1),
                              self.grads.index_select(1, torch.tensor([0],device=dev)), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, 1],
                               self.grad_dims)

        self.opt.step()
