### A copy of common.py from https://github.com/facebookresearch/GradientEpisodicMemory. 
### We leveraged the same architecture and weight initialization for all of our experiments. 

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import relu, avg_pool2d
import pdb

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)

def find_groups(args,W_ij_ab,W_ij_bc,debug=False):
    
    A = (np.abs(W_ij_ab) > args.xi).astype(int)
    B = (np.abs(W_ij_bc) > args.xi).astype(int)

    layer_a, layer_b, layer_c = W_ij_ab.shape[0], W_ij_ab.shape[1], W_ij_bc.shape[1]
    q_kc = np.random.dirichlet(np.ones(args.c),size=layer_b)
    prev = q_kc
    diff=0
    for t in range(args.iterations):
        #print(np.argmax(q_kc,axis=1))
        #Eq 8
        Pi = q_kc.sum(axis=0)/q_kc.shape[0]

        tau_a = A.dot(q_kc)
        tau_a = (tau_a/tau_a.sum(axis=0,keepdims=True))

        tau_b = q_kc.transpose().dot(B).transpose()
        tau_b = (tau_b/tau_b.sum(axis=0,keepdims=True))


        #Eq 9
        #q_kc = np.array([Pi[c]*np.prod(tau_b[:,c:(c+1)]**B.transpose(),axis=0)*np.prod(tau_a[:,c:(c+1)]**A,axis=0) for c in range(args.c)])
        #q_kc = (q_kc/q_kc.sum(axis=0)).transpose()
        log_q_kc =np.array([np.log(Pi[c])+np.sum(np.log(tau_b[:,c:(c+1)]**B.transpose()),axis=0)+np.sum(np.log(tau_a[:,c:(c+1)]**A),axis=0)for c in range(args.c)])
        #print("middle stuff:{}".format(np.array([1/((np.exp(log_q_kc - log_q_kc[c,:])).sum(axis=0)) for c in range(args.c)])))
        #q_kc = np.array([1/((np.exp(log_q_kc - log_q_kc[c,:])).sum(axis=0)) for c in range(args.c)]).transpose()
        #print("logqkc:{}".format(np.exp(log_q_kc).sum(axis=0)))
        q_kc = np.array(np.exp(log_q_kc)/np.exp(log_q_kc).sum(axis=0)).transpose()
        print("t:{}  qkc:{}".format(t,q_kc))
        memberships = np.argmax(q_kc,axis=1)
        if np.where(memberships==memberships[0])[0].shape[0] == memberships.shape[0]:
            break

        diff = (np.abs(prev - q_kc)).sum() 
        #print("diff:{}".format(diff))       
        if diff< args.threshold:
            tau_a = np.where(tau_a==0, 1e-10, tau_a)
            tau_b = np.where(tau_b==0, 1e-10, tau_b)

            var1 = np.log(Pi[memberships]).sum() 
            var2 = sum([sum(A[:,i]*np.log(tau_a[:,memberships[i]])) for i in range(memberships.shape[0])]) 
            var3 = sum([sum(B[i,:]*np.log(tau_b[:,memberships[i]])) for i in range(memberships.shape[0])])
            
            likelihood = np.log(Pi[memberships]).sum() + \
                sum([sum(A[:,i]*np.log(tau_a[:,memberships[i]])) for i in range(memberships.shape[0])]) + \
                sum([sum(B[i,:]*np.log(tau_b[:,memberships[i]])) for i in range(memberships.shape[0])])

            return True,np.argmax(q_kc,axis=1),diff,t,likelihood

        prev = q_kc
    return False,[-1]*layer_b,diff,t,-np.Inf

def find_likelihood_of_groups(args,groups,W_ij_ab,W_ij_bc):
    #pdb.set_trace()

    A = (np.abs(W_ij_ab) > args.xi).astype(int)
    B = (np.abs(W_ij_bc) > args.xi).astype(int)

    layer_a, layer_b, layer_c = W_ij_ab.shape[0], W_ij_ab.shape[1], W_ij_bc.shape[1]

    q_kc = np.full((layer_b,len(groups)),0.1/len(groups))
    for i in range(len(groups)):
        q_kc[groups[i],i] = 1 -0.1

    prev = q_kc
    
    #print(np.argmax(q_kc,axis=1))
    #Eq 8
    Pi = q_kc.sum(axis=0)/q_kc.shape[0]

    tau_a = A.dot(q_kc)
    tau_a = (tau_a/tau_a.sum(axis=0,keepdims=True))

    tau_b = q_kc.transpose().dot(B).transpose()
    tau_b = (tau_b/tau_b.sum(axis=0,keepdims=True))


    #Eq 9
    #q_kc = np.array([Pi[c]*np.prod(tau_b[:,c:(c+1)]**B.transpose(),axis=0)*np.prod(tau_a[:,c:(c+1)]**A,axis=0) for c in range(args.c)])
    #q_kc = (q_kc/q_kc.sum(axis=0)).transpose()

    log_q_kc =np.array([np.log(Pi[c])+np.sum(np.log(tau_b[:,c:(c+1)]**B.transpose()),axis=0)+np.sum(np.log(tau_a[:,c:(c+1)]**A),axis=0)for c in range(len(groups))])
    memberships = np.argmax(q_kc,axis=1)
    temp = np.log(Pi[memberships]).sum() 
    temp = sum([sum(A[:,i]*np.log(tau_a[:,memberships[i]])) for i in range(memberships.shape[0])]) 
    temp = sum([sum(B[i,:]*np.log(tau_b[:,memberships[i]])) for i in range(memberships.shape[0])])

    likelihood = np.log(Pi[memberships]).sum() + \
        sum([sum(A[:,i]*np.log(tau_a[:,memberships[i]])) for i in range(memberships.shape[0])]) + \
        sum([sum(B[i,:]*np.log(tau_b[:,memberships[i]])) for i in range(memberships.shape[0])])

    return likelihood

class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []
        #pdb.set_trace()
        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        print("layers number {}".format(len(layers)))
        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


class MLP_layerWise(nn.Module):
    def __init__(self, sizes):
        super(MLP_layerWise, self).__init__()
        #pdb.set_trace()
        layers = []
        self.output_index = []
        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
                self.output_index += [0,1]
            else:
                self.output_index += [1]
        print("layers number {}".format(len(layers)))
        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        #pdb.set_trace()
        output = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if self.output_index[i] == 1:
                output.append(x)

        return output

class ResNet_layerWise(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet_layerWise, self).__init__()
        self.in_planes = nf

        #self.conv1 = conv3x3(3, nf * 1)
        self.conv1 = conv3x3(1, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 10, num_blocks[3], stride=2)
        #new layer
        self.linear0 = nn.Linear(nf * 10 * block.expansion, nf * 8 * block.expansion)
        self.relu0 = nn.ReLU()        
        self.linear1 = nn.Linear(nf * 8 * block.expansion, nf * 6 * block.expansion)
        self.relu1 = nn.ReLU()        
        self.linear2 = nn.Linear(nf * 6 * block.expansion, num_classes)

        #self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #pdb.set_trace()        
        output = []
        bsz = x.size(0)
        #out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = relu(self.bn1(self.conv1(x.view(bsz, 1, 28, 28))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear0(out)
        out = self.relu0(out)
        output.append(out)
        out = self.linear1(out)
        out = self.relu1(out)
        output.append(out)
        out = self.linear2(out)
        output.append(out)
        return output

def ResNet18_layerWise(nclasses, nf=20):
    return ResNet_layerWise(BasicBlock, [2, 2, 2, 2], nclasses, nf)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        #self.conv1 = conv3x3(3, nf * 1)
        self.conv1 = conv3x3(1, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 10, num_blocks[3], stride=2)
        #new layer
        self.linear0 = nn.Linear(nf * 10 * block.expansion, nf * 8 * block.expansion)
        self.relu0 = nn.ReLU()        
        self.linear1 = nn.Linear(nf * 8 * block.expansion, nf * 6 * block.expansion)
        self.relu1 = nn.ReLU()        
        self.linear2 = nn.Linear(nf * 6 * block.expansion, num_classes)

        #self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #pdb.set_trace()        
        bsz = x.size(0)
        #out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = relu(self.bn1(self.conv1(x.view(bsz, 1, 28, 28))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear0(out)
        out = self.relu0(out)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        return out


def ResNet18(nclasses, nf=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)



def von_Neumann_divergence(A,B):

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
    #pdb.set_trace()
    return Divergence

class Covariance():
    def __init__(self,dim=None):
        self.initialized = False
        if not dim is None:
            self.dim = dim
            self.N = 0 
            self.mean = np.zeros((1,dim))
            self.C = np.zeros((dim,dim))
            self.initialized = True
    
    def update(self,x):
        if not self.initialized:
            self.__init__(x.shape[1])
        #pdb.set_trace()
        for i in range(x.shape[0]):
            self.update_(x[i,:])

    def update_(self,x):
        self.N+=1
        self.mean += (x - self.mean)/self.N
        temp = (x-self.mean)
        if np.isnan(temp).any():
            pdb.set_trace()
        if np.isnan(temp*temp.transpose()).any():
            pdb.set_trace()
        self.C +=temp*temp.transpose()

    def get_cov(self):
        return self.C/self.N
