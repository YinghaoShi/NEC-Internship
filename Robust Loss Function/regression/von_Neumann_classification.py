# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 14:23:19 2021

@author: yu
"""


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy as np
from sklearn.datasets import make_classification
from torch.autograd import Variable

def logm(A):
    e,v = torch.symeig(A, eigenvectors=True)
    e = torch.diag(torch.log(e))
    return torch.mm(v,torch.mm(e,torch.inverse(v)))

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()

def cov_x_y(x,y):
    y = y.view(-1,1)
    x_y = torch.cat((y,x),dim=1)
    return cov(x_y)

def loss_function(x,y_true,y_pred):
    
    cov_true = cov_x_y(x,y_true)
    cov_estimate = cov_x_y(x,y_pred)
    
    #divergence between cov_true and cov_estimate
    diff_logm_1 = logm(cov_true)-logm(cov_estimate)
    inner_first_term_1 = torch.mm(cov_true, diff_logm_1)
    div_1 = torch.trace(inner_first_term_1-cov_true+cov_estimate)
    
    #divergence between cov_estimate and cov_true
    diff_logm_2 = logm(cov_estimate)-logm(cov_true)
    inner_first_term_2 = torch.mm(cov_estimate, diff_logm_2)
    div_2 = torch.trace(inner_first_term_2-cov_estimate+cov_true)
    
    div_total = div_1+div_2
    return div_total
#%%

torch.manual_seed(5)  
""" x_test = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y_test = x_test.pow(2)               # test without noise """


""" def train_generator(noise_rate):
    x_train = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape=(1000, 1)
    y_train = x_train.pow(2) + noise_rate*torch.rand(x_train.size())
    return [x_train,y_train]     """          


def train_generator(noise_rate):
    x_train,y_train = make_classification(n_samples=1000, n_features=2, n_informative=2,n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,class_sep=2,flip_y=noise_rate,weights=[0.5,0.5], random_state=17)
    x_train=Variable(torch.Tensor(x_train).float())
    y_train=Variable(torch.Tensor(y_train).float())
    #y_train=y_train[...,None]
    return [x_train,y_train]


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
#%% von_Neumann divergence
torch.manual_seed(4)
net = Net(n_feature=2, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
#loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


plt.ion()   # something about plotting

noise_rate_choice=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

loss_total=[[0]*50 for _ in range(11)]

for index in range(len(noise_rate_choice)):
    for number in range(50):
        for t in range(2):
            prediction = net(train_generator(noise_rate_choice[index])[0])     # input x and predict based on x
            print(train_generator(noise_rate_choice[index])[0].shape)
            print(prediction.shape)
            loss = loss_function(train_generator(noise_rate_choice[index])[0],train_generator(noise_rate_choice[index])[1],prediction)   # must be (1. nn output, 2. target)
            #loss = loss_func(y,prediction)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            loss_total[index][number]=loss
        
print(loss_total)

mean_loss_total=[[0] for _ in range(11)]
for row in range(11):
    mean_loss_total[row]=torch.mean(torch.Tensor(loss_total[row][:]))

print(mean_loss_total)


#bias = torch.mean(y_train - prediction)
#bias = 0
#test_prediction = net(x_test)  
#y_pred_test = test_prediction+bias
#pred = prediction+bias
#loss_mse = loss_func(pred, y_train)
#plt.scatter(x_train.numpy(), y_train.numpy())
#plt.plot(x_train.numpy(), pred.detach().numpy(),'r-', lw=5)
#plt.text(0.5, 0, 'Loss=%.4f' % loss_mse.data.numpy(), fontdict={'size': 20, 'color':  'red'})
#plt.ioff()
#plt.show()

#%% MSE
torch.manual_seed(4)  

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net_mse = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net_mse.parameters(), lr=0.2)
loss_func = torch.nn.CrossEntropyLoss()  # this is for regression mean squared loss


plt.ion()   # something about plotting

loss_total_2=[[0]*50 for _ in range(11)]

for index in range(len(noise_rate_choice)):
    for number in range(50):
        for t in range(2):
            prediction = net_mse(train_generator(noise_rate_choice[index])[0])     # input x and predict based on x
            #loss = loss_function(x,y,prediction)   # must be (1. nn output, 2. target)
            loss = loss_func(prediction,train_generator(noise_rate_choice[index])[1].long())

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            loss_total_2[index][number]=loss

print(loss_total_2)

mean_loss_total_2=[[0] for _ in range(11)]
for row in range(11):
    mean_loss_total_2[row]=torch.mean(torch.Tensor(loss_total_2[row][:]))

print(mean_loss_total_2)

#plt.scatter(x_train.numpy(), y_train.numpy())
#plt.plot(x_train.numpy(), prediction.data.numpy(),'r-', lw=5)
#plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})

#plt.ioff()
#plt.show()

""" plt.plot(noise_rate_choice,mean_loss_total,'b-',lw=4,label='Von_Neumann')
plt.plot(noise_rate_choice,mean_loss_total_2,'r-',lw=4,label='MSE')
plt.xlabel('noise_rate')
plt.ylabel('loss')
plt.legend(prop={'size':20})
plt.ioff()
plt.show()
print('end') """

#%%compare in test dataset

""" prediction = net_mse(x_test)
prediction_new = net(x_test)
pred_train = net(x_train)
bias = torch.mean(y_train - pred_train)
pred_vN = prediction_new+bias
#bias = 0

plt.plot(x_test.numpy(), pred_vN.data.numpy(),'b--', lw=4,label='Von_Neumann')
plt.plot(x_test.numpy(), prediction.data.numpy(),'r-.', lw=4,label='MSE')
plt.plot(x_test.numpy(), y_test.numpy(),color='black',lw=4,label='Ground Truth (test)')

plt.legend()


test_prediction = net(x_test)  
y_pred_test = test_prediction+bias
loss_VN = loss_func(pred_vN, y_test)
loss_mse = loss_func(prediction, y_test)
plt.text(-0.5, 0.7, 'VN_Loss=%.4f' % loss_VN.data.numpy(), fontdict={'size': 15, 'color':  'red'})
plt.text(-0.5, 0.6, 'MSE_Loss=%.4f' % loss_mse.data.numpy(), fontdict={'size': 15, 'color':  'red'}) """























# %%
