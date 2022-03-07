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
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def logm(A):
    e,v = torch.symeig(A, eigenvectors=True)  #torch.symeig返回(eigenvalues,eigenvectors)，可设置是否需要eigenvectors
    e = torch.diag(torch.log(e))              #torch.diag()将数值放在一个零矩阵的对角线上
    return torch.mm(v,torch.mm(e,torch.inverse(v)))   #torch.mm(a,b) 矩阵乘法a乘b

def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()   #tensor.t()矩阵的转置

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
        avg = torch.mean(x, 0)  #torch.mean(x,0) 例子：[4,3]的矩阵对列求和取均值结果为[1,3]的矩阵

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))  # tensor下，a=b.sub(c)为a=b-c #x=x.expand_as(y)将x扩展为和y一样的维度  例子：如果x为大小为[1,3]的矩阵，y为[4,3]的矩阵，则拓展后x大小也为[4,3]，且同一列的数值都一样

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze() #tensor.squeeze()去掉维度中为1的维度 例子：大小为[1,2,1,3]会变为[2,3]

#协方差的求解步骤：(1)对x的每一列求均值   avg=torch.mean(x,0)
#               (2)对x的每一行减去均值   y=x.sub(avg.expand_as(x))
#               (3)cov=1/(m-1)*y.t()*y


def cov_x_y(x,y):
    y = y.view(-1,1)  #tensor增加第二维的方法
    x_y = torch.cat((y,x),dim=1) #torch.cat((a,b),0)就是横着拼，得保证列维度相同，所以0就是表示改变的是第一个维度；同理，torch((a,b),1)就是竖着拼，改变的是第二个维度
    #a=torch.diag_embed(torch.rand(x_y.shape[1]))
    result=cov(x_y)
    #print(result)
    return result

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

#loss function的公式为D=trace((   cov(xy)-cov(xy^)  )   *  (  logm(cov(xy)-logm(cov(xy^))  )    ))
#diff_logm_1就是公式后面的logm的部分，diff_logm_2只是对logm部分取了一个负数将前面的-cov(xy^)的负号抵消了，所以是div1+div2

#%%

torch.manual_seed(4)  
""" x_test = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y_test = x_test.pow(2)               # test without noise


def train_generator(noise_rate):
    x_train = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  # x data (tensor), shape=(1000, 1)
    y_train = x_train.pow(2) + noise_rate*torch.rand(x_train.size())
    return [x_train,y_train]   """

x,y=load_boston(return_X_y=True)
#x,y=load_diabetes(return_X_y=True)
#x=x[:,:1]

#x_stand = StandardScaler().fit_transform(x)
#sklearn_pca = sklearnPCA(n_components=6)
#x_pca = sklearn_pca.fit_transform(x_stand)

mean=np.mean(x,0)
std=np.std(x,0)
x=(x-mean)/std
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

x_train=torch.from_numpy(x_train).float()
y_train=torch.from_numpy(y_train).float()
x_test=torch.from_numpy(x_test).float()
y_test=torch.from_numpy(y_test).float()

def train_generator(noise_rate):
    datasets = torch.utils.data.TensorDataset(x_train, y_train+noise_rate*torch.rand(y_train.size()))
    train_iter = torch.utils.data.DataLoader(datasets, batch_size=20, shuffle=False)
    return train_iter

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden_1= torch.nn.Linear(n_feature, 128)   # hidden layer
        self.hidden_2 = torch.nn.Linear(128, 64)   # output layer
        #self.hidden_3 = torch.nn.Linear(64, 10)
        self.predict = torch.nn.Linear(64,n_output)
        #self.bat_1=torch.nn.BatchNorm1d(128)
        #self.bat_2=torch.nn.BatchNorm1d(64)
        self.do=torch.nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))# activation function for hidden layer
        x = F.relu(self.hidden_2(x))
        #x = F.relu(self.hidden_3(x))
        #x=self.do(x)
        x = self.predict(x)             # linear output
        return x
#%% von_Neumann divergence
torch.manual_seed(4)
net = Net(n_feature=13, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


plt.ion()   # something about plotting

#noise_rate_choice=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
noise_rate_choice=[0,1,2,3,4,5,6,7,8,9,10]

loss_total=[[0]*2 for _ in range(11)]

for index in range(len(noise_rate_choice)):
    for number in range(2):
        for epoch in range(200):
            for inputs,labels in train_generator(noise_rate_choice[index]):
                prediction = net(inputs)     # input x and predict based on x
            
                print(prediction.size())

                loss = loss_function(inputs,labels,prediction.squeeze())   # must be (1. nn output, 2. target)
                #loss = loss_func(y,prediction)

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                #torch.nn.utils.clip_grad_norm_(net.parameters(),2)
                optimizer.step()        # apply gradients

        test_prediction=net(x_test)
        loss_total[index][number]=loss_func(y_test,test_prediction.squeeze())

        
#print(loss_total)

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

#noise_rate_choice=[0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5]
noise_rate_choice=[0,1,2,3,4,5,6,7,8,9,10]

#noise_rate_choice=[0]

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden_1= torch.nn.Linear(n_feature, 128)   # hidden layer
        self.hidden_2 = torch.nn.Linear(128, 64)   # output layer
        #self.hidden_3 = torch.nn.Linear(64, 10)
        self.predict = torch.nn.Linear(64,n_output)
        #self.bat_1=torch.nn.BatchNorm1d(128)
        #self.bat_2=torch.nn.BatchNorm1d(64)
        self.do=torch.nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))# activation function for hidden layer
        x = F.relu(self.hidden_2(x))
        #x = F.relu(self.hidden_3(x))
        #x=self.do(x)
        x = self.predict(x)             # linear output
        return x

net_mse = Net(n_feature=13, n_output=1)     # define the network
print(net_mse)  # net architecture

optimizer = torch.optim.Adam(net_mse.parameters(), lr=0.0005)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

loss_total_2=[[0]*2 for _ in range(11)]

for index in range(len(noise_rate_choice)):
    for number in range(2):
        for epoch in range(200):
            for inputs,labels in train_generator(noise_rate_choice[index]):
                prediction = net_mse(inputs)     # input x and predict based on x
                #print(prediction.size())
                #loss = loss_function(x,y,prediction)   # must be (1. nn output, 2. target)
                loss = loss_func(prediction.squeeze(),labels)

                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
            
        test_prediction=net_mse(x_test)
        #loss_total_2[index][number]=loss_func(y_test+noise_rate_choice[index]*torch.rand(y_test.size()),test_prediction.squeeze())
        loss_total_2[index][number]=loss_func(y_test,test_prediction.squeeze())

#plot to test whether the prediction describes data distribution well based on MSE 0 noise level
""" predic=net_mse(x)
print(predic.shape)
plt.plot(np.linspace(0,506,506),predic.detach().numpy(),'b-',lw=2,label='prediction')
plt.scatter(np.linspace(0,506,506),y.detach().numpy(),label='ground_truth')
plt.xlabel('data')
plt.ylabel('value')
plt.legend() """

print(loss_total_2)

mean_loss_total_2=[[0] for _ in range(11)]
for row in range(11):
    mean_loss_total_2[row]=torch.mean(torch.Tensor(loss_total_2[row][:]))

print(mean_loss_total_2)

#from yu
#plt.scatter(x_train.numpy(), y_train.numpy())
#plt.plot(x_train.numpy(), prediction.data.numpy(),'r-', lw=5)
#plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
#plt.ioff()
#plt.show()

""" plt.plot(noise_rate_choice,mean_loss_total_2,'b-',lw=2)
plt.xlabel('noise level')
plt.ylabel('performance')
plt.ioff()
plt.show() """

#plot comparsion of Ours and MSE
plt.plot(noise_rate_choice,mean_loss_total,'b-',lw=2,label='Ours',marker='o',markerfacecolor='gray',markersize=10)
plt.plot(noise_rate_choice,mean_loss_total_2,'r--',lw=2,label='MSE',marker='^',markerfacecolor='white',markersize=10)
#plt.plot(epoch,mean_loss_total_2,'r--',lw=2,label='MSE',marker='^',markerfacecolor='white',markersize=10)

plt.xlabel('noise_level',{'size':15})
plt.ylabel('performance(MSE)',{'size':15})
plt.legend(prop={'size':20})
plt.yticks(fontproperties = 'Times New Roman', size = 15)
plt.xticks(fontproperties = 'Times New Roman', size = 15)
plt.ioff()
plt.show() 

print('end') 

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
