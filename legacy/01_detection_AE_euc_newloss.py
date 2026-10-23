# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:57:05 2021

@author: LAI Chenyang
"""

# =============================================================================
# EHSA position signal reconstruction by AE
# =============================================================================

import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset # creat custom dataset
from torch.utils.data import DataLoader
# from thr_definition import euc_thr, mal_thr, hau_thr
from thr_definition_new import euc_thr, mal_thr, hau_thr
# spatial distance
# from similarity import newLoss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

# load data

# load healthy data
health_AME_dis = scio.loadmat('./healthy.mat')['AME_dis_mat']
health_Current = scio.loadmat('./healthy.mat')['current_mat']
health_para = scio.loadmat('./healthy.mat')['para_mat']
health_Simu_dis = scio.loadmat('./healthy.mat')['Simu_dis_mat']
health_Spool_dis = scio.loadmat('./healthy.mat')['Spool_dis_mat']

# load degradation data
deg_AME_dis = scio.loadmat('./degradation.mat')['AME_dis_mat']
deg_Current = scio.loadmat('./degradation.mat')['current_mat']
deg_para = scio.loadmat('./degradation.mat')['para_mat']
deg_Simu_dis = scio.loadmat('./degradation.mat')['Simu_dis_mat']
deg_Spool_dis = scio.loadmat('./degradation.mat')['Spool_dis_mat']

# plots original data
deg_severity=deg_para[:,6]
index1=(deg_severity<0.4)  # leakage 0.3mm
index2=np.logical_and(deg_severity<0.7,deg_severity>0.5)  # leakage 0.6mm
index3=np.logical_and(deg_severity<1.0,deg_severity>0.8)  # leakage 0.9mm
index4=np.logical_and(deg_severity<1.3,deg_severity>1.1)  # leakage 1.2mm
index5=np.logical_and(deg_severity<1.6,deg_severity>1.4)  # leakage 1.5mm

# plt.plot(health_AME_dis.T,'b')
# plt.plot(deg_AME_dis[index1,:].T,'b')
# plt.plot(deg_AME_dis[index2,:].T,'k')
# # plt.plot(deg_AME_dis[index3,:].T,'g')
# # plt.plot(deg_AME_dis[index4,:].T,'y')
# plt.plot(deg_AME_dis[index5,:].T,'r')
# plt.show()

# resample and combine dis_current data
index=np.linspace(0,1800,91,dtype=(int))
health_AME_dis=health_AME_dis[:,index[1:]]
health_Current=health_Current[:,index[1:]]
deg_AME_dis=deg_AME_dis[:,index[1:]]
deg_Current=deg_Current[:,index[1:]]

health_cur_dis=np.hstack((health_AME_dis,health_Current))
deg_cur_dis=np.hstack((deg_AME_dis,deg_Current))

deg1=deg_AME_dis[index1,:];deg1_cur=deg_Current[index1,:]
deg2=deg_AME_dis[index2,:];deg2_cur=deg_Current[index2,:]
deg3=deg_AME_dis[index3,:];deg3_cur=deg_Current[index3,:]
deg4=deg_AME_dis[index4,:];deg4_cur=deg_Current[index4,:]
deg5=deg_AME_dis[index5,:];deg5_cur=deg_Current[index5,:]

def data_loader_config(X_data,Y_data,batch_size):
    
    # train_val_set, test_set split 
    train_x,test_x,train_y,test_y=train_test_split(X_data,Y_data,random_state=10,test_size=0.2)
    
    # # split the train_val_set into train set and validation set
    # train_x,val_x,train_y,val_y=train_test_split(train_val_x,train_val_y,random_state=10)
    
    # x normalization
    x_scaler = MinMaxScaler()
    x_scaler.fit(train_x)
    x_train=x_scaler.transform(train_x)
    x_test=x_scaler.transform(test_x)
    # y normalization
    y_scaler = MinMaxScaler()
    y_scaler.fit(train_y)
    y_train=y_scaler.transform(train_y)
    y_test=y_scaler.transform(test_y)
    
    # torch dataset of train/test set
    xtrain=torch.from_numpy(x_train).type(torch.FloatTensor)
    xtest=torch.from_numpy(x_test).type(torch.FloatTensor)
    ytrain=torch.from_numpy(y_train).type(torch.FloatTensor)
    ytest=torch.from_numpy(y_test).type(torch.FloatTensor)
    
    Data=[xtrain,xtest,ytrain,ytest]
    
    train_set=TensorDataset(xtrain,ytrain)
    test_set=TensorDataset(xtest,ytest)
    
    # torch dataloader of train/test set
    train_loader=DataLoader(train_set,batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(test_set,batch_size=batch_size)
    
    return Data,train_loader, test_loader, x_scaler, y_scaler

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(90, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, 3),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, 90),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# generate data loader
X_data=health_AME_dis     # X: position Y: current
Y_data=health_Current
batch_size=64
Data, train_loader, test_loader, x_scaler, y_scaler=data_loader_config(X_data, Y_data, batch_size)
xtrain=Data[0]
xtest=Data[1]
ytrain=Data[2]
ytest=Data[3]

# index of data used in phy_loss functions
sinwave_ind=np.array(range(39))
step_ind=np.array(range(39,70))
ramp_ind=np.array(range(70,89))

# sinwave_ind=np.array(range(8,18))
# step_ind=np.array(range(36,50))
# ramp_ind=np.array(range(65,80))

# inverse normalization 这里反归一化后只用于生成阈值 (back to numpy)
xtrain_thr=xtrain*(x_scaler.data_max_-x_scaler.data_min_)+x_scaler.data_min_
ytrain_thr=ytrain*(y_scaler.data_max_-y_scaler.data_min_)+y_scaler.data_min_
xtrain_thr=xtrain_thr.numpy()
ytrain_thr=ytrain_thr.numpy()

# get current and velocity
health_vel=np.diff(xtrain_thr, n=1,axis=1)/0.02
health_cur=(ytrain_thr[:,:-1]+ytrain_thr[:,1:])/2

# generate euc threshold
perc=50
thr_euc_sinwave, mean_cordi_euc_sinwave=euc_thr(health_cur,health_vel,sinwave_ind,perc)
thr_euc_step, mean_cordi_euc_step=euc_thr(health_cur,health_vel,step_ind,perc)
thr_euc_ramp, mean_cordi_euc_ramp=euc_thr(health_cur,health_vel,ramp_ind,perc)

# physical loss functions
def euc_loss(decoded,y,ind,mean_cordi,thr,x_scaler, y_scaler):
    
    # inverse transform
    decoded=decoded*(torch.from_numpy(x_scaler.data_max_-x_scaler.data_min_))+torch.from_numpy(x_scaler.data_min_)
    y=y*(torch.from_numpy(y_scaler.data_max_-y_scaler.data_min_))+torch.from_numpy(y_scaler.data_min_)
    
    thr=torch.from_numpy(thr).type(torch.FloatTensor)
    mean_cordi=torch.from_numpy(mean_cordi).type(torch.FloatTensor)
    
    decoded_vel=torch.diff(decoded, n=1,axis=1)/0.02
    cur=(y[:,:-1]+y[:,1:])/2
    
    cur_part=cur[:,ind]
    vel_part=decoded_vel[:,ind]

    cur_sq=(cur_part-mean_cordi[:,0]).pow(2)    
    decoded_sq=(vel_part-mean_cordi[:,1]).pow(2)
    
    dist=torch.sqrt(cur_sq+decoded_sq)
    
    relu = nn.ReLU()
    loss=torch.mean(relu(dist-thr))

    return loss

## Autoencoder training
u_seq = [1,1e1,1e2,1e3,1e4,1e5]
# u_seq = [1e3]

for u in u_seq:
    autoencoder = AutoEncoder()
    LR = 0.002       # learning rate
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    
    # mse loss func
    loss_func = nn.MSELoss()
    # training
    epochs=100
    m_dist=[]
    weight_seq=[]
    # u=1e3 # weight of physical loss   scale: 1e2-1e5
    # u_seq = [1,1e1,1e2,1e3,1e4,1e5]
    # u_seq = [1e3]

    train_loss_seq=[]
    test_loss_seq=[]
    sinwave_loss_seq=[]
    step_loss_seq=[]
    ramp_loss_seq=[]
    
    # data_loss_seq=[]
    # cv_loss_seq=[]
    
    for epoch in range(epochs):
        # data_loss = 0
        # cv_loss = 0
        for x,cur in train_loader:
            encoded, decoded = autoencoder(x)
            mse_loss=loss_func(decoded,x)
            # sinwave part / step part / ramp part loss
            sinwave_loss=euc_loss(decoded,cur,sinwave_ind,mean_cordi_euc_sinwave,thr_euc_sinwave,x_scaler, y_scaler)
            step_loss=euc_loss(decoded,cur,step_ind,mean_cordi_euc_step,thr_euc_step,x_scaler, y_scaler)
            ramp_loss=euc_loss(decoded,cur,ramp_ind,mean_cordi_euc_ramp,thr_euc_ramp,x_scaler, y_scaler)
            phy_loss=sinwave_loss+step_loss+ramp_loss
            # mixed loss
            loss=mse_loss+u*phy_loss
            optimizer.zero_grad() # gradients to zero
            loss.backward()
            optimizer.step()
            # see thr gradient
            grad_ele=autoencoder.encoder[0].weight.grad[0,10]
            # print(grad_ele)
            weight_seq.append(np.array(grad_ele))
            
            # data_loss += mse_loss
            # cv_loss += phy_loss
            
        
            
        with torch.no_grad():
            # mse loss of training / test dataset
            encoded, decoded = autoencoder(xtrain)
            train_loss=loss_func(decoded,xtrain).data.item()
            encoded_test, decoded_test = autoencoder(xtest)
            test_loss=loss_func(decoded_test,xtest).data.item()
            train_loss_seq.append(train_loss)
            test_loss_seq.append(test_loss)
            # phy loss of training set
            training_sinwave_loss=euc_loss(decoded,ytrain,sinwave_ind,mean_cordi_euc_sinwave,thr_euc_sinwave,x_scaler, y_scaler)
            training_step_loss=euc_loss(decoded,ytrain,step_ind,mean_cordi_euc_step,thr_euc_step,x_scaler, y_scaler)
            training_ramp_loss=euc_loss(decoded,ytrain,ramp_ind,mean_cordi_euc_ramp,thr_euc_ramp,x_scaler, y_scaler)
            # phy loss of test set
            test_sinwave_loss=euc_loss(decoded_test,ytest,sinwave_ind,mean_cordi_euc_sinwave,thr_euc_sinwave,x_scaler, y_scaler)
            test_step_loss=euc_loss(decoded_test,ytest,step_ind,mean_cordi_euc_step,thr_euc_step,x_scaler, y_scaler)
            test_ramp_loss=euc_loss(decoded_test,ytest,ramp_ind,mean_cordi_euc_ramp,thr_euc_ramp,x_scaler, y_scaler)
            # store
            sinwave_loss_seq.append([training_sinwave_loss,test_sinwave_loss])
            step_loss_seq.append([training_step_loss,test_step_loss])
            ramp_loss_seq.append([training_ramp_loss,test_ramp_loss])
            # print
            print('epoch: ',epoch,'test_loss: ',round(test_loss,4),'phy_loss: ',round(test_sinwave_loss.data.item(),4)
                  ,round(test_step_loss.data.item(),4),round(test_ramp_loss.data.item(),4))

    train_phy_loss=np.array(sinwave_loss_seq)[:,0]+np.array(step_loss_seq)[:,0]+np.array(ramp_loss_seq)[:,0]
    np.save("test_phy_loss_{0}".format(u),train_phy_loss)
    np.save("test_loss_seq_{0}".format(u),train_loss_seq)

# np.save("train_0.npy",phy_loss)
train_phy_loss=np.array(sinwave_loss_seq)[:,0]+np.array(step_loss_seq)[:,0]+np.array(ramp_loss_seq)[:,0]
test_phy_loss=np.array(sinwave_loss_seq)[:,1]+np.array(step_loss_seq)[:,1]+np.array(ramp_loss_seq)[:,1]
plt.figure()
plt.plot(test_phy_loss)
plt.show()
plt.figure()
plt.plot(train_loss_seq)
plt.show()

import matplotlib.ticker as mticker
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
fmt = mticker.FuncFormatter(g)
# u_seq = [1,1e1,1e2,1e3,1e4,1e5]
plt.figure()
test_loss_seq = np.load("test_phy_loss_{0}.npy".format(1))
plt.plot(test_loss_seq, label="$\mu$={}".format(fmt(1)))
for u in u_seq[1:]:
    test_loss_seq = np.load("test_phy_loss_{0}.npy".format(u))
    plt.plot(test_loss_seq, label="$\mu$={}".format(fmt(u)))
plt.legend(loc='upper right')
plt.ticklabel_format(useOffset=False)
plt.xlabel('epoch')
plt.ylabel('physical loss')
plt.show()

plt.figure()
test_loss_seq = np.load("test_loss_seq_{0}.npy".format(1))
plt.plot(test_loss_seq, label="$\mu$={}".format(fmt(1)))
for u in u_seq[1:]:
    test_loss_seq = np.load("test_loss_seq_{0}.npy".format(u))
    plt.plot(test_loss_seq, label="$\mu$={}".format(fmt(u)))
plt.legend(loc='upper right')
plt.ticklabel_format(useOffset=False)
plt.xlabel('epoch')
plt.ylabel('physical loss')
plt.show()

plt.figure()
test_loss_seq = np.load("test_loss_seq_{0}.npy".format(1))  + np.load("test_loss_seq_{0}.npy".format(u))
plt.plot(test_loss_seq, label="$\mu$={}".format(fmt(1))) 
for u in u_seq[1:]:
    test_loss_seq = np.load("test_loss_seq_{0}.npy".format(u)) + u*np.load("test_loss_seq_{0}.npy".format(u))
    plt.plot(test_loss_seq, label="$\mu$={}".format(fmt(u)))
plt.legend(loc='upper right')
plt.ticklabel_format(useOffset=False)
plt.xlabel('epoch')
plt.ylabel('physical loss')
plt.show()


#%%
# signal reconstruction error
encoded_xtest, decoded_xtest = autoencoder(xtest)
decoded_inver=x_scaler.inverse_transform(decoded_xtest.detach().numpy()).T
xtest_inver=x_scaler.inverse_transform(xtest).T
plt.figure()
plt.plot(xtest_inver,'b')
plt.plot(decoded_inver,'r')
plt.show()

sum_error=np.sum(np.absolute(decoded_inver-xtest_inver),axis=0)
# sum_error=np.sum(decoded_inver-xtest_inver,axis=0)


deg_=x_scaler.transform(deg3)
deg_=torch.from_numpy(deg_).type(torch.FloatTensor)
deg_encoded, deg_decoded = autoencoder(deg_)
deg_decoded=x_scaler.inverse_transform(deg_decoded.detach().numpy()).T
plt.figure()
plt.plot(deg1.T,'b')
plt.plot(deg_decoded,'r')
plt.show()

sum_error_deg=np.sum(np.absolute(deg_decoded-deg3.T),axis=0)
# sum_error_deg=np.sum(deg_decoded-deg1.T,axis=0)

# error visual
plt.figure()
plt.plot(sum_error,'b')
plt.plot(sum_error_deg,'r')
plt.show()

# AUC calculation
def roc(A,B): # A: recon error; B: deg recon error

    num=100
    thr_min=min(A);
    thr_max=max(A);
    thr_seq=np.linspace(thr_min,thr_max,num);
    
    
    x=np.zeros([num,1])
    y=np.zeros([num,1])
    
    for i,thr in enumerate(thr_seq):
        # TPR
        TPR=sum(B>thr)/len(B)
        # FPR
        FPR=sum(A>thr)/len(A)
        y[i]=TPR
        x[i]=FPR
        
    # AUC
    AUC=-sum(np.diff(x,axis=0)*((y[:-1]+y[1:])/2))
    
    return x,y,AUC

x,y,AUC=roc(sum_error,sum_error_deg)

# healthy_x = np.load("euc_deg1_x.npy")
# healthy_y = np.load("euc_deg1_y.npy")

plt.figure()
plt.plot(x,y,'r')
# plt.plot(healthy_x,healthy_y,'b')
plt.show()

# np.save("deg1_x.npy",x)
# np.save("deg1_y.npy",y)

# threshold-based model
mean_curve=np.vstack((mean_cordi_euc_sinwave,mean_cordi_euc_step,mean_cordi_euc_ramp))
# test set
# inverse normalization
xtest_thr=xtest*(x_scaler.data_max_-x_scaler.data_min_)+x_scaler.data_min_
ytest_thr=ytest*(y_scaler.data_max_-y_scaler.data_min_)+y_scaler.data_min_
xtest_thr=xtest_thr.numpy()
ytest_thr=ytest_thr.numpy()

test_vel=np.diff(xtest_thr, n=1,axis=1)/0.02
test_cur=(ytest_thr[:,:-1]+ytest_thr[:,1:])/2

d_seq=[]
for vel,cur in zip(test_vel,test_cur):
    d=np.sum(np.sqrt(np.square(cur-mean_curve[:,0])+np.square(vel-mean_curve[:,1])))
    d_seq.append(d)

# degradation set
x=deg1
c=deg1_cur

vel=np.diff(x, n=1,axis=1)/0.02
cur=(c[:,:-1]+c[:,1:])/2

deg_seq=[]
for vel,cur in zip(vel,cur):
    d=np.sum(np.sqrt(np.square(cur-mean_curve[:,0])+np.square(vel-mean_curve[:,1])))
    deg_seq.append(d)


plt.figure()
plt.plot(d_seq)
plt.plot(deg_seq)




