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
import os
# from thr_definition import euc_thr, mal_thr, hau_thr
# from thr_definition_new import euc_thr, mal_thr, hau_thr
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

#%% load data

# load healthy data
health_AME_dis = scio.loadmat('./healthy.mat')['AME_dis_mat']
health_Current = scio.loadmat('./healthy.mat')['current_mat']
health_para = scio.loadmat('./healthy.mat')['para_mat']
health_Simu_dis = scio.loadmat('./healthy.mat')['Simu_dis_mat']
health_Spool_dis = scio.loadmat('./healthy.mat')['Spool_dis_mat']

# plt.figure()
# plt.plot(health_AME_dis.T,'b')
# plt.show()

# plt.figure()
# plt.plot(health_Current.T,'b')
# plt.show()

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

plt.figure()
plt.plot(health_AME_dis.T,'b')
plt.plot(deg_AME_dis[index1,:].T,'b')
plt.plot(deg_AME_dis[index2,:].T,'k')
plt.plot(deg_AME_dis[index3,:].T,'g')
# plt.plot(deg_AME_dis[index4,:].T,'y')
# plt.plot(deg_AME_dis[index5,:].T,'r')
plt.show()

# resample and combine dis_current data
index=np.linspace(0,1800,91,dtype=(int))
health_AME_dis=health_AME_dis[:,index[1:]]
health_Current=health_Current[:,index[1:]]
deg_AME_dis=deg_AME_dis[:,index[1:]]
deg_Current=deg_Current[:,index[1:]]

plt.figure()
plt.plot(health_AME_dis.T,'b')
plt.show()

plt.figure()
plt.plot(health_Current.T,'b')
plt.show()


health_cur_dis=np.hstack((health_AME_dis,health_Current))
deg_cur_dis=np.hstack((deg_AME_dis,deg_Current))

deg1=deg_AME_dis[index1,:];deg1_cur=deg_Current[index1,:]
deg2=deg_AME_dis[index2,:];deg2_cur=deg_Current[index2,:]
deg3=deg_AME_dis[index3,:];deg3_cur=deg_Current[index3,:]
deg4=deg_AME_dis[index4,:];deg4_cur=deg_Current[index4,:]
deg5=deg_AME_dis[index5,:];deg5_cur=deg_Current[index5,:]

#%% 

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
            nn.Linear(90*2, 60*2),
            nn.Tanh(),
            nn.Linear(60*2, 30*2),
            nn.Tanh(),
            nn.Linear(30*2, 10*2),
            nn.Tanh(),
            nn.Linear(10*2, 3*2),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(3*2, 10*2),
            nn.Tanh(),
            nn.Linear(10*2, 30*2),
            nn.Tanh(),
            nn.Linear(30*2, 60*2),
            nn.Tanh(),
            nn.Linear(60*2, 90*2),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def euc_thr(health_cur,health_vel,perc):  # calculate the distance btw validation set and average of training set
    
    cur_part=health_cur
    vel_part=health_vel
    # split train set and validation set
    cur_train, cur_val, vel_train, vel_val = train_test_split(cur_part, vel_part, test_size=0.15, random_state=42)
    # get average curve of training data curves
    cordi_training=np.stack((cur_part,vel_part),2)
    mean_cordi_training=np.mean(cordi_training,axis=0)
    # calculate the euc distance btw validation set and mean curve
    cur_sq=np.square(cur_val-mean_cordi_training[:,0])
    vel_sq=np.square(vel_val-mean_cordi_training[:,1])
    dist=np.sqrt(cur_sq+vel_sq)
    
    # percentile based threshold
    thr=np.percentile(dist, perc,axis=0)
    
    return thr, mean_cordi_training

# physical loss functions
def euc_loss(decoded,y,mean_cordi,thr,x_scaler, y_scaler):
    
    # inverse transform
    decoded=decoded*(torch.from_numpy(x_scaler.data_max_-x_scaler.data_min_))+torch.from_numpy(x_scaler.data_min_)
    y=y*(torch.from_numpy(y_scaler.data_max_-y_scaler.data_min_))+torch.from_numpy(y_scaler.data_min_)
    
    thr=torch.from_numpy(thr).type(torch.FloatTensor)
    mean_cordi=torch.from_numpy(mean_cordi).type(torch.FloatTensor)
    
    decoded_vel=torch.diff(decoded, n=1,axis=1)/0.02
    cur=(y[:,:-1]+y[:,1:])/2
    
    cur_part=cur
    vel_part=decoded_vel

    cur_sq=(cur_part-mean_cordi[:,0]).pow(2)    
    decoded_sq=(vel_part-mean_cordi[:,1]).pow(2)
    
    dist=torch.sqrt(cur_sq+decoded_sq)
    
    relu = nn.ReLU()
    loss=torch.mean(relu(dist-thr))

    return loss

#%% data preparation
# generate data loader
X_data=health_AME_dis     # X: position Y: current
Y_data=health_Current
batch_size=64
Data, train_loader, test_loader, x_scaler, y_scaler=data_loader_config(X_data, Y_data, batch_size)
# xtrain=torch.vstack((Data[0],Data[1]))
xtrain=Data[0]
xtest=Data[1]
# ytrain=torch.vstack((Data[2],Data[3]))
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

#%% Model training
# perc_seq = [0,10,20,30,40,50,60,70,80,90]
# u_seq = [0, 10, 100, 1000, 1e4, 1e5, 1e6]

perc_seq = [0]
u_seq = [0]

setup_seed(22)
for perc in perc_seq: 
    # generate euc threshold
    # perc=30
    thr_euc, mean_cordi_euc=euc_thr(health_cur,health_vel,perc)
    ## Autoencoder training
    for u in u_seq:
        print(perc, u)
        autoencoder = AutoEncoder()
        LR = 0.001       # learning rate
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
        phy_loss_seq=[]
        
        # data_loss_seq=[]
        # cv_loss_seq=[]
        
        for epoch in range(epochs):
            # data_loss = 0
            # cv_loss = 0
            for x,cur in train_loader:
                comb_input = torch.hstack((x,cur))
                encoded, decoded = autoencoder(comb_input)
                mse_loss=loss_func(decoded,comb_input)
                # sinwave part / step part / ramp part loss
                # phy_loss=euc_loss(decoded,cur,mean_cordi_euc,thr_euc,x_scaler, y_scaler)
    
                # mixed loss
                loss=mse_loss #+u*phy_loss
                optimizer.zero_grad() # gradients to zero
                loss.backward()
                optimizer.step()
                # see thr gradient
                # grad_ele=autoencoder.encoder[0].weight.grad[0,10]
                # print(grad_ele)
                # weight_seq.append(np.array(grad_ele))
                
                # data_loss += mse_loss
                # cv_loss += phy_loss
                
            with torch.no_grad():
                # mse loss of training / test dataset
                
                comb_train = torch.hstack((xtrain,ytrain))
                encoded, decoded = autoencoder(comb_train)
                train_loss=loss_func(decoded,comb_train).data.item()
                
                comb_test = torch.hstack((xtest,ytest))
                encoded_test, decoded_test = autoencoder(comb_test)
                test_loss=loss_func(decoded_test,comb_test).data.item()
                train_loss_seq.append(train_loss)
                test_loss_seq.append(test_loss)
                # phy loss of training set
                # training_phy_loss=euc_loss(decoded,ytrain,mean_cordi_euc,thr_euc,x_scaler, y_scaler)
                # phy loss of test set
                # test_phy_loss=euc_loss(decoded_test,ytest,mean_cordi_euc,thr_euc,x_scaler, y_scaler)
                # store
                # phy_loss_seq.append([training_phy_loss.data.item(),test_phy_loss.data.item()])
                # print
                print('epoch: ',epoch,'test_loss: ',round(test_loss,4))
        
        # save model
        # creat dir path
        path = 'DAE_dis_models'
        os.makedirs(path, exist_ok = True) 
        # 这里会存储迄今最优模型的参数
        torch.save(autoencoder.state_dict(), 'DAE_dis_models/model_parms_{0}_{1}.pt'.format(perc,u))    
        # 这里会存储迄今最优的模型
        torch.save(autoencoder, 'DAE_dis_models/model_{0}_{1}.pkl'.format(perc,u))

        # save losses
        # creat dir path
        path = 'DAE_dis_losses'
        os.makedirs(path, exist_ok = True)   
        # np.save("DAE_dis_losses/phy_loss_seq_{0}_{1}".format(perc,u),phy_loss_seq)
        np.save("DAE_dis_losses/train_loss_seq_{0}_{1}".format(perc,u),train_loss_seq)
        np.save("DAE_dis_losses/test_loss_seq_{0}_{1}".format(perc,u),test_loss_seq)

plt.figure()
plt.plot(train_loss_seq)
plt.show()

plt.figure()
plt.plot(test_loss_seq)
plt.show()

#%% recons train
comb_train = torch.hstack((xtrain,ytrain))

_,decoded_train = autoencoder(comb_train[:, None, :])

recons_x = decoded_train[:, 0, :90]
recons_cur = decoded_train[:, 0, 90:]

decoded_inver=x_scaler.inverse_transform(torch.squeeze(recons_x).detach().numpy()).T

xtrain_inver=x_scaler.inverse_transform(xtrain)

plt.figure()
plt.plot(xtrain_inver.T, 'b')
plt.plot(decoded_inver, 'r')
plt.show()

decoded_inver_cur=y_scaler.inverse_transform(torch.squeeze(recons_cur).detach().numpy()).T

ytrain_inver=y_scaler.inverse_transform(ytrain)

plt.figure()
plt.plot(ytrain_inver.T, 'b')
plt.plot(decoded_inver_cur, 'r')
plt.show()



















