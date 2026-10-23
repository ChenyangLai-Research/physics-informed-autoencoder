# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:13:25 2021

@author: LAI Chenyang
"""
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset # creat custom dataset
from torch.utils.data import DataLoader
from scipy import stats


# euclidean distance
def euc_thr(health_cur,health_vel,ind,perc):  # calculate the distance btw validation set and average of training set
    
    cur_part=health_cur[:,ind]
    vel_part=health_vel[:,ind]
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


# Mahalanobis distance
def mal_thr(health_cur,health_vel,ind,perc):
    cur_part=health_cur[:,ind]
    vel_part=health_vel[:,ind]
    # split train set and validation set
    cur_train, cur_val, vel_train, vel_val = train_test_split(cur_part, vel_part, test_size=0.2, random_state=42)
    # stack the current and velocity
    cur_vel_train=np.stack((cur_train,vel_train),axis=2).transpose(1,0,2) # (ind,sample,(x,y))
    cur_vel_val=np.stack((cur_val,vel_val),axis=2).transpose(1,0,2) # (ind,sample,(x,y))
    # calculate mahal dist btw validation set and train set
    malah_val=[]
    mea_mat=[]
    icv_mat=[]
    for sample_train, sample_val in zip(cur_vel_train,cur_vel_val):
        a=sample_train; b=sample_val
        mea=np.mean(a,axis=0)
        icv=np.linalg.inv(np.cov(a.T))
        mea_mat.append(mea)
        icv_mat.append(icv)
        malah=[]
        for sam in b:
            m=np.sqrt(np.dot(((sam-mea).dot(icv)),(sam-mea).T))
            malah.append(m)
        malah_val.append(malah)
    malah_val=np.array(malah_val).T
    # percentile based threshold
    thr=np.percentile(malah_val, perc, axis=0)

    return thr, mea_mat, icv_mat

# shape distance (average hausdorff distance)
def cdist(x, y): # euclidian distance
    '''
    Input: x is a Nxd Tensor
           y is a Mxd Tensor
    Output: dist is a NxM matrix where dist[i,j] is the norm
           between x[i,:] and y[j,:]
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    '''
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    
    # differences = x.unsqueeze(1) - y.unsqueeze(0)
    # distances = np.sum(differences**2, -1).sqrt()
    return distances

def averageHDD(set1,set2):
    
    d2_matrix = cdist(set1, set2)

    # Modified Chamfer Loss
    term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
    term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

    res = term_1 + term_2
    
    return res

def hau_thr(health_cur,health_vel,ind,perc):
    cur_part=health_cur[:,ind]
    vel_part=health_vel[:,ind]
    # split train set and validation set
    cur_train, cur_val, vel_train, vel_val = train_test_split(cur_part, vel_part, test_size=0.2, random_state=42)
    # get the mean of train set (cur and vel)
    cur_train_mean=np.mean(cur_train,axis=0)
    vel_train_mean=np.mean(vel_train,axis=0)
    traj_mean=np.array([cur_train_mean,vel_train_mean]).T # mean cur-vel (x,y)
    # train/val set coordinates
    traj_train=np.stack((cur_train,vel_train),axis=2)
    traj_val=np.stack((cur_val,vel_val),axis=2)
    
    # calculate the average hausdorff distance
    hau_dist=[]
    for sam_val in traj_val:
        # h_d=averageHDD(sam_val,traj_mean)
        h_d=averageHDD(torch.tensor(sam_val),torch.tensor(traj_mean))
        hau_dist.append(np.array(h_d))

    hau_dist=np.array(hau_dist)
        
    # percentile based threshold
    thr=np.percentile(hau_dist, perc)

    return thr, hau_dist, traj_mean













