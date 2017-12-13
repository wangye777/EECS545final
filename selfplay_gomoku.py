import torch
import torch_init
import torchvision
import torch.nn as nn
from torch.autograd import Variable

from myunet import UNet_gomoku

import numpy as np
import random
import os
import glob

def check_gomoku(cb1, cb2):
    sl = cb1.shape[0]   # side length
    assert(cb1.shape[0] == cb2.shape[0] and cb1.shape[0] == cb1.shape[1]),"different or non-square classboard"

    cb1_s = torch.squeeze(cb1)
    cb2_s = torch.squeeze(cb2)

    result = -1
#   cb1_s_fp = cb1_s
#   cb2_s_fp = cb2_s
#   for ind in range(sl):
#    	cb1_s_fp[:, ind] = cb1_s[:, sl - ind - 1]
#    	cb2_s_fp[:, ind] = cb2_s[:, sl - ind - 1]
#   for ind in range(sl):
#     	row_1 = cb1_s[ind, :]
#     	col_1 = cb1_s[:, ind]
#       sla_1_1 = torch.diag(cb1_s, ind)
#       sla_1_2 = torch.diag(cb1_s, -ind)
#       sla_1_3 = torch.diag(cb1_s_fp, ind)
#       sla_1_4 = torch.diag(cb1_s_fp, -ind)
    for i in range(sl):
        for j in range(sl):
            if j+4 < sl:
                if ((cb1_s[i,j]==1)+(cb1_s[i,j+1]==1)+(cb1_s[i,j+2]==1)+(cb1_s[i,j+3]==1)+(cb1_s[i,j+4]==1) == 5):
                    result = 0
                if ((cb1_s[j,i]==1)+(cb1_s[j+1,i]==1)+(cb1_s[j+2,i]==1)+(cb1_s[j+3,i]==1)+(cb1_s[j+4,i]==1) == 5):
                    result = 0
                if ((cb2_s[i,j]==1)+(cb2_s[i,j+1]==1)+(cb2_s[i,j+2]==1)+(cb2_s[i,j+3]==1)+(cb2_s[i,j+4]==1) == 5):
                    result = 1
                if ((cb2_s[j,i]==1)+(cb2_s[j+1,i]==1)+(cb2_s[j+2,i]==1)+(cb2_s[j+3,i]==1)+(cb2_s[j+4,i]==1) == 5):
                    result = 1
        for k in range(i-5):
            if i >= 5:
                if (cb1_s[i-1-k,k]==1)+(cb1_s[i-2-k,k+1]==1)+(cb1_s[i-3-k,k+2]==1)+(cb1_s[i-4-k,k+3]==1)+(cb1_s[i-5-k,k+4]==1)==5:
                    result=0
                if (cb1_s[sl-i+k,sl-1-k]==1)+(cb1_s[sl-i+k+1,sl-1-k-1]==1)+(cb1_s[sl-i+k+2,sl-1-k-2]==1)+(cb1_s[sl-i+k+3,sl-1-k-3]==1)+(cb1_s[sl-i+k+4,sl-1-k-4]==1)==5:
                    result=0
                if (cb2_s[i-1-k,k]==1)+(cb2_s[i-2-k,k+1]==1)+(cb2_s[i-3-k,k+2]==1)+(cb2_s[i-4-k,k+3]==1)+(cb2_s[i-5-k,k+4]==1)==5:
                    result=1
                if (cb2_s[sl-i+k,sl-1-k]==1)+(cb2_s[sl-i+k+1,sl-1-k-1]==1)+(cb2_s[sl-i+k+2,sl-1-k-2]==1)+(cb2_s[sl-i+k+3,sl-1-k-3]==1)+(cb2_s[sl-i+k+4,sl-1-k-4]==1)==5:
                    result=1
                if (cb1_s[i-1-k,k]==1)+(cb1_s[i-2-k,k+1]==1)+(cb1_s[i-3-k,k+2]==1)+(cb1_s[i-4-k,k+3]==1)+(cb1_s[i-5-k,k+4]==1)==5:
                    result=0
                if (cb1_s[sl-i+k,sl-1-k]==1)+(cb1_s[sl-i+k+1,sl-1-k-1]==1)+(cb1_s[sl-i+k+2,sl-1-k-2]==1)+(cb1_s[sl-i+k+3,sl-1-k-3]==1)+(cb1_s[sl-i+k+4,sl-1-k-4]==1)==5:
                    result=0
                if (cb2_s[i-1-k,k]==1)+(cb2_s[i-2-k,k+1]==1)+(cb2_s[i-3-k,k+2]==1)+(cb2_s[i-4-k,k+3]==1)+(cb2_s[i-5-k,k+4]==1)==5:
                    result=1
                if (cb2_s[sl-i+k,sl-1-k]==1)+(cb2_s[sl-i+k+1,sl-1-k-1]==1)+(cb2_s[sl-i+k+2,sl-1-k-2]==1)+(cb2_s[sl-i+k+3,sl-1-k-3]==1)+(cb2_s[sl-i+k+4,sl-1-k-4]==1)==5:
                    result=1

    return result

numMax = 50
thre_max = 5
iterNum = 150000

model = torch.load('/home/luotg/Desktop/kaggle_carvana/wy/gomoku/unet_multi_lr/model/Unet_gomoku-step-500000.pt')
model.eval()

lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.99)

it = 0
iter = 0

while it < iterNum:
    it+=1
    inputs = torch.zeros(numMax, 2, 15, 15)
    labels = torch.zeros(numMax, 1, 15, 15)
    input1 = torch.zeros(1, 2, 15, 15)
    rand_ind_x = torch.randperm(3)[0]
    rand_ind_y = torch.randperm(3)[0]
    input1[0, 1, 6 + rand_ind_x, 6 + rand_ind_y] = 1

    for i in range(numMax):
        
        inputs[i, :, :, :] = input1
        input_v = Variable(input1.cuda())
        action = model(input_v)

        mask = torch.add(torch.neg(torch.add(input1[0, 0, :, :], 1, input1[0, 1, :, :])), 1) # mask of taken place
        action_valid = torch.mul(action.data[0], mask.cuda())
        action_valid_flat = action_valid.view(1, 1, -1)

        imax_ind = torch.zeros(thre_max,1)
        for j in range(thre_max):
            imax, imax_ind[j] = torch.max(action_valid_flat,2)
            action_valid_flat[0,0,int(imax_ind[j])] = 0
        next_ind = imax_ind[torch.randperm(thre_max)[0]]
        next_state_flat = torch.zeros(1, 1, 15 * 15)
        next_state_flat[0,0,int(next_ind)] = 1
        next_state = next_state_flat.view(1, 1, 15, 15)
        
        labels[i, :, :, :] = next_state
       #print next_state
       #print input1
        input1[0, 0, :, :] = torch.add(input1[0, 0, :, :], next_state)
       #print input1
        result = check_gomoku(input1[0, 0, :, :], input1[0, 1, :, :])
        if result == -1:
            slice1 = torch.zeros(1,1,15,15)
            slice2 = torch.zeros(1,1,15,15)
            slice1[0,0,:,:] = input1[:,1,:,:]
            slice2[0,0,:,:] = input1[:,0,:,:]
            input1 = torch.cat((slice1,slice2),1)           
           # temp = input1[0,0,:,:]
           # input1[0,0,:,:]=input1[0,1,:,:]
           # input1[0,1,:,:]=temp
        else:
            if result == 0.5:
                black_win = 0.5
            else:
                black_win = (i % 2) ^ (result)
            last_num = i
            break
        #print input1
       #input.permute(0, 1, 3, 2)
    if i == numMax-1:
            last_num = numMax
    if last_num < numMax and result >= 0:

        iter+= 1

        for k in range(last_num):
            if black_win == 0.5:
                lr_reward = 0.9
                scale = float(last_num - k)/float(last_num)
            else:
                if (k % 2) == black_win:
                    lr_reward = 1
                    scale = float(last_num + j)/float(2 * last_num)
                else:
                    lr_reward = -0.1
                    scale = float(2 * j - last_num)/float(last_num)
            inputs_t = torch.zeros(1,2,15,15)
            labels_t = torch.zeros(1,1,15,15)  
            inputs_t[0,:,:,:] = inputs[k, :, :, :]
            labels_t[0,:,:,:] = labels[k, :, :, :]

            optimizer = torch.optim.SGD(model.parameters(), lr * lr_reward * scale, momentum = 0.99)
            optimizer.zero_grad()
            inputs_t = Variable(inputs_t.cuda())
            labels_t = Variable(labels_t.cuda())
            outputs = model(inputs_t)
            loss = torch.nn.functional.mse_loss(outputs, labels_t, size_average=True)
            loss.backward()
            optimizer.step()

        print('loss: %.7f (step: %d, process step: %d, lr: %.8f, num of try: %d, black win: %d)' % (loss.data[0], iter, k, lr*lr_reward*scale, last_num, black_win))
        title = 'loss (step: %d process step: %d lr: %f)' % (iter, k, lr*lr_reward*scale)

        if iter % 5000 == 0 and iter != 0:
            checkRoot = '/home/luotg/Desktop/kaggle_carvana/wy/gomoku/unet_sp/snapshot'
            filename = ('%s/Unet_gomoku-step-%d.pt' \
            	    % (checkRoot, iter))
            torch.save(model, filename)
            print('save: (step: %d)' % (iter))
    else:
        print('fail')






    		

  

     	
     













