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

numMax = 60
thre_max = 5
iterNum = 50000

model = torch.load('/home/luotg/Desktop/kaggle_carvana/wy/gomoku/unet_multi_lr/model/Unet_gomoku-step-500000.pt')
model.eval()

lr = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.99)

it = 0

while it < iterNum:
    inputs = torch.zeros(numMax, 2, 15, 15)
    labels = torch.zeros(numMax, 1, 15, 15)
    input = torch.zeros(1, 2, 15, 15)
    rand_ind_x = torch.randperm(3)[0]
    rand_ind_y = torch.randperm(3)[0]
    input[0, 1, 6 + rand_ind_x, 6 + rand_ind_y] = 1

    for i in range(numMax):

        inputs[i, :, :, :] = input
        input_v = Variable(input.cuda())
        action = model(input_v)

        mask = torch.add(torch.neg(torch.add(input[0, 0, :, :], 1, input[0, 1, :, :])), 1) # mask of taken place
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

        input[0, 0, :, :] = torch.add(input[0, 0, :, :], next_state)
        result = check_gomoku(input[0, 0, :, :], input[0, 1, :, :])

        if result == -1:
            continue
        else:
            if result == 0.5:
                black_win = 0.5
            else:
                black_win = (j % 2) ^ (result)
            last_num = i
            break

        input.permute(0, 1, 3, 2)

    if last_num < numMax and result >= 0:

        it+= 1

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

            inputs_t = inputs[k, :, :, :]
            labels_t = labels[k, :, :, :]

            optimizer = torch.optim.SGD(model.parameters(), lr * lr_reward * scale, momentum = 0.99)
            optimizer.zero_grad()

            outputs = model(inputs_t)
            loss = torch.nn.functional.mse_loss(outputs, labels_t, size_average=True)
            epoch_loss.append(loss.data[0])
            loss.backward()
            optimizer.step()

        print('loss: %.7f (step: %d, process step: %d, lr: %.8f, num of try: %d, black win: %d)' % (loss.data[0], it, k, lr*lr_reward*scale, last_num, black_win))
        title = 'loss (step: %d process step: %d lr: %f)' % (it, k, lr*lr_reward*scale)

        if it % 5000 == 0 and it != 0:
            checkRoot = '/home/luotg/Desktop/kaggle_carvana/wy/gomoku/selfplay/model'
            filename = ('%s/Unet_gomoku-step-%d.pt' \
            	    % (checkRoot, it))
            torch.save(model, filename)
            print('save: (step: %d)' % (it))
    else:
        print('fail')



def check_gomoku(cb1, cb2): 

    sl = cb1.shape[2]   # side length
    assert(cb1.shape[2] == cb2.shape[2] and cb1.shape[2] == cb1.shape[3]),"different or non-square classboard"

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
        for k in range(sl-i-5):
            if sl-i >= 5:
                if (cb1_s[0+k,i+k]==1)+(cb1_s[0+k+1,i+k+1]==1)+(cb1_s[0+k+2,i+k+2]==1)+(cb1_s[0+k+3,i+k+3]==1)+(cb1_s[0+k+4,i+k+4]==1)==5:
                    result = 0
                if (cb1_s[i+k,0+k]==1)+(cb1_s[i+k+1,0+k+1]==1)+(cb1_s[i+k+2,0+k+2]==1)+(cb1_s[i+k+3,0+k+3]==1)+(cb1_s[i+k+4,0+k+4]==1)==5:
                    result = 0
                if (cb2_s[0+k,i+k]==1)+(cb2_s[0+k+1,i+k+1]==1)+(cb2_s[0+k+2,i+k+2]==1)+(cb2_s[0+k+3,i+k+3]==1)+(cb2_s[0+k+4,i+k+4]==1)==5:
                    result = 1
                if (cb2_s[i+k,0+k]==1)+(cb2_s[i+k+1,0+k+1]==1)+(cb2_s[i+k+2,0+k+2]==1)+(cb2_s[i+k+3,0+k+3]==1)+(cb2_s[i+k+4,0+k+4]==1)==5:
                    result = 1
        for k in range(i-5):
            if i >= 5:
                if (cb1_s[0+k,i-k]==1)+(cb1_s[0+k-1,i-k+1]==1)+(cb1_s[0+k-2,i-k+2]==1)+(cb1_s[0+k-3,i-k+3]==1)+(cb1_s[0+k-4,i-k+4]==1)==5:
                    return 0
                if (cb1_s[sl-1-k,i+k]==1)+(cb1_s[sl-1-k-1,i+k+1]==1)+(cb1_s[sl-1-k-2,i+k+2]==1)+(cb1_s[sl-1-k-3,i+k+3]==1)+(cb1_s[sl-1-k-4,i+k+4]==1)==5:
                    return 0
                if (cb2_s[0+k,i-k]==1)+(cb2_s[0+k-1,i-k+1]==1)+(cb2_s[0+k-2,i-k+2]==1)+(cb2_s[0+k-3,i-k+3]==1)+(cb2_s[0+k-4,i-k+4]==1)==5:
                    return 1
                if (cb2_s[sl-1-k,i+k]==1)+(cb2_s[sl-1-k-1,i+k+1]==1)+(cb2_s[sl-1-k-2,i+k+2]==1)+(cb2_s[sl-1-k-3,i+k+3]==1)+(cb2_s[sl-1-k-4,i+k+4]==1)==5:
                    return 1

    return result


    		

  

     	
     













