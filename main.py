import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torch_init

from myunet import UNet_gomoku
from myfunc import imsave

import numpy as np
import random
import os
import glob

Train = True
dataset_dir = '/home/luotg/Desktop/kaggle_carvana/wy/gomoku/dataset'
data_dir = glob.glob(dataset_dir + "/*.npy")
"""parameters"""
Epoch = 3
iterNum = len(data_dir)
seed = 2112
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
model = UNet_gomoku()
getattr(torch_init, 'he_normal')(model)
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.99)
model = model.cuda()

if Train == True:
    """train"""
    item=[]
    for i in range(iterNum):
      #  if i % 2 == 0:
         item.append(i)
    random.shuffle(item)
    dir_root='../trainingset/'
    for it in range(iterNum*Epoch):
        #model.train()
        if (it % 10000 ==0 and it!=0 ):
            lr = lr/5
        epoch_loss = []
        num = item[it%(iterNum)]
        data_gomoku = np.load(data_dir[num])
        black_win = data_gomoku[0, 0, np.shape(data_gomoku)[2]-1]
        inputs = np.zeros((1,2,15,15),dtype=float)
        inputs_ = np.zeros((1,2,15,15),dtype=float)
        N = np.shape(data_gomoku)[2]-1
        scale = 0.0
        for j in range(np.shape(data_gomoku)[2]-2):
            if black_win==0.5:
                lr_reward = 0.9
                scale = float(N-j)/float(N)
            else:
                if j%2 == black_win:
                    lr_reward = 1
                    scale = float(N+j)/float(2*N)
                else:
                    lr_reward = -0.1
                    scale = float(2*j-N)/float(N)

            if j==0:
                inputs[0,1,:,:]=data_gomoku[:,:,0]
                labels = data_gomoku[:,:,1]
            else:
                inputs_[0,0,:,:] = inputs[0,1,:,:]
                inputs_[0,1,:,:] = inputs[0,0,:,:]
                inputs[0,0,:,:] = inputs_[0,0,:,:] + data_gomoku[:,:,j-1]
                inputs[0,1,:,:] = inputs_[0,1,:,:] + data_gomoku[:,:,j]
                labels = data_gomoku[:,:,j+1]
            inputs_t = torch.from_numpy(inputs)
            labels_t = torch.from_numpy(labels)
            inputs_t, labels_t = inputs_t.float(), labels_t.float()
            inputs_t, labels_t = Variable(inputs_t.cuda()), Variable(labels_t.cuda())
       
            optimizer = torch.optim.SGD(model.parameters(),lr*lr_reward*scale, momentum=0.99)
            optimizer.zero_grad()
            outputs = model(inputs_t)
            loss = torch.nn.functional.mse_loss(outputs, labels_t, size_average=True)
            epoch_loss.append(loss.data[0])
            loss.backward()
            optimizer.step()
            average = sum(epoch_loss) / len(epoch_loss)
        print('loss: %.7f (step: %d, process step: %d, lr: %.8f)' % (loss.data[0], it, j, lr*lr_reward*scale))
        epoch_loss.append(average) 
        title = 'loss (step: %d process step: %d lr: %f)' % (it, j, lr*lr_reward*scale)
        if it % 10000 == 0 and it != 0:
            checkRoot = '/home/luotg/Desktop/kaggle_carvana/wy/gomoku/unet/model'
            filename = ('%s/Unet_gomoku-step-%d.pt' \
                    % (checkRoot, it))
            torch.save(model, filename)
            print('save: (step: %d)' % (it))
else:
    for ib, data in enumerate(loader):
        print('testing batch %d' % ib)
        inputs = Variable(data[0]).cuda()
        outputs = model(inputs)
        hhh = color_transform(outputs[0].cpu().max(0)[1].data)
        imsave(os.path.join(outputRoot, data[1][0] + '.png'), hhh)
