import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import random
import os
from myunet import UNet128
model1=torch.load('/home/luotg/flowlearning/flowlearning_Unet/model/FCN-step-3501.pt')
#model2=torch.load('/home/luotg/flowlearning/flowlearning_fcn/model/FCN-step-51.pt')
num=650
dir_root='/home/luotg/flowlearning/trainingset/'
dir_in ='input_dist_%06d'%(num)
dir_tar = 'output4_%06d'%(num)
input_temp = np.loadtxt(dir_root+dir_in+'.txt',dtype='float')
output_temp = np.loadtxt(dir_root+dir_tar+'.txt',dtype='float')
inputs = np.reshape(input_temp,[1,1,128,256])
outputs = np.reshape(output_temp,[1,1,128,256])
inputs = torch.from_numpy(inputs)
inputs = inputs.float()
inputs = Variable(inputs.cuda())
outputs = torch.from_numpy(outputs)
outputs = outputs.float()
labels_ = Variable(outputs.cuda())
label_1 = torch.clamp(labels_, min = 0.0)
label_2 = torch.clamp(torch.mul(labels_, -1), min = 0.0)
labels = torch.cat((label_1, label_2), 1)
model1.eval()
#model2.eval()
output1 = model1(inputs)
#output2 = model2(inputs)
out = output1.data[0]
mse = output1.data[0]-labels.data[0]
mse = torch.abs(mse)
perc = torch.clamp(torch.div(mse,out),max = 1)
mse_sum = torch.sum(mse)
base = torch.abs(labels.data[0])
base_sum = torch.sum(base)
ans = mse_sum/base_sum
print out
print labels.data[0]
print mse
print perc
print ans
#final = mse./base
#err = torch.sum(final)/(128*256)
#print err
#print torch.sum(torch.abs(output1.data[0]))
#torch.save('t.csv',labels_,'ascii')
np.savetxt('1_t.txt',outputs)
np.savetxt('1_o.txt',output1.data[0])
print output1.size(2)
