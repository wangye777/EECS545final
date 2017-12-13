# in dataset, the last channel in [0,0] is 1 if black wins, otherwise 0 or 0.5. b,w,b,w,b,w,b...
import numpy as np
filename = "./renjunet_v10_20171118.rif"
file = open(filename,"r")
list_arr = file.readlines()
l = len(list_arr)
print l
num = 0
for i in range(l):
    list_arr[i] = list_arr[i].split(" ")
    if list_arr[i][0][:6]== '<move>' and len(list_arr[i])>5:
        history1 = np.zeros((15, 15, len(list_arr[i]) + 1),dtype=int)
        history2 = np.zeros((15, 15, len(list_arr[i]) + 1),dtype=int)
        history3 = np.zeros((15, 15, len(list_arr[i]) + 1),dtype=int)
        history4 = np.zeros((15, 15, len(list_arr[i]) + 1),dtype=int)
        x = ord(list_arr[i][0][6:7]) - 96 - 1
        y = int(list_arr[i][0][7:]) - 1
        history1[x, y, 0] = 1
        history2[14-x, y, 0] = 1
        history3[x, 14-y, 0] = 1
        history4[14-x, 14-y, 0] = 1
        for j in range(1,len(list_arr[i])-1):
            x = ord(list_arr[i][j][0:1]) - 96 - 1
            y = int(list_arr[i][j][1:]) - 1
            history1[x, y, j] = 1
            history2[14 - x, y, j] = 1
            history3[x, 14 - y, j] = 1
            history4[14 - x, 14 - y, j] = 1
        x = ord(list_arr[i][len(list_arr[i])-1][0:1]) - 96 - 1
        y = int(list_arr[i][len(list_arr[i])-1][1:-8]) - 1
        history1[x, y, len(list_arr[i])-1] = 1
        history2[14 - x, y, len(list_arr[i])-1] = 1
        history3[x, 14 - y, len(list_arr[i])-1] = 1
        history4[14 - x, 14 - y, len(list_arr[i])-1] = 1
        for k in range(len(list_arr[i-1])):
            if list_arr[i-1][k][0:2]=='br':
                label = float(list_arr[i-1][k][9:-1])
        history1[0, 0, len(list_arr[i])] = label
        history2[0, 0, len(list_arr[i])] = label
        history3[0, 0, len(list_arr[i])] = label
        history4[0, 0, len(list_arr[i])] = label
        num += 1
        name = str(num).rjust(7,'0')
        np.save('./dataset/'+ name + '_1', history1)
        np.save('./dataset/' + name + '_2', history2)
        np.save('./dataset/' + name + '_3', history3)
        np.save('./dataset/' + name + '_4', history4)
        print num , i
print 'finished with ' + str(num) + ' examples'
