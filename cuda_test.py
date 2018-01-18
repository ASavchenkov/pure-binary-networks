import torch
import binutils


x = torch.ByteTensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]).cuda()
y = torch.IntTensor(x.size()).cuda()

binutils.popc(x, y, 15)
print(x,y)
