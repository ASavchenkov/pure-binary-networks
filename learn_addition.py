import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

import binary_layers as bl
from binary_SGD import B_SGD


def bit_batch(x):
    x = np.unpackbits(x,-1)
    x = np.packbits(x,0)
    return x

def un_bit_batch(x):
    x = np.unpackbits(x,0)
    x = np.packbits(x,-1)
    return x

def generate_data(batch_size):
    x =  np.random.randint(0,2,(batch_size,2), dtype = np.uint8)
    y = np.sum(x,axis=1, keepdims=True, dtype=np.uint8)

    x,y = bit_batch(x),bit_batch(y)
    x = x[:,[7,-1]]
    y = y[:,[6,7]]
    return torch.from_numpy(x), torch.from_numpy(y)

#this is a lie. It either generates 0 or 255
#this is because shuffling within a byte sucks
def gen_rand_bits(shape):
    return (torch.rand(shape)*2).byte()*255

#swaps along the channel dimension
def swap(a):
    mid = a.size(-1)//2
    a = torch.cat((a[:,mid:],a[:,:mid]),dim=1)
    return a

def transpose2(a):
    a = a.view(a.size(0),a.size(1)//2,2)
    a = a.transpose(1,2)
    a = a.contiguous().view(a.size(0),-1)
    return a

#does both orientations in a row
class Residual_Binary(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w1 = nn.Parameter(gen_rand_bits(width))
        self.w2 = nn.Parameter(gen_rand_bits(width))
        self.b1 = nn.Parameter(gen_rand_bits(width))
        self.b2 = nn.Parameter(gen_rand_bits(width))

    def forward(self, x):
        z = bl.b_xnor(x,self.w1)
        z = bl.b_and(z,self.b1)
        z = swap(z)
        x = bl.b_or(x,z)
        x = transpose2(x)
        z = bl.b_xnor(x,self.w2)
        z = bl.b_or(z,self.b2)
        z = swap(z)
        x = bl.b_and(x,z)
        x = transpose2(x)
        return x


class Net(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.layers = nn.Sequential(*[Residual_Binary(width) for i in range(depth)])
        # self.layers = Residual_Binary(width)

        
    def forward(self, x):
        h = self.layers(x)
        return h
        


if __name__ == '__main__':
    
    model_width = 2**2
    model = Net(model_width,1)
    model = model.cuda()

    lr = 1
    optimizer = B_SGD(model.parameters(),lr = lr) #lr is again related to batch size

    xx, yy =    generate_data(8)
    xx, yy =    xx.cuda(), yy.cuda()
    x = torch.cat([xx]*(model_width//2),dim = 1)
    y = torch.cat([yy]*(model_width//2), dim = 1)

    x,y = Variable(x), Variable(y)

    last_loss = 0
    for i in range(1):

        #I'm too lazy to write layers that squeeze,
        #so it's easier to tile the inputs and outputs.
        #same result, different code.


        # optimizer.zero_grad()
        
        h = model(x)
        loss = bl.b_loss(h,y)
        print_loss = loss.data.numpy()[0]
         

        loss.backward()
        optimizer.step() 

        if(print_loss==last_loss and lr<500):
            lr+=1
            optimizer.lr = lr
        else:
            print(i,lr,print_loss)
            pass
        last_loss = print_loss
        
