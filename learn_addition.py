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
    x =  np.random.randint(0,127,(batch_size,2), dtype = np.uint8)
    y = np.sum(x,axis=1, keepdims=True, dtype=np.uint8)

    x,y = bit_batch(x),bit_batch(y)
    return torch.from_numpy(x), torch.from_numpy(y)

#this is a lie. It either generates 0 or 255
#this is because shuffling within a byte sucks
def gen_rand_bits(shape):
    return (torch.rand(shape)*2).byte()*255

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
        x = bl.b_or(x,z)
        z = bl.b_xnor(x,self.w2)
        z = bl.b_or(z,self.b2)
        x = bl.b_and(x,z)
        return x


class Net(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        # self.layers = nn.Sequential(*[Residual_Binary(width) for i in range(depth)])
        self.layers = Residual_Binary(width)

        
    def forward(self, x):
        h = self.layers(x)
        return h
        


if __name__ == '__main__':
    
    model_width = 16
    model = Net(model_width,1)
    model = model.cuda()

    optimizer = B_SGD(model.parameters(),lr = 0.3) #lr is again related to batch size

    xx, yy =    generate_data(16)
    xx, yy =    xx.cuda(), yy.cuda()
    for i in range(10):

        #I'm too lazy to write layers that squeeze,
        #so it's easier to tile the inputs and outputs.
        #same result, different code.
        x = torch.cat([xx]*(model_width//16))
        y = torch.cat([yy]*(model_width//8), dim = 1)

        x,y = Variable(x), Variable(y)

        # optimizer.zero_grad()
        
        h = model(x)
        loss = bl.b_loss(h,y)
        print(loss)

        loss.backward()
        optimizer.step() 

        
