import torch
import torch.nn as nn

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

def gen_rand_bits(shape):
    return (torch.rand()*2).byte()*255

#does both orientations in a row
class Residual_Binary(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w1 = gen_rand_bits(width)
        self.w2 = gen_rand_bits(width)
        self.b = gen_rand_bits(width)
        self.b2 = gen_rand_bits(width)

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
        self.layers = nn.Sequential(*[Residual_Binary(width) for i in range(depth)])

        
    def forward(self, x):
        x = torch.cat([x]*self.tiling, dim = 1)
        h = self.layers(x) #output is also tiled, and so will the label be
        return h
        


if __name__ == '__main__':

    model = Net(256,32)

    optimizer = B_SGD(model.parameters(),lr = 0.01)

    for i in range(1):
        x,y = generate_data(8)
        h = net(x)
        tiling = h.size()[1]//y.size()[1]
        loss = bl.b_loss(x,torch.cat([y]*tiling))
        loss.backward
    

        
