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
    # x = np.array([[0,0,0,0,1,1,1,1],
                  # [0,0,1,1,0,0,1,1]], dtype = np.uint8).T
    y = np.sum(x,axis=1, keepdims=True, dtype=np.uint8)

    x,y = bit_batch(x),bit_batch(y)
    x = x[:,[7,-1]]
    y = y[:,[7,7]]

    return torch.from_numpy(x), torch.from_numpy(y)

#this is a lie. It either generates 0 or 255
#this is because shuffling within a byte sucks
def gen_rand_bits(shape, prob1=0.5):
    return (torch.rand(shape)+prob1).byte()*255

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
class Regular_Binary(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w11 = nn.Parameter(gen_rand_bits(width))
        self.w12 = nn.Parameter(gen_rand_bits(width))
        self.w21 = nn.Parameter(gen_rand_bits(width))
        self.w22 = nn.Parameter(gen_rand_bits(width))

        self.b11 = nn.Parameter(gen_rand_bits(width,1))
        self.b12 = nn.Parameter(gen_rand_bits(width,0.5))
        self.b21 = nn.Parameter(gen_rand_bits(width,0))
        self.b22 = nn.Parameter(gen_rand_bits(width,0.5))

    def forward(self, x):
        print(x)
        z1,z2 = bl.b_split_or(x)
        z1 = bl.b_xnor(z1,self.w11)
        z2 = bl.b_xnor(z2,self.w12)
        z2 = swap(z2)
        x = bl.b_or(z1,z2)
        # x = bl.b_and(z,self.b11)
 
        z1,z2 = bl.b_split_and(x)
        z1 = bl.b_xnor(z1,self.w21)
        z2 = bl.b_xnor(z2,self.w22)
        z2 = swap(z2)
        x = bl.b_and(z1,z2)
        # x = bl.b_or(z,self.b21)

        x = transpose2(x)
        return x

class Residual_Binary(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w1 = nn.Parameter(gen_rand_bits(width))
        self.w2 = nn.Parameter(gen_rand_bits(width))
        
        #start biases off at "do nothing"
        self.b1 = nn.Parameter(gen_rand_bits(width,0.1))
        self.b2 = nn.Parameter(gen_rand_bits(width,0.9))

    def forward(self, x):
        # print(x)
        x,z = bl.b_split_or(x)
        z = bl.b_xnor(z,self.w1)
        z = bl.b_and(z,self.b1)
        z = swap(z)
        x = bl.b_or(x,z)
 
        x,z = bl.b_split_and(x)
        z = bl.b_xnor(z,self.w2)
        z = bl.b_or(z,self.b2)
        z = swap(z)
        x = bl.b_and(x,z)

        x = transpose2(x)
        return x

class Net(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.layers = nn.Sequential(*[Regular_Binary(width) for i in range(depth)])
        self.b1 = nn.Parameter(gen_rand_bits(width,0))
        self.b2 = nn.Parameter(gen_rand_bits(width,1))

        
    def forward(self, x):
        x = self.layers(x)
        # x = bl.b_or(x,self.b1)
        # x = bl.b_and(x,self.b2)
        print('-----------------------------------------------------------------')
        return x


#specifically built to learn XOR
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        #first layer weights
        self.w1 = nn.Parameter(gen_rand_bits(2,0.5))
        self.w2 = nn.Parameter(gen_rand_bits(2,0.5))

        # self.w1 = nn.Parameter(torch.ByteTensor([255,0]))
        # self.w2 = nn.Parameter(torch.ByteTensor([0,255]))
        
        #second layer weights
        self.w3 = nn.Parameter(gen_rand_bits(2,0.5))
        self.w4 = nn.Parameter(gen_rand_bits(2,0.5))

        # self.w3 = nn.Parameter(torch.ByteTensor([255,255]))
        # self.w4 = nn.Parameter(torch.ByteTensor([255,255]))

        
    def forward(self, x):
        
        x1,x2 = bl.b_split_and(x)
        
        z1 = bl.b_xnor(x1,self.w1)
        z2 = bl.b_xnor(x2,self.w2)

        z1 = swap(z1)
        
        x = bl.b_or(z1,z2)
        x1,x2 = bl.b_split_or(x)
        z1 = bl.b_xnor(x1,self.w3)
        z2 = bl.b_xnor(x2,self.w4)

        z1 = swap(z1)

        x = bl.b_and(z1,z2)
        
        
        # x = bl.b_or(x,self.b1)
        # x = bl.b_and(x,self.b2)
        return x

if __name__ == '__main__':
    
    model_width = 2**1
    # model = Net(model_width,8)
    model = XORNet()
    model = model.cuda()

    lr = 1
    optimizer = B_SGD(model.parameters(),lr = lr) #lr is again related to batch size

    last_loss = 0
    for i in range(1000):

        xx, yy =    generate_data(2**3)
        xx, yy =    xx.cuda(), yy.cuda()
        x = torch.cat([xx]*(model_width//2),dim = 1)
        y = torch.cat([yy]*(model_width//2), dim = 1)

        x,y = Variable(x), Variable(y)


        #I'm too lazy to write layers that squeeze,
        #so it's easier to tile the inputs and outputs.
        #same result, different code.


        # optimizer.zero_grad()
        
        h = model(x)
        loss = bl.b_loss(h,y)
        print_loss = loss.data.numpy()[0]
        
        loss.backward()
        max_count, max_idx = optimizer.step() 

        print_h = np.unpackbits(h.data.cpu().numpy(),axis=0)
        # print(y,print_h)
        print(i,max_count,max_idx,print_loss)
        last_loss = print_loss
        
