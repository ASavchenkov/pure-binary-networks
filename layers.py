import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter


class Factorized_Linear(nn.Module):
   
    #in_shape is an array of integers
    #out_shape is the same
    #reduction is an integer stating how many dimensions to reduce along
    #reduction happens along left side to allow broadcasting during multiplication

    def __init__(self, in_shape, out_shape, reduction, bias=False):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.reduction = reduction


        self.weight = Parameter(torch.Tensor(*(out_shape + in_shape)))
        if bias:
            self.bias = Parameter(torch.Tensor(*(out_shape + in_shape[:-self.reduction])))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(float(torch.prod(torch.Tensor(self.in_shape[-self.reduction:]))))
        stdv = 1. / math.sqrt(float(torch.prod(torch.Tensor(self.in_shape))))
        print(stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        for i in range(len(self.out_shape)):
            input = torch.unsqueeze(input,1)
        z = input * self.weight #this is broadcasted. Weight is higher dim

        
        for i in range(self.reduction):
            z = torch.sum(z,-1) #sum self.reduction times along the right axis

        if self.bias:
            z = z + self.bias

        return z 
 

        # return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_shape) + ' )'




class Factorized_Conv2d(nn.Module):
    #this variant actually does transposition and makes use of
    #included convolution functionality instead of rewriting it all

    def __init__(self, in_channels, out_channels, kernel_size , group_size, padding = 0, bias = False):

        super().__init__()
        self.group_size = group_size

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding, groups = in_channels/group_size, bias = bias)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self,input):

        output = self.conv(input)

        #transpose along channels
        N, C, H, W = output.size()
        output = output.view(N, C/self.group_size, self.group_size, H, W)
        output.permute(0, 2, 1, 3, 4)
        output = output.view(N, C, H, W)

        return output


class Factorized_BN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.bn = nn.BatchNorm1d(self.input_size)

    def reset_parameters(self):
        self.bn.reset_parameters()

    def forward(self,input):
        orig_shape = input.size()
        input = input.view(-1,self.input_size)
        output = self.bn(input).view(orig_shape)
        return output


