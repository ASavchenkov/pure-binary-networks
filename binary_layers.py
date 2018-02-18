import torch
from torch.autograd import Function, Variable
import torch.nn as nn
from torch.nn import Parameter

from popc_cuda import popc

#For the sake of intuition, gradients are passed back not as a gradient
#but as "error", meaning the bits getting passed back are whether or not
#the bit is wrong. We assume you can easily figure out what the bit
#should be by taking what the bit is, and XORing it with the "error"


def print_count(x):
    print(torch.sum(popc(x)))

#inputs are N dimensional tensors of bytes
class XNOR(Function):

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a,b)
        
        out  = (a ^ b) ^ 255 #definition of xnor. Use xnor with 255 to do not
        return out

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, grad_output):
        a , b = ctx.saved_variables
        ga = gb = None

        if ctx.needs_input_grad[0]:
            ga = grad_output
        if ctx.needs_input_grad[1]:
            gb = grad_output.clone()
        return ga, gb

b_xnor = XNOR.apply

class AND(Function):

    @staticmethod
    def forward(ctx, a,b):
        h= a & b
        ctx.save_for_backward(a,b,h)
        return h

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, gh):

        a, b, h = ctx.saved_variables
        ga = gb = None

        if ctx.needs_input_grad[0]:
            ga = ((h ^ a)^255) & gh
        if ctx.needs_input_grad[1]:
            gb = ((h ^ b)^255) & gh
        return ga, gb
b_and = AND.apply

class OR(Function):

    @staticmethod
    def forward(ctx, a,b):
        h = a | b
        ctx.save_for_backward(a,b,h)
        return h

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, gh):

        a, b, h = ctx.saved_variables
        
        ga = gb = None

        if ctx.needs_input_grad[0]:
            ga = ((h ^ a)^255) & gh
        if ctx.needs_input_grad[1]:
            gb = ((h ^ b)^255) & gh

        return ga, gb
b_or = OR.apply

#to make sure backprop uses binary OR/AND
#instead of integer addition
#also needs to go through another function
#to split into variables because pytorch apparently can't handle multiple
#gradient inputs into a function once view() gets called -.-
class SPLIT_OR(Function):

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return torch.stack((a,a),len(a.size()))

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_variables
        
        ga = None

        if ctx.needs_input_grad[0]:
            grad_output_1, grad_output_2 =  torch.unbind(grad_output,dim=-1)
            ga = grad_output_1 | grad_output_2

        return ga


def b_split_or(a):
    return torch.unbind(SPLIT_OR.apply(a),dim=-1)

#same thing as with OR. This is to bring balance to the force
class SPLIT_AND(Function):

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        #this gets passed into a function that splits it into 2 variables 
        return torch.stack((a,a),len(a.size()))

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_variables
        
        ga = None

        if ctx.needs_input_grad[0]:
            grad_output_1, grad_output_2 =  torch.unbind(grad_output,dim=-1)
            ga = grad_output_1 & grad_output_2

        return ga

def b_split_and(a):
    return torch.unbind(SPLIT_AND.apply(a),dim=-1)

#gives reduced loss because pytorch whines when you don't return a scalar
class XORLoss(Function):

    @staticmethod
    def forward(ctx, h, y):
        
        counts = popc(h ^ y)
        ctx.save_for_backward(h,y)
        return torch.IntTensor([torch.sum(counts)])

    @staticmethod
    def backward(ctx, grad_output):

        h , y = ctx.saved_variables
        
        #this ends up being computed twice but we don't care since it's so
        #small and pytorch gets weird with saving intermediate values for backward
        gy = gh = None
        if ctx.needs_input_grad[0]:
            gh = y ^ h
        if ctx.needs_input_grad[1]:
            gy = h ^ y

        return gh , gy
b_loss = XORLoss.apply



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


class Residual_Binary(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w1 = nn.Parameter(gen_rand_bits(width))
        self.w2 = nn.Parameter(gen_rand_bits(width))
        
        #start biases off at "do nothing"
        self.b1 = nn.Parameter(gen_rand_bits(width,0))
        self.b2 = nn.Parameter(gen_rand_bits(width,1))

    def forward(self, x):
        x,z = b_split_or(x)
        z = b_xnor(z,self.w1)
        z = b_and(z,self.b1)
        z = swap(z)
        x = b_or(x,z)
        x.register_hook(print_count) 
        x,z = b_split_or(x)
        z.register_hook(print_count)
        z = b_xnor(z,self.w2)
        z.register_hook(print_count)
        z = b_or(z,self.b2)
        z.register_hook(print_count)
        z = swap(z)
        x = b_and(x,z)
        x.register_hook(print_count)


        x = transpose2(x)
        return x

#for testing gradients
if __name__ == '__main__':

    h = Parameter(torch.ByteTensor([1]).cuda(), requires_grad=True)
    y = Variable(torch.ByteTensor([3]).cuda(), requires_grad=False)

    z = b_and(h,y)
    z = b_loss(z,Variable(torch.ByteTensor([3]).cuda()))
    print(z)
    z.backward()
    

    pass
