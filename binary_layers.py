import torch
from torch.autograd import Function, Variable
from torch.nn import Parameter

from popc_cuda import popc

#For the sake of intuition, gradients are passed back not as a gradient
#but as "error", meaning the bits getting passed back are whether or not
#the bit is wrong. We assume you can easily figure out what the bit
#should be by taking what the bit is, and XORing it with the "error"


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
            ga = (h ^ a) & gh
        if ctx.needs_input_grad[1]:
            gb = (h ^ b) & gh

        return ga, gb
b_or = OR.apply

#to make sure backprop uses binary OR/AND
#instead of integer addition
class SPLIT_OR(Function):

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a, a.clone()

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, grad_output_1, grad_output_2):

        a, = ctx.saved_variables
        
        ga = None

        if ctx.needs_input_grad[0]:
            ga = grad_output_1 | grad_output_2

        return ga
b_split_or = SPLIT_OR.apply

#same thing as with OR. This is to provide balance to the force
class SPLIT_AND(Function):

    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return a, a.clone()

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, grad_output_1, grad_output_2):

        a, = ctx.saved_variables
        
        ga = None

        if ctx.needs_input_grad[0]:
            ga = grad_output_1 & grad_output_2

        return ga
b_split_and = SPLIT_AND.apply

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

#for testing gradients
if __name__ == '__main__':

    h = Parameter(torch.ByteTensor([1]).cuda(), requires_grad=True)
    y = Variable(torch.ByteTensor([3]).cuda(), requires_grad=False)

    z = b_and(h,y)
    z = b_loss(z,Variable(torch.ByteTensor([3]).cuda()))
    print(z)
    z.backward()
    

    pass
