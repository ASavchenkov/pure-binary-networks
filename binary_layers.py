import torch

from torch.autograd import Function, Variable

#For the sake of intuition, gradients are passed back not as a gradient
#but as a solution, meaning the bits getting passed back are what that
#bit "should" be in order to reduce the loss function.

#inputs are N dimensional tensors of integers
class XNOR(Function):

    @staticmethod
    def forward(ctx, a,b):
        ctx.save_for_backward(a,b)
        return (a ^ b) ^ 255 #definition of xnor. Use xnor with 255 to do not

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, grad_output):
        print(grad_output)
        a , b = ctx.saved_variables
        ga = gb = None

        if ctx.needs_input_grad[0]:
            ga = b ^ grad_output
        if ctx.needs_input_grad[1]:
            gb = a ^ grad_output
        return ga, gb

b_xnor = XNOR.apply

class AND(Function):

    @staticmethod
    def forward(ctx, a,b):
        ctx.save_for_backward(a,b)
        return a & b #definition of and

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, grad_output):

        a , b = ctx.saved_variables
        ga = gb = None

        if ctx.needs_input_grad[0]:
            ga = grad_output
        if ctx.needs_input_grad[1]:
            gb = grad_output

        return ga, gb
b_and = AND.apply

class OR(Function):

    @staticmethod
    def forward(ctx, a,b):
        ctx.save_for_backward(a,b)
        return a | b #definition of or

    # This gradient is actually the solution since it's a binary output
    @staticmethod
    def backward(ctx, grad_output):

        a , b = ctx.saved_variables
        
        ga = gb = None

        if ctx.needs_input_grad[0]:
            ga = grad_output
        if ctx.needs_input_grad[1]:
            gb = grad_output

        return ga, gb
b_or = OR.apply


#gives unreduced loss. gh = y because the intent is to get bits correct
#regardless of what the network output.
class XORLoss(Function):

    @staticmethod
    def forward(ctx, h, y):
        ctx.save_for_backward(h,y)

        return h ^ y #definition of or

    @staticmethod
    def backward(ctx, grad_output):

        h , y = ctx.saved_variables
        
        gy = gh = None

        if ctx.needs_input_grad[0]:
            gh = y
        if ctx.needs_input_grad[1]:
            gy = h

        return gh , gy
b_loss = XORLoss.apply

#for testing gradients
if __name__ == '__main__':

    h = Variable(torch.ByteTensor([1]), requires_grad=True)
    y = Variable(torch.ByteTensor([3]), requires_grad=False)

    z = b_loss(h,y)
    z.backward()
    # print(z)

    pass
