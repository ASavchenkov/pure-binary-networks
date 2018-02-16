import torch
from torch.autograd import Function
from torch.nn  import Parameter

import binary_layers as bl

#it turns out that pytorch has default backpropagation
#behavior for variables being used multiple times.
#It does addition. This is totally reasonable in literally
#every case but ours (and how would it know anyways?)
#so all variable reuse has to go through new "split"
#functions that backprop with bitwise and or or

def gen_rand_bits(shape):
    return (torch.rand(shape)*2).byte()*255

x = Parameter(torch.ByteTensor([8 + 32]).cuda())
y = Parameter(torch.ByteTensor([4 + 8]).cuda())
x = x.view(x.size())
y = y.view(x.size())
x1,x2 = bl.b_split_and(x)
y1,y2 = bl.b_split_and(y)
a = bl.b_and(x1,y1)
b = bl.b_or(x2,y2)
h = bl.b_loss(a,b)

a.register_hook(print)
b.register_hook(print)
x.register_hook(print)
y.register_hook(print)
print(h)
h.backward()

