import torch
from torch.optimizer import Optimizer, required

# import popc_cuda 
from popc_cuda import popc

class B_SGD(Optimizer):
    r"""Implements Binary SGD

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()


    lr defines percentage correct below which a bit is flipped
    don't ever make it 0.5 or higher. That makes the network "unlearn"
    """

    def __init__(self, params, lr=required):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self):
        """ Performs a single optimization step.
            only changes bits with a high enough error
        """
        loss = None

        for group in self.param_groups:
             
            for p in group['params']:
                if p.grad is None:
                    continue


                solution = p.grad.data #this is still bitwise
                
                error = solution ^ p.data #should be same shape up until N
                
                max_count = error.size()[0]*8 #figure out what the theshold should be
                threshold = int(max_count*(1-self.lr))

                counts = torch.sum(popc(error),dim = 0) #apply popc to get integer errors, sum over batches
                flip = torch.clamp(counts/threshold,0,1).byte()*255 #threshold and expand
                p.data = p.data ^ flip #XOR flips things. That's how it works.
                
                #THRESHOLD MUST ALWAYS BE ABOVE HALF
                

        return loss
