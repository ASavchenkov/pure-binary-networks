import torch
from torch.optim import Optimizer

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

    def __init__(self, params, lr=0.01):
        self.lr = lr
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
    
    #this flips bits based on "wrongness"
    def _set_by_error(self,p):
        solution = p.grad.data
        error = solution ^ p.data

        max_count = error.size()[0]*8 #figure out what the theshold should be
        threshold = int(max_count*(1-self.lr))

        counts = torch.sum(popc(error),dim = 0) #apply popc to get integer errors, sum over batches
        flip = torch.clamp(counts/threshold,0,1).byte()*255 #threshold and expand
        p.data = p.data ^ flip #XOR'l flip ya. flip ya fo real. *taptaptap* Can ya hear me in the back?
    
    #this sets bits based on "confidence". Doesn't care about actual value of p.data
    #the most theoretically pure one, since the other one technically uses a second
    #bit since it has access to the original value.
    def _set_by_confidence(self,p):
        
        solution = p.grad.data
        max_count = solution.size()[0]*8 
        threshold = int(max_count*self.lr)

        counts = torch.sum(popc(solution),dim=0)

        mask0 = torch.clamp(counts/threshold,0,1).byte()*255 #should be few zeroes
        mask1 = torch.clamp(counts/(max_count-threshold),0,1).byte()*255 #should be few ones
        p.data = (p.data  & mask0 ) | mask1 #sets the selected bits to selected values


    def step(self):
        """ Performs a single optimization step.
            only changes bits with a high enough error
        """
        loss = None

        for group in self.param_groups:
             
            for p in group['params']:
                if p.grad is None:
                    continue
                
                #there are 2 functions that operate differently
                self._set_by_confidence(p)
                

                
                

        return loss
