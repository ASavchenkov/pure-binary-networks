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

    def __init__(self, params, lr=1):
        self.lr = lr
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
    
    def _get_max_flip(self,p):
        solution = p.grad.data
        error = solution ^ p.data

        counts = torch.sum(popc(error),dim = 0) #apply popc to get integer errors, sum over N
        # counts = counts.float() * torch.Tensor(counts.size()).normal_(0,1).cuda()
        return torch.max(counts)


    #this flips bits based on "wrongness"
    def _set_by_error(self, p, threshold = None):
        solution = p.grad.data
        error = solution ^ p.data

        counts = torch.sum(popc(error),dim = 0) #apply popc to get integer errors, sum over N
        if(not threshold):
            threshold = torch.max(counts)
        flip = torch.clamp(counts/threshold,0,1).byte()*255 #threshold and expand
        p.data = p.data ^ flip
        #XOR'l flip ya. flip ya fo real. *tap tap tap* Can ya hear me in the back?
    
    #this sets bits based on "confidence". Doesn't care about actual value of p.data
    #the most theoretically pure one, since the other one technically uses a second
    #bit since it has access to the original value.
    def _set_by_confidence(self,p):
        
        solution = p.grad.data
        max_count = solution.size()[0]*8 
        # threshold = int(max_count*self.lr)
        threshold = self.lr

        counts = torch.sum(popc(solution),dim=0)

        mask0 = torch.clamp(counts/threshold,0,1).byte()*255 #should be few zeroes
        mask1 = torch.clamp(counts/(max_count-threshold),0,1).byte()*255 #should be few ones
        p.data = (p.data  & mask0 ) | mask1 #sets the selected bits to selected values


    def step(self):
        """ Performs a single optimization step.
            only changes bits with a high enough error
        """

        for group in self.param_groups:
            max_count = 0
            max_idx = 0
            for i,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                this_max = self._get_max_flip(p)
                if(this_max>max_count):
                    max_count = this_max 
                    max_idx = i
            for p in group['params']:
                if p.grad is None:
                    continue

                self._set_by_error(p,max_count)

        
        return max_count,max_idx
