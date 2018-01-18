from .optimizer import Optimizer, required

import numpy as np


class BinarySGD(Optimizer):
    r"""Implements Binary SGD

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()


    lr defines the threshold for when bits flip. Lower LR means lower threshold
    means more bits flip.
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

                error = np.unpack(error.numpy())
                total_error = np.sum(error,(0,-1)) #sum over bits and batch
                
                

                p.data.add_(-group['lr'], d_p)
                

        return loss