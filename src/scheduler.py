
import numpy as np


class LRRangeTest:

    def __init__(self, optimizer, total_steps, start=-12, end=-2):
        self.lrs = 10**np.linspace(start, end, total_steps+1)
        self.optimizer = optimizer
        self.current_step = 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lrs[self.current_step]

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lrs[self.current_step]
        self.current_step += 1

    def get_last_lr(self):
        return [self.lrs[self.current_step]]


