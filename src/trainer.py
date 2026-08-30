
from torch.nn.utils import clip_grad_norm_
import torch

from src.utils import RollingCounter


class Trainer:

    def __init__(self, params, opt, obj, sch=None, grad_acc=1):
        self.grad_acc = grad_acc
        self.params = params
        self.opt = opt
        self.obj = obj
        self.metrics = { 'loss': RollingCounter() }

        self.sch = sch if sch is not None else None

    def step(self, preds, labels, step):

        if step % self.grad_acc == 0:
            self.opt.zero_grad()

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = self.calc_loss(preds, labels)
            loss /= self.grad_acc

        clip_grad_norm_(self.params, 5)
        loss.backward()

        if (step + 1) % self.grad_acc == 0:
            self.opt.step()

            if self.sch is not None:
                self.sch.step()

        return self.get_metrics()

    def calc_loss(self, preds, labels):
        loss = self.obj(preds, labels)
        self.metrics['loss'].add(loss.item())
        return loss

    def get_metrics(self):
        metrics = {
            f'{key}': self.metrics[key].rolling_average() \
            for key in self.metrics
        }

        if self.sch is not None:
            metrics['lr'] = self.sch.get_last_lr()[-1]

        return metrics
