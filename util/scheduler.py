import math
from typing import List

from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineLR(_LRScheduler):

    base_lrs: List[float]
    last_epoch: int

    def __init__(
        self,
        optimizer,
        warmup_iters: int,
        max_iters: int,
        warmup_ratio: float = 5e-4,
        eta: float = 0.01,
        last_epoch: int = -1,
    ):

        self.warmup_iters = warmup_iters
        self.max_iters = max_iters

        assert self.warmup_iters < self.max_iters

        self.warmup_ratio = warmup_ratio
        self.eta = eta
        self._consine_max_iter = self.max_iters - self.warmup_iters
        super().__init__(optimizer, last_epoch=last_epoch)
        return
    
    def _get_ratio(self) -> float:

        # ramp
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            return self.warmup_ratio + (1. - self.warmup_ratio) * alpha
        
        if self.last_epoch < self.max_iters:
            real_last_epoch = self.last_epoch - self.warmup_iters
            alpha = real_last_epoch / self._consine_max_iter
            return self.eta + (1 - self.eta) * 0.5 * (1 + math.cos(math.pi * alpha))

        return self.eta

    def get_lr(self):
        ratio = self._get_ratio()
        return [lr * ratio for lr in self.base_lrs]
    