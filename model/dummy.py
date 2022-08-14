import torch
from torch import nn
from torch import Tensor

class Dummy(nn.Module):
    """always output zeros for evaluation for baseline"""

    def __init__(self, output_shape: tuple):
        """

        Args:
            output_shape (tuple): shape of output (without batch_size)
        """
        super().__init__()
        self.output_shape = output_shape
        return
    
    def forward(self, x: Tensor):

        B = x.shape[0]

        return torch.zeros([B, *self.output_shape]).type_as(x).to(x.device)
    