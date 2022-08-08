from typing import List

from torch import Tensor
from torch import nn

from .mppp import Encoder

class MP2net(nn.Module):

    def __init__(
        self,
        src_dim: int,
        max_input_len: int,
        encoder_output_len: int,
        encoder_blocks_params: List[Encoder.Params],
        **_,
    ):

        super().__init__()
        self.encoder = Encoder(max_input_len, encoder_output_len, src_dim, encoder_blocks_params)
        self.fc = nn.Linear(encoder_output_len, 2)
        return
    
    def forward(self, x: Tensor) -> Tensor:

        x: Tensor = self.encoder(x)
        # B x T

        return self.fc(x)
    