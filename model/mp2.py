from typing import List

from torch import Tensor
from torch import nn

from .mppp import Encoder
from .ast import AstEncoder

class MP2net(nn.Module):

    def __init__(
        self,
        src_dim: int,
        max_input_len: int,
        encoder_output_len: int,
        pretrained_encoder_name: str = None,
        encoder_blocks_params: List[Encoder.Params] = None,
        stochastic_depth_rate: float = None,
        dropout_rate: float = None,
        **_,
    ):
        super().__init__()

        assert bool(encoder_blocks_params is None) ^ bool(pretrained_encoder_name is None)

        self.encoder = Encoder(max_input_len, encoder_output_len, src_dim, encoder_blocks_params)\
            if pretrained_encoder_name is None else (
                AstEncoder(
                    pretrained_encoder_name,
                    input_time_len=max_input_len,
                    input_freq_bins=src_dim,
                    output_dim=encoder_output_len,
                    stochastic_depth_rate=(stochastic_depth_rate or 0.0),
                    dropout_rate=(dropout_rate or 0.0),
                )
            )

        self.fc = nn.Linear(encoder_output_len, 2)
        return
    
    def forward(self, x: Tensor) -> Tensor:

        x: Tensor = self.encoder(x)
        # B x T

        return self.fc(x)
    