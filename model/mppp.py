from typing import List, TypedDict
import logging

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from .base import Block

logger = logging.getLogger(__name__)

class Encoder(nn.Module):

    class Params(TypedDict):
        dim: int
        head: int
        head_dim: int
        mlp_dim: int
        out_dim: int
        sr_ratio: int
        dropout: float
        stochastic_depth_rate: float

    def __init__(
        self,
        max_input_len: int,
        target_len: int,
        dim: int,
        block_params: List[Params],
    ):
        super().__init__()

        assert len(block_params) >= 1

        self.blocks = nn.Sequential(*[Block(**params) for params in block_params])
        
        self.pos_embedding = nn.Parameter(torch.randn(max_input_len, dim))

        self.proj = nn.Linear(block_params[-1]['out_dim'], target_len)

        return
    
    def forward(self, x: Tensor):
        """
        Shapes:
            B: batch
            N_IN: input length
            N_: input length after blocks
            D_IN: input dim
            D_: input dim after blocks
            T: target encoder output length
        
        Args:
            x (Tensor): B x N x D
        """

        _B, N_IN, _D_IN = x.shape

        x += self.pos_embedding[:N_IN]

        x = self.blocks(x)
        # B x N_ x D_

        x = torch.mean(x, dim=1)
        # B x D_
        
        x = self.proj(x)
        # B x T
        
        return x


class Decoder(nn.Module):

    class Params(TypedDict):
        head: int
        head_dim: int
        mlp_dim: int
        # out_dim: int
        # sr_ratio: int
        dropout: float

    def __init__(
        self,
        seq_len: int,
        encoder_out_dim: int,
        blocks_params: List[Params],
        pe_dim: int = 1,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, pe_dim))

        dim = 1 + pe_dim + encoder_out_dim

        self.blocks = nn.Sequential(*[Block(dim, **params) for params in blocks_params])
        self.linear = nn.Linear(1 + pe_dim + encoder_out_dim, 1)
        return
    
    def forward(self, prev_output: Tensor, encoder_out: Tensor):
        """
        Shapes:
            B: batch
            T: decoder output (or input) length
            D: decoder embeded dim
            S: encoder output dim

        Args:
            prev_output (Tensor): B x T x 1
            encoder_out (Tensor): B x S
        
        Returns:
            (Tensor): B x T x 1
        """

        B, T, _ = prev_output.shape
        B, S = encoder_out.shape

        encoder_out = encoder_out.reshape(B, 1, S).repeat(1, T, 1)
        # B x T x S

        pe = self.pos_embedding.repeat(B, 1, 1)

        x = torch.cat((prev_output, pe, encoder_out), dim=-1)
        # B x T x (1 + D_pos + S)

        logger.debug(f'x before blocks ({x.shape}):')
        logger.debug(x)
        x: Tensor = self.blocks(x)
        # B x T x (1 + D_pos + S)

        logger.debug(f'x_before_linear ({x.shape}):')
        logger.debug(x)
        x = self.linear(x) # B x T x 1
        # logger.debug(f'x_after_linear ({x.shape}):')
        # logger.debug(x)

        return F.relu(x)
    
    def inference_forward(self, prev_output: Tensor, encoder_out: Tensor):

        _B, T, _ = prev_output.shape

        logger.debug('encoder_out:')
        logger.debug(encoder_out)
        for i in range(1, T+1):

            output = self.forward(prev_output, encoder_out)

            if i <= 2:
                logger.debug(f'output_{i}')
                logger.debug(output.squeeze())
            prev_output[:, :i] = output[:, :i]
        
        return prev_output
        


class MPPPnet(nn.Module):

    def __init__(
        self,
        src_dim: int,
        max_input_len: int,
        feature_dim: int,
        encoder_blocks_params: List[Encoder.Params],
        decoder_seq_len: int,
        decoder_pe_dim: int,
        decoder_blocks_params: List[Decoder.Params],
        **_,
    ):
        super().__init__()

        self.encoder = Encoder(max_input_len, feature_dim, src_dim, encoder_blocks_params)
        self.decoder = Decoder(decoder_seq_len, feature_dim, decoder_blocks_params, decoder_pe_dim)

        return
    
    def forward(self, encode_src: Tensor, decode_src: Tensor) -> Tensor:
        """
        Args:
            encode_src (Tensor): B x N x D
            decoded_src (Tensor): B x T x 1
        
        Returns:
            (Tensor): B x T x 1
        """

        features: Tensor = self.encoder(encode_src)

        return self.decoder(decode_src, features)
    
    def inference_forward(self, encode_src: Tensor, decode_src: Tensor) -> Tensor:

        features: Tensor = self.encoder(encode_src)

        return self.decoder.inference_forward(decode_src, features)
