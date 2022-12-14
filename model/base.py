import logging
from typing import Callable
import torch
from torch import nn
from torch import Tensor

logger = logging.getLogger(__name__)

# borrow from Segformer.mit.Attention
class Attention(nn.Module):
    def __init__(self, dim: int, head: int, head_dim: int = None, sr_ratio: int = 1):
        super().__init__()
        self.head_dim = head_dim or dim // head
        inner_dim = self.head_dim * head
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, inner_dim)
        self.kv = nn.Linear(dim, inner_dim*2)
        self.proj = nn.Linear(inner_dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv1d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)
        return

    q: Callable[[Tensor], Tensor]
    sr: Callable[[Tensor], Tensor]
    kv: Callable[[Tensor], Tensor]

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        q: Tensor = self.q(x).reshape(B, N, self.head, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1) # B x D x N
            x = self.sr(x).permute(0, 2, 1)
            x = self.norm(x)
        
            
        k, v = self.kv(x).reshape(B, -1, 2, self.head, self.head_dim).permute(2, 0, 3, 1, 4)


        attn: Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.head_dim * self.head)
        x = self.proj(x)
        return x

class MLP(nn.Module):

    def __init__(self, dim: int, mlp_dim: int, out_dim: int = None, dropout: float = 0.0):
        super().__init__()

        out_dim = out_dim or dim

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, out_dim),
            nn.Dropout(dropout),
        )
        return
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class Block(nn.Module):

    def __init__(
        self,
        dim: int,
        head: int,
        mlp_dim: int,
        out_dim: int = None,
        sr_ratio: int = 1,
        head_dim: int = None,
        dropout: float = 0.0,
        stochastic_depth_rate: float = 0.0,
    ):
        super().__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(dim),
            Attention(
                dim,
                head=head,
                head_dim=head_dim,
                sr_ratio=sr_ratio
            ),
        )
        self.ff = MLP(dim, mlp_dim, out_dim=out_dim, dropout=dropout)

        self.attention_residual = (sr_ratio == 1)
        self.ff_residual = (dim == out_dim or out_dim is None)

        if stochastic_depth_rate > 0.:
            assert self.attention_residual or self.ff_residual

        self.stoch_depth = StochasticDepth(stochastic_depth_rate)\
            if stochastic_depth_rate > 0 else nn.Identity()
        return
    
    def forward(self, x: Tensor) -> Tensor:

        residual = x
        # logger.debug(f'x before attention ({x.shape}):')
        # logger.debug(x)
        x = self.attention(x)

        if self.attention_residual:
            x = self.stoch_depth(x) + residual

        residual = x
        x = self.ff(x)
        if self.ff_residual:
            x = self.stoch_depth(x) + residual

        return x

class StochasticDepth(nn.Module):
    def __init__(self, stochastic_depth_rate:float):
        super().__init__()

        assert 0.0 <= stochastic_depth_rate <= 1.0
        # self.drop_rate = stoch_depth_rate
        self.survival_rate = stochastic_depth_rate
        return

    def forward(self, x: Tensor):
        if not self.training:
            return x

        size = [1] * x.ndim
        is_drop = torch.empty(size, dtype=x.dtype, device=x.device)
        is_drop = is_drop.bernoulli_(self.survival_rate)
        
        return x * is_drop
