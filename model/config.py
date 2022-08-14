from typing import List

from model_utils.base import BaseConfig

from .mppp import Encoder, Decoder


EB0 = Encoder.Params(
    dim=80, head=8, head_dim=16, mlp_dim=128, out_dim=80, sr_ratio=1, dropout=0.3,
)
EB1 = Encoder.Params(
    dim=80, head=8, head_dim=16, mlp_dim=64, out_dim=32, sr_ratio=4, dropout=0.3,
)
EB1_ = Encoder.Params(
    dim=80, head=8, head_dim=16, mlp_dim=128, out_dim=64, sr_ratio=4, dropout=0.2,
)
EB2 = Encoder.Params(
    dim=32, head=2, head_dim=16, mlp_dim=32, out_dim=32, sr_ratio=1, dropout=0.5,
)
EB2_ = Encoder.Params(
    dim=64, head=2, head_dim=16, mlp_dim=128,
    out_dim=64, sr_ratio=1, dropout=0.4, stochastic_depth_rate=0.2,
)

DB = Decoder.Params(head=2, mlp_dim=64, out_dim=None, head_dim=None, dropout=0.0)

class MPPPConfig(BaseConfig):

    src_dim: int = 80

    max_input_len: int = 2800

    feature_dim: int = 32

    encoder_blocks_params: List[Encoder.Params] = [EB1, EB2]

    decoder_seq_len: int = 104

    decoder_pe_dim: int = 1

    decoder_blocks_params: List[Decoder.Params] = [DB, DB]

class MP2Config(BaseConfig):

    src_dim: int = 80

    max_input_len: int = 2800

    feature_dim: int = 32

    # encoder_blocks_params: List[Encoder.Params] = [EB0, EB1, EB2]
    encoder_blocks_params: List[Encoder.Params] = [EB1_, *([EB2_] * 5)]

    @property
    def encoder_output_len(self):
        return self.feature_dim
    