"""adapted from https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py"""
from torch import nn
from torch import Tensor
import timm
from timm.models.vision_transformer import VisionTransformer, PatchEmbed as VPatchEmbed

class PatchEmbed(VPatchEmbed):

    def __init__(self, **kwargs):
        kwargs.update(in_chans=1)
        super().__init__(**kwargs)
        return

    def forward(self, x: Tensor):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class AstEncoder(nn.Module):

    def __init__(
        self,
        pretrained_model_name: str,
        input_time_len: int = 2800,
        input_freq_bins: int = 128,
        output_dim: int = 32,
        stochastic_depth_rate: float = 0.0,
        dropout_rate: float = 0.0,
        **_,
    ):
        super().__init__()
        self.vit: VisionTransformer = timm.create_model(
            pretrained_model_name,
            pretrained=True,
            num_classes=output_dim,
            drop_rate=dropout_rate,
            drop_path_rate=stochastic_depth_rate,
            embed_layer=PatchEmbed,
            img_size=(input_time_len, input_freq_bins),
        )
        # use fc_norm
        self.vit.fc_norm, self.vit.norm = self.vit.norm, self.vit.fc_norm

        # use avg pooling
        self.vit.global_pool = 'avg'
        return
    
    def forward(self, x: Tensor) -> Tensor:
        # x: B x T x F
        x = x.unsqueeze(1)
        # B x 1 X T x F
        
        # x = x.transpose(2, 3) # seems unnecessary

        return self.vit.forward(x)
    