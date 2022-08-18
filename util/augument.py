import random
from torchaudio import transforms as T
from torchaudio import functional as F
import torch
from torch import nn
from torch import Tensor
from model_utils.base import BaseConfig, UNIMPLEMENTED


class AugConfig(BaseConfig):

    freq_mask_param: int = UNIMPLEMENTED

    time_mask_param: int = UNIMPLEMENTED

    gain_param: float = UNIMPLEMENTED

    rolling_distance: int = UNIMPLEMENTED

    add_noise: bool = UNIMPLEMENTED

    max_len: int = UNIMPLEMENTED


class Transform(nn.Module):

    def __init__(self, config: AugConfig):

        super().__init__()
        self.fm = self.tm = nn.Identity()

        # since the fliter banks output is in shape (time x freq_bank),
        # instead of (freq x time) therefore `T.FreqencyMasking` would actually work as
        # `T.TimeMasking`
        if config.freq_mask_param > 0:
            self.fm = T.TimeMasking(config.freq_mask_param)
        
        if config.time_mask_param > 0:
            self.tm = T.FrequencyMasking(config.time_mask_param)
        
        self.config = config

        return
    
    def forward(self, wav: Tensor):

        wav = self.fm(self.tm(wav))

        if self.config.gain_param > 0:
            gain = random.uniform(- self.config.gain_param, self.config.gain_param)

            wav = F.gain(wav, gain)
        
        assert self.config.add_noise is False
        if self.config.add_noise:
            # TODO
            # wav += torch.rand_like(wav) * random.uniform() / 10
            pass
        
        if (d := self.config.rolling_distance) > 0:
            wav = torch.roll(wav, random.randint(-d, d), 0)

        pre_trim = (max(len(wav), self.config.max_len) - self.config.max_len) // 2

        return wav[pre_trim : (self.config.max_len + pre_trim)]
