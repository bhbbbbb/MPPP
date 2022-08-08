import random
from torchaudio import transforms as T
from torchaudio import functional as F
from torch import nn
from torch import Tensor

class Transform(nn.Module):

    def __init__(self, freq_mask_param: int, time_mask_param: int, gain_param: float):

        super().__init__()
        self.fm = self.tm = nn.Identity()

        # since the fliter banks output is in shape (time x freq_bank),
        # instead of (freq x time) therefore `T.FreqencyMasking` would actually work as
        # `T.TimeMasking`
        if freq_mask_param > 0:
            self.fm = T.TimeMasking(freq_mask_param)
        
        if time_mask_param > 0:
            self.tm = T.FrequencyMasking(time_mask_param)
        
        self.gain_param = gain_param
        return
    
    def forward(self, wav: Tensor):

        wav = self.fm(self.tm(wav))

        gain = random.uniform(- self.gain_param, self.gain_param)

        wav = F.gain(wav, gain)

        return wav


