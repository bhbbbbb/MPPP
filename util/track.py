from __future__ import annotations
import os
import re
from datetime import date, timedelta
from typing import Iterable, Iterator, Union
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
try:
    from torchaudio.backend import soundfile_backend as backend
except ImportError:
    from torchaudio.backend import sox_io_backend as backend
from torchaudio import transforms as T
import torchaudio.compliance.kaldi as ta_kaldi

from model_utils.base.config import BaseConfig, UNIMPLEMENTED, NOT_NECESSARY

class TrackConfig(BaseConfig):

    src_sample_rate = 44100

    sample_rate = 16000

    min_track_length: int = 104
    """num of weeks"""

    padding_val: float = 0.0
    """padding value used for null (val after log10)"""

    now_date: date = UNIMPLEMENTED
    # now_date: date = date(2022, 7, 20)
    """date that data was fetched"""

    cache_acoustic_features: bool = True

    max_acoustic_features_len: int = 2800

    use_region: str = UNIMPLEMENTED

    stream_root: str = UNIMPLEMENTED

    audio_root: str = NOT_NECESSARY


class InvalidDateException(BaseException):
    pass

DATE_PATTERN = r'(\d{4})[\/\\-](\d{2})[\/\\-](\d{2})'
def parse_date(date_string: str) -> Union[date, None]:
    match = re.match(DATE_PATTERN, date_string)
    
    if match:
        return date(int(match[1]), int(match[2]), int(match[3]))
    return None


class Track:

    track_id: str
    start_date: date
    stream: np.ndarray
    config: TrackConfig

    def __init__(self, track_id: str, config: TrackConfig = None, **kwargs):

        self.track_id = track_id
        
        self.config = config

        if self.config is None:
            self.config = TrackConfig(**kwargs)
            self.config.check_and_freeze()
        
        elif len(kwargs):
            d = config.asdict()
            d.update(**kwargs)
            self.config = TrackConfig(**d)
            self.config.check_and_freeze()



        self.stream, self.start_date = Track._load_stream(
            track_id,
            self.config.use_region,
            self.config.stream_root
        )

        self._pre_transform()
        self._features = None

        return
    
    @property
    def acoustic_features(self):

        if self.config.cache_acoustic_features:
            if self._features is None:
                self._features = self.extract_acoustic_features()
            return self._features
        
        return self.extract_acoustic_features()


    def is_valid(self):
        return (
            self.start_date
            < self.config.now_date - timedelta(days=7) * self.config.min_track_length
        )
    
    def _pre_transform(self):
        self._trim_left()
        self._fix_right()
        return

    def __len__(self):
        return len(self.stream)

    def plot(self):
        plt.plot(list(range(len(self.stream))), self.stream)
        plt.title(str(self.track_id))
        plt.ylim((3.5, 6.5))
        plt.show()
        return

    def _trim_left(self):
        start = 0
        while start < len(self.stream):
            if self.stream[start] != 0:
                break
            start += 1
        
        self.start_date += start * timedelta(days=7)
        self.stream = self.stream[start:]
        return
        
        
        # end = len(track) - 1
        # while end >= 0:
        #     if track[end] != 0:
        #         break
        #     end -= 1
        
        # end += 1
        
        # assert end - start > 0, f'expect (end - start) > 0. got ({end} - {start}) = {end - start}'
        # return track[start:end]
    
    def _fix_right(self):
        """Trim right part if len(stream) > MIN_TRACK_LENGTH;
        Pad with zero otherwise
        """
        self.stream = self.stream[:self.config.min_track_length]

        if (diff := self.config.min_track_length - len(self.stream)) > 0:
            self.stream = np.pad(self.stream, (0, diff))
        return


    @staticmethod
    def _load_stream(track_id: str, use_region: str, stream_root: str):

        track_df = pd.read_csv(
            os.path.join(stream_root, f'{track_id}.csv'),
            usecols=['date', use_region.lower()]
        )

        track_df['date'] = track_df['date'].map(parse_date)

        start_date: date = track_df['date'][0]
        # assert (start_date - release_date).days >= 0,\
        #     f'track_id: {track_id}, start_date: {start_date}, release_date: {release_date}'
        
        end_date: date = track_df['date'][len(track_df) - 1]

        num_weeks = (end_date - start_date) / timedelta(days=7)

        # assert num_weeks.is_integer()
        if not num_weeks.is_integer():
            raise InvalidDateException

        num_weeks = int(num_weeks) + 1

        streams = np.empty([num_weeks])
        streams_idx = 0

        for cur_date, stream in zip(track_df['date'], track_df[use_region]):

            stream: float

            while cur_date > streams_idx * timedelta(days=7) + start_date:
                streams[streams_idx] = 0.0
                streams_idx += 1

            # assert cur_date == streams_idx * timedelta(days=7) + start_date
            if cur_date != streams_idx * timedelta(days=7) + start_date:
                raise InvalidDateException

            streams[streams_idx] = 0.0 if np.isnan(stream) else math.log10(stream)
            streams_idx += 1

        # return streams, (start_date, end_date)
        return streams, start_date

    # def extract_acoustic_features(self, output_dir: str = None):
    def extract_acoustic_features(self):

        audio_path = os.path.join(self.config.audio_root, f'{self.track_id}.wav')

        assert os.path.exists(audio_path)

        wav, sample_rate = backend.load(audio_path, format='WAV')

        assert sample_rate == self.config.src_sample_rate, (
            f'expect sample_rate = {self.config.src_sample_rate}'
            f'. got sample_rate = {sample_rate}'
        )
        wav: Tensor = T.Resample(sample_rate, self.config.sample_rate)(wav)

        # if wav.shape[0] != 1:
        #     wav = torch.mean(wav, dim=0, keepdim=True)
        features = ta_kaldi.fbank(wav, num_mel_bins=80, sample_frequency=self.config.sample_rate)

        # output_dir = output_dir or os.path.join(self.config.audio_root, '..', 'ac_features')
        # if output_dir is not None:
        #     os.makedirs(output_dir, exist_ok=True)
        #     torch.save(features, os.path.join(output_dir, f'{self.track_id}.pt'))

        return features[:self.config.max_acoustic_features_len]
    
    @staticmethod
    def from_set(
        set_path: str,
        config: TrackConfig,
        indices: Iterable[int] = None,
        **kwargs,
    ) -> Iterator[Track]:

        ids = pd.read_csv(set_path)['track_id']

        if indices is not None:
            ids = ids[indices]

        return (Track(track_id, config, **kwargs) for track_id in ids)
