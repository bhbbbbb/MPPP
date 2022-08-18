from __future__ import annotations
import os
import re
from datetime import date, timedelta
from typing import NamedTuple, Union, List

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

    min_stream_length: int = 104
    """num of weeks"""

    now_date: date = UNIMPLEMENTED
    # now_date: date = date(2022, 7, 20)
    """date that data was fetched"""

    cache_acoustic_features: bool = True
    cache_streams: bool = True

    stream_root: str = UNIMPLEMENTED

    audio_root: str = NOT_NECESSARY

    num_mel_bins: int = 128


class InvalidDateException(BaseException):
    pass

_DATE_PATTERN = r'(\d{4})[\/\\-](\d{2})[\/\\-](\d{2})'
def _parse_date(date_string: str) -> Union[date, None]:
    match = re.match(_DATE_PATTERN, date_string)
    
    if match:
        return date(int(match[1]), int(match[2]), int(match[3]))
    return None


class Stream:

    def __init__(
        self,
        data: np.ndarray,
        region: str,
        start_date: date,
        required_len: int,
    ):
        self.data = data
        self.region = region
        self.start_date = start_date
        self._trim_left_()
        self._fix_right_(required_len)
        return
    
    # def scale_(self, scaler: Callable[[np.ndarray], np.ndarray]):
    #     self.data = scaler(self.data)
    #     return self

    def plot(self):
        print(self.data)
        plt.plot(list(range(len(self.data))), self.data)
        # plt.ylim((3., 7.))
        plt.show()
        return

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data[idx]
    
    def __str__(self):
        return (
            f'<\nstart_date: {self.start_date} at region: {self.region}\n'
            f'for {str(self.data)}'
            '\n>'
        )

    def _trim_left_(self):
        start = 0
        while start < len(self):
            if self.data[start] != 0:
                break
            start += 1
        
        self.start_date += start * timedelta(days=7)

        self.data = self.data[start:]
        return
        
    def _fix_right_(self, target_len: int):
        """Trim right part if len(stream) > MIN_STREAM_LENGTH;
        Pad with zero otherwise
        """
        self.data = self.data[:target_len]

        if (diff := target_len - len(self)) > 0:
            self.data = np.pad(self.data, (0, diff))
        return
    

class Track:

    track_id: str
    start_date: date
    _streams: dict[str, Stream]
    config: TrackConfig

    def __init__(self, track_id: str, config: TrackConfig):

        self.track_id = track_id
        
        self.config = config

        self.track_df = pd.read_csv(
            os.path.join(self.config.stream_root, f'{track_id}.csv'),
        )

        self.track_df['date'] = self.track_df['date'].map(_parse_date)

        self.start_date = self.track_df['date'][0]

        def to_stamp(d: date) -> int:
            return (d - self.start_date).days

        self.track_df['date_stamp'] = self.track_df['date'].map(to_stamp)

        if not self.is_date_valid(self.start_date):
            raise InvalidDateException
        
        self._features = None
        self._streams = {}

        return
    
    @property
    def acoustic_features(self):

        if self.config.cache_acoustic_features:
            if self._features is None:
                self._features = self.extract_acoustic_features()
            return self._features
        
        return self.extract_acoustic_features()


    def is_date_valid(self, start_date: date):
        return (
            start_date
            < self.config.now_date - timedelta(days=7) * (self.config.min_stream_length + 1)
        )
    

    def _load_unscaled_stream(
        self,
        region: str,
    ) -> Union[np.ndarray, None]:

        streams: List[float] = []
        class Point(NamedTuple):
            stamp: int
            stream: float
        
        points = map(
            lambda ss: Point(ss[0], 0.0 if np.isnan(ss[1]) else ss[1]),
            zip(self.track_df['date_stamp'], self.track_df[region])
        )

        prev = cur = next(points)
        idx = 0
        while True:
            streams.append(0.0)
            target_stamp = idx * 7

            try:
                while cur.stamp < target_stamp:
                    prev, cur = cur, next(points)
            except StopIteration:
                break

            if cur.stamp == target_stamp:
                streams[idx] = cur.stream
            
            else:
                if prev.stamp > target_stamp - 7:
                    streams[idx] += prev.stream * (7 - (target_stamp - prev.stamp)) / 7
                
                if cur.stamp < target_stamp + 7:
                    streams[idx] += cur.stream * (7 - (cur.stamp - target_stamp)) / 7
            
            idx += 1
        
        return np.array(streams)
            
    def get_stream(self, region: str):
        """get stream for specific region.

        Returns:
            return None if that stream is not legal
        """
        
        if self.config.cache_streams and region in self._streams:
            return self._streams[region]
        
        raw_stream = self._load_unscaled_stream(region)
        print(raw_stream)
        stream = Stream(
            raw_stream,
            region=region,
            start_date=self.start_date,
            required_len=self.config.min_stream_length,
        )

        if not self.is_date_valid(stream.start_date):
            raise InvalidDateException(str(stream))

        if self.config.cache_streams:
            self._streams[region] = stream
        return stream

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
        features = ta_kaldi.fbank(
            wav,
            num_mel_bins=self.config.num_mel_bins,
            sample_frequency=self.config.sample_rate
        )

        # output_dir = output_dir or os.path.join(self.config.audio_root, '..', 'ac_features')
        # if output_dir is not None:
        #     os.makedirs(output_dir, exist_ok=True)
        #     torch.save(features, os.path.join(output_dir, f'{self.track_id}.pt'))

        return features
    
    # @staticmethod
    # def from_set(
    #     set_path: str,
    #     config: TrackConfig,
    #     indices: Iterable[int] = None,
    #     **kwargs,
    # ) -> Iterator[Track]:

    #     ids = pd.read_csv(set_path)['track_id']

    #     if indices is not None:
    #         ids = ids[indices]

    #     return (Track(track_id, config, **kwargs) for track_id in ids)
