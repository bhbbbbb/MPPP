# import sklearn.model_selection as sk
import os
import math
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset as BaseDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from util.track import Track
from util.augument import Transform
from .mode import Mode as M
from .config import DatasetConfig

class MPPPDataset(BaseDataset):

    def __init__(self, config: DatasetConfig, mode: M):
        super().__init__()

        self.config = config
        self.mode = mode

        self.tracks = MPPPDataset._load_tracks(mode, config)
        self.transfrom = None
        if mode is M.TRAIN:
            self.transfrom = Transform(
                self.config.freq_mask_param,
                self.config.time_mask_param,
                self.config.random_gain_param,
            )
        return
    
    @property
    def batch_size(self):
        return self.config.batch_size[self.mode]

    @staticmethod
    def _load_tracks(mode: M, config: DatasetConfig):

        set_path = None
        if mode is M.TRAIN:
            set_path = config.TRAIN_SET_PATH
        
        elif mode is M.VALID:
            set_path = config.VALID_SET_PATH
        
        elif mode is M.TEST:
            set_path = config.TEST_SET_PATH
        
        assert set_path is not None

        if not config.use_mixed_region:
            return list(Track.from_set(set_path, config))
        
        df = pd.read_csv(set_path, usecols=['track_id', 'region'])

        return [
            Track(track_id, config, use_region=region) for track_id, region\
                in zip(df['track_id'], df['region'])
        ]
        

    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, index) -> Tuple[Tensor, np.ndarray]:

        track = self.tracks[index]

        features = track.acoustic_features
        if self.transfrom is not None:
            features = self.transfrom(features)

        return features, track.stream
    
    @property
    def dataloader(self) -> Iterator[Tuple[Tensor, Tensor]]:
        return DataLoader(
            self,
            batch_size=self.config.batch_size[self.mode],
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers,
            shuffle=(self.mode is M.TRAIN),
            pin_memory=self.config.pin_memory,
            drop_last=(self.mode is M.TRAIN and self.config.drop_last),
        )

class MP2Dataset(MPPPDataset):

    def __init__(self, config: DatasetConfig, mode: M):
        super().__init__(config, mode)
        self.scaler = self._get_scalers()
        return

    dataloader: Iterator[Tuple[Tensor, Tensor]]

    def _get_non_transformed_item(self, index):

        features, stream = super().__getitem__(index)

        debut: float = stream[0]

        delog_stream: np.ndarray = np.vectorize(lambda x: 10 ** x)(stream)

        sumation = math.log10(delog_stream.sum())
        
        return features, [sumation, debut]

    def __getitem__(self, index):

        features, sum_debut = self._get_non_transformed_item(index)
        
        return features, self.scaler.transform([sum_debut])[0]
    
    def _get_scalers(self):

        if os.path.exists(self.config.mp2_scaler_path):
            scaler: StandardScaler = joblib.load(self.config.mp2_scaler_path)
        
        else:
            assert self.mode == M.TRAIN
            tem = (self._get_non_transformed_item(idx) for idx in range(len(self)))
            items = [(sumation, debut) for _, (sumation, debut) in tqdm(tem, total=len(self))]
            scaler = StandardScaler()
            scaler.fit(items)
            os.makedirs(os.path.dirname(self.config.mp2_scaler_path), exist_ok=True)
            joblib.dump(scaler, self.config.mp2_scaler_path)

        print('scaler.mean, scaler.var, scaler.scale')
        print(scaler.mean_, scaler.var_, scaler.scale_)
        return scaler
    