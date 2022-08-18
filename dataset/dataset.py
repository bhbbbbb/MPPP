# import sklearn.model_selection as sk
import os
import math
from typing import Iterable, Iterator, Tuple, Dict

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import (
    IterableDataset,
    DataLoader,
    get_worker_info,
    RandomSampler,
    BatchSampler,
    WeightedRandomSampler
)
from torch.nn import Identity
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

from util.track import Track
from util.augument import Transform
from .mode import Mode as M
from .config import DatasetConfig

class MPPPDataset(IterableDataset):

    region_weights: Dict[str, float]
    def __init__(self, config: DatasetConfig, mode: M):
        super().__init__()

        self.config = config
        self.mode = mode

        self.transfrom = Transform(config) if mode == M.TRAIN else Identity()

        set_path = MPPPDataset._get_set_path(mode, config)
        self.df = pd.read_csv(set_path)

        self.region_weights = self.config.use_regions
        if not isinstance(self.config.use_regions, dict):
            self.region_weights = {region: 1 for region in self.config.use_regions}

        self.tracks = [Track(track_id, config) for track_id in self.df['track_id']]
        self.indices_for_workers = None
        return
    
    @property
    def batch_size(self):
        return self.config.batch_size[self.mode]

    @staticmethod
    def _get_set_path(mode: M, config: DatasetConfig):

        if mode is M.TRAIN:
            return config.TRAIN_SET_PATH
        
        if mode is M.VALID:
            return config.VALID_SET_PATH
        
        if mode is M.TEST:
            return config.TEST_SET_PATH
        
        raise Exception('123')
        
    def __len__(self):
        if self.mode == M.TRAIN:
            return len(self.tracks) * self.config.num_samples_per_track
        
        return None

    def __iter__(self):
        worker_info = get_worker_info()

        worker_id = getattr(worker_info, 'id', 0)
        
        for idx in self.indices_for_workers[worker_id]:
            yield from self.__getitem__(idx)

    def __getitem__(self, index: int) -> Iterable[Tuple[Tensor, np.ndarray]]:

        track = self.tracks[index]
        regions = list(self.region_weights.keys())
        region_availability = self.df.iloc[index][regions]

        features = track.acoustic_features
        if self.transfrom is not None:
            features = self.transfrom(features)

        if self.mode == M.TRAIN:
            sampler = WeightedRandomSampler(
                weights=(region_availability * list(self.region_weights.values())),
                num_samples=min(self.config.num_samples_per_track, int(region_availability.sum())),
                replacement=False,
            )

            regions_to_get = (regions[i] for i in sampler)
        
        else:
            regions_to_get = (r for r, a in zip(regions, region_availability) if bool(a))
        

        yield from (
            (self.transfrom(features), track.get_stream(r)) for r in regions_to_get
        )
        return
    
    @property
    def dataloader(self) -> Iterator[Tuple[Tensor, Tensor]]:
        self.indices_for_workers = list(
            BatchSampler(
                RandomSampler(self.tracks, replacement=False) if self.mode == M.TRAIN\
                    else range(len(self.tracks)),
                batch_size=math.ceil(len(self.tracks) / (self.config.num_workers or 1)),
                drop_last=False,
            )
        )
        return DataLoader(
            self,
            batch_size=self.config.batch_size[self.mode],
            num_workers=self.config.num_workers,
            persistent_workers=self.config.persistent_workers,
            pin_memory=self.config.pin_memory,
            drop_last=(self.mode == M.TRAIN),
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
    