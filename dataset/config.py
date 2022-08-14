import os
from datetime import date
from typing import Dict

from model_utils.base import UNIMPLEMENTED, NOT_NECESSARY

from util.track import TrackConfig
from .mode import Mode

class DatasetConfig(TrackConfig):

    # -----------------  track config -----------------------
    sample_rate = 16000

    min_track_length: int = 104
    """num of weeks"""

    padding_val: float = 0.0
    """padding value used for null (val after log10)"""

    now_date: date = UNIMPLEMENTED
    # now_date: date = date(2022, 7, 20)
    """date that data was fetched"""

    use_region: str = UNIMPLEMENTED

    stream_root: str = UNIMPLEMENTED

    audio_root: str = NOT_NECESSARY
    
    # ----------------------

    TRAIN_SET_PATH: str = UNIMPLEMENTED
    VALID_SET_PATH: str = UNIMPLEMENTED
    TEST_SET_PATH: str = UNIMPLEMENTED

    use_mixed_region: bool = UNIMPLEMENTED

    # ------------------

    # ---------- DataLoader -------------------
    batch_size: Dict[Mode, int] = UNIMPLEMENTED

    num_workers: int = 4 if os.name == 'nt' else 2

    persistent_workers: bool = (os.name == 'nt')

    pin_memory: bool = True

    drop_last: bool = True

    # ----------------------------------------

    mp2_scaler_path: str = UNIMPLEMENTED

    # --------------------------------------------

    freq_mask_param: int = int(80 * 0.2)
    time_mask_param: int = int(2800 * 0.2)
    random_gain_param: float = 7.

@DatasetConfig.register_checking_hook
def check_region_config(config: DatasetConfig):
    """Deambiguous"""
    print(
        f'checkfor (use_mixed_region, use_region) = {config.use_mixed_region}, {config.use_region}'
    )
    if config.use_mixed_region:
        assert not config.use_region, f'use_region = {config.use_region}'
    return

@DatasetConfig.register_checking_hook
def check_worker_setup(config: DatasetConfig):
    print(
        'check for presistent_workers, num_workers'
        f' = {config.persistent_workers}, {config.num_workers}'
    )
    if config.persistent_workers:
        assert config.num_workers > 0
    