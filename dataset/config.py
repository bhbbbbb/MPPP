import os
from datetime import date
from typing import Dict, List, Union

from model_utils.base import UNIMPLEMENTED, NOT_NECESSARY

from util.track import TrackConfig
from util.augument import AugConfig
from .mode import Mode

class DatasetConfig(TrackConfig, AugConfig):

    # -----------------  track config -----------------------
    sample_rate = 16000

    min_track_length: int = 104
    """num of weeks"""

    now_date: date = UNIMPLEMENTED
    # now_date: date = date(2022, 7, 20)
    """date that data was fetched"""

    use_regions: Union[List[str], Dict[str, int]] = UNIMPLEMENTED
    """
    e.g.
    List:
        ['us', 'jp', 'tw']
    
    Dict(with weight):
        {
            'us': 4,
            'jp': 2,
            'tw': 1,
        }
    """
    
    num_samples_per_track: int = 1

    stream_root: str = UNIMPLEMENTED

    audio_root: str = NOT_NECESSARY
    
    # ----------------------

    TRAIN_SET_PATH: str = UNIMPLEMENTED
    VALID_SET_PATH: str = UNIMPLEMENTED
    TEST_SET_PATH: str = UNIMPLEMENTED

    # ------------------

    # ---------- DataLoader -------------------
    batch_size: Dict[Mode, int] = UNIMPLEMENTED

    num_workers: int = 4 if os.name == 'nt' else 2

    persistent_workers: bool = (os.name == 'nt')

    pin_memory: bool = True

    # ----------------------------------------

    mp2_scaler_path: str = UNIMPLEMENTED

    # -------------- AugConfig --------------------


    freq_mask_param: int = int(128 * 0.2)

    time_mask_param: int = int(2800 * 0.2)

    gain_param: float = 7.

    rolling_distance: int = 256

    add_noise: bool = False

    max_len: int = 2800

    # -----------------------------------------


@DatasetConfig.register_checking_hook
def check_worker_setup(config: DatasetConfig):
    print(
        'check for presistent_workers, num_workers'
        f' = {config.persistent_workers}, {config.num_workers}'
    )
    if config.persistent_workers:
        assert config.num_workers > 0
    