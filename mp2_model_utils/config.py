from typing import Tuple
from model_utils.config import ModelUtilsConfig, UNIMPLEMENTED

class MP2ModelUtilsConfig(ModelUtilsConfig):

    device: str = 'cuda:0'
    """Device to use, cpu or gpu"""

    learning_rate: float = UNIMPLEMENTED

    epochs_per_checkpoint: int = 0
    """num of epochs per checkpoints

        Example:
            1: stand for save model every epoch
            0: for not save until finish
    """

    log_dir: str = 'log'
    """dir for saving checkpoints and log files"""

    logging: bool = True
    """whether log to file 'log.log'. It's useful to turn this off when inference on kaggle"""

    epochs_per_eval: int = 1
    """Number of epochs per evalution"""

    # ----------- Early Stoping Config -----------------------
    early_stopping: bool = False
    """whether enable early stopping"""

    early_stopping_threshold: int = 0
    """Threshold for early stopping mode. Only matter when EARLY_STOPPING is set to True."""

    save_best: bool = False
    """set True to save every time when the model reach best valid score."""


    # ----------------------------------------------------

    stream_scale: float = 1/6
    """scale to be applied to stream
    
        stream' = stream * stream_scale
    """

    loss_item_weights: Tuple[float, float] = (4., 1.)

    # ------------------------------------------------------------
    # scheduler

    _max_epochs: int = 50
    _warmup_epochs: int = 3

    iters_per_epoch: int = UNIMPLEMENTED

    @property
    def max_iters(self) -> int:
        return self._max_epochs * self.iters_per_epoch

    @property
    def warmup_iters(self) -> int:
        return self._warmup_epochs * self.iters_per_epoch


    # --------------------------------------------------------------
