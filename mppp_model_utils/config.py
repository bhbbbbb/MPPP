from model_utils.config import ModelUtilsConfig, UNIMPLEMENTED

class MPPPModelUtilsConfig(ModelUtilsConfig):

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


    start_token: float = -1.0

    stream_scale: float = 1/6
    """scale to be applied to stream
    
        stream' = stream * stream_scale
    """

    huber_delta = .5
    stream_null_value = 0.
    huber_weight = .1
    
    decoder_seq_len = 104

    loss_slope_factor = 4
