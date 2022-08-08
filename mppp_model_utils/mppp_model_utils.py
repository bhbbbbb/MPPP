import math
from typing import Callable

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast
from tqdm import tqdm
from model_utils import BaseModelUtils
from model_utils.base.criteria import Criteria, Loss
from matplotlib import pyplot as plt

from dataset import MPPPDataset
from model.mppp import MPPPnet
from .config import MPPPModelUtilsConfig
from .loss import HuberMSELoss

class MPPPModelUtils(BaseModelUtils):

    criterion: Callable
    config: MPPPModelUtilsConfig
    model: MPPPnet

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = HuberMSELoss(
            self.config.huber_delta,
            self.config.huber_weight,
            self.config.stream_null_value,
            MPPPModelUtils._get_loss_item_weights(
                self.config.loss_slope_factor,
                self.config.decoder_seq_len,
            ).to(self.config.device),
        )
        return

    @staticmethod
    def _get_loss_item_weights(slope_factor: float, seq_len: int):
        """
        item_weight = -log10(x/seq_len) + 1 , for x in [1, seq_len]
        """
        return Tensor(
            [1 - slope_factor * math.log10(i / seq_len) for i in range(1, seq_len + 1)]
        ).unsqueeze(-1)
    
    @staticmethod
    def _get_optimizer(model: nn.Module, config: MPPPModelUtilsConfig):
        return AdamW(model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def _eval_epoch(self, eval_dataset: MPPPDataset) -> Criteria:

        self.model.eval()

        pbar = tqdm(eval_dataset.dataloader)

        overall_loss = 0.0
        steps = 0

        for features, stream in pbar:

            steps += 1

            features, stream = features.to(self.config.device), stream.to(self.config.device)

            stream = stream.unsqueeze(-1) * self.config.stream_scale

            blank_stream = torch.full_like(stream, self.config.start_token, dtype=torch.float16)

            with autocast():
                output = self.model.inference_forward(features, blank_stream)

                loss: Tensor = self.criterion(output, stream)

            running_loss = loss.item()
            overall_loss += running_loss

            pbar.set_description(f'L: {running_loss: .4e}')
        
        overall_loss /= steps

        return Criteria(Loss(overall_loss))
    
    def _train_epoch(self, train_dataset: MPPPDataset) -> Criteria:

        self.model.train()

        pbar = tqdm(train_dataset.dataloader)

        overall_loss = 0.0
        steps = 0

        for features, stream in pbar:

            steps += 1

            self.optimizer.zero_grad()

            features, stream = features.to(self.config.device), stream.to(self.config.device)

            stream = stream.unsqueeze(-1) * self.config.stream_scale

            tf_stream = MPPPModelUtils._to_teacher_forcing_labels(stream, self.config.start_token)
            tf_stream = tf_stream.float()

            # with autocast():
            output = self.model(features, tf_stream)

            loss: Tensor = self.criterion(output, stream)

            loss.backward()

            self.optimizer.step()

            running_loss = loss.item()
            overall_loss += running_loss

            pbar.set_description(f'L: {running_loss: .4e}')
        
        overall_loss /= steps

        return Criteria(Loss(overall_loss))

    @staticmethod
    def _to_teacher_forcing_labels(stream: Tensor, start_token_value: float):
        """

        Args:
            stream (Tensor): B x T x 1
        """

        padding = (0, 0, 1, 0)
        return F.pad(stream, padding, value=start_token_value)[:, :-1]
    
    @torch.inference_mode()
    def inference(self, dataset: MPPPDataset, num: int = None):

        self.model.eval()

        idx = 0
        for features, stream in tqdm(dataset.dataloader):

            features, stream = features.to(self.config.device), stream.to(self.config.device)

            stream = stream.unsqueeze(-1) * self.config.stream_scale

            blank_stream = torch.full_like(stream, self.config.start_token, dtype=torch.float16)

            results = self.model.inference_forward(features, blank_stream)

            loss: Tensor = self.criterion(results, stream)

            # running_loss = loss.item()

            # ----------------------------------------------------
            # features = features.to(self.config.device)

            # blank_stream = torch.full_like(
            #     stream,
            #     self.config.start_token,
            #     dtype=torch.float16,
            #     device=self.config.device,
            # ).unsqueeze(-1)

            # results = self.model.inference_forward(features, blank_stream)
            # -------------------------------------------------------------

            # results /= self.config.stream_scale

            # title = f'{loss[0].item()}, {loss[1].item()}, {loss[2].item()}'
            title = str(loss.item())
            MPPPModelUtils._plot_comparsion(results.cpu(), stream.cpu(), title)

            idx += 1

            if num is not None and idx >= num:
                return

        return
    
    @staticmethod
    def _plot_comparsion(predicted: Tensor, target: Tensor, title: str = None):

        predicted, target = predicted.squeeze(), target.squeeze()

        x_axis = list(range(len(predicted)))

        if title is not None:
            plt.title(title)
        plt.ylim((0, 1.2))
        plt.plot(x_axis, predicted, label='predicted')
        plt.plot(x_axis, target, label='ground-turth')
        plt.legend()
        plt.show()
        return
