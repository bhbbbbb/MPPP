from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
from model_utils import BaseModelUtils
from model_utils.base.criteria import Criteria

from dataset.dataset import MP2Dataset
from model.mp2 import MP2net
from util.scheduler import WarmupCosineLR
from .loss import WeightedMSELoss
from .config import MP2ModelUtilsConfig
from .criteria import Loss, DebutLoss, SumationLoss

class MP2ModelUtils(BaseModelUtils):

    criterion: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]
    config: MP2ModelUtilsConfig
    model: MP2net

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = WeightedMSELoss(
            Tensor(self.config.loss_item_weights).to(self.config.device)
        )
        self.scaler = GradScaler(enabled=False)
        return

    @staticmethod
    def _get_optimizer(model: nn.Module, config: MP2ModelUtilsConfig):
        return AdamW(model.parameters(), lr=config.learning_rate)
    
    @staticmethod
    def _get_scheduler(optimizer, config: MP2ModelUtilsConfig):
        return WarmupCosineLR(
            optimizer,
            config.warmup_iters,
            config.max_iters,
        )

    def _train_epoch(self, train_dataset: MP2Dataset) -> Criteria:

        self.model.train()

        torch.cuda.empty_cache()
        pbar = tqdm(train_dataset.dataloader)

        overall_loss = 0.0
        overall_sumation_loss = 0.0
        overall_debut_loss = 0.0
        steps = 0

        for features, targets in pbar:

            steps += 1

            self.optimizer.zero_grad()

            features, targets = features.to(self.config.device), targets.to(self.config.device)

            targets = targets.float()

            with autocast():
                output = self.model(features)

                total_loss, sumation_loss, debut_loss = self.criterion(output, targets)

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            # total_loss.backward()
            # self.optimizer.step()

            lr_ratio = self.scheduler.get_last_lr()[0] / self.config.learning_rate

            running_loss = total_loss.item()
            overall_loss += running_loss
            overall_sumation_loss += sumation_loss.item()
            overall_debut_loss += debut_loss.item()

            pbar.set_description(f'LR: {lr_ratio:.2f}, L: {running_loss: .4e}')
        
        overall_loss /= steps
        overall_debut_loss /= steps
        overall_sumation_loss /= steps


        return Criteria(
            Loss(overall_loss),
            SumationLoss(overall_sumation_loss),
            DebutLoss(overall_debut_loss),
        )
    
    @torch.no_grad()
    def _eval_epoch(self, eval_dataset: MP2Dataset) -> Criteria:

        self.model.eval()

        pbar = tqdm(eval_dataset.dataloader)

        overall_loss = 0.0
        overall_sumation_loss = 0.0
        overall_debut_loss = 0.0
        steps = 0
        torch.cuda.empty_cache()

        for features, targets in pbar:

            steps += 1

            features, targets = features.to(self.config.device), targets.to(self.config.device)

            output = self.model(features)

            total_loss, sumation_loss, debut_loss = self.criterion(output, targets)

            running_loss = total_loss.item()
            overall_loss += running_loss
            overall_sumation_loss += sumation_loss.item()
            overall_debut_loss += debut_loss.item()

            pbar.set_description(f'L: {running_loss: .4e}')
        
        overall_loss /= steps
        overall_sumation_loss /= steps
        overall_debut_loss /= steps

        return Criteria(
            Loss(overall_loss),
            SumationLoss(overall_sumation_loss),
            DebutLoss(overall_debut_loss),
        )
    
    @torch.inference_mode()
    def inference(self, dataset: MP2Dataset):

        self.model.eval()

        pbar = tqdm(dataset.dataloader)

        idx = 0

        for features, targets in pbar:
            

            features, targets = features.to(self.config.device), targets.to(self.config.device)

            output: Tensor = self.model(features)

            _total_loss, sumation_loss, debut_loss = self.criterion(output, targets)

            track = dataset.tracks[idx]

            title = (
                f'{track.track_id}, '
                f'L_sum: {sumation_loss.item(): .3f}, '
                f'L_debut: {debut_loss.item(): .3f}'
            )

            MP2ModelUtils._plot_results(
                track.stream,
                dataset.scaler.inverse_transform(output.cpu())[0],
                dataset.scaler.inverse_transform(targets.cpu())[0],
                title,
            )

            idx += 1

        
    @staticmethod
    def _plot_results(
        streams: np.ndarray,
        predicted: Tensor,
        targets: Tensor,
        title: str = None,
    ):

        x_axis = list(range(len(streams)))

        if title is not None:
            plt.title(title)

        plt.ylim((0, 7.5))
        plt.annotate(
            (
                'ground-turth:\n'
                f'sumation = {targets[0]}\n'
                f'debut = {targets[1]}'
            ),
            (0, targets[1])
        )
        text = (
            'predicted:\n'
            f'sum_p = {predicted[0].item()}\n'
            f'debut_p = {predicted[-1].item()}'
        )
        plt.text(.5, .5, text)
        plt.plot(x_axis, streams, label='streams')
        plt.legend()
        plt.show()
        return
    