from torch import nn
from torch import Tensor

class WeightedMSELoss(nn.Module):

    def __init__(
        self,
        item_weights: Tensor,
    ):
        super().__init__()
        assert item_weights.shape == (2,)
        self.mse = nn.MSELoss(reduction='none')
        self.item_weights = item_weights
        return

    def forward(self, predicted: Tensor, target: Tensor):

        loss: Tensor = self.mse(predicted, target) * self.item_weights

        loss1 = loss[:, 0].mean()
        loss2 = loss[:, 1].mean()
        total = loss1 + loss2

        return total, loss1, loss2
