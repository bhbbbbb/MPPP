from torch import nn
from torch import Tensor



class HuberMSELoss(nn.Module):

    def __init__(
        self,
        huber_delta: float,
        huber_weight: float,
        null_value: float,
        item_weights: Tensor,
    ):
        super().__init__()
        self.huber = nn.HuberLoss(delta=huber_delta, reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        # * 2 due to huberLoss = 0.5 * MSE (if x < delta)
        self.weight = huber_weight * 2
        self.null_value = null_value
        self.item_weights = item_weights
        return

    def forward(self, predicted: Tensor, target: Tensor):

        is_null = target <= self.null_value

        mse_loss: Tensor = self.mse(predicted, target) * self.item_weights
        huber_loss: Tensor = self.huber(predicted, target) * self.weight * self.item_weights

        n_value = 1.

        for d in target.shape:
            n_value *= d

        # print(self.weight)
        # print(huber_loss.squeeze())
        # print(mse_loss.squeeze())
        # print(f'is_null count: {is_null.sum()}')
        # h_loss = huber_loss[is_null].sum()
        # mse_loss = mse_loss[~is_null].sum()
        # all_loss = (h_loss + mse_loss) / n_value
        # return h_loss, mse_loss, all_loss
        
        return (huber_loss[is_null].sum() + mse_loss[~is_null].sum()) / n_value
