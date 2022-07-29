import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class L1Loss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self) -> None:
        super(L1Loss, self).__init__()


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target)#, reduction=self.reduction