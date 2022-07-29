import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones import hypernet,HyperNet
from ..builder import HEADS
from ..utils import is_tracing
from .reg_head import RegHead


@HEADS.register_module()
class LinearRegHead(RegHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """
    # num_classes,
    # in_channels,
    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LinearRegHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        # self.hypernet = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()


        # if self.num_classes <= 0:
        #     raise ValueError(
        #         f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)#self.in_channels, self.num_classes

    def simple_test(self, img):
        """Test without augmentation."""
        # pred = self.hypernet(img)
        pred = img

        # if isinstance(cls_score, list):
        #     cls_score = sum(cls_score) / float(len(cls_score))
        # pred = F.softmax(cls_score, dim=1) if cls_score is not None else None

        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return pred
        # pred = list(pred.detach().cpu().numpy())
        pred = pred.detach().cpu().tolist()
        return pred

    def forward_train(self, x, gt_label):
        pred_scores = []
        gt_scores = []
        epoch_loss = []
        pred_scores = pred_scores + x.cpu().tolist()
        gt_scores = gt_scores + gt_label.cpu().tolist()
        # cls_score = x
        # print('='*50)
        # print(cls_score)
        # print(gt_label)
        losses = self.loss(x, gt_label)
        # epoch_loss.append(losses.item())
        # losses1 = sum(epoch_loss) / len(epoch_loss)
        # losses.backward()
        return losses
