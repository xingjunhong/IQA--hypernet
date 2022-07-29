import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class Conv(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """
    def __init__(self,in_channels=24, out_channels=1, kernel_size=(3,3), stride=(2,2),padding=1):
        super(Conv, self).__init__()

        self.cov = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,ceil_mode=True)

    # def __init__(self, kernel_size=3, stride=None, padding=0, dilation=1,
    #             ceil_mode=True, return_indices=False):
    #     super(MaxPooling, self).__init__()
    #     self.mp = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
    #             ceil_mode=ceil_mode,return_indices=return_indices)

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.cov(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            b = inputs.size()
            outs = self.cov(inputs)
            outs = self.mp(outs)
            outs = outs.view(inputs.size(0), -1)

        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs

# import torch
# import torch.nn as nn
# import numpy as np
# from ..builder import NECKS
#
#
# @NECKS.register_module()
# class Conv(nn.Module):
#     """Global Average Pooling neck.
#
#     Note that we use `view` to remove extra channel after pooling.
#     We do not use `squeeze` as it will also remove the batch dimension
#     when the tensor has a batch dimension of size 1, which can lead to
#     unexpected errors.
#     """
#     def __init__(self,in_channels=2048, out_channels=1, kernel_size=(3,3), stride=(2,2),padding=1):
#         super(Conv, self).__init__()
#
#         self.cov = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
#         self.mp = nn.MaxPool2d(kernel_size=5, stride=2, padding=1,ceil_mode=True)
#         self.cov1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1), stride=1)
#
#     # def __init__(self, kernel_size=3, stride=None, padding=0, dilation=1,
#     #             ceil_mode=True, return_indices=False):
#     #     super(MaxPooling, self).__init__()
#     #     self.mp = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
#     #             ceil_mode=ceil_mode,return_indices=return_indices)
#
#     def init_weights(self):
#         pass
#
#     def forward(self, inputs):
#         if isinstance(inputs, tuple):
#             outs = tuple([self.cov(x) for x in inputs])
#             outs = tuple(
#                 [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
#         elif isinstance(inputs, torch.Tensor):
#             b = inputs.size()
#             outs = self.cov(inputs)
#             outs = self.mp(outs)
#             outs = outs.cpu()
#             outs = outs.detach().numpy()
#             outs = outs.transpose(0, 2, 1, 3)
#             outs = torch.from_numpy(outs)
#             outs = outs.cuda().float()
#             outs = self.cov1(outs)
#             outs = outs.view(inputs.size(0), -1)
#             # outs =
#
#
#         else:
#             raise TypeError('neck inputs should be tuple or torch.tensor')
#         return outs
