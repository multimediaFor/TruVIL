import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule_3d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 act=True):
        super(ConvModule_3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Decoder_3d(nn.Module):
    """
    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'trilinear'.
    """

    def __init__(self,
                 interpolate_mode='trilinear',
                 out_channels=3,
                 in_channels=None,
                 channels=256,
                 in_index=[0, 1, 2, 3, 4],
                 dropout_ratio=0.1):
        super().__init__()

        self.interpolate_mode = interpolate_mode
        self.in_channels = in_channels
        self.channels = channels
        self.in_index = in_index

        num_inputs = len(self.in_channels)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule_3d(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=(1, 1, 1),
                    stride=1))

        self.fusion_conv = ConvModule_3d(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=(1, 1, 1))

        self.linear_pred = nn.Conv3d(channels, out_channels, kernel_size=(1, 1, 1))
        self.dropout = nn.Dropout3d(dropout_ratio)

    def forward(self, inputs):
        """

        :param inputs: (N, C1, T/2, H/4, W/4)、(N, C2, T/2, H/8, W/8)、(N, C3, T/2, H/16, W/16)、(N, C4, T/2, H/32, W/32)
        :return: (N, 4C, T/2, H/4, W/4)
        """

        inputs = [inputs[i] for i in self.in_index]
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                F.interpolate(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=True))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.dropout(out)
        out = self.linear_pred(out)
        return out
