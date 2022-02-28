import torch
import torch.nn as nn
import torch.nn.functional as functional

from .spec_norm import SpectralNorm


def DisConvLayer(in_channels, out_channels, kernel_size, stride, padding, spec_norm, bias):
    if spec_norm:
        conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    return conv

class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        y = y + residual
        y = self.relu(y)

        return y

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, channels):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(channels * self.expansion, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y += identity
        y = self.relu(y)

        return y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        return y


class UpsampleConvBlock(nn.Module):
    """Based on UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, final=False):
        super(UpsampleConvBlock, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(1)

        mid_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.final = final

    def forward(self, x, x_skip=None):
        # _, c, h, w = x_skip.shape
        # feature_r = functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)     # Activate interpolate during evaluation in case the input size is not a multiple of 32
        if x_skip is not None:
            x = torch.cat((x, x_skip), 1)
        x = functional.interpolate(x, mode='nearest', scale_factor=2)
        x = self.reflection_pad(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if not self.final:
            out = self.relu(out)

        return out


class SubPixelUpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final=False):
        super(SubPixelUpsamplingBlock, self).__init__()
        shuffle_channels = out_channels if not final else in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, shuffle_channels * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(shuffle_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)
        self.final = final

    def forward(self, x, x_skip=None, hint=None):
        if x_skip is not None:
            x = torch.cat((x, x_skip), 1)
        out = self.relu(self.conv1(x))
        out = self.upsample(out)
        out = self.conv2(out)

        if not self.final:
            out = self.relu(out)
        return out

class AttentionUpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpsamplingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(2)
        self.attn_layer = Attention(out_channels, 512, mode='channel')
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_skip, hint=None):
        x = torch.cat((x, x_skip), 1)
        out = self.relu(self.conv1(x))
        out = self.upsample(out)
        out = out +  out * self.sigmoid(self.attn_layer(out.mean(dim=[2, 3], keepdims=True), hint))
        out = self.relu(self.conv2(out))
        return out