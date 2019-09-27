import torch
from torch import nn


class DepthSeparableConv2d(nn.Module):
    """Depth separable convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, meta=False):
        super(DepthSeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                          padding, dilation, groups=in_channels, bias=False)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                          padding=0, dilation=1, groups=1, bias=bias)

        self.kernel_size = self.pointwise_conv2d.kernel_size
        self.out_channels = self.pointwise_conv2d.out_channels

        self.weight = self.pointwise_conv2d.weight
        if bias:
            self.bias = self.pointwise_conv2d.bias
        else:
            self.register_parameter('bias', None)

        self.meta = meta
        if self.meta:
            for param in self.pointwise_conv2d.parameters():
                param.requires_grad = False

    def forward(self, x):
        depthwise = self.depthwise_conv2d(x)
        pointwise = self.pointwise_conv2d(depthwise)
        return pointwise


class DepthwiseAndConv2d(nn.Module):
    """Depthwise followed normal convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, meta=False):
        super(DepthwiseAndConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                                          padding=padding, dilation=1, groups=in_channels, bias=False)
        self.normal_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=1, bias=bias)

        self.kernel_size = self.normal_conv2d.kernel_size
        self.out_channels = self.normal_conv2d.out_channels

        self.weight = self.normal_conv2d.weight
        if bias:
            self.bias = self.normal_conv2d.bias
        else:
            self.register_parameter('bias', None)

        self.meta = meta
        if self.meta:
            for param in self.normal_conv2d.parameters():
                param.requires_grad = False

    def forward(self, x):
        depthwise = self.depthwise_conv2d(x)
        out = self.normal_conv2d(depthwise)
        return out


class ShiftScaleConv2d(nn.Module):
    """Shift and scale convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, meta=False):
        super(ShiftScaleConv2d, self).__init__()
        self.normal_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=1, bias=bias)
        self.scale = nn.Parameter(torch.ones(size=(out_channels, 1, 1), requires_grad=True))
        self.shift = nn.Parameter(torch.zeros(size=(out_channels, 1, 1), requires_grad=True))

        self.kernel_size = self.normal_conv2d.kernel_size
        self.out_channels = self.normal_conv2d.out_channels

        self.weight = self.normal_conv2d.weight
        if bias:
            self.bias = self.normal_conv2d.bias
        else:
            self.register_parameter('bias', None)

        self.meta = meta
        if self.meta:
            for param in self.normal_conv2d.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.normal_conv2d(x)
        x = self.scale * x + self.shift
        return x


def conv3x3(Conv, in_planes, out_planes, stride=1, meta=False):
    """3x3 convolution with padding and specific conv type"""
    return Conv(in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                meta=meta)


def depth_separable_conv3x3(in_planes, out_planes, stride=1, meta=False):
    """3x3 convolution with padding and specific DepthSeparableConv2d type"""

    return conv3x3(DepthSeparableConv2d, in_planes, out_planes, stride,
                   meta=meta)


def depthwise_and_conv3x3(in_planes, out_planes, stride=1, meta=False):
    """3x3 convolution with padding and specific DepthwiseAndConv2d type"""

    return conv3x3(DepthwiseAndConv2d, in_planes, out_planes, stride,
                   meta=meta)


def ss_conv3x3(in_planes, out_planes, stride=1, meta=False):
    """3x3 convolution with padding and specific ShiftScaleConv2d type"""

    return conv3x3(ShiftScaleConv2d, in_planes, out_planes, stride,
                   meta=meta)


if __name__ == '__main__':
    batch_size = 5
    x = torch.rand(batch_size, 3, 80, 80)
    dc = depth_separable_conv3x3(3, 64, meta=False)
    dac = depthwise_and_conv3x3(3, 64, meta=False)
    ssc = ss_conv3x3(3, 64, meta=False)


    def print_info(x, net):
        print('x -> ', net(x).size())
        for para in net.parameters():
            print(para.size(), ' req grad ->', para.requires_grad)
        print()


    print_info(x, dc)
    print_info(x, dac)
    print_info(x, ssc)
