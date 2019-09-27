import torch.nn as nn
import math

from models.ops import depth_separable_conv3x3
from models.ops import depthwise_and_conv3x3
from models.ops import ss_conv3x3
from models.ops import DepthSeparableConv2d
from models.ops import DepthwiseAndConv2d
from models.ops import ShiftScaleConv2d

__all__ = ['get_network', 'get_network_outputs_dim']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def disable_gradient(param_list):
    if not param_list:
        return
    for param in param_list:
        if param is not None:
            for p in param.parameters():
                p.requires_grad = False


class Block(nn.Module):
    expansion = 1

    def __init__(self, conv_func, inplanes, planes, stride=1, downsample=None, meta=False):
        super(Block, self).__init__()
        if conv_func != conv3x3:
            self.conv1 = conv_func(inplanes, planes, stride, meta=meta)
        else:
            self.conv1 = conv_func(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if conv_func != conv3x3:
            self.conv2 = conv_func(planes, planes, meta=meta)
        else:
            self.conv2 = conv_func(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.meta = meta
        if self.meta:
            self._disable_gradients()

    def _disable_gradients(self):
        to_be_disables = []
        if isinstance(self.conv1, nn.Conv2d):
            to_be_disables.append(self.conv1)
        if isinstance(self.conv2, nn.Conv2d):
            to_be_disables.append(self.conv2)

        to_be_disables.extend([self.bn1, self.bn2, self.downsample])
        disable_gradient(to_be_disables)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(Block):
    expansion = 1

    def __init__(self, *args, **kwargs):
        super(BasicBlock, self).__init__(conv3x3, *args, **kwargs)


class DSBlock(Block):
    expansion = 1

    def __init__(self, *args, **kwargs):
        super(DSBlock, self).__init__(depth_separable_conv3x3, *args, **kwargs)


class DABlock(Block):
    expansion = 1

    def __init__(self, *args, **kwargs):
        super(DABlock, self).__init__(depthwise_and_conv3x3, *args, **kwargs)


class SSBlock(Block):
    expansion = 1

    def __init__(self, *args, **kwargs):
        super(SSBlock, self).__init__(ss_conv3x3, *args, **kwargs)


class Neck(nn.Module):
    expansion = 4

    def __init__(self, Conv2d, inplanes, planes, stride=1, downsample=None, meta=False):
        super(Neck, self).__init__()
        if Conv2d != nn.Conv2d:
            self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False, meta=meta)
        else:
            self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if Conv2d != nn.Conv2d:
            self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False, meta=meta)
        else:
            self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if Conv2d != nn.Conv2d:
            self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False, meta=meta)
        else:
            self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.meta = meta

        if self.meta:
            self._disable_gradients()

    def _disable_gradients(self):
        to_be_disables = []
        if isinstance(self.conv1, nn.Conv2d):
            to_be_disables.append(self.conv1)
        if isinstance(self.conv2, nn.Conv2d):
            to_be_disables.append(self.conv2)
        if isinstance(self.conv3, nn.Conv2d):
            to_be_disables.append(self.conv3)

        to_be_disables.extend([self.bn1, self.bn2, self.bn3, self.downsample])
        disable_gradient(to_be_disables)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(Neck):
    expansion = 4

    def __init__(self, *args, **kwargs):
        super(Bottleneck, self).__init__(nn.Conv2d, *args, **kwargs)


class DSBottleneck(Neck):
    expansion = 4

    def __init__(self, *args, **kwargs):
        super(DSBottleneck, self).__init__(DepthSeparableConv2d, *args, **kwargs)


class DABottleneck(Neck):
    expansion = 4

    def __init__(self, *args, **kwargs):
        super(DABottleneck, self).__init__(DepthwiseAndConv2d, *args, **kwargs)


class SSBottleneck(Neck):
    expansion = 4

    def __init__(self, *args, **kwargs):
        super(SSBottleneck, self).__init__(ShiftScaleConv2d, *args, **kwargs)


class MiniResNet(nn.Module):

    def __init__(self, blocks, layers, num_classes=1000, meta=False, return_base=False):
        assert len(blocks) == 4, 'expect the blocks as 4, but got `%d`' % len(blocks)
        self._assert_expansions_equal(blocks)
        self.inplanes = 64
        self.meta = meta
        self.return_base = return_base
        super(MiniResNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[0], 64, layers[0])
        self.layer2 = self._make_layer(blocks[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(blocks[3], 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.fc = nn.Linear(512 * blocks[-1].expansion, num_classes)

        if self.meta:
            self._disable_gradients()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, DepthSeparableConv2d, DepthwiseAndConv2d, ShiftScaleConv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _disable_gradients(self):
        to_be_disables = [self.conv1, self.bn1, self.fc]
        disable_gradient(to_be_disables)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, meta=self.meta)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, meta=self.meta))

        return nn.Sequential(*layers)

    @staticmethod
    def _assert_expansions_equal(blocks):
        expansions = [block.expansion for block in blocks]
        for i in range(1, len(expansions)):
            if expansions[i] != expansions[i - 1]:
                raise ValueError('expansions `%s` not same for all blocks' % str(expansions))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if not self.return_base:
            x = self.fc(x)

        return x


def mini_resnet12(blocks, meta=False, **kwargs):
    """Constructs a MiniResNet-12 model.

    Args:
        blocks:  blocks to construct the network.
        meta (bool): If True, returns a model in meta mode.
    """
    model = MiniResNet(blocks, [2, 1, 1, 1], meta=meta, **kwargs)
    return model


def mini_resnet18(blocks, meta=False, **kwargs):
    """Constructs a MiniResNet-18 model.

    Args:
        blocks:  blocks to construct the network.
        meta (bool): If True, returns a model in meta mode.
    """
    model = MiniResNet(blocks, [2, 2, 2, 2], meta=meta, **kwargs)
    return model


def mini_resnet34(blocks, meta=False, **kwargs):
    """Constructs a MiniResNet-34 model.

    Args:
        blocks:  blocks to construct the network.
        meta (bool): If True, returns a model in meta mode.
    """
    model = MiniResNet(blocks, [3, 4, 6, 3], meta=meta, **kwargs)
    return model


def mini_resnet50(Bottlenecks, meta=False, **kwargs):
    """Constructs a MiniResNet-50 model.

    Args:
        Bottlenecks: bottlenecks to construct the network.
        meta (bool): If True, returns a model in meta mode.
    """
    model = MiniResNet(Bottlenecks, [3, 4, 6, 3], meta=meta, **kwargs)
    return model


def mini_resnet101(Bottlenecks, meta=False, **kwargs):
    """Constructs a MiniResNet-101 model.

    Args:
        Bottlenecks: bottlenecks to construct the network.
        meta (bool): If True, returns a model in meta mode.
    """
    model = MiniResNet(Bottlenecks, [3, 4, 23, 3], meta=meta, **kwargs)
    return model


def mini_resnet152(Bottlenecks, meta=False, **kwargs):
    """Constructs a MiniResNet-152 model.

    Args:
        Bottlenecks: bottlenecks to construct the network.
        meta (bool): If True, returns a model in meta mode.
    """
    model = MiniResNet(Bottlenecks, [3, 8, 36, 3], meta=meta, **kwargs)
    return model


def mini_ss_resnet12(meta=False, return_base=True, **kwargs):
    """Constructs a MiniResNet-12 model with 2 ss blocks.

        Args:
            return_base:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet12([BasicBlock, BasicBlock, SSBlock, SSBlock], meta=meta, return_base=return_base, **kwargs)


def mini_ss_resnet18(meta=False, return_base=True, **kwargs):
    """Constructs a MiniResNet-18 model with 2 ss blocks.

        Args:
            return_base:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet18([BasicBlock, BasicBlock, SSBlock, SSBlock], meta=meta, return_base=return_base, **kwargs)


def mini_ds_resnet18(meta=False, **kwargs):
    """Constructs a MiniResNet-18 model with 2 ds blocks.

        Args:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet18([BasicBlock, BasicBlock, DSBlock, DSBlock], meta=meta, **kwargs)


def mini_da_resnet18(meta=False, **kwargs):
    """Constructs a MiniResNet-18 model with 2 da blocks.

        Args:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet18([BasicBlock, BasicBlock, DABlock, DABlock], meta=meta, **kwargs)


def mini_ss_resnet34(meta=False, return_base=True, **kwargs):
    """Constructs a MiniResNet-50 model with 2 ss bottlenecks.

        Args:
            return_base:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet34([Bottleneck, Bottleneck, SSBottleneck, SSBottleneck],
                         meta=meta, return_base=return_base, **kwargs)


def mini_ss_resnet50(meta=False, return_base=True, **kwargs):
    """Constructs a MiniResNet-50 model with 2 ss bottlenecks.

        Args:
            return_base:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet50([Bottleneck, Bottleneck, SSBottleneck, SSBottleneck],
                         meta=meta, return_base=return_base, **kwargs)


def mini_ds_resnet50(meta=False, **kwargs):
    """Constructs a MiniResNet-50 model with 2 ds bottlenecks.

        Args:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet50([Bottleneck, Bottleneck, DSBottleneck, DSBottleneck], meta=meta, **kwargs)


def mini_da_resnet50(meta=False, **kwargs):
    """Constructs a MiniResNet-50 model with 2 da bottlenecks.

        Args:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet50([Bottleneck, Bottleneck, DABottleneck, DABottleneck], meta=meta, **kwargs)


def mini_ss_resnet101(meta=False, return_base=True, **kwargs):
    """Constructs a MiniResNet-50 model with 2 ss bottlenecks.

        Args:
            return_base:
            meta (bool): If True, returns a model in meta mode.
    """
    return mini_resnet101([Bottleneck, Bottleneck, SSBottleneck, SSBottleneck],
                          meta=meta, return_base=return_base, **kwargs)


nets = {
    'mini_ss_resnet12': mini_ss_resnet12,
    'mini_ss_resnet18': mini_ss_resnet18,
    'mini_ss_resnet34': mini_ss_resnet34,
    'mini_ss_resnet50': mini_ss_resnet50,
    'mini_ss_resnet101': mini_ss_resnet101,
}

# where using the following mini resnet architecture
# self.conv1: kernel_size=(7x7), padding=3, stride=2
# self.avgpool: kernel_size=(5x5)
# nets_output_dim = {
#     'mini_ss_resnet12': 512,
#     'mini_ss_resnet18': 512,
#     'mini_ss_resnet34': 2048,
#     'mini_ss_resnet50': 2048,
#     'mini_ss_resnet101': 2048,
# }

# where using the following mini resnet architecture
# self.conv1: kernel_size=(3x3), padding=2, stride=1
# self.avgpool: kernel_size=(5x5)
nets_output_dim = {
    'mini_ss_resnet12': 2048,
    'mini_ss_resnet18': 2048,
    'mini_ss_resnet34': 8192,
    'mini_ss_resnet50': 8192,
    'mini_ss_resnet101': 8192,
}


def get_network(name):
    if name not in nets:
        raise ValueError('`%s` network not found.' % name)
    return nets[name]


def get_network_outputs_dim(name):
    if name not in nets_output_dim:
        raise ValueError('`%s` network not found.' % name)
    return nets_output_dim[name]


if __name__ == '__main__':
    test = mini_ss_resnet101(meta=True, num_classes=64)
    import torch

    batch_size = 5
    x = torch.rand(5, 3, 80, 80)
    x = test(x)
    print(x.size())
    # for name, param in test.named_parameters():
    #     print('name ->', name, 'size -> ', param.size(), 'requires_grad->', param.requires_grad)
