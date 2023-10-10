import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as weight_init
from functools import partial
import math

class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), has_bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), has_bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), has_bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), has_bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(out_channels, out_channels * 4, kernel_size=(1, 1, 1), has_bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
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

        out = out + residual
        out = self.relu(out)

        return out

def conv3x3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), has_bias=False)

def conv1x1x1(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=stride, has_bias=False)

def downsample_basic_block(x, out_channels, stride):
    out = ops.AvgPool3d(kernel_size=1, stride=stride)(x)
    zero_pads = ops.Pad(paddings=((0, 0), (0, out_channels - x.shape[1]), (0, 0), (0, 0), (0, 0)))(out)
    return zero_pads

class ResNet(nn.Cell):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 channels,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            channels,
            64,
            kernel_size=(7, 7, 7),
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            has_bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = max(int(math.ceil(sample_duration / 16)), 1)
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Dense(512 * block.expansion, num_classes)

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv3d):
                cell.weight = weight_init.initializer(weight_init.XavierUniform(), cell.weight.shape)
            elif isinstance(cell, nn.BatchNorm3d):
                cell.gamma.set_data('ones')
                cell.beta.set_data('zeros')

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    out_channels=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.SequentialCell(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=(1, 1, 1),
                        stride=(stride, stride, stride),
                        has_bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.cells_and_names():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def resnet10(**kwargs):
    """Constructs a ResNet-10 model."""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-152 model."""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-200 model."""
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
