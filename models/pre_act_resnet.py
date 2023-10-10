import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=(3, 3, 3),
        stride=stride,
        padding=1,
        pad_mode="pad",
        has_bias=False
    )

def downsample_basic_block(x, planes, stride):
    avg_pool3d = ops.AvgPool3d(kernel_size=1, stride=stride, pad_mode="valid")
    out = avg_pool3d(x)
    zero_pads = nn.ZerosLike()(out)
    out = ops.Concat(1)((out, zero_pads))
    return out

class PreActivationBasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out

class PreActivationBottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActivationBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, has_bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, has_bias=False)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return out

class PreActivationResNet(nn.Cell):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 channels,
                 shortcut_type='B',
                 num_classes=400):
        super(PreActivationResNet, self).__init__()
        self.inplanes = 64
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

        for m in self.get_parameters():
            if isinstance(m, nn.Conv3d):
                m.weight = Normal()(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.gamma.default_input = 1
                m.beta.default_input = 0

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.SequentialCell([
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        has_bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
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
        return model.get_parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.name_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'learning_rate': 0.0})

    return parameters

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = PreActivationResNet(PreActivationBasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = PreActivationResNet(PreActivationBasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-200 model.
    """
    model = PreActivationResNet(PreActivationBottleneck, [3, 24, 36, 3], **kwargs)
    return model
