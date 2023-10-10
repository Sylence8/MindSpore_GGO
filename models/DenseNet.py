import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU())
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, pad_mode='same', has_bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU())
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, pad_mode='same', has_bias=False))
        self.drop_rate = drop_rate

    def construct(self, x):
        new_features = super().construct(x)
        if self.drop_rate > 0:
            new_features = ops.Dropout()(new_features)
        return ops.Concat(1)((x, new_features))

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer{}'.format(i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, pad_mode='same', has_bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseNet(nn.Cell):
    def __init__(self, num_classes=1000, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, drop_rate=0, n_input_channels=1):
        super(DenseNet, self).__init__()

        self.features = []
        self.features.append(('conv1', nn.Conv3d(n_input_channels, num_init_features, kernel_size=(7, 7, 7), stride=(1, 2, 2), pad_mode='same', has_bias=False)))
        self.features.append(('norm1', nn.BatchNorm3d(num_init_features)))
        self.features.append(('relu1', nn.ReLU()))
        self.features.append(('pool1', nn.MaxPool3d(kernel_size=3, stride=2, pad_mode='same')))
        self.features = nn.SequentialCell(self.features)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        self.features.add(nn.BatchNorm3d(num_features))
        self.features.add(nn.ReLU())

        self.classifier = nn.Dense(num_features, num_classes)

        for module in self.features:
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif isinstance(module, nn.BatchNorm3d):
                module.gamma.set_value(1)
                module.beta.set_value(0)
        
        for module in self.classifier:
            if isinstance(module, nn.Dense):
                module.bias.set_value(0)

    def construct(self, x):
        x = self.features(x)
        x = ops.Reshape()(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x
