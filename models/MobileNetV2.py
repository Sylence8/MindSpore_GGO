import mindspore.nn as nn
from mindspore import context
from mindspore.common.initializer import Normal
import mindspore.ops.operations as P

def conv_bn(inp, oup, stride):
    return nn.SequentialCell([
        nn.Conv3d(in_channels=inp, out_channels=oup, kernel_size=(3, 3, 3), stride=stride, pad_mode="same", padding=1, has_bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6()
    ])

def conv_1x1x1_bn(inp, oup):
    return nn.SequentialCell([
        nn.Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, has_bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6()
    ])

class InvertedResidual(nn.Cell):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup
        hidden_dim = round(inp * expand_ratio)

        if expand_ratio == 1:
            self.conv = nn.SequentialCell([
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, group=hidden_dim, has_bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, has_bias=False),
                nn.BatchNorm3d(oup),
            ])
        else:
            self.conv = nn.SequentialCell([
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, has_bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(),
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, group=hidden_dim, has_bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(),
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, has_bias=False),
                nn.BatchNorm3d(oup),
            ])

    def construct(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Cell):
    def __init__(self, num_classes=1000, sample_size=224, width_mult=1.0, in_channels=3):
        super(MobileNetV2, self).__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1,  16, 1, (1,1,1)],
            [6,  24, 2, (2,2,2)],
            [6,  32, 3, (2,2,2)],
            [6,  64, 4, (2,2,2)],
            [6,  96, 3, (1,1,1)],
            [6, 160, 3, (2,2,2)],
            [6, 320, 1, (1,1,1)],
        ]

        assert sample_size % 16 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(in_channels, input_channel, (1, 2, 2))]
        
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.features = nn.SequentialCell(self.features)

        self.classifier = nn.SequentialCell([
            nn.Dropout(0.2),
            nn.Dense(self.last_channel, num_classes),
        ])

        self._initialize_weights()

    def construct(self, x):
        x = self.features(x)
        x = P.ReduceMean(keep_dims=False)(x, (-3, -2, -1))
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.get_parameters():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.default_input = Normal(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.default_input = 0
            elif isinstance(m, nn.BatchNorm3d):
                m.gamma.default_input = 1
                m.beta.default_input = 0
            elif isinstance(m, nn.Dense):
                m.weight.default_input = Normal(0, 0.01)
                m.bias.default_input = 0

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.get_parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.name_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'learning_rate': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

    
def get_model(**kwargs):
    model = MobileNetV2(**kwargs)
    return model

if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    model = get_model(num_classes=600, sample_size=112, width_mult=1.0, in_channels=3)
    print(model)

    input_data = Tensor(np.random.randn(8, 3, 16, 112, 112).astype(np.float32))
    output = model(input_data)
    print(output.shape)
