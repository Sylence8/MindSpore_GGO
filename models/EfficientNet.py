import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import Tensor
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv3d,
    get_model_params,
    efficientnet_params,
    Swish,
    MemoryEfficientSwish,
)

class MBConvBlock3D(nn.Cell):
    def __init__(self, block_args, global_params):
        super(MBConvBlock3D, self).__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, pad_mode='same', has_bias=False)
            self._bn0 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, epsilon=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv3d(
            in_channels=oup, out_channels=oup, group=oup, kernel_size=k, stride=s, pad_mode='same', has_bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, epsilon=self._bn_eps)

        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv3d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv3d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = Conv3d(in_channels=oup, out_channels=final_oup, kernel_size=1, pad_mode='same', has_bias=False)
        self._bn2 = nn.BatchNorm3d(num_features=final_oup, momentum=self._bn_mom, epsilon=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def construct(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = ops.ReduceMean(keep_dims=True)(x, (2, 3, 4))
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = ops.Sigmoid()(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet3D(nn.Cell):
    def __init__(self, blocks_args=None, global_params=None, in_channels=3):
        super(EfficientNet3D, self).__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=2, pad_mode='same', has_bias=False)
        self._bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, epsilon=bn_eps)

        self._blocks = nn.CellList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, pad_mode='same', has_bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, epsilon=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Dense(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def construct(self, inputs):
        bs = inputs.shape[0]
        x = self.extract_features(inputs)

        if self._global_params.include_top:
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import Tensor
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv3d,
    get_model_params,
    efficientnet_params,
    Swish,
    MemoryEfficientSwish,
)

class MBConvBlock3D(nn.Cell):
    def __init__(self, block_args, global_params):
        super(MBConvBlock3D, self).__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, pad_mode='same', has_bias=False)
            self._bn0 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, epsilon=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv3d(
            in_channels=oup, out_channels=oup, group=oup, kernel_size=k, stride=s, pad_mode='same', has_bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_mom, epsilon=self._bn_eps)

        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv3d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv3d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = Conv3d(in_channels=oup, out_channels=final_oup, kernel_size=1, pad_mode='same', has_bias=False)
        self._bn2 = nn.BatchNorm3d(num_features=final_oup, momentum=self._bn_mom, epsilon=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def construct(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = ops.ReduceMean(keep_dims=True)(x, (2, 3, 4))
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = ops.Sigmoid()(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet3D(nn.Cell):
    def __init__(self, blocks_args=None, global_params=None, in_channels=3):
        super(EfficientNet3D, self).__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=2, pad_mode='same', has_bias=False)
        self._bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, epsilon=bn_eps)

        self._blocks = nn.CellList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, pad_mode='same', has_bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=out_channels, momentum=bn_mom, epsilon=bn_eps)

        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Dense(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def construct(self, inputs):
        bs = inputs.shape[0]
        x = self.extract_features(inputs)

        if self._global_params.include_top:
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
