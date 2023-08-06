import collections
import copy
import math
from itertools import repeat
from dataclasses import dataclass
from functools import partial

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
import torch.utils.model_zoo as model_zoo

efficientnet_param = {
    "b0": [ 1.0, 1.0, 0.2, 224, None, "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"],
    "b1": [ 1.0, 1.1, 0.2, 240, None, "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth"],
    "b2": [ 1.1, 1.2, 0.3, 288, None, "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth"],
    "b3": [ 1.2, 1.4, 0.3, 300, None, "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth"],
    "b4": [ 1.4, 1.8, 0.4, 380, None, "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth"],
    "b5": [ 1.6, 2.2, 0.4, 456, partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth"],
    "b6": [ 1.8, 2.6, 0.5, 528, partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth"],
    "b7": [ 2.0, 3.1, 0.5, 600, partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth"],
    "v2_s": [ -1.0, -1.0, 0.2, 384, partial(nn.BatchNorm2d, eps=1e-03), "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"],
    "v2_m": [ -1.0, -1.0, 0.3, 512, partial(nn.BatchNorm2d, eps=1e-03), "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth"],
    "v2_l": [ -1.0, -1.0, 0.4, 512, partial(nn.BatchNorm2d, eps=1e-03), "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth"]
}

class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return self.stochastic_depth(input, self.p, self.mode, self.training)

    def stochastic_depth(self, input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
        """
        Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
        <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
        branches of residual architectures.
        Args:
            input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                        being its batch i.e. a batch with ``N`` rows.
            p (float): probability of the input to be zeroed.
            mode (str): ``"batch"`` or ``"row"``.
                        ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                        randomly selected rows from the batch.
            training: apply stochastic depth if is ``True``. Default: ``True``
        Returns:
            Tensor[N, ...]: The randomly zeroed tensor.
        """
        if p < 0.0 or p > 1.0:
            raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
        if mode not in ["batch", "row"]:
            raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
        if not training or p == 0.0:
            return input

        survival_rate = 1.0 - p
        if mode == "row":
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        else:
            size = [1] * input.ndim
        noise = torch.empty(size, dtype=input.dtype, device=input.device)
        noise = noise.bernoulli_(survival_rate)
        if survival_rate > 0.0:
            noise.div_(survival_rate)
        # print('using stochastic depth')
        return input * noise

class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input

class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = self._make_ntuple(kernel_size, _conv_dim)
                dilation = self._make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

    def _make_ntuple(self, x: Any, n: int) -> Tuple[Any, ...]:
        """
        Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
        Otherwise, we will make a tuple of length n, all with value of x.
        reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
        Args:
            x (Any): input value
            n (int): length of the resulting tuple
        """
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))

class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        stochastic_depth_prob: float = 0.2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        """
        EfficientNet V1 and V2 just the backbone

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, 
                norm_layer=norm_layer, activation_layer=nn.SiLU))

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        self.keep_features = []
        self.out_features_size = []
        stride_now_tmp = 1
        last_stride_tmp = 1
        keep_features_tmp = [8, 16]
        for i, cnf in enumerate(inverted_residual_setting):
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)
                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
                stride_now_tmp *= block_cnf.stride
            if last_stride_tmp!=stride_now_tmp and stride_now_tmp in keep_features_tmp:
                self.out_features_size.append(inverted_residual_setting[i - 1].out_channels)
                last_stride_tmp = stride_now_tmp
                self.keep_features.append(i)

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )
        self.out_features_size.append(lastconv_output_channels)

        self.features = nn.Sequential(*layers)

        self.init_model_weights()

    def init_model_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        features = []
        for i, each_block in enumerate(self.features):
            x = each_block(x)
            # print(i, x.size())
            if i in self.keep_features:
                features.append(x)
        features.append(x)
        return features

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    last_channel: Optional[int],
    arch: str,
    pretrainedPath:str,
    **kwargs: Any,) -> EfficientNet:
    model = EfficientNet(inverted_residual_setting, last_channel=last_channel, **kwargs)
    if pretrainedPath is not None:
        the_url = efficientnet_param[arch][5]
        b0_state_dict = model_zoo.load_url(the_url, model_dir=pretrainedPath)
        weights_load = {}
        state = model.state_dict()
        state_keys = list(state.keys())
        for i, the_keys in enumerate(state_keys):#b0_state_dict.keys()):
            if state_keys[i].find('features') == -1:
                continue #ignore the classifier part
            weights_load[state_keys[i]] = b0_state_dict[the_keys]
        state.update(weights_load)
        model.load_state_dict(state)
    return model

def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("b"):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel

def Get_EfficientNet_Backbone(arch, pretrained_path):
    if efficientnet_param[arch][0] != -1.0 or efficientnet_param[arch][1] != -1.0:
        inverted_residual_setting, last_channel = _efficientnet_conf(arch, width_mult=efficientnet_param[arch][0], depth_mult=efficientnet_param[arch][1])
    else:
        inverted_residual_setting, last_channel = _efficientnet_conf(arch)

    return _efficientnet(inverted_residual_setting, last_channel,
                         arch, pretrained_path,
                         norm_layer = efficientnet_param[arch][4]), efficientnet_param[arch][3]

if __name__ == "__main__":
    model_type = 'v2_l'
    model, input_size = Get_EfficientNet_Backbone(model_type, 'weights/')
    fake_input = torch.rand(1, 3, input_size, input_size, requires_grad=True)
    fake_output = model(fake_input)
    print(model_type, input_size, model.out_features_size, fake_output[0].size(), fake_output[1].size(), fake_output[2].size())