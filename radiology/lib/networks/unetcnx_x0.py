import warnings
from typing import Optional, Callable, List, Any, Sequence
from functools import partial

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.convnext import ConvNeXt, CNBlockConfig
from torchvision.ops.stochastic_depth import StochasticDepth

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock, UnetrPrUpBlock


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
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

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv3dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution3d-Normalization-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input video.
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm3d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm3d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
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
            torch.nn.Conv3d,
        )


class LayerNorm3d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class CN3DBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 4, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 4, 1, 2, 3]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class UNETCNX_X0(ConvNeXt):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        patch_size=4,
        spatial_dims=3,
        norm_name='instance',
        stochastic_depth_prob: float=0.4,
        layer_scale: float=1e-6,
        num_classes: int=1000,
        block: Optional[Callable[..., nn.Module]]=CN3DBlock,
        norm_layer: Optional[Callable[..., nn.Module]]=None,
        **kwargs: Any
    ) -> None:
        block_setting: List[CNBlockConfig] = [
            CNBlockConfig(feature_size * 2, feature_size * 4, 3),
            CNBlockConfig(feature_size * 4, feature_size * 8, 3),
            CNBlockConfig(feature_size * 8, feature_size * 16, 9),
            CNBlockConfig(feature_size * 16, feature_size * 32, 3)
        ]

        if norm_layer is None:
            norm_layer = partial(LayerNorm3d, eps=1e-6)

        super().__init__(
            block_setting,
            stochastic_depth_prob,
            layer_scale,
            num_classes,
            block,
            norm_layer
        )

        firstconv_output_channels = block_setting[0].input_channels
        self.features[0] = Conv3dNormActivation(
            in_channels,
            firstconv_output_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            norm_layer=norm_layer,
            activation_layer=None,
            bias=True,
        )


        self.dec_features = []
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for block_num, cnf in enumerate(block_setting, 1):
            # mdf orginal 2d downsampling to 3d downsampling
            if cnf.out_channels is not None:
                self.features[block_num * 2] =nn.Sequential(
                    norm_layer(cnf.input_channels),
                    nn.Conv3d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                )

        
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.inc = self.features[0]

        self.encoder1 = nn.Sequential(
            self.features[1],
            self.features[2],
        )

        self.encoder2 = nn.Sequential(
            self.features[3],
            self.features[4],
        )

        self.encoder3 = nn.Sequential(
            self.features[5],
            self.features[6],
        )

        self.encoder4 = nn.Sequential(
            self.features[7],
            self.features[8]
        )

        self.encoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 32,
            out_channels=feature_size * 16,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)


    def forward(self, x):
        enc0 = self.encoder0(x)

        inc = self.inc(x)
        enc1 = self.encoder1(inc)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        dec4 = self.encoder5(enc4, enc3)

        dec3 = self.decoder4(dec4, enc2)
        dec2 = self.decoder3(dec3, enc1)
        dec1 = self.decoder2(dec2, inc)
        dec0 = self.decoder1(dec1, enc0)
        out = self.out(dec0)
        return out

    def encode(self, x):
        inc = self.inc(x)
        enc1 = self.encoder1(inc)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        return enc4

        