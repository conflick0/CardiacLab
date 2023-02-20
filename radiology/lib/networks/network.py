from monai.networks.nets import SwinUNETR,UNETR
from lib.networks.unetcnx_x0 import UNETCNX_X0


def network(model_name, in_channels, out_channels, inp_size=(96, 96, 96)):
    print(f'model: {model_name}')

    if model_name == 'unetcnx_x0':
        return UNETCNX_X0(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=24,
            patch_size=2
        )
    elif model_name == 'swinunetr':
        return SwinUNETR(
            img_size=inp_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            use_checkpoint=True,
        )
    elif model_name == 'unetr':
        return UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=inp_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        )
    else:
        raise ValueError(f'not found model name: {model_name}')
