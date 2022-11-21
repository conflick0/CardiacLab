from networks.unetcnx import UNETCNX


def network(model_name, in_channels, out_channels):
    print(f'model: {model_name}')

    if model_name == 'unetcnx':
        return UNETCNX(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            patch_size=4
        )
    else:
        raise ValueError(f'not found model name: {model_name}')