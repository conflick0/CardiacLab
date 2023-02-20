import os
from pathlib import Path
import gdown


def download_ckp(ckp_pth, id):
    if not os.path.exists(ckp_pth):
        print('download ...')
        model_dir = Path(ckp_pth).parent
        os.makedirs(model_dir, exist_ok=True)

        if id is None:
            raise ValueError(f'Invalid download id: {id} !')

        gdown.download(id=id, output=ckp_pth, quiet=False)
