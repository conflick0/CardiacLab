import os
from pathlib import Path
from config import config

ROOT_DIR = Path(__file__).parent.parent

config = config['segmentation_cardiac'][config['model']]

os.system(
f'monailabel start_server \
--app {os.path.join(ROOT_DIR, config["app"])} \
--studies {config["studies"]} \
--conf models {config["models"]} \
--conf network {config["network"]} \
--conf download_ckp_id {config["download_ckp_id"]} \
--conf target_spacing "{config["target_spacing"]}" \
--conf spatial_size "{config["spatial_size"]}" \
--conf intensity "{config["intensity"]}"'
)

