import os
from pathlib import Path
import tomli

ROOT_DIR = Path(__file__).parent.parent

file_path = os.path.join(ROOT_DIR, 'run', 'config.toml')

with open(file_path, "rb") as f:
    config = tomli.load(f)
