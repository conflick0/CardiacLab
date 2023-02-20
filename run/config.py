import os.path
import tomli

file_path = os.path.join('config.toml')

with open(file_path, "rb") as f:
    config = tomli.load(f)
