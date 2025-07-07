from pathlib import Path
import tomllib

def get_config():
    config_path = Path(__file__).parent.parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)["crumbot"]