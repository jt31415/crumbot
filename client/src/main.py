from config import logging_config
logging_config.configure_logging()

import openwakeword
openwakeword.utils.download_models()

import asyncio
from pipeline.pipeline import run_mic

if __name__ == "__main__":
    try:
        asyncio.run(run_mic())
    except KeyboardInterrupt:
        print("\nExiting...")
