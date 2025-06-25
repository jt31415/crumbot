import config
config.configure_logging()

import asyncio
from pipeline.pipeline import run_mic

if __name__ == "__main__":
    try:
        asyncio.run(run_mic())
    except KeyboardInterrupt:
        print("\nExiting...")
