import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_script(script_name: str):
    """
    Run a PowerShell script located in the scripts directory.
    :param script_name: The name of the script to run.
    """
    script_path = Path(__file__).parent / "scripts" / script_name
    if script_path.exists():
        try:
            subprocess.run(["powershell", script_path], shell=True, check=True)
            logger.info(f"Successfully ran script: {script_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running script {script_name}: {e}")
    else:
        logger.error(f"Script {script_name} does not exist at path: {script_path}")