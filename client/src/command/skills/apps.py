from langchain_core.tools import tool
from command.powershell import run_script

@tool
def launch_minecraft():
    """
    Launches Minecraft. Only call this tool if the user explicitly mentions Minecraft.
    """
    run_script("minecraft.ps1")
    return True

@tool
def launch_discord():
    """
    Launches Discord.
    """
    run_script("discord.ps1")
    return True