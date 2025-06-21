from langchain_core.tools import tool
from voice_assistant import display

@tool
def display_on():
    """Turns on the display."""
    display.display_on()

@tool
def display_off():
    """Turns off the display."""
    display.display_off()