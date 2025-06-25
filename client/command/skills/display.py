from langchain_core.tools import tool
from utils import display

@tool
def display_on():
    """Turns on the display. Initially, the display is off, so this should be called if the user says, 'display'."""
    display.display_on()
    return True

@tool
def display_off():
    """Turns off the display."""
    display.display_off()
    return True