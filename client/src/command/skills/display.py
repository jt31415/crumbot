from langchain_core.tools import tool
from utils import display

@tool
def display_on():
    """Turns on the display. Should be called whenever the user mentions 'display'."""
    display.display_on()
    return True

@tool
def display_off():
    """
    Turns off the display.
    If the prompt contains the term 'playoff', use this function, since it's probably a speech-to-text mistake.
    """
    display.display_off()
    return True