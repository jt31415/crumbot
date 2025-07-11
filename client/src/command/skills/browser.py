from langchain_core.tools import tool
import webbrowser
import logging
from typing import Annotated

logger = logging.getLogger(__name__)

@tool
def open_in_browser(
    query: Annotated[str, "Query to look up"]
):
    """
    Open Google in a web browser with a specific query, does not return any result.
    Only use this tool if user explicitly asks to lookup something in a browser.
    """

    webbrowser.open(f"https://www.google.com/search?q={query}")
    logger.info(f"Searching for: {query}")
    return True