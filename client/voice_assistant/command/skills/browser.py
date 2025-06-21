from langchain_core.tools import tool
import webbrowser
import logging
from typing import Annotated

logger = logging.getLogger(__name__)

@tool
def google_search(
    query: Annotated[str, "Query to look up"]
):
    """
    Open Google in a web browser with a specific query, does not return any result.
    """

    webbrowser.open(f"https://www.google.com/search?q={query}")
    logger.info(f"Searching for: {query}")