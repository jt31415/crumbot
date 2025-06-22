import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from pathlib import Path
import importlib
from typing import Iterator, Generator, Any

logger = logging.getLogger(__name__)

def strip_thinking(message: str) -> str:
    """
    Strip the thinking part from an LLM response.
    
    :param message: The message string to process.
    :return: The processed message without thinking.
    """

    if "<think>" in message:
        start = message.index("<think>")
        if "</think>" not in message:
            end = start + len("<think>")
        else:
            end = message.index("</think>") + len("</think>")
            
        return (message[:start] + message[end:]).strip()
    
    return message.strip()

class OllamaCommandProcessor:
    def __init__(self, model_name: str):
        """
        Initialize the OllamaCommandProcessor with a specified model name.

        :param model_name: The name of the Ollama model to use for command processing.
        """
        self.model = ChatOllama(model=model_name)
        self._create_agent()

    def _discover_tools(self) -> list[BaseTool]:
        """
        Discover and return the tools available for command processing.
        This method can be extended to include more tools as needed.

        :return: A list of callable tools for command processing.
        """

        discovered_tools = {}
        
        skills_dir = Path(__file__).parent / "skills"
        for skill in skills_dir.glob("*.py"):
            if skill == "__init__.py":
                continue
            
            module = importlib.import_module(f".skills.{skill.stem}", package="voice_assistant.command")

            for name, val in module.__dict__.items():
                if isinstance(val, BaseTool):
                    discovered_tools[name] = val

        self.tools = discovered_tools

        return discovered_tools.values()
    
    def _create_agent(self):
        """
        Bind the necessary tools to the Ollama model for command processing.
        This method can be extended to include more tools as needed.
        """
        tools = self._discover_tools()
        checkpointer = InMemorySaver()
        prompt = open(Path(__file__).parent / "system_prompt").read()

        self.agent = create_react_agent(
            model=self.model,
            tools=tools,
            checkpointer=checkpointer,
            prompt=prompt
        )

    def _agent_stream(self, it: Iterator[dict[str, Any]]) -> Generator[str, None, None]:
        """
        Process the stream of messages from the agent and yield the LLM's response.
        
        :param it: The stream of updates from the agent.
        :return: A generator yielding the LLM's response.
        """
        try:
            for update in it:
                for key in update:
                    if "messages" in update[key]:
                        for message in update[key]["messages"]:
                            if isinstance(message, AIMessage):
                                yield strip_thinking(message.content)
        except GraphRecursionError as e:
            logger.error(f"Graph recursion error: {e}")
            yield "An error occurred while processing the command. Please try again."

    def process_prompt(self, prompt: str) -> Generator[str, None, None]:
        """
        Process the given prompt string to execute commands.

        :param prompt: The command prompt string to process.
        :return: A string indicating the result of the command execution.
        """

        logger.info(f"Processing prompt.")

        config = {"configurable": {"thread_id": "default"}}
        response = self.agent.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
            stream_mode="updates"
        )

        yield from self._agent_stream(response)