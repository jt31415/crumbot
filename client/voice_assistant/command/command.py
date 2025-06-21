import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from collections.abc import Callable
from pathlib import Path
import importlib

logger = logging.getLogger(__name__)


class OllamaCommandProcessor:
    def __init__(self, model_name: str):
        """
        Initialize the OllamaCommandProcessor with a specified model name.

        :param model_name: The name of the Ollama model to use for command processing.
        """
        self.model = ChatOllama(model=model_name)
        self._bind_tools()

    def _discover_tools(self) -> list[Callable]:
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
    
    def _bind_tools(self):
        """
        Bind the necessary tools to the Ollama model for command processing.
        This method can be extended to include more tools as needed.
        """
        self.model = self.model.bind_tools(self._discover_tools())

    def process_prompt(self, prompt: str) -> str:
        """
        Process the given prompt string to execute commands.

        :param prompt: The command prompt string to process.
        :return: A string indicating the result of the command execution.
        """

        # TODO: use langgraph's create_react_agent and memory with InMemorySaver, as well as using a ChatPromptTemplate

        logger.info(f"Processing prompt.")

        messages = [
            SystemMessage(content="You are a helpful voice assistant/agent named 'Crumbot'. You have several tools that allow you to control the user's computer. If a user asks to do a task, respond with result of the task briefly. If they ask a question, then answer the question. Keep responses as brief as possible, and don't use any markdown."),
            HumanMessage(prompt)
        ]
        
        done = False

        while not done:
            ai_msg = self.model.invoke(messages, think=False)
            messages.append(ai_msg)

            for tool_call in ai_msg.tool_calls:
                selected_tool = self.tools[tool_call["name"].lower()]
                tool_msg = selected_tool.invoke(tool_call)
                messages.append(tool_msg)

            if not ai_msg.tool_calls:
                done = True

        logger.info(f"Done processing.")

        return messages[-1].content