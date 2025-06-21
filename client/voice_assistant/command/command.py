import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
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
        self.agent = create_react_agent(
            model=self.model,
            tools=tools,
            checkpointer=checkpointer,
            prompt="You are a helpful voice assistant/agent named 'Crumbot'. You have several tools that allow you to control the user's computer. If a user asks to do a task, respond with result of the task briefly. If they ask a question, then answer the question. If what they are saying doesn't make sense, respond with '<empty>', it was probably a false wakeword activation. Keep responses as brief as possible, and don't use markdown."  # ChatPromptTemplate can be used here if needed
        )

    def process_prompt(self, prompt: str) -> str:
        """
        Process the given prompt string to execute commands.

        :param prompt: The command prompt string to process.
        :return: A string indicating the result of the command execution.
        """

        logger.info(f"Processing prompt.")

        config = {"configurable": {"thread_id": "default"}}
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config
        )

        logger.info(f"Done processing.")

        return response["messages"][-1].content