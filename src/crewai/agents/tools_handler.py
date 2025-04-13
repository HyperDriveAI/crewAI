from typing import Any, Dict

from langchain.callbacks.base import BaseCallbackHandler

from ..tools.cache_tools import CacheTools
from .cache.cache_handler import CacheHandler


class ToolsHandler(BaseCallbackHandler):
    """Callback handler for tool usage."""

    last_used_tool: Dict[str, Any] = {}
    cache: CacheHandler

    def __init__(self, cache: CacheHandler, **kwargs: Any):
        """Initialize the callback handler."""
        self.cache = cache
        super().__init__(**kwargs)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when a tool starts running.

        This function is called at the beginning of a tool's execution. It
        extracts the name of the tool from the serialized data and checks if it
        is either "invalid_tool" or "_Exception". If not, it logs the tool usage
        by storing the tool name and input string in `self.last_used_tool`.

        Args:
            serialized (Dict[str, Any]): A dictionary containing serialized information about the tool.
            input_str (str): The input string provided to the tool.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of running the tool start process, which is typically `None`
                or a specific value depending on the implementation.
        """
        name = serialized.get("name")
        if name not in ["invalid_tool", "_Exception"]:
            tools_usage = {
                "tool": name,
                "input": input_str,
            }
            self.last_used_tool = tools_usage

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when a tool ends running.

        Checks if the output contains specific error messages. If not, it caches
        the tool, input, and output if the last used tool is different from the
        current one.

        Args:
            output (str): The output string of the tool.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: The result of the function.
        """
        if (
            "is not a valid tool" not in output
            and "Invalid or incomplete response" not in output
            and "Invalid Format" not in output
        ):
            if self.last_used_tool["tool"] != CacheTools().name:
                self.cache.add(
                    tool=self.last_used_tool["tool"],
                    input=self.last_used_tool["input"],
                    output=output,
                )
