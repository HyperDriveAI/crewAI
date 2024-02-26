from typing import Any, Dict

from langchain.callbacks.base import BaseCallbackHandler

from ..tools.cache_tools import CacheTools
from .cache.cache_handler import CacheHandler


class ToolsHandler(BaseCallbackHandler):
    """Callback handler for tool usage."""

    last_used_tool: Dict[str, Any] = {}
    cache: CacheHandler

    def __init__(self, cache: CacheHandler, **kwargs: Any):
        """        Save the processed files map to a JSON file.

        Function parameters should be documented in the ``Args`` section. The name
        of each parameter is required. The type and description of each parameter
        is optional, but should be included if not obvious.

        Args:
            dictionary (dict): The processed files map.

        Returns:
            bool: True if successful, False otherwise.
                The return type is optional and may be specified at the beginning of
                the ``Returns`` section followed by a colon.
                
                The ``Returns`` section may span multiple lines and paragraphs.
                Following lines should be indented to match the first line.
                
                The ``Returns`` section supports any reStructuredText formatting,
                including literal blocks::
                
                    {
                        'param1': param1,
                        'param2': param2
                    }
        """
        self.cache = cache
        super().__init__(**kwargs)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """        Save the processed files map to a JSON file.

        Function parameters should be documented in the ``Args`` section. The name
        of each parameter is required. The type and description of each parameter
        is optional, but should be included if not obvious.

        Args:
            dictionary (dict): The processed files map.

        Returns:
            bool: True if successful, False otherwise.
                The return type is optional and may be specified at the beginning of
                the ``Returns`` section followed by a colon.
                
                The ``Returns`` section may span multiple lines and paragraphs.
                Following lines should be indented to match the first line.
                
                The ``Returns`` section supports any reStructuredText formatting,
                including literal blocks::
                
                    {
                        'param1': param1,
                        'param2': param2
                    }
        """
        name = serialized.get("name")
        if name not in ["invalid_tool", "_Exception"]:
            tools_usage = {
                "tool": name,
                "input": input_str,
            }
            self.last_used_tool = tools_usage

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """        Save the processed files map to a JSON file.

        Function parameters should be documented in the ``Args`` section. The name
        of each parameter is required. The type and description of each parameter
        is optional, but should be included if not obvious.

        Args:
            dictionary (dict): The processed files map.

        Returns:
            bool: True if successful, False otherwise.
                The return type is optional and may be specified at the beginning of
                the ``Returns`` section followed by a colon.
                
                The ``Returns`` section may span multiple lines and paragraphs.
                Following lines should be indented to match the first line.
                
                The ``Returns`` section supports any reStructuredText formatting,
                including literal blocks::
                
                    {
                        'param1': param1,
                        'param2': param2
                    }
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
