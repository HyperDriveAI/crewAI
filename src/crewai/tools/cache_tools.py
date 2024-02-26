from langchain.tools import Tool
from pydantic import BaseModel, ConfigDict, Field

from crewai.agents.cache import CacheHandler


class CacheTools(BaseModel):
    """Default tools to hit the cache."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(
        description="Cache Handler for the crew",
        default=CacheHandler(),
    )

    def tool(self):
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

        return Tool.from_function(
            func=self.hit_cache,
            name=self.name,
            description="Reads directly from the cache",
        )

    def hit_cache(self, key):
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

        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)
