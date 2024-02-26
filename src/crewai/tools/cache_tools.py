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
        """        Returns a Tool object created from the function.

        This function returns a Tool object created from the function 'hit_cache' with the specified name and description.
        The Tool object represents reading directly from the cache.

        Returns:
            Tool: A Tool object created from the function.
        """

        return Tool.from_function(
            func=self.hit_cache,
            name=self.name,
            description="Reads directly from the cache",
        )

    def hit_cache(self, key):
        """        Read data from cache based on the provided key.

        Args:
            key (str): The key used to retrieve data from the cache.

        Returns:
            The data retrieved from the cache based on the provided key.
        """

        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)
