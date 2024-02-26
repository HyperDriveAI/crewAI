from typing import ClassVar

from langchain.prompts import PromptTemplate, BasePromptTemplate
from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent with support for different languages."""

    i18n: I18N = Field(default=I18N())

    SCRATCHPAD_SLICE: ClassVar[str] = "\n{agent_scratchpad}"

    def task_execution_with_memory(self) -> BasePromptTemplate:
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
        return self._build_prompt(["role_playing", "tools", "memory", "task"])

    def task_execution_without_tools(self) -> BasePromptTemplate:
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
        return self._build_prompt(["role_playing", "task"])

    def task_execution(self) -> BasePromptTemplate:
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
        return self._build_prompt(["role_playing", "tools", "task"])

    def _build_prompt(self, components: list[str]) -> BasePromptTemplate:
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
        prompt_parts = [self.i18n.slice(component) for component in components]
        prompt_parts.append(self.SCRATCHPAD_SLICE)
        return PromptTemplate.from_template("".join(prompt_parts))
