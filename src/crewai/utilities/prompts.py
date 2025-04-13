from typing import ClassVar

from langchain.prompts import PromptTemplate, BasePromptTemplate
from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent with support for different languages."""

    i18n: I18N = Field(default=I18N())

    SCRATCHPAD_SLICE: ClassVar[str] = "\n{agent_scratchpad}"

    def task_execution_with_memory(self) -> BasePromptTemplate:
        """Generate a prompt for task execution with memory components.

        This function constructs and returns a prompt template that includes
        role-playing, tools, memory, and task elements. It utilizes a private
        method `_build_prompt` to assemble the required components into a
        cohesive prompt structure.

        Returns:
            BasePromptTemplate: A prompt template configured for executing tasks with memory.
        """
        return self._build_prompt(["role_playing", "tools", "memory", "task"])

    def task_execution_without_tools(self) -> BasePromptTemplate:
        """Generate a prompt for task execution without tools components.

        This function constructs a prompt by combining specific components such
        as role-playing and task-related elements. It utilizes an internal
        method `_build_prompt` to assemble the necessary parts of the prompt.

        Returns:
            BasePromptTemplate: The constructed prompt template.
        """
        return self._build_prompt(["role_playing", "task"])

    def task_execution(self) -> BasePromptTemplate:
        """Generate a standard prompt for task execution.

        This method constructs a prompt template by combining elements such as
        role playing, tools, and specific tasks. It invokes an internal helper
        method `_build_prompt` with a predefined list of components to generate
        the final prompt.

        Returns:
            BasePromptTemplate: The generated prompt template ready for use in task execution.
        """
        return self._build_prompt(["role_playing", "tools", "task"])

    def _build_prompt(self, components: list[str]) -> BasePromptTemplate:
        """Constructs a prompt string from specified components.

        This function takes a list of string components, slices each component
        using self.i18n.slice(), appends a scratchpad slice, and then joins all
        parts to form a final prompt template. The resulting prompt is returned
        as an instance of BasePromptTemplate.

        Args:
            components (list[str]): A list of string components to be included in the prompt.

        Returns:
            BasePromptTemplate: A BasePromptTemplate object constructed from the given components.
        """
        prompt_parts = [self.i18n.slice(component) for component in components]
        prompt_parts.append(self.SCRATCHPAD_SLICE)
        return PromptTemplate.from_template("".join(prompt_parts))
