from langchain_core.exceptions import OutputParserException

from crewai.utilities import I18N


class TaskRepeatedUsageException(OutputParserException):
    """Exception raised when a task is used twice in a roll."""

    i18n: I18N = I18N()
    error: str = "TaskRepeatedUsageException"
    message: str

    def __init__(self, i18n: I18N, tool: str, tool_input: str, text: str):
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

        self.i18n = i18n
        self.text = text
        self.tool = tool
        self.tool_input = tool_input
        self.message = self.i18n.errors("task_repeated_usage").format(
            tool=tool, tool_input=tool_input
        )

        super().__init__(
            error=self.error,
            observation=self.message,
            send_to_llm=True,
            llm_output=self.text,
        )

    def __str__(self):
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

        return self.message
