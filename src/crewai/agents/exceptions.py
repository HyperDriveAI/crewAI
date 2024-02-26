from langchain_core.exceptions import OutputParserException

from crewai.utilities import I18N


class TaskRepeatedUsageException(OutputParserException):
    """Exception raised when a task is used twice in a roll."""

    i18n: I18N = I18N()
    error: str = "TaskRepeatedUsageException"
    message: str

    def __init__(self, i18n: I18N, tool: str, tool_input: str, text: str):
        """        Initialize the class instance with the provided parameters.

        Args:
            i18n (I18N): The internationalization object.
            tool (str): The tool being used.
            tool_input (str): The input for the tool.
            text (str): The text associated with the instance.
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
        """        Return a string representation of the object.

        This method returns the string representation of the object by returning the 'message' attribute.

        Returns:
            str: The string representation of the object.
        """

        return self.message
