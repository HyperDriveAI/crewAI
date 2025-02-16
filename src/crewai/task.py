import uuid
from typing import Any, List, Optional

from pydantic import UUID4, BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from crewai.agent import Agent
from crewai.tasks.task_output import TaskOutput
from crewai.utilities import I18N


class Task(BaseModel):
    """Class that represent a task to be executed."""

    __hash__ = object.__hash__  # type: ignore
    i18n: I18N = I18N()
    description: str = Field(description="Description of the actual task.")
    callback: Optional[Any] = Field(
        description="Callback to be executed after the task is completed.", default=None
    )
    agent: Optional[Agent] = Field(
        description="Agent responsible for execution the task.", default=None
    )
    expected_output: Optional[str] = Field(
        description="Clear definition of expected output for the task.",
        default=None,
    )
    context: Optional[List["Task"]] = Field(
        description="Other tasks that will have their output used as context for this task.",
        default=None,
    )
    output: Optional[TaskOutput] = Field(
        description="Task output, it's final result after being executed", default=None
    )
    tools: List[Any] = Field(
        default_factory=list,
        description="Tools the agent is limited to use for this task.",
    )
    id: UUID4 = Field(
        default_factory=uuid.uuid4,
        frozen=True,
        description="Unique identifier for the object, not set by user.",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _deny_user_set_id(cls, v: Optional[UUID4]) -> None:
        """Deny setting a user-defined ID for the class.

        This method checks if a user-defined ID is being set. If a value is
        provided, it raises a custom error indicating that the field should not
        be set by the user. This is typically used to enforce that certain
        fields are managed internally and should not be modified externally.

        Args:
            v (Optional[UUID4]): The user-defined ID to be set.

        Raises:
            PydanticCustomError: If a value is provided for the user-defined ID.
        """

        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def check_tools(self):
        """Check and initialize tools if they are not set.

        This method checks if the `tools` attribute is empty. If it is, and if
        the `agent` attribute exists and has its own tools, it extends the
        `tools` list with the agent's tools. This ensures that the object has
        the necessary tools available for its operations.

        Returns:
            self: The instance of the class, allowing for method chaining.
        """
        if not self.tools and self.agent and self.agent.tools:
            self.tools.extend(self.agent.tools)
        return self

    def execute(self, agent: Agent | None = None, context: Optional[str] = None) -> str:
        """Execute the task.

        This method executes a task using the specified agent. If no agent is
        provided, it defaults to the instance's agent. If the agent is not
        available, an exception is raised. The context can be provided to
        include additional information relevant to the task execution. The
        result of the task execution is stored and can be accessed through the
        output attribute.

        Args:
            agent (Agent | None): An optional agent to execute the task. If not provided, the instance's
                agent is used.
            context (Optional[str]): An optional string providing context for the task execution.

        Returns:
            str: The output of the executed task.

        Raises:
            Exception: If no agent is assigned to execute the task.
        """

        agent = agent or self.agent
        if not agent:
            raise Exception(
                f"The task '{self.description}' has no agent assigned, therefore it can't be executed directly and should be executed in a Crew using a specific process that support that, like hierarchical."
            )

        if self.context:
            context = "\n".join([task.output.result for task in self.context])

        result = self.agent.execute_task(
            task=self._prompt(), context=context, tools=self.tools
        )

        self.output = TaskOutput(description=self.description, result=result)
        self.callback(self.output) if self.callback else None
        return result

    def _prompt(self) -> str:
        """Prompt the task description and expected output.

        This method generates a prompt for the task by combining the task's
        description with the expected output, if provided. It constructs a list
        of strings that includes the description and, if applicable, the
        formatted expected output. Finally, it joins these strings with newline
        characters to create a single prompt string.

        Returns:
            str: The combined prompt of the task description and expected output.
        """
        tasks_slices = [self.description]

        if self.expected_output:
            output = self.i18n.slice("expected_output").format(
                expected_output=self.expected_output
            )
            tasks_slices = [self.description, output]
        return "\n".join(tasks_slices)
