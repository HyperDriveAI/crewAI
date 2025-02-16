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
        """Deny setting a user-defined ID for a specific field.

        This method checks if a value is provided for the user-defined ID. If a
        value is present, it raises a custom error indicating that this field
        should not be set by the user. This is typically used in scenarios where
        certain fields are managed internally and should not be modified by
        external input.

        Args:
            v (Optional[UUID4]): The value to be set for the user-defined ID, which is expected

        Raises:
            PydanticCustomError: If a value is provided, indicating that the field cannot be set
        """

        if v:
            raise PydanticCustomError(
                "may_not_set_field", "This field is not to be set by the user.", {}
            )

    @model_validator(mode="after")
    def check_tools(self):
        """Check if the tools are set.

        This method verifies whether the tools attribute is empty. If it is
        empty and the agent attribute is set and has its own tools, it extends
        the tools list with the agent's tools. This ensures that the current
        instance has the necessary tools available for its operations.

        Returns:
            self: The current instance of the class, allowing for method chaining.
        """
        if not self.tools and self.agent and self.agent.tools:
            self.tools.extend(self.agent.tools)
        return self

    def execute(self, agent: Agent | None = None, context: Optional[str] = None) -> str:
        """Execute the task.

        This method executes a task using the specified agent. If no agent is
        provided, it defaults to using the instance's agent. The method also
        handles the context by joining the results of previous tasks if
        available. If no agent is assigned, an exception is raised indicating
        that the task cannot be executed directly.

        Args:
            agent (Agent | None): The agent responsible for executing the task. If None, the instance's
                agent is used.
            context (Optional[str]): An optional context string that may provide additional information for
                task execution.

        Returns:
            str: The output of the executed task.

        Raises:
            Exception: If no agent is assigned to the task, indicating that it cannot be
                executed directly.
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

        This method constructs a prompt for the task by combining the task's
        description with its expected output, if provided. It formats the
        expected output using internationalization (i18n) support. The final
        prompt is returned as a single string, with each component separated by
        a newline.

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
